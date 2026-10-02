#!/usr/bin/env python3
"""
LLM Policy Simulation Pipeline
Reads data/llm_experiment_inputs.xlsx → calls provider APIs → saves outputs/simulation_outputs.csv
Supports resumption from partial outputs via run_id.
"""

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import openpyxl
from dotenv import load_dotenv

load_dotenv()
import headroom_integration

# ── Paths ────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
INPUT_XLSX = PROJECT / "data" / "llm_experiment_inputs.xlsx"
OUTPUT_CSV = PROJECT / "outputs" / "simulation_outputs.csv"

# ── Provider configs ─────────────────────────────────────────────
# Each provider gets a client factory + model-specific settings.
# API keys come from environment variables.

PROVIDER_CONFIG = {
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,  # uses default
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
    },
    "Anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
    },
    "Google": {
        "env_key": "GEMINI_API_KEY",
        "base_url": None,
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
    },
    "Qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
        "model_override": "qwen-max",  # qwen3.6-72b-instruct not on DashScope; use flagship
    },
    "Meta": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
        "model_override": "meta-llama/llama-3.1-70b-instruct",  # xlsx has Together AI naming
    },
    "MiMo": {
        "env_key": "MIMO_API_KEY",
        "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
        "model_override": "mimo-v2.5-pro",
    },
    "LMStudio": {
        "env_key": "LMSTUDIO_API_KEY",
        "base_url": "http://127.0.0.1:1234/v1",
        "temperature_field": "temperature",
        "temperature": 0,
        "extra_body": {},
        "model_override": "llama-3.3-70b-instruct-abliterated",
    },
}

# Providers that use the OpenAI client shape (/v1/chat/completions)
OPENAI_COMPATIBLE = {"OpenAI", "DeepSeek", "Qwen", "Meta", "MiMo", "LMStudio"}

# ── Prompt template ──────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a study participant in a public-health policy simulation. "
    "You will be given a description of a person and a scenario about vaccine appointment registration. "
    "You must role-play that person and decide their willingness to get vaccinated given the scenario. "
    "Respond ONLY with a single JSON object and nothing else. "
    'The JSON must have exactly two keys: "willingness" (an integer from 0 to 100) '
    'and "rationale" (a concise string explaining the decision in 1-3 sentences).'
)

USER_PROMPT_TEMPLATE = """Person profile:
{profile_text}

Scenario:
{burden_text}

Based on the person described above, what is their willingness (0-100) to proceed with vaccination given this registration scenario?"""

# ── Output columns ───────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "run_id", "model", "provider", "api_model",
    "json_valid", "score_valid", "non_refusal",
    "willingness", "rationale", "raw_output", "error", "timestamp_utc",
]

# ── Rate-limit / retry config ────────────────────────────────────
MAX_RETRIES = 5
BASE_DELAY = 2  # seconds
MAX_DELAY = 120  # seconds


def load_completed_runs() -> set:
    """Return set of run_ids already present in the output CSV."""
    if not OUTPUT_CSV.exists():
        return set()
    completed = set()
    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only skip rows that succeeded (no error)
            if not row.get("error", "").strip():
                completed.add(row.get("run_id", ""))
    return completed


def read_simulation_plan(xlsx_path: Path) -> list[dict]:
    """Read the 'Simulation Plan' sheet and return rows as list of dicts."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Simulation Plan"]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) for h in rows[0]]
    data = []
    for row in rows[1:]:
        record = dict(zip(headers, row))
        data.append(record)
    wb.close()
    return data


def build_prompt(record: dict) -> str:
    """Build the user prompt from profile_text and burden_text."""
    return USER_PROMPT_TEMPLATE.format(
        profile_text=record["profile_text"],
        burden_text=record["scenario_text"],
    )


def parse_response(raw: str) -> dict:
    """
    Attempt to extract a JSON object from the raw response.
    Returns {"json_valid": bool, "score_valid": bool, "non_refusal": bool,
             "willingness": int|None, "rationale": str}
    """
    result = {
        "json_valid": False,
        "score_valid": False,
        "non_refusal": True,
        "willingness": None,
        "rationale": "",
    }
    if not raw:
        return result

    # Try to find JSON block
    candidate = raw.strip()
    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
    else:
        # Try to find first { ... }
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            candidate = m.group(0).strip()

    try:
        parsed = json.loads(candidate)
        result["json_valid"] = True
        if "willingness" in parsed:
            w = parsed["willingness"]
            if isinstance(w, (int, float)) and 0 <= w <= 100:
                result["score_valid"] = True
                result["willingness"] = int(w)
        if "rationale" in parsed:
            result["rationale"] = str(parsed["rationale"])
    except (json.JSONDecodeError, TypeError):
        result["json_valid"] = False

    # Detect refusals (model declines to role-play).
    # Avoid matching persona role-playing ("I cannot afford..." is legitimate).
    refusal_patterns = [
        r"cannot\s+(fulfil|role.play|simulate|provide|comply|assist)",
        r"unable\s+to\s+(fulfil|role.play|simulate|provide|comply|assist)",
        r"as\s+an\s+(AI|artificial\s+intelligence|language\s+model)",
        r"i\s+(?:am\s+)?(?:not\s+)?(?:able|programmed|designed)\s+to\s+(?:role.play|simulate|pretend)",
        r"i\s+don'?t\s+feel\s+comfortable\s+(?:role.play|simulating|pretending)",
        r"against\s+(?:my|the)\s+(?:guidelines|policies|ethical)",
    ]
    lower_raw = raw.lower()
    if any(re.search(p, lower_raw) for p in refusal_patterns):
        result["non_refusal"] = False

    return result


def call_openai_compatible(record: dict, config: dict) -> dict:
    """Call OpenAI-compatible APIs (OpenAI, DeepSeek, DashScope, Together)."""
    from openai import OpenAI

    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing env var: {config['env_key']}")

    client_kwargs = {"api_key": api_key}
    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]
    client = OpenAI(**client_kwargs)

    system_prompt = SYSTEM_PROMPT
    user_prompt = build_prompt(record)

    # Allow provider config to override the model ID
    model_id = config.get("model_override", record["api_model"])
    kwargs = {
        "model": model_id,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        config["temperature_field"]: config["temperature"],
    }
    if config.get("extra_body"):
        kwargs["extra_body"] = config["extra_body"]

    # Gemini via OpenAI-compatible needs special handling if it's Google provider
    if record["provider"] == "Google":
        # For Gemini, we use google-genai directly, but if using OpenAI compat...
        # We handle Google separately below.
        pass

    kwargs["messages"] = headroom_integration.compress_messages(kwargs["messages"], model=model_id)
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content or ""
    return {"raw_output": content}


def call_anthropic(record: dict, config: dict) -> dict:
    """Call Anthropic Messages API."""
    from anthropic import Anthropic

    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing env var: {config['env_key']}")

    client = Anthropic(api_key=api_key)

    system_prompt = SYSTEM_PROMPT
    user_prompt = build_prompt(record)

    msgs = headroom_integration.compress_messages(
        [{"role": "user", "content": user_prompt}],
        model=record["api_model"],
    )
    response = client.messages.create(
        model=record["api_model"],
        max_tokens=512,
        temperature=config["temperature"],
        system=system_prompt,
        messages=msgs,
    )
    content = response.content[0].text if response.content else ""
    return {"raw_output": content}


def call_google(record: dict, config: dict) -> dict:
    """Call Google Gemini API via google.genai (current SDK)."""
    from google import genai

    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing env var: {config['env_key']}")

    system_prompt = SYSTEM_PROMPT
    user_prompt = build_prompt(record)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=record["api_model"],
        contents=full_prompt,
        config={
            "temperature": config["temperature"],
            # Note: max_output_tokens omitted — google.genai SDK has a bug
            # where setting it causes immediate MAX_TOKENS truncation.
        },
    )
    content = response.text if response.text else ""
    return {"raw_output": content}


def call_api(record: dict) -> dict:
    """Dispatch to the correct provider and return raw_output."""
    provider = record["provider"]
    config = PROVIDER_CONFIG.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")

    if provider == "Anthropic":
        return call_anthropic(record, config)
    elif provider == "Google":
        return call_google(record, config)
    elif provider in OPENAI_COMPATIBLE:
        return call_openai_compatible(record, config)
    else:
        raise ValueError(f"No call handler for provider: {provider}")


def run_one(record: dict) -> dict:
    """Run one simulation row with retries. Returns the output row dict."""
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            api_result = call_api(record)
            raw = api_result["raw_output"]
            parsed = parse_response(raw)
            return {
                "run_id": record["run_id"],
                "model": record["model"],
                "provider": record["provider"],
                "api_model": record["api_model"],
                "json_valid": parsed["json_valid"],
                "score_valid": parsed["score_valid"],
                "non_refusal": parsed["non_refusal"],
                "willingness": parsed["willingness"] if parsed["willingness"] is not None else "",
                "rationale": parsed["rationale"],
                "raw_output": raw,
                "error": "",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
                print(f"  [retry {attempt}/{MAX_RETRIES}] {last_error[:120]} — waiting {delay}s")
                time.sleep(delay)
            else:
                print(f"  [FAILED after {MAX_RETRIES} retries] {last_error[:200]}")

    return {
        "run_id": record["run_id"],
        "model": record["model"],
        "provider": record["provider"],
        "api_model": record["api_model"],
        "json_valid": False,
        "score_valid": False,
        "non_refusal": False,
        "willingness": "",
        "rationale": "",
        "raw_output": "",
        "error": last_error,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def append_to_csv(row: dict):
    """Append a single result row to the output CSV."""
    file_exists = OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    # Check that at least one API key is set
    providers_needed = set()
    plan = read_simulation_plan(INPUT_XLSX)
    for row in plan:
        providers_needed.add(row["provider"])
    print(f"Providers needed: {providers_needed}")

    # Warn about missing keys
    for provider in providers_needed:
        config = PROVIDER_CONFIG.get(provider)
        if config and not os.environ.get(config["env_key"]):
            print(f"⚠  WARNING: {config['env_key']} not set — {provider} calls will fail")

    completed = load_completed_runs()
    print(f"Already completed: {len(completed)} rows")
    print(f"Total plan: {len(plan)} rows")

    n_done = 0
    n_skipped = 0
    for i, record in enumerate(plan):
        rid = record["run_id"]
        if rid in completed:
            n_skipped += 1
            continue

        print(f"[{i+1}/{len(plan)}] {rid} | {record['model']} | profile={record['profile_id']} burden={record['burden_level']} rep={record['repetition']}")
        result = run_one(record)
        append_to_csv(result)
        n_done += 1

        # Small delay between calls to be gentle on APIs
        time.sleep(0.2)

    print(f"\nDone. New: {n_done}, Skipped: {n_skipped}, Total: {len(plan)}")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
