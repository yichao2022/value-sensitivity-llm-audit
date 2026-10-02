#!/usr/bin/env python3
"""
Policy-Value Orientation Battery Pipeline
Reads data/orientation_responses.csv → 1 API call per model (20 items) → fills scores.
Supports resumption from outputs/orientation_responses_filled.csv.
"""

import csv
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()
import headroom_integration

# ── Paths ────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "orientation_responses.csv"
OUTPUT_CSV = PROJECT / "outputs" / "orientation_responses_filled.csv"

# ── Provider configs ─────────────────────────────────────────────
PROVIDER_CONFIG = {
    "OpenAI": {
        "env_key": "OPENAI_API_KEY",
        "base_url": None,
        "temperature": 0,
        "extra_body": {},
    },
    "Anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": None,
        "temperature": 0,
        "extra_body": {},
    },
    "Google": {
        "env_key": "GEMINI_API_KEY",
        "base_url": None,
        "temperature": 0,
        "extra_body": {},
    },
    "DeepSeek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "temperature": 0,
        "extra_body": {},
    },
    "Qwen": {
        "env_key": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "qwen-max",
    },
    "Meta": {
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0,
        "extra_body": {},
        "model_override": "meta-llama/llama-3.1-70b-instruct",
    },
}

OPENAI_COMPATIBLE = {"OpenAI", "DeepSeek", "Qwen", "Meta"}

# ── Prompts ──────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a policy analyst completing a survey. "
    "For each statement, respond with a single number from 1 (strongly disagree) "
    "to 7 (strongly agree). Output only valid JSON."
)

MAX_RETRIES = 5
BASE_DELAY = 2
MAX_DELAY = 120


def load_input() -> list[dict]:
    """Read orientation_responses.csv and group items by model."""
    rows = []
    with open(INPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def group_by_model(rows: list[dict]) -> list[dict]:
    """Group rows into per-model batches. Returns list of dicts with model info + items."""
    groups = defaultdict(lambda: {"model": "", "provider": "", "api_model": "", "items": []})
    for r in rows:
        key = r["model"]
        groups[key]["model"] = r["model"]
        groups[key]["provider"] = r["provider"]
        groups[key]["api_model"] = r["api_model"]
        groups[key]["items"].append({
            "item_order": int(r["item_order"]),
            "item_id": r["item_id"],
            "item_text": r["item_text"],
        })
    # Sort items by item_order
    result = []
    for model, group in groups.items():
        group["items"].sort(key=lambda x: x["item_order"])
        result.append(group)
    # Sort by model name for deterministic order
    result.sort(key=lambda x: x["model"])
    return result


def build_user_prompt(items: list[dict]) -> str:
    """Build the user prompt listing all 20 items."""
    lines = ["Please rate the following 20 statements on a scale from 1 (strongly disagree) to 7 (strongly agree).\n"]
    for item in items:
        lines.append(f"{item['item_order']}. [{item['item_id']}] {item['item_text']}")
    lines.append("\nRespond ONLY with a single JSON object mapping item_id to score, like:")
    lines.append('{"AP1": 6, "BS2": 3, ...}')
    return "\n".join(lines)


def load_completed_models() -> set:
    """Return set of model names already present in the output CSV with scores."""
    if not OUTPUT_CSV.exists():
        return set()
    completed = set()
    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("raw_score", "").strip():
                completed.add(row["model"])
    return completed


# ── API call dispatchers ─────────────────────────────────────────

def call_openai_compatible(group: dict, config: dict) -> str:
    from openai import OpenAI
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing env var: {config['env_key']}")
    client_kwargs = {"api_key": api_key}
    if config["base_url"]:
        client_kwargs["base_url"] = config["base_url"]
    client = OpenAI(**client_kwargs)

    model_id = config.get("model_override", group["api_model"])
    items = group["items"]
    user_prompt = build_user_prompt(items)

    msgs = headroom_integration.compress_messages(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": user_prompt}],
        model=model_id)
    response = client.chat.completions.create(
        model=model_id, max_tokens=1024, messages=msgs,
        temperature=config["temperature"],
    )
    return response.choices[0].message.content or ""


def call_anthropic(group: dict, config: dict) -> str:
    from anthropic import Anthropic
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing env var: {config['env_key']}")
    client = Anthropic(api_key=api_key)

    items = group["items"]
    user_prompt = build_user_prompt(items)

    msgs = headroom_integration.compress_messages(
        [{"role": "user", "content": user_prompt}],
        model=group["api_model"])
    response = client.messages.create(
        model=group["api_model"], max_tokens=1024,
        temperature=config["temperature"], system=SYSTEM_PROMPT,
        messages=msgs,
    )
    return response.content[0].text if response.content else ""


def call_google(group: dict, config: dict) -> str:
    from google import genai
    api_key = os.environ.get(config["env_key"])
    if not api_key:
        raise RuntimeError(f"Missing env var: {config['env_key']}")
    client = genai.Client(api_key=api_key)

    items = group["items"]
    user_prompt = build_user_prompt(items)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    response = client.models.generate_content(
        model=group["api_model"],
        contents=full_prompt,
        config={"temperature": config["temperature"]},
    )
    return response.text if response.text else ""


def call_api(group: dict) -> str:
    provider = group["provider"]
    config = PROVIDER_CONFIG.get(provider)
    if not config:
        raise ValueError(f"Unknown provider: {provider}")
    if provider == "Anthropic":
        return call_anthropic(group, config)
    elif provider == "Google":
        return call_google(group, config)
    elif provider in OPENAI_COMPATIBLE:
        return call_openai_compatible(group, config)
    else:
        raise ValueError(f"No handler for: {provider}")


# ── JSON parsing ─────────────────────────────────────────────────

def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from raw response. Returns dict of item_id → score or None."""
    candidate = raw.strip()
    # Try fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
    else:
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m:
            candidate = m.group(0).strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return {}


def process_model(group: dict) -> list[dict]:
    """Run one model batch and return filled rows."""
    # Build ordered items list
    items = sorted(group["items"], key=lambda x: x["item_order"])

    raw = ""
    last_error = ""
    parsed = {}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_api(group)
            parsed = parse_json_response(raw)
            if parsed:
                break
            last_error = "JSON parse failed"
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
                print(f"  [retry {attempt}/{MAX_RETRIES}] {last_error[:120]} — waiting {delay}s")
                time.sleep(delay)
            else:
                print(f"  [FAILED] {last_error[:200]}")

    ts = datetime.now(timezone.utc).isoformat()
    results = []

    for item in items:
        item_id = item["item_id"]
        score = parsed.get(item_id)
        if isinstance(score, (int, float)) and 1 <= score <= 7:
            raw_score = int(score)
            score_valid = 1
        else:
            raw_score = ""
            score_valid = 0

        results.append({
            "model": group["model"],
            "provider": group["provider"],
            "deployment": "official_api",
            "api_model": group["api_model"],
            "item_order": item["item_order"],
            "item_id": item_id,
            "dimension": "",  # filled from original CSV later
            "reverse_coded": "",  # filled from original CSV later
            "coding_direction": "",
            "item_text": item["item_text"],
            "raw_score": raw_score,
            "score_valid": score_valid,
            "score_recoded": "",
            "_raw_output": raw,
            "_error": last_error,
            "_timestamp": ts,
        })

    return results


def main():
    rows = load_input()
    groups = group_by_model(rows)
    completed = load_completed_models()

    print(f"Models: {len(groups)}, already completed: {len(completed)}")

    # Build a lookup of original rows to get dimension/reverse_coded
    original_lookup = {}
    for r in rows:
        original_lookup[(r["model"], r["item_id"])] = r

    all_results = []

    for group in groups:
        model = group["model"]
        if model in completed:
            print(f"[SKIP] {model} — already completed")
            # Re-read existing rows for this model
            with open(OUTPUT_CSV, "r", newline="") as f:
                for row in csv.DictReader(f):
                    if row["model"] == model:
                        all_results.append(row)
            continue

        print(f"[RUN] {model} ({group['provider']}, {group['api_model']})")
        results = process_model(group)

        # Fill dimension/reverse_coded from original CSV
        for r in results:
            orig = original_lookup.get((r["model"], r["item_id"]), {})
            r["dimension"] = orig.get("dimension", "")
            r["reverse_coded"] = orig.get("reverse_coded", "")
            r["coding_direction"] = orig.get("coding_direction", "")

        all_results.extend(results)

        # Write incremental output
        fieldnames = [
            "model", "provider", "deployment", "api_model",
            "item_order", "item_id", "dimension", "reverse_coded", "coding_direction",
            "item_text", "raw_score", "score_valid", "score_recoded",
        ]
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)

        # Small delay
        time.sleep(0.3)

    print(f"\nDone. {len(all_results)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
