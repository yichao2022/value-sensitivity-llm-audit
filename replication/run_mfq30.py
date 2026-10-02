#!/usr/bin/env python3
"""
MFQ-30 Fielding Pipeline
Reads data/mfq30_items.csv → 1 API call per model (30 scored + 2 catch items) → fills scores.
Supports resumption from outputs/mfq30_responses_filled.csv.
Response scale: 1-7 (strongly disagree → strongly agree), matching the PVOC pipeline.

Usage:
    python3 run_mfq30.py
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

# ── Paths ────────────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "data" / "mfq30_items.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_responses_filled.csv"

# ── Provider configs (same as run_orientation.py) ───────────────
PROVIDER_CONFIG = {
    "OpenAI": {"env_key": "OPENAI_API_KEY", "base_url": None, "temperature": 0, "extra_body": {}},
    "Anthropic": {"env_key": "ANTHROPIC_API_KEY", "base_url": None, "temperature": 0, "extra_body": {}},
    "Google": {"env_key": "GEMINI_API_KEY", "base_url": None, "temperature": 0, "extra_body": {}},
    "DeepSeek": {"env_key": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com", "temperature": 0, "extra_body": {}},
    "Qwen": {"env_key": "DASHSCOPE_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "temperature": 0, "extra_body": {}, "model_override": "qwen-max"},
    "Meta": {"env_key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1", "temperature": 0, "extra_body": {}, "model_override": "meta-llama/llama-3.1-70b-instruct"},
}

OPENAI_COMPATIBLE = {"OpenAI", "DeepSeek", "Qwen", "Meta"}

SYSTEM_PROMPT = (
    "You are a research participant completing a moral attitudes survey. "
    "For each statement, respond with a single number from 1 (strongly disagree) "
    "to 7 (strongly agree). Output only valid JSON."
)

MAX_RETRIES = 5
BASE_DELAY = 2
MAX_DELAY = 120


def load_input() -> list[dict]:
    with open(INPUT_CSV, "r", newline="") as f:
        return list(csv.DictReader(f))


def group_by_model(rows: list[dict]) -> list[dict]:
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
            "format": r["format"],
        })
    result = []
    for model, group in groups.items():
        group["items"].sort(key=lambda x: x["item_order"])
        result.append(group)
    result.sort(key=lambda x: x["model"])
    return result


def build_user_prompt(items: list[dict]) -> str:
    """MFQ-30 has two parts: relevance (R) and judgment (J). We use a single
    1-7 agreement scale; the instruction states relevance items ask how much
    the consideration matters when judging right/wrong."""
    lines = [
        "Part 1. When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking?",
        "Rate each from 1 (not at all relevant) to 7 (extremely relevant).",
        "",
        "Part 2. Please indicate your agreement with the following statements.",
        "Rate each from 1 (strongly disagree) to 7 (strongly agree).",
        "",
        "Rate ALL items below with a single number 1-7.",
    ]
    for item in items:
        part = "R" if item["format"] == "relevance" else "J"
        lines.append(f"{part}{item['item_order']}. [{item['item_id']}] {item['item_text']}")
    lines.append("\nRespond ONLY with a single JSON object mapping item_id to score, like:")
    lines.append('{"CA01": 6, "FA02": 3, ...}')
    return "\n".join(lines)


def load_completed_models() -> set:
    if not OUTPUT_CSV.exists():
        return set()
    completed = set()
    with open(OUTPUT_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("raw_score", "").strip():
                completed.add(row["model"])
    return completed


# ── API call dispatchers (identical to run_orientation.py) ──────

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
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}]
    kwargs = {}
    # Kimi K2.6 needs reasoning disabled to return JSON directly (see EXPANSION_SUMMARY)
    if "kimi" in model_id.lower() and "moonshot" in group.get("api_model", "").lower():
        kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    response = client.chat.completions.create(
        model=model_id, max_tokens=2048, messages=msgs,
        temperature=config["temperature"], **kwargs,
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
    msgs = [{"role": "user", "content": user_prompt}]
    response = client.messages.create(
        model=group["api_model"], max_tokens=2048,
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


def parse_json_response(raw: str) -> dict:
    candidate = raw.strip()
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
            "dimension": "",
            "dimension_name": "",
            "format": item["format"],
            "reverse_coded": "",
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

    original_lookup = {}
    for r in rows:
        original_lookup[(r["model"], r["item_id"])] = r

    all_results = []
    written_models = set()

    for group in groups:
        model = group["model"]
        if model in completed:
            print(f"[SKIP] {model} — already completed")
            with open(OUTPUT_CSV, "r", newline="") as f:
                for row in csv.DictReader(f):
                    if row["model"] == model:
                        all_results.append(row)
            written_models.add(model)
            continue

        print(f"[RUN] {model} ({group['provider']}, {group['api_model']})")
        results = process_model(group)

        for r in results:
            orig = original_lookup.get((r["model"], r["item_id"]), {})
            r["dimension"] = orig.get("dimension", "")
            r["dimension_name"] = orig.get("dimension_name", "")
            r["reverse_coded"] = orig.get("reverse_coded", "")
            r["coding_direction"] = orig.get("coding_direction", "")

        all_results.extend(results)
        written_models.add(model)

        # 每处理完一个模型就写一次（增量，避免中断丢全部）
        fieldnames = [
            "model", "provider", "deployment", "api_model",
            "item_order", "item_id", "dimension", "dimension_name", "format",
            "reverse_coded", "coding_direction",
            "item_text", "raw_score", "score_valid", "score_recoded",
        ]
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)

        time.sleep(0.3)

    print(f"\nDone. {len(all_results)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
