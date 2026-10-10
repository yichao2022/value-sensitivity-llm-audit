#!/usr/bin/env python3
"""Run H3 placebo experiment. Resumes rows that already have a response."""
import csv
import os
import time
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()
HOME = Path(__file__).resolve().parent
PLAN_PATH = HOME / "data" / "h3_placebo_plan.csv"
OUTPUT_PATH = HOME / "outputs" / "h3_placebo_results.csv"
FIELDS = [
    "run_id", "model", "frame", "profile_id", "burden_level", "repetition",
    "raw_response", "tokens_in", "tokens_out", "error",
]
# Plan stores Together / native IDs. The calls that previously succeeded
# used these OpenRouter IDs.
MODEL_IDS = {
    "Llama 3.1 70B Instruct": "meta-llama/llama-3.1-70b-instruct",
    "Mistral Large": "mistralai/mistral-large",
}
OPENROUTER = "https://openrouter.ai/api/v1"


def endpoint(provider):
    if provider == "OpenAI":
        return os.getenv("OPENAI_API_KEY"), None
    if provider == "Meta":
        return os.getenv("OPENROUTER_API_KEY"), OPENROUTER
    if provider == "Mistral":
        # .env has no MISTRAL_API_KEY; prior successful runs used OpenRouter.
        return os.getenv("MISTRAL_API_KEY") or os.getenv("OPENROUTER_API_KEY"), OPENROUTER
    if provider == "Qwen":
        return os.getenv("DASHSCOPE_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return None, None


def load_done():
    done = {}
    if not OUTPUT_PATH.exists():
        return done
    with open(OUTPUT_PATH, newline="") as f:
        for row in csv.DictReader(f):
            prev = done.get(row["run_id"])
            if prev is None or len(row.get("raw_response") or "") > len(prev.get("raw_response") or ""):
                done[row["run_id"]] = row
    return done


def usable(row):
    return bool((row.get("raw_response") or "").strip()) and not (row.get("error") or "").strip()


def save(rows):
    tmp = OUTPUT_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(OUTPUT_PATH)


def call(row, api_key, base_url, api_model):
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    headers = {}
    if base_url and "openrouter" in base_url:
        headers = {
            "HTTP-Referer": "https://github.com/yichao2022/value-sensitivity-llm-audit",
            "X-Title": "LLM Policy Simulation Study",
        }
    last = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=api_model,
                messages=[
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]},
                ],
                temperature=0,
                max_tokens=500,
                extra_headers=headers,
            )
            return {
                "run_id": row["run_id"],
                "model": row["model"],
                "frame": row["frame"],
                "profile_id": row["profile_id"],
                "burden_level": row["burden_level"],
                "repetition": row["repetition"],
                "raw_response": response.choices[0].message.content or "",
                "tokens_in": response.usage.prompt_tokens if response.usage else 0,
                "tokens_out": response.usage.completion_tokens if response.usage else 0,
                "error": "",
            }
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    return {
        "run_id": row["run_id"],
        "model": row["model"],
        "frame": row["frame"],
        "profile_id": row["profile_id"],
        "burden_level": row["burden_level"],
        "repetition": row["repetition"],
        "raw_response": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "error": str(last),
    }


def main():
    with open(PLAN_PATH, newline="") as f:
        plan = list(csv.DictReader(f))
    done = load_done()
    pending = [r for r in plan if not usable(done.get(r["run_id"], {}))]
    print(f"Plan {len(plan)}, already ok {len(plan) - len(pending)}, pending {len(pending)}", flush=True)

    completed = failed = 0
    for i, row in enumerate(pending, 1):
        api_key, base_url = endpoint(row["provider"])
        api_model = MODEL_IDS.get(row["model"], row["api_model"])
        if not api_key:
            result = {k: row.get(k, "") for k in FIELDS}
            result["raw_response"] = ""
            result["tokens_in"] = result["tokens_out"] = 0
            result["error"] = f"No API key for {row['provider']}"
        else:
            result = call(row, api_key, base_url, api_model)
        done[row["run_id"]] = result
        if result["error"]:
            failed += 1
            print(f"FAIL {row['run_id']} {api_model}: {result['error'][:160]}", flush=True)
        else:
            completed += 1
        if i % 10 == 0 or i == len(pending):
            save([done[r["run_id"]] for r in plan if r["run_id"] in done])
            print(f"Progress {i}/{len(pending)} ok {completed} failed {failed}", flush=True)
        time.sleep(0.3)

    print(f"Complete: {completed} succeeded, {failed} failed", flush=True)


if __name__ == "__main__":
    main()
