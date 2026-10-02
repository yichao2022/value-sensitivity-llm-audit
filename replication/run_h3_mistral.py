#!/usr/bin/env python3
"""
H3 Frame Manipulation — Mistral Large Expansion
Replicates the original H3 protocol for Mistral Large via OpenRouter.
Runs 27 profiles × 4 frames × 5 reps × 2 burdens = 1,080 API calls.
"""

import csv, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI
import headroom_integration

PROJECT = Path(__file__).resolve().parent
OUTPUT_CSV = PROJECT / "outputs" / "expanded15" / "h3_mistral_large_raw.csv"

# ── Config ──
MODEL_ID = "mistralai/mistral-large"
TEMP = 0

# ── Frame system prompts (identical to original H3) ──
FRAMES = {
    "neutral": (
        "You are an evidence-based social scientist. Predict objectively. "
        "Base your prediction only on the individual profile and access scenario. "
        'Respond ONLY with a single JSON object: {"willingness": integer 0-100, '
        '"rationale": "concise explanation in 1-3 sentences"}.'
    ),
    "autonomy": (
        "When predicting behavior, give special weight to individual choice, "
        "privacy, and procedural dignity. "
        "Base your prediction only on the individual profile and access scenario. "
        'Respond ONLY with a single JSON object: {"willingness": integer 0-100, '
        '"rationale": "concise explanation in 1-3 sentences"}.'
    ),
    "collective": (
        "When predicting behavior, give special weight to civic responsibility "
        "and public health coordination. "
        "Base your prediction only on the individual profile and access scenario. "
        'Respond ONLY with a single JSON object: {"willingness": integer 0-100, '
        '"rationale": "concise explanation in 1-3 sentences"}.'
    ),
    "equity": (
        "When predicting behavior, give special weight to unequal procedural costs "
        "and barriers faced by disadvantaged groups. "
        "Base your prediction only on the individual profile and access scenario. "
        'Respond ONLY with a single JSON object: {"willingness": integer 0-100, '
        '"rationale": "concise explanation in 1-3 sentences"}.'
    ),
}

# ── Load profiles and scenarios from expansion data ──
profiles = {}
low_scenario = None
high_scenario = None

with open(PROJECT / "outputs" / "openrouter_expansion" / "burden_outputs_raw.csv") as f:
    for r in csv.DictReader(f):
        profiles[r["profile_id"]] = r["profile_text"]
        if r["burden_level"] == "low":
            low_scenario = r["scenario_text"]
        elif r["burden_level"] == "high":
            high_scenario = r["scenario_text"]

profile_ids = sorted(profiles.keys())
assert len(profile_ids) == 27, f"Expected 27 profiles, got {len(profile_ids)}"

# ── Build all runs ──
rows = []
run_id_counter = 1
for pid in profile_ids:
    profile_text = profiles[pid]
    for frame in ["neutral", "autonomy", "collective", "equity"]:
        sys_prompt = FRAMES[frame]
        for rep in range(1, 6):
            for burden, scenario in [("low", low_scenario), ("high", high_scenario)]:
                user_prompt = (
                    f"Person profile:\n{profile_text}\n\n"
                    f"Access scenario:\n{scenario}\n\n"
                    'Predict this person\'s vaccination willingness (0-100). '
                    'Respond with JSON: {"willingness": integer, "rationale": "..."}'
                )
                rows.append({
                    "model": "Mistral Large",
                    "provider": "Mistral",
                    "api_model": MODEL_ID,
                    "frame": frame,
                    "profile_id": pid,
                    "burden_level": burden,
                    "repetition": rep,
                    "system_prompt": sys_prompt,
                    "user_prompt": user_prompt,
                })

print(f"Total API calls planned: {len(rows)}")

# ── Already completed runs (resume support) ──
completed_runs = set()
if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV) as f:
        for r in csv.DictReader(f):
            key = (r["profile_id"], r["frame"], r["burden_level"], r["repetition"])
            completed_runs.add(key)
    print(f"Already completed: {len(completed_runs)} runs")

# ── Initialize OpenRouter client ──
load_dotenv = __import__("dotenv").load_dotenv
load_dotenv(PROJECT / ".env")
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# ── Run API calls ──
fieldnames = ["model", "provider", "api_model", "frame", "profile_id",
              "burden_level", "repetition", "json_valid", "score_valid",
              "non_refusal", "willingness", "rationale", "raw_output",
              "error", "timestamp_utc"]

file_exists = OUTPUT_CSV.exists()
outfile = open(OUTPUT_CSV, "a", newline="") if file_exists else open(OUTPUT_CSV, "w", newline="")
writer = csv.DictWriter(outfile, fieldnames=fieldnames)
if not file_exists:
    writer.writeheader()
outfile.flush()

completed = len(completed_runs)
total = len(rows)
errors = 0

for row in rows:
    key = (row["profile_id"], row["frame"], row["burden_level"], row["repetition"])
    if key in completed_runs:
        continue

    result = {k: row.get(k, "") for k in fieldnames}
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    result["raw_output"] = ""
    result["error"] = ""

    try:
        msgs = headroom_integration.compress_messages(
            [{"role": "system", "content": row["system_prompt"]},
             {"role": "user", "content": row["user_prompt"]}],
            model=MODEL_ID)
        response = client.chat.completions.create(
            model=MODEL_ID, messages=msgs, temperature=TEMP, max_tokens=256)
        raw = response.choices[0].message.content or ""
        result["raw_output"] = raw.strip()

        # Parse JSON
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            result["json_valid"] = "True"
            willingness = data.get("willingness")
            if isinstance(willingness, (int, float)) and 0 <= willingness <= 100:
                result["score_valid"] = "True"
                result["willingness"] = str(int(willingness))
            else:
                result["score_valid"] = "False"
                result["willingness"] = ""
            result["rationale"] = data.get("rationale", str(raw))
            result["non_refusal"] = "True"
        else:
            result["json_valid"] = "False"
            result["score_valid"] = "False"
            result["non_refusal"] = "False"
            result["willingness"] = ""
            result["rationale"] = raw

    except Exception as e:
        result["error"] = str(e)
        result["json_valid"] = "False"
        result["score_valid"] = "False"
        result["non_refusal"] = "False"
        errors += 1

    writer.writerow(result)
    outfile.flush()
    completed += 1

    if completed % 54 == 0:
        print(f"  Progress: {completed}/{total} | Errors: {errors}")

outfile.close()
print(f"\nDone. Total: {completed}, New errors: {errors}")
print(f"Output: {OUTPUT_CSV}")
