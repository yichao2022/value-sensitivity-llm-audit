#!/usr/bin/env python3
"""Generate frame_manipulation_plan.csv for H3 within-model frame manipulation."""

import csv, openpyxl
from itertools import product
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
INPUT_XLSX = PROJECT / "data" / "llm_experiment_inputs.xlsx"

# ── Load data ────────────────────────────────────────────────────

wb = openpyxl.load_workbook(INPUT_XLSX, read_only=True)

# Models
ws = wb['Models']
mheaders = [str(h) for h in next(ws.iter_rows(values_only=True))]
models = [dict(zip(mheaders, row)) for row in ws.iter_rows(values_only=True)]
# Filter: GPT-4.1, Llama 3.1 70B Instruct, Qwen3.7 Plus
h3_models = [m for m in models if m['model'] in (
    'GPT-4.1', 'Llama 3.1 70B Instruct', 'Qwen3.7 Plus'
)]
print(f"H3 models: {[m['model'] for m in h3_models]}")

# Profiles
ws = wb['Profiles']
prows = list(ws.iter_rows(values_only=True))
pheaders = [str(h) for h in prows[0]]
profiles = [dict(zip(pheaders, row)) for row in prows[1:]]
print(f"Profiles: {len(profiles)}")

# Burden Scenarios
ws = wb['Burden Scenarios']
srows = list(ws.iter_rows(values_only=True))
bheaders = [str(h) for h in srows[0]]
scenarios = [dict(zip(bheaders, row)) for row in srows[1:]]
print(f"Scenarios: {len(scenarios)}")

wb.close()

# ── Frames ───────────────────────────────────────────────────────

FRAMES = {
    "neutral": (
        "You are an evidence-based social scientist. Predict objectively. "
        "Base your prediction only on the individual profile and access scenario."
    ),
    "autonomy": (
        "You are an evidence-based social scientist. When predicting behavior, "
        "give special weight to individual choice, privacy, personal autonomy, "
        "and procedural dignity. Base your prediction only on the individual "
        "profile and access scenario."
    ),
    "collective": (
        "You are an evidence-based social scientist. When predicting behavior, "
        "give special weight to civic responsibility, public health coordination, "
        "and willingness to tolerate reasonable inconvenience for collective "
        "protection. Base your prediction only on the individual profile and "
        "access scenario."
    ),
    "equity": (
        "You are an evidence-based social scientist. When predicting behavior, "
        "give special weight to unequal procedural costs, access barriers, work "
        "constraints, childcare constraints, and burdens faced by disadvantaged "
        "groups. Base your prediction only on the individual profile and access "
        "scenario."
    ),
}

# User prompt template (same as main simulation)
USER_PROMPT_TEMPLATE = """Person profile:
{profile_text}

Scenario:
{burden_text}

Based on the person described above, what is their willingness (0-100) to proceed with vaccination given this registration scenario?"""

# ── Generate plan ────────────────────────────────────────────────

REPETITIONS = 5

rows = []
run_id = 1

for model_info in h3_models:
    for frame_name in ["neutral", "autonomy", "collective", "equity"]:
        for profile in profiles:
            for scenario in scenarios:
                for rep in range(1, REPETITIONS + 1):
                    user_prompt = USER_PROMPT_TEMPLATE.format(
                        profile_text=profile['profile_text'],
                        burden_text=scenario['scenario_text'],
                    )
                    rows.append({
                        'run_id': f"H3_{run_id:05d}",
                        'model': model_info['model'],
                        'provider': model_info['provider'],
                        'deployment': model_info['deployment'],
                        'api_model': model_info['api_model'],
                        'access_channel': model_info.get('access_channel', ''),
                        'frame': frame_name,
                        'system_prompt': FRAMES[frame_name],
                        'profile_id': profile['profile_id'],
                        'profile_text': profile['profile_text'],
                        'burden_level': scenario['burden_level'],
                        'scenario_text': scenario['scenario_text'],
                        'repetition': rep,
                        'user_prompt': user_prompt,
                    })
                    run_id += 1

# Save
fields = ['run_id', 'model', 'provider', 'deployment', 'api_model', 'access_channel',
          'frame', 'system_prompt', 'profile_id', 'profile_text',
          'burden_level', 'scenario_text', 'repetition', 'user_prompt']

with open(PROJECT / "data" / "frame_manipulation_plan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print(f"\nframe_manipulation_plan.csv: {len(rows)} rows")
# Verify
from collections import Counter
mc = Counter((r['model'], r['frame']) for r in rows)
for k, c in sorted(mc.items()):
    print(f"  {k[0]:<30} {k[1]:<12} {c}")
print(f"Total: {sum(mc.values())}")
print(f"Expected: {len(h3_models)} × {len(FRAMES)} × {len(profiles)} × {len(scenarios)} × {REPETITIONS} = {len(h3_models)*len(FRAMES)*len(profiles)*len(scenarios)*REPETITIONS}")
