#!/usr/bin/env python3
"""Burden simulation for 6 OpenRouter expansion models. 1620 calls total."""

import csv, json, os, re, time, requests, openpyxl
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import headroom_integration
PROJECT = Path(__file__).resolve().parent
KEY = os.environ['OPENROUTER_API_KEY']
OR_BASE = "https://openrouter.ai/api/v1"

# Load models
MODELS = list(csv.DictReader(open(PROJECT / "inputs/models_expanded_openrouter_additions.csv")))

# Load profiles and scenarios from xlsx
wb = openpyxl.load_workbook(PROJECT / "data/llm_experiment_inputs.xlsx", read_only=True)
ws = wb['Profiles']; prows = list(ws.iter_rows(values_only=True))
profiles = [dict(zip([str(h) for h in prows[0]], row)) for row in prows[1:]]
ws = wb['Burden Scenarios']; srows = list(ws.iter_rows(values_only=True))
scenarios = [dict(zip([str(h) for h in srows[0]], row)) for row in srows[1:]]
wb.close()

# Prompts
SYSTEM_PROMPT = (
    "You are an evidence-based social scientist. Given a person's profile and "
    "a vaccine appointment registration scenario, predict how willing this person "
    "would be to get vaccinated. Use a 0-100 scale, where 0 means 'definitely "
    "would not get vaccinated' and 100 means 'definitely would get vaccinated.' "
    "Return only valid JSON in the following format: "
    '{"willingness": 72, "rationale": "..."} '
    "The willingness value must be an integer from 0 to 100. The rationale should "
    "be one to three sentences."
)

USER_TEMPLATE = """Person profile:
{profile_text}

Scenario:
{scenario_text}

Based on the person described above, what is their willingness (0-100) to proceed with vaccination given this registration scenario?"""

REPETITIONS = 5
OUT_FILE = PROJECT / "outputs/openrouter_expansion/burden_outputs_raw.csv"
OUT_FIELDS = ['model','provider','access_route','requested_model_id','returned_model_id',
              'route_provider','profile_id','burden_level','repetition',
              'profile_text','scenario_text','raw_response',
              'json_valid','willingness','score_valid','rationale','non_refusal',
              'timestamp','temperature','top_p','metadata_json']

MAX_RETRIES = 4

def call_or(model_id, model_name):
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ""},  # filled per call
        ],
        "temperature": 0, "max_tokens": 500,
    }
    if 'kimi' in model_name.lower():
        body['reasoning'] = {'enabled': False}
    return headers, body

def parse_response(raw):
    if not raw: return {"json_valid":False,"score_valid":False,"non_refusal":True,"willingness":"","rationale":""}
    result = {"json_valid":False,"score_valid":False,"non_refusal":True,"willingness":"","rationale":""}
    candidate = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if m: candidate = m.group(1).strip()
    else:
        m = re.search(r"\{.*\}", candidate, re.DOTALL)
        if m: candidate = m.group(0).strip()
    try:
        p = json.loads(candidate); result["json_valid"] = True
        if "willingness" in p:
            w = p["willingness"]
            if isinstance(w,(int,float)) and 0<=w<=100:
                result["score_valid"]=True; result["willingness"]=int(w)
        if "rationale" in p: result["rationale"]=str(p["rationale"])
    except: pass
    refusal = [r"cannot\s+(fulfil|role\.play|simulate|provide|comply|assist)",
               r"unable\s+to\s+(fulfil|role\.play|simulate|provide|comply|assist)",
               r"as\s+an\s+(AI|artificial\s+intelligence|language\s+model)"]
    if any(re.search(p,raw.lower()) for p in refusal): result["non_refusal"]=False
    return result

# Generate plan
plan = []
for m in MODELS:
    for p in profiles:
        for s in scenarios:
            for rep in range(1, REPETITIONS+1):
                plan.append({
                    'model': m['model'], 'provider': m['provider'],
                    'access_route': m['access_route'],
                    'requested_model_id': m['requested_model_id'],
                    'profile_id': p['profile_id'], 'profile_text': p['profile_text'],
                    'burden_level': s['burden_level'], 'scenario_text': s['scenario_text'],
                    'repetition': rep,
                })

print(f"Plan: {len(plan)} calls ({len(MODELS)} models × {len(profiles)} profiles × {len(scenarios)} burden × {REPETITIONS} reps)")

# Resume
completed = set()
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        for row in csv.DictReader(f):
            if not row.get('error','').strip() and row.get('json_valid','').lower() in ('true','1'):
                completed.add((row['model'], row['profile_id'], row['burden_level'], str(row['repetition'])))
print(f"Already completed: {len(completed)}")

file_exists = OUT_FILE.exists()
n_new = 0

for i, row in enumerate(plan):
    key = (row['model'], row['profile_id'], row['burden_level'], str(row['repetition']))
    if key in completed: continue

    print(f"[{i+1}/{len(plan)}] {row['model'][:25]:<25} {row['profile_id']} {row['burden_level']} rep={row['repetition']}")

    headers, body_tpl = call_or(row['requested_model_id'], row['model'])
    body_tpl['messages'][1]['content'] = USER_TEMPLATE.format(
        profile_text=row['profile_text'], scenario_text=row['scenario_text'])

    raw = ""; metadata = {}; returned_id = ""; route = ""
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = requests.post(f"{OR_BASE}/chat/completions", headers=headers, json=body_tpl, timeout=120)
            j = r.json()
            raw = j['choices'][0]['message'].get('content','') or ''
            returned_id = j.get('model','')
            metadata = {'id':j.get('id',''),'model':j.get('model',''),'provider':j.get('provider','')}
            break
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  [retry {attempt}] {str(e)[:80]}")
                time.sleep(2**attempt)
            else:
                print(f"  [FAILED] {str(e)[:120]}")

    parsed = parse_response(raw)
    out = {
        'model': row['model'], 'provider': row['provider'], 'access_route': row['access_route'],
        'requested_model_id': row['requested_model_id'], 'returned_model_id': returned_id,
        'route_provider': metadata.get('provider',''),
        'profile_id': row['profile_id'], 'burden_level': row['burden_level'],
        'repetition': row['repetition'],
        'profile_text': row['profile_text'], 'scenario_text': row['scenario_text'],
        'raw_response': raw, 'json_valid': parsed['json_valid'],
        'willingness': parsed['willingness'], 'score_valid': parsed['score_valid'],
        'rationale': parsed['rationale'], 'non_refusal': parsed['non_refusal'],
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'temperature': 0, 'top_p': 'default',
        'metadata_json': json.dumps(metadata),
    }

    with open(OUT_FILE, 'a' if file_exists else 'w', newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        if not file_exists: w.writeheader(); file_exists = True
        w.writerow(out)
    n_new += 1
    time.sleep(0.15)

print(f"\nDone. New: {n_new}, Total: {len(completed)+n_new}")
