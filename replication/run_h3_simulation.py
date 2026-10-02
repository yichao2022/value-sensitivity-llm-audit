#!/usr/bin/env python3
"""H3 Within-Model Frame Manipulation — API runner. Supports resume by run_id."""

import csv, json, os, re, time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import headroom_integration

PROJECT = Path(__file__).resolve().parent
PLAN_CSV = PROJECT / "data" / "frame_manipulation_plan.csv"
OUTPUT_CSV = PROJECT / "outputs" / "frame_manipulation_outputs.csv"

PROVIDER_CONFIG = {
    "OpenAI": {"env_key": "OPENAI_API_KEY", "base_url": None},
    "Anthropic": {"env_key": "ANTHROPIC_API_KEY", "base_url": None},
    "Google": {"env_key": "GEMINI_API_KEY", "base_url": None},
    "DeepSeek": {"env_key": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com"},
    "Qwen": {"env_key": "DASHSCOPE_API_KEY", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
              "model_override": "qwen-max"},
    "Meta": {"env_key": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1",
             "model_override": "meta-llama/llama-3.1-70b-instruct"},
    "MiMo": {"env_key": "MIMO_API_KEY", "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
              "model_override": "mimo-v2.5-pro"},
    "LMStudio": {"env_key": "LMSTUDIO_API_KEY", "base_url": "http://127.0.0.1:1234/v1",
                  "model_override": "llama-3.3-70b-instruct-abliterated"},
}
OPENAI_COMPATIBLE = {"OpenAI", "DeepSeek", "Qwen", "Meta", "MiMo", "LMStudio"}
MAX_RETRIES, BASE_DELAY, MAX_DELAY = 5, 2, 120

OUTPUT_COLS = ["run_id","model","provider","api_model","frame","profile_id","burden_level",
               "repetition","json_valid","score_valid","non_refusal",
               "willingness","rationale","raw_output","error","timestamp_utc"]

def load_completed():
    if not OUTPUT_CSV.exists(): return set()
    done = set()
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            if not row.get("error","").strip():
                done.add(row["run_id"])
    return done

def parse_response(raw):
    result = {"json_valid":False,"score_valid":False,"non_refusal":True,"willingness":None,"rationale":""}
    if not raw: return result
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

def call_openai_compat(row):
    from openai import OpenAI
    cfg = PROVIDER_CONFIG[row["provider"]]
    client = OpenAI(api_key=os.environ[cfg["env_key"]],
                    base_url=cfg.get("base_url"))
    model_id = cfg.get("model_override", row["api_model"])
    msgs = headroom_integration.compress_messages(
        [{"role":"system","content":row["system_prompt"]},
         {"role":"user","content":row["user_prompt"]}],
        model=model_id)
    r = client.chat.completions.create(
        model=model_id, max_tokens=512, temperature=0, messages=msgs)
    return r.choices[0].message.content or ""

def call_anthropic(row):
    from anthropic import Anthropic
    cfg = PROVIDER_CONFIG[row["provider"]]
    client = Anthropic(api_key=os.environ[cfg["env_key"]])
    msgs = headroom_integration.compress_messages(
        [{"role":"user","content":row["user_prompt"]}],
        model=row["api_model"])
    r = client.messages.create(
        model=row["api_model"], max_tokens=512, temperature=0,
        system=row["system_prompt"], messages=msgs)
    return r.content[0].text if r.content else ""

def call_google(row):
    from google import genai
    cfg = PROVIDER_CONFIG[row["provider"]]
    client = genai.Client(api_key=os.environ[cfg["env_key"]])
    full = f"{row['system_prompt']}\n\n{row['user_prompt']}"
    r = client.models.generate_content(model=row["api_model"], contents=full,
                                        config={"temperature":0})
    return r.text if r.text else ""

def call_api(row):
    p = row["provider"]
    if p == "Anthropic": return call_anthropic(row)
    elif p == "Google": return call_google(row)
    elif p in OPENAI_COMPATIBLE: return call_openai_compat(row)
    raise ValueError(f"No handler: {p}")

def main():
    plan = list(csv.DictReader(open(PLAN_CSV)))
    done = load_completed()
    print(f"Plan: {len(plan)} rows, completed: {len(done)}")

    n_new = 0
    for i, row in enumerate(plan):
        rid = row["run_id"]
        if rid in done: continue

        print(f"[{i+1}/{len(plan)}] {rid} | {row['model']:<28} frame={row['frame']:<12} {row['profile_id']} {row['burden_level']} rep={row['repetition']}")

        raw = ""; err = ""
        for attempt in range(1, MAX_RETRIES+1):
            try:
                raw = call_api(row); break
            except Exception as e:
                err = str(e)
                if attempt < MAX_RETRIES:
                    d = min(BASE_DELAY*(2**(attempt-1)), MAX_DELAY)
                    print(f"  [retry {attempt}] {err[:100]} — {d}s")
                    time.sleep(d)
                else:
                    print(f"  [FAILED] {err[:150]}")

        parsed = parse_response(raw)
        out = {
            "run_id":rid,"model":row["model"],"provider":row["provider"],
            "api_model":row["api_model"],"frame":row["frame"],
            "profile_id":row["profile_id"],"burden_level":row["burden_level"],
            "repetition":row["repetition"],
            "json_valid":parsed["json_valid"],"score_valid":parsed["score_valid"],
            "non_refusal":parsed["non_refusal"],
            "willingness":parsed["willingness"] if parsed["willingness"] is not None else "",
            "rationale":parsed["rationale"],"raw_output":raw,"error":err,
            "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        }
        exists = OUTPUT_CSV.exists()
        with open(OUTPUT_CSV,"a",newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUTPUT_COLS)
            if not exists: w.writeheader()
            w.writerow(out)
        n_new += 1
        # time.sleep(0.2)  # removed for speed

    print(f"\nDone. New: {n_new}, Total: {len(done)+n_new}")

if __name__ == "__main__":
    main()
