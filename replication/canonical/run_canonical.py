#!/usr/bin/env python3
"""
Canonical rerun runner: 15-model baseline burden simulation + orientation battery + H3
frame manipulation, all under the frozen protocol in protocol.py.

Subcommands
  lock    write protocol.lock.json (prompt hashes) and resolved model roster
  probe   1 cheap call per model to resolve the actual endpoint id that answers
  plan    build canonical/plans/{sim,h3,orient}.csv (randomized, frozen before running)
  sim     run the baseline administrative-burden simulation (15 x 27 x 2 x 5 = 4050)
  h3      run the frame manipulation (4 x 27 x 4 x 5 = 2160), randomized frame order
  orient  run the orientation battery (15 calls, 20 items each)
  all     plan + sim + h3 + orient

Logging: every row records the SHA256 of the exact system/user prompt sent, the requested and
returned model id, the endpoint host, and the order index, so provenance is auditable per call.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

import protocol as P

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
OUT = PROJECT / "outputs" / "canonical"
PLANS = HERE / "plans"
OUT.mkdir(parents=True, exist_ok=True)
PLANS.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT / ".env")

ENDPOINTS = {
    "openai":      ("https://api.openai.com/v1",                  "OPENAI_API_KEY"),
    "anthropic":   ("https://api.anthropic.com/v1",                "ANTHROPIC_API_KEY"),
    "gemini":      ("https://generativelanguage.googleapis.com",  "GEMINI_API_KEY"),
    "deepseek":    ("https://api.deepseek.com",                    "DEEPSEEK_API_KEY"),
    "dashscope":   ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
    "openrouter":  ("https://openrouter.ai/api/v1",                "OPENROUTER_API_KEY"),
}
WORKERS = 6
MAX_RETRIES = 4
REFRESH_PATTERNS = [
    r"cannot\s+(fulfil|role\.play|simulate|provide|comply|assist)",
    r"unable\s+to\s+(fulfil|role\.play|simulate|provide|comply|assist)",
    r"as\s+an\s+(AI|artificial\s+intelligence|language\s+model)",
]

_write_lock = threading.Lock()


def commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def load_models() -> list:
    with open(HERE / "models.csv", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["model"].strip()]


# ------------------------------------------------------------------ API calls
def _post(url, headers, body, timeout=180):
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def _usage(route: str, j: dict) -> tuple:
    """(tokens_in, tokens_out) from whichever usage shape the provider returned."""
    if route == "gemini":
        u = j.get("usageMetadata") or {}
        return u.get("promptTokenCount", ""), u.get("candidatesTokenCount", "")
    if route == "anthropic":
        u = j.get("usage") or {}
        return u.get("input_tokens", ""), u.get("output_tokens", "")
    u = j.get("usage") or {}
    return u.get("prompt_tokens", ""), u.get("completion_tokens", "")


def call_model(route: str, model_id: str, system_prompt: str, user_prompt: str,
               max_tokens: int = None, temperature: float = None):
    """Return (text, returned_model_id, endpoint_host, (tokens_in, tokens_out)). System prompt
    is sent as a real system role on every provider (Gemini via systemInstruction)."""
    base, env_key = ENDPOINTS[route]
    key = os.environ.get(env_key)
    if not key:
        raise RuntimeError(f"missing env {env_key}")
    mt = max_tokens or P.MAX_TOKENS
    temp = P.TEMPERATURE if temperature is None else temperature

    if route == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        body = {"model": model_id, "max_tokens": mt, "temperature": temp,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]}
        try:
            j = _post(f"{base}/messages", headers, body)
        except RuntimeError as e:
            # Newer Anthropic models (e.g. claude-opus-4-8) reject `temperature` outright.
            if "temperature" not in str(e):
                raise
            body.pop("temperature", None)
            j = _post(f"{base}/messages", headers, body)
        text = "".join(b.get("text", "") for b in j.get("content", []))
        return text, j.get("model", ""), "api.anthropic.com", _usage("anthropic", j)

    if route == "gemini":
        # NOTE: 2.5 Pro/Flash are thinking models; output tokens are shared with the thinking
        # budget, so never send a token cap below 512 or the visible answer comes back empty
        # (candidates[0].content has only thought parts). Thought parts are filtered out below.
        j = _post(f"{base}/v1beta/models/{model_id}:generateContent?key={key}",
                  {"content-type": "application/json"},
                  {"systemInstruction": {"parts": [{"text": system_prompt}]},
                   "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                   "generationConfig": {"temperature": temp, "maxOutputTokens": max(mt, 4096)}})
        cands = j.get("candidates") or []
        text = ""
        if cands:
            for part in ((cands[0].get("content") or {}).get("parts") or []):
                if not part.get("thought"):
                    text += part.get("text", "") or ""
        return text, j.get("modelVersion", model_id), "generativelanguage.googleapis.com", _usage("gemini", j)

    body = {"model": model_id, "max_tokens": mt, "temperature": temp,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}]}
    if route == "openrouter" and "kimi" in model_id:
        body["reasoning"] = {"enabled": False}
    j = _post(f"{base}/chat/completions",
              {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body)
    text = (j["choices"][0]["message"].get("content") or "")
    return text, j.get("model", ""), base.split("//")[-1].split("/")[0], _usage(route, j)


def parse_score(raw: str) -> dict:
    res = {"json_valid": False, "score_valid": False, "non_refusal": True,
           "willingness": "", "rationale": ""}
    if not raw:
        return res
    cand = raw.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cand, re.DOTALL) or re.search(r"\{.*\}", cand, re.DOTALL)
    if m:
        cand = m.group(1) if m.re.groups else m.group(0)
        cand = cand.strip()
    try:
        obj = json.loads(cand)
        res["json_valid"] = True
        w = obj.get("willingness")
        if isinstance(w, (int, float)) and 0 <= w <= 100:
            res["score_valid"] = True
            res["willingness"] = int(w)
        if "rationale" in obj:
            res["rationale"] = str(obj["rationale"])
    except Exception:
        pass
    if any(re.search(p, raw.lower()) for p in REFRESH_PATTERNS):
        res["non_refusal"] = False
    return res


def parse_numbers(raw: str, n: int) -> list:
    nums = [int(x) for x in re.findall(r"\b([1-7])\b", raw or "")]
    return nums[:n]


# ------------------------------------------------------------------ plan build
def _rng(model: str) -> random.Random:
    seed = int.from_bytes(hashlib.sha256(f"{P.SEED}:{model}".encode()).digest()[:8], "big")
    return random.Random(seed)


def inputs():
    import openpyxl
    wb = openpyxl.load_workbook(PROJECT / "data/llm_experiment_inputs.xlsx", read_only=True)
    ws = wb["Profiles"]; rows = list(ws.iter_rows(values_only=True))
    profiles = [dict(zip([str(h) for h in rows[0]], r)) for r in rows[1:]]
    ws = wb["Burden Scenarios"]; rows = list(ws.iter_rows(values_only=True))
    scenarios = [dict(zip([str(h) for h in rows[0]], r)) for r in rows[1:]]
    return profiles, scenarios


def build_plan_sim(models, profiles, scenarios):
    plan, i = [], 0
    for m in models:
        rng = _rng(m["model"])
        cells = [(p, s, rep) for p in profiles for s in scenarios
                 for rep in range(1, P.REPETITIONS + 1)]
        rng.shuffle(cells)
        for p, s, rep in cells:
            i += 1
            plan.append({"run_id": f"CAN_SIM_{i:05d}", "order_index": i, "block": "sim",
                         "model": m["model"], "provider": m["provider"], "route": m["route"],
                         "requested_model_id": m["requested_model_id"], "frame": "baseline",
                         "profile_id": p["profile_id"], "profile_text": p["profile_text"],
                         "burden_level": s["burden_level"], "scenario_text": s["scenario_text"],
                         "repetition": rep})
    return plan


def build_plan_h3(models, profiles, scenarios):
    """H3 unit = one (profile, frame, repetition) burden contrast, so each unit needs the
    low AND high scenario under the SAME frame. Frame order is randomized per (profile, rep);
    the low/high order inside a pair is randomized too. Result: 27 x 5 x 4 x 2 = 1080 calls/model."""
    h3_models = {"GPT-4.1", "Llama 3.1 70B Instruct", "Qwen3.7 Plus", "Mistral Large"}
    by_level = {s["burden_level"]: s["scenario_text"] for s in scenarios}
    plan, i = [], 0
    for m in models:
        if m["model"] not in h3_models:
            continue
        rng = _rng("h3:" + m["model"])
        cells = []
        for p in profiles:
            for rep in range(1, P.REPETITIONS + 1):
                for frame in rng.sample(list(P.FRAME_CLAUSES), len(P.FRAME_CLAUSES)):
                    levels = ["low", "high"]
                    rng.shuffle(levels)
                    for lvl in levels:
                        cells.append((p, rep, frame, lvl))
        rng.shuffle(cells)
        for p, rep, frame, lvl in cells:
            i += 1
            plan.append({"run_id": f"CAN_H3_{i:05d}", "order_index": i, "block": "h3",
                         "model": m["model"], "provider": m["provider"], "route": m["route"],
                         "requested_model_id": m["requested_model_id"], "frame": frame,
                         "profile_id": p["profile_id"], "profile_text": p["profile_text"],
                         "burden_level": lvl, "scenario_text": by_level[lvl], "repetition": rep})
    return plan


def build_plan_orient(models):
    rows = list(csv.DictReader(open(PROJECT / "data/orientation_responses.csv")))
    items, seen = [], set()
    for r in rows:
        if r["item_id"] not in seen:
            seen.add(r["item_id"])
            items.append(r)
    items.sort(key=lambda r: int(r["item_order"]))
    listing = "\n".join(f"{r['item_order']}. {r['item_text']}" for r in items)
    user = P.ORIENTATION_USER_TEMPLATE.format(n=len(items), items=listing)
    return items, user, [{"run_id": f"CAN_ORI_{i:03d}", "order_index": i, "block": "orient",
                          "model": m["model"], "provider": m["provider"], "route": m["route"],
                          "requested_model_id": m["requested_model_id"]}
                         for i, m in enumerate(models, 1)]


def write_plan(plan, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0].keys()))
        w.writeheader()
        w.writerows(plan)
    return path


# ------------------------------------------------------------------ runner
FIELDS = ["run_id", "order_index", "block", "model", "provider", "route", "requested_model_id",
          "returned_model_id", "endpoint_host", "frame", "profile_id", "burden_level", "repetition",
          "system_prompt_sha256", "user_prompt_sha256", "temperature", "protocol_sha256",
          "script_commit", "tokens_in", "tokens_out", "raw_response", "json_valid", "score_valid",
          "non_refusal", "willingness", "rationale", "error", "timestamp_utc"]


def run_block(plan, out_path, system_for, user_for, parse_fn, label):
    sys_hashes = {f: P.sha256(P.system_prompt(f)) for f in P.FRAME_CLAUSES}
    proto_sha = P.sha256(json.dumps(P.lock_payload(), sort_keys=True))
    done = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # "done" is keyed by CELL, not run_id: run_id is just the position in the current
                # plan, so a re-planned run (e.g. after a model rename) shifts the id->cell mapping
                # and an id-keyed done-set silently skips work or reruns finished work.
                if not (r.get("error") or "").strip() and str(r.get("json_valid", "")).lower() in ("true", "1"):
                    done.add((r.get("model"), r.get("frame", ""), r.get("profile_id", ""),
                              r.get("burden_level", ""), str(r.get("repetition", ""))))
    new_file = not out_path.exists()
    fh = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); fh.flush()
    todo = [r for r in plan
            if (r["model"], r.get("frame", ""), r.get("profile_id", ""),
                r.get("burden_level", ""), str(r.get("repetition", ""))) not in done]
    print(f"[{label}] total={len(plan)} done={len(done)} to_run={len(todo)} workers={WORKERS}")

    def one(row):
        sp = system_for(row)
        up = user_for(row)
        err, raw, rid, host, usage = "", "", "", "", ("", "")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw, rid, host, usage = call_model(row["route"], row["requested_model_id"], sp, up)
                err = ""
                break
            except Exception as e:
                err = str(e)[:400]
                if attempt < MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
        parsed = parse_fn(raw)
        out = {"run_id": row["run_id"], "order_index": row["order_index"], "block": row["block"],
               "model": row["model"], "provider": row["provider"], "route": row["route"],
               "requested_model_id": row["requested_model_id"], "returned_model_id": rid,
               "endpoint_host": host, "frame": row.get("frame", ""),
               "profile_id": row.get("profile_id", ""), "burden_level": row.get("burden_level", ""),
               "repetition": row.get("repetition", ""),
               "system_prompt_sha256": sys_hashes.get(row.get("frame", "neutral"), P.sha256(sp)),
               "user_prompt_sha256": P.sha256(up), "temperature": P.TEMPERATURE,
               "protocol_sha256": proto_sha, "script_commit": commit_hash(),
               "tokens_in": usage[0], "tokens_out": usage[1],
               "raw_response": raw[:4000], "error": err,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(), **parsed}
        with _write_lock:
            w.writerow(out); fh.flush()
        return out

    n = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for fu in as_completed(futs):
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  [{label}] {n}/{len(todo)}", flush=True)
    fh.close()
    print(f"[{label}] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["lock", "probe", "plan", "sim", "h3", "orient", "all"])
    ap.add_argument("--only", default="", help="comma-separated model names; restricts this run")
    a = ap.parse_args()
    models = load_models()
    keep = {s.strip() for s in a.only.split(",") if s.strip()}

    if a.cmd in ("lock", "all"):
        p = P.write_lock(HERE / "protocol.lock.json")
        print("shield/protocol:", p, "baseline_sha=", P.sha256(P.BASE_SYSTEM_PROMPT))

    if a.cmd in ("probe", "all"):
        rows = [{"model": m["model"], "route": m["route"], "requested_model_id": m["requested_model_id"],
                 "returned_model_id": "", "ok": "", "error": "", "host": ""} for m in models]
        alts = {m["model"]: [x for x in (m.get("alt_ids") or "").split(";") if x] for m in models}
        def probe(r):
            cands = [(r["route"], r["requested_model_id"])]
            for x in alts.get(r["model"], []):
                cands.append(tuple(x.split(":", 1)) if ":" in x else (r["route"], x))
            for rt, cand in cands:
                try:
                    t, rid, host, _u = call_model(rt, cand, P.BASE_SYSTEM_PROMPT,
                                                  "Reply with the single word: ok", max_tokens=512)
                    r.update({"route": rt, "requested_model_id": cand, "returned_model_id": rid,
                              "ok": "yes", "host": host, "error": ""})
                    return r
                except Exception as e:
                    r["error"] = f"{rt}:{cand}: {str(e)[:150]}"
            r["ok"] = "no"
            return r
        with ThreadPoolExecutor(max_workers=6) as ex:
            rows = list(ex.map(probe, rows))
        with open(HERE / "probe_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        for r in rows:
            print(f"  {r['ok']:>3} {r['model']:<24} req={r['requested_model_id']:<38} "
                  f"ret={r['returned_model_id']:<44} {r['error'][:60]}")
        if a.cmd == "probe":
            return

    profiles, scenarios = inputs()
    sim_plan = build_plan_sim(models, profiles, scenarios)
    h3_plan = build_plan_h3(models, profiles, scenarios)
    items, orient_user, ori_plan = build_plan_orient(models)
    # --only filters the plan rows (not the model list) so run_ids keep matching the frozen plan.
    if keep:
        sim_plan = [r for r in sim_plan if r["model"] in keep]
        h3_plan = [r for r in h3_plan if r["model"] in keep]
        ori_plan = [r for r in ori_plan if r["model"] in keep]

    if a.cmd in ("plan", "all"):
        print(write_plan(sim_plan, PLANS / "plan_sim.csv"), len(sim_plan))
        print(write_plan(h3_plan, PLANS / "plan_h3.csv"), len(h3_plan))
        print(write_plan(ori_plan, PLANS / "plan_orient.csv"), len(ori_plan))
        print("frame counts per model (h3):",
              {m: sum(1 for r in h3_plan if r["model"] == m) for m in {r["model"] for r in h3_plan}})
        if a.cmd == "plan":
            return

    if a.cmd in ("sim", "all"):
        run_block(sim_plan, OUT / "sim_raw.csv",
                  lambda r: P.system_prompt("neutral"),
                  lambda r: P.USER_TEMPLATE.format(profile_text=r["profile_text"],
                                                   scenario_text=r["scenario_text"]),
                  parse_score, "sim")
    if a.cmd in ("h3", "all"):
        run_block(h3_plan, OUT / "h3_raw.csv",
                  lambda r: P.system_prompt(r["frame"]),
                  lambda r: P.USER_TEMPLATE.format(profile_text=r["profile_text"],
                                                   scenario_text=r["scenario_text"]),
                  parse_score, "h3")
    if a.cmd in ("orient", "all"):
        o_rows = []
        def orow(r):
            return r
        def run_one_or(r):
            raw, rid, host, usage = "", "", "", ("", "")
            try:
                raw, rid, host, usage = call_model(r["route"], r["requested_model_id"],
                                                   P.ORIENTATION_SYSTEM_PROMPT, orient_user)
            except Exception as e:
                raw = ""
                print("orient fail", r["model"], str(e)[:120])
            nums = parse_numbers(raw, len(items))
            return {"run_id": r["run_id"], "order_index": r["order_index"], "block": "orient",
                    "model": r["model"], "provider": r["provider"], "route": r["route"],
                    "requested_model_id": r["requested_model_id"], "returned_model_id": rid,
                    "endpoint_host": host, "tokens_in": usage[0], "tokens_out": usage[1],
                    "system_prompt_sha256": P.sha256(P.ORIENTATION_SYSTEM_PROMPT),
                    "user_prompt_sha256": P.sha256(orient_user), "raw_response": raw[:4000],
                    "n_parsed": len(nums), "scores": json.dumps(nums),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        with ThreadPoolExecutor(max_workers=6) as ex:
            o_rows = list(ex.map(run_one_or, ori_plan))
        p = OUT / "orientation_raw.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(o_rows[0].keys())); w.writeheader(); w.writerows(o_rows)
        print("[orient] wrote", p, "parsed:", {r["model"]: r["n_parsed"] for r in o_rows})


def scenarios_by_id(scenarios):
    return {s["burden_level"]: s for s in scenarios}


def _h3_scenario(scenarios):
    return next(s["scenario_text"] for s in scenarios)


if __name__ == "__main__":
    main()
