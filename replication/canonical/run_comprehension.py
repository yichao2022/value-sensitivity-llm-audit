#!/usr/bin/env python3
"""Auxiliary burden-classification (comprehension) diagnostic.

NOT part of the frozen canonical administration. This block was generated and fielded
separately (see the timestamp column) and reuses the canonical endpoint clients, model list,
decoding parameters and protocol lock, so the only thing that differs is the prompt.

Design: 15 endpoints x 27 profiles x 2 burden levels = 810 calls, one word returned.
Scoring: a response is malformed if it contains neither "low" nor "high"; otherwise the
first occurrence decides the label, and the response is correct if the label matches the
burden level assigned to the scenario that was shown.

    python3 run_comprehension.py plan   # write canonical/plans/plan_comp.csv
    python3 run_comprehension.py run    # append to outputs/canonical/comp_raw.csv
    python3 run_comprehension.py all
"""
import argparse
import csv
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P          # noqa: E402
import run_canonical as R     # noqa: E402  (loads .env, endpoint clients, plan helpers)

HERE = Path(__file__).resolve().parent
OUT = R.OUT
PLANS = R.PLANS
PLAN_PATH = PLANS / "plan_comp.csv"
RAW_PATH = OUT / "comp_raw.csv"
BLOCK = "comp"
REPETITION = 1

SYSTEM_PROMPT = ("You are answering a single factual question about a vaccination access scenario. "
                 "Answer with one word only.")

USER_TEMPLATE = """Individual profile:

{profile}

Vaccination access scenario:

{scenario}

Question:

Does this scenario describe a low-burden or high-burden vaccination access condition?

Return only one word: low or high."""

FIELDS = ["run_id", "order_index", "block", "model", "provider", "route", "requested_model_id",
          "returned_model_id", "endpoint_host", "profile_id", "burden_level",
          "system_prompt_sha256", "user_prompt_sha256", "temperature", "protocol_sha256",
          "script_commit", "tokens_in", "tokens_out", "raw_response",
          "expected_label", "returned_label", "correct", "malformed", "error", "timestamp_utc"]


def build_plan(models, profiles, scenarios):
    """One cell per (model, profile, burden level), order randomised per model from the protocol seed."""
    plan, i = [], 0
    for m in models:
        rng = R._rng(f"comp:{m['model']}")
        cells = [(p, s) for p in profiles for s in scenarios]
        rng.shuffle(cells)
        for p, s in cells:
            i += 1
            plan.append({"run_id": f"CAN_COMP_{i:04d}", "order_index": i, "block": BLOCK,
                         "model": m["model"], "provider": m["provider"], "route": m["route"],
                         "requested_model_id": m["requested_model_id"],
                         "profile_id": p["profile_id"], "profile_text": p["profile_text"],
                         "burden_level": s["burden_level"], "scenario_text": s["scenario_text"],
                         "repetition": REPETITION})
    return plan


def parse_label(raw: str) -> dict:
    """First standalone low/high token decides; empty or neither -> malformed."""
    m = re.search(r"\b(low|high)\b", (raw or "").lower())
    label = m.group(1) if m else ""
    return {"returned_label": label, "malformed": not label}


def run(plan):
    sys_sha = P.sha256(SYSTEM_PROMPT)
    proto_sha = P.sha256(json.dumps(P.lock_payload(), sort_keys=True))
    done = set()
    if RAW_PATH.exists():
        with open(RAW_PATH, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not (r.get("error") or "").strip() and r.get("returned_label"):
                    done.add((r["model"], r["profile_id"], r["burden_level"]))
    todo = [r for r in plan if (r["model"], r["profile_id"], r["burden_level"]) not in done]
    print(f"[{BLOCK}] total={len(plan)} done={len(done)} to_run={len(todo)} workers={R.WORKERS}", flush=True)

    new_file = not RAW_PATH.exists()
    fh = open(RAW_PATH, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    lock = threading.Lock()
    if new_file:
        w.writeheader(); fh.flush()

    def one(row):
        up = USER_TEMPLATE.format(profile=row["profile_text"], scenario=row["scenario_text"])
        err, raw, rid, host, usage = "", "", "", "", ("", "")
        for attempt in range(1, R.MAX_RETRIES + 1):
            try:
                raw, rid, host, usage = R.call_model(row["route"], row["requested_model_id"],
                                                     SYSTEM_PROMPT, up, max_tokens=512)
                err = ""
                break
            except Exception as e:
                err = str(e)[:400]
                if attempt < R.MAX_RETRIES:
                    time.sleep(min(2 ** attempt, 30))
        parsed = parse_label(raw)
        out = {"run_id": row["run_id"], "order_index": row["order_index"], "block": BLOCK,
               "model": row["model"], "provider": row["provider"], "route": row["route"],
               "requested_model_id": row["requested_model_id"], "returned_model_id": rid,
               "endpoint_host": host, "profile_id": row["profile_id"],
               "burden_level": row["burden_level"], "system_prompt_sha256": sys_sha,
               "user_prompt_sha256": P.sha256(up), "temperature": P.TEMPERATURE,
               "protocol_sha256": proto_sha, "script_commit": R.commit_hash(),
               "tokens_in": usage[0], "tokens_out": usage[1], "raw_response": raw[:2000],
               "expected_label": row["burden_level"], "correct": parsed["returned_label"] == row["burden_level"],
               "error": err, "timestamp_utc": datetime.now(timezone.utc).isoformat(), **parsed}
        with lock:
            w.writerow(out); fh.flush()
        return out

    n = 0
    with ThreadPoolExecutor(max_workers=R.WORKERS) as ex:
        futs = [ex.submit(one, r) for r in todo]
        for _ in as_completed(futs):
            n += 1
            if n % 50 == 0 or n == len(todo):
                print(f"  [{BLOCK}] {n}/{len(todo)}", flush=True)
    fh.close()
    print(f"[{BLOCK}] wrote {RAW_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "all"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    models = R.load_models()
    keep = {s.strip() for s in a.only.split(",") if s.strip()}
    if keep:
        models = [m for m in models if m["model"] in keep]
    profiles, scenarios = R.inputs()
    plan = build_plan(models, profiles, scenarios)
    if a.cmd in ("plan", "all"):
        print(R.write_plan(plan, PLAN_PATH), len(plan))
    if a.cmd in ("run", "all"):
        run(plan)
