#!/usr/bin/env python3
"""Abliteration activation collector for Colibrì + GLM-5.2.

Runs harmful prompts through the engine in COLLECT mode and dumps the
post-layer residual stream of the final prompt token, for every layer.

Usage:
  COLI_MODEL=/Volumes/ORICO/glm52_i4 python3 collect_activations.py --n 128 --out /Volumes/ORICO/abliteration/collect
  COLI_MODEL=... python3 collect_activations.py --n 128 --out ... --kind harmless   # uses chosen=refusal
"""
import os, sys, subprocess, json, struct, argparse, time
from pathlib import Path
LOG_PATH = os.environ.get("COLLECT_LOG", "/tmp/collect_harmless.log")
def log(msg):
    print(msg, flush=True)
    try:
        with open(LOG_PATH, "a") as f: f.write(msg + "\n")
    except: pass

HERE = Path(__file__).resolve().parent
COLI = HERE / "c" / "coli"
MODEL = os.environ.get("COLI_MODEL", "/Volumes/ORICO/glm52_i4")

def load_prompts(n, kind="harmful"):
    from datasets import load_dataset
    if kind == "harmful":
        ds = load_dataset("LLM-LAT/harmful_dataset", split="train")
        # harmful prompts naturally elicit refusal; the model's response is the refusal.
        prompts = [ds[i]["prompt"] for i in range(min(n, len(ds)))]
    else:
        # harmless: everyday instructions from Alpaca (no safety refusal expected).
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        prompts = [ds[i]["instruction"] for i in range(min(n, len(ds)))]
    return prompts

def run_one(prompt_text, out_dir, label, idx):
    """Run one prompt in COLLECT mode; residual written to out_dir/residuals.bin."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["COLI_COLLECT_DIR"] = str(out)
    env["NGEN"] = "1"   # only generate 1 token: we want the prefill residual only
    # write label for later PCA
    (out / "label.txt").write_text(label)
    t0 = time.time()
    r = subprocess.run([str(COLI), "run", "--ngen", "1", prompt_text], env=env, capture_output=True, text=True, timeout=600)
    dt = time.time() - t0
    ok = (out / "residuals.bin").exists()
    return ok, dt, r.stderr[-500:] if r.stderr else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--out", default="/Volumes/ORICO/abliteration/collect")
    ap.add_argument("--kind", default="harmful", choices=["harmful","harmless"])
    args = ap.parse_args()

    prompts = load_prompts(args.n, args.kind)
    log(f"[collect] {len(prompts)} prompts, kind={args.kind}, out={args.out}")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    # save manifest
    with open(Path(args.out) / "manifest.jsonl", "w") as mf:
        for i, p in enumerate(prompts):
            mf.write(json.dumps({"idx": i, "prompt": p}) + "\n")

    ok_n = 0
    for i, p in enumerate(prompts):
        d = Path(args.out) / f"p{i:04d}"
        ok, dt, err = run_one(p, d, args.kind, i)
        if ok:
            ok_n += 1
            log(f"[collect] {i+1}/{len(prompts)} OK {dt:.1f}s")
        else:
            log(f"[collect] {i+1}/{len(prompts)} FAIL: {err[-200:]}")
    log(f"[collect] done: {ok_n}/{len(prompts)} ok")

if __name__ == "__main__":
    main()
