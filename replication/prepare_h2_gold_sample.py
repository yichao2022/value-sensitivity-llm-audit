#!/usr/bin/env python3
"""Build a blinded 200-rationale gold-coding sheet for H2 frame validation.

Sampling: stratified random sample of the 2,019 high-burden rationales in the
canonical H2 sample, allocated proportionally across the 15 model endpoints,
seed 20260917 (reproducible). The coding sheet is blinded: it contains only a
randomized rationale id, the rationale text, and empty label columns. The
algorithmic labels, model identity, profile and repetition live in the separate
key file so a human coder can work without seeing them.

After the sheet is filled in by a human coder, run score_h2_gold_coding.py.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path(__file__).resolve().parent
OUT = HOME / "outputs"
SEED = 20260917
N_SAMPLE = 200
FRAMES = ["access_barriers", "collective_responsibility", "coercive_backlash",
          "autonomy_infringement", "procedural_legitimacy"]


def load_rows():
    spec = importlib.util.spec_from_file_location("h2", HOME / "analyze_canonical_h2.py")
    h2 = importlib.util.module_from_spec(spec)
    sys.modules["h2"] = h2
    spec.loader.exec_module(h2)
    return pd.DataFrame(h2.load())


def main():
    df = load_rows()
    df = df[df["rationale"].astype(str).str.strip() != ""].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    picks = []
    for model, grp in df.groupby("model", sort=True):
        k = int(round(N_SAMPLE * len(grp) / len(df)))
        idx = rng.choice(grp.index.values, size=min(k, len(grp)), replace=False)
        picks.extend(idx)
    # trim / top up to exactly N_SAMPLE
    picks = list(dict.fromkeys(picks))
    if len(picks) > N_SAMPLE:
        picks = list(rng.choice(picks, size=N_SAMPLE, replace=False))
    while len(picks) < N_SAMPLE:
        extra = rng.choice([i for i in df.index if i not in picks], size=N_SAMPLE - len(picks), replace=False)
        picks.extend(extra.tolist())

    sample = df.loc[picks].copy()
    ids = rng.permutation(len(sample))            # blinded, randomized order
    sample["rationale_id"] = [f"R{i:03d}" for i in ids]

    sheet = sample[["rationale_id", "rationale"]].copy()
    for f in FRAMES:
        sheet[f] = ""
    sheet.columns = ["rationale_id", "rationale_text"] + [f"human_{f}" for f in FRAMES]

    key = sample[["rationale_id", "model", "profile_id", "repetition", "delta"] + FRAMES].copy()
    key.columns = ["rationale_id", "model", "profile_id", "repetition", "burden_effect"] + [f"algo_{f}" for f in FRAMES]

    sheet.to_csv(OUT / "h2_gold_coding_sheet.csv", index=False, quoting=csv.QUOTE_ALL)
    key.to_csv(OUT / "h2_gold_coding_key.csv", index=False)

    print(f"sheet: {OUT/'h2_gold_coding_sheet.csv'}  ({len(sheet)} rationales, blinded)")
    print(f"key:   {OUT/'h2_gold_coding_key.csv'}  (algorithmic labels; do not open while coding)")
    print("\nframe composition of the sample (algorithmic labels, for reference only):")
    for f in FRAMES:
        print(f"  {f:<26} {int(key[f'algo_{f}'].sum()):3d} / {len(key)}  ({100*key[f'algo_{f}'].mean():.1f}%)")
    print("\nmodels represented:", key.model.nunique(), "| profiles:", key.profile_id.nunique())


if __name__ == "__main__":
    main()
