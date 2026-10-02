#!/usr/bin/env python3
"""Score the human gold-coding sheet against the algorithmic H2 frame codes.

Usage: fill outputs/h2_gold_coding_sheet.csv (columns human_access_barriers,
human_collective_responsibility, human_coercive_backlash, ... with 0/1), then:
    python3 score_h2_gold_coding.py

Reports, per frame: n positive (human vs algorithm), precision, recall, F1,
exact agreement, Cohen's kappa, and the 2x2 confusion counts. Writes
outputs/h2_gold_validation.csv.

A frame is only scored where the human left a value (0/1); blank cells are
treated as not yet coded and are dropped for that frame.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

HOME = Path(__file__).resolve().parent
OUT = HOME / "outputs"
FRAMES = ["access_barriers", "collective_responsibility", "coercive_backlash",
          "autonomy_infringement", "procedural_legitimacy"]


def cohens_kappa(y_true, y_pred):
    a = y_true == y_pred
    po = a.mean()
    pe = ((y_true.mean() * y_pred.mean()) + ((1 - y_true.mean()) * (1 - y_pred.mean())))
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    sheet = pd.read_csv(OUT / "h2_gold_coding_sheet.csv")
    key = pd.read_csv(OUT / "h2_gold_coding_key.csv")
    df = sheet.merge(key, on="rationale_id", how="inner", validate="one_to_one")
    print(f"merged {len(df)} coded rationales")

    rows = []
    for f in FRAMES:
        hcol, acol = f"human_{f}", f"algo_{f}"
        d = df[[hcol, acol]].apply(pd.to_numeric, errors="coerce").dropna()
        if d.empty:
            print(f"{f:<26} not scored yet (no human labels)")
            continue
        h, a = d[hcol].astype(int).values, d[acol].astype(int).values
        tp = int(((h == 1) & (a == 1)).sum()); fp = int(((h == 0) & (a == 1)).sum())
        fn = int(((h == 1) & (a == 0)).sum()); tn = int(((h == 0) & (a == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (tp + fp and tp + fn and prec + rec) else float("nan")
        rows.append(dict(frame=f, n=len(d), human_positive=int(h.sum()), algo_positive=int(a.sum()),
                         tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1,
                         agreement=(h == a).mean(), kappa=cohens_kappa(h, a)))
        print(f"{f:<26} human+={int(h.sum()):3d} algo+={int(a.sum()):3d}  "
              f"precision={prec:.3f} recall={rec:.3f} F1={f1:.3f}  "
              f"agreement={(h == a).mean():.3f} kappa={cohens_kappa(h, a):+.3f}")

    if rows:
        res = pd.DataFrame(rows)
        res.to_csv(OUT / "h2_gold_validation.csv", index=False)
        print(f"\nwrote {OUT/'h2_gold_validation.csv'}")
        print("Interpretation guide: for frame-level indicators with low prevalence, report precision and F1 rather than accuracy.")
    else:
        print("\nNo human labels found yet — fill the sheet first (0/1 in each human_* column).")


if __name__ == "__main__":
    main()
