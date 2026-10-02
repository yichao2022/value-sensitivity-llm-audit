#!/usr/bin/env python3
"""Summarise the auxiliary burden-classification diagnostic (outputs/canonical/comp_raw.csv).

Reports per model and pooled: N scored, correct, malformed (no low/high token returned),
accuracy with an exact (Clopper-Pearson) 95% confidence interval, and the mean returned
token count. Writes outputs/canonical/comp_summary.csv and prints a LaTeX-ready block.
"""
import csv
from pathlib import Path

import pandas as pd
from scipy import stats as st

HOME = Path(__file__).resolve().parent
CAN = HOME / "outputs" / "canonical"
RAW = CAN / "comp_raw.csv"


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple:
    lo = 0.0 if k == 0 else st.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else st.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def main() -> None:
    rows = list(csv.DictReader(open(RAW, newline="")))
    scored = [r for r in rows if (r.get("returned_label") or "").strip()]
    malformed = [r for r in rows if not (r.get("returned_label") or "").strip()]
    errored = [r for r in rows if (r.get("error") or "").strip()]

    out = []
    for model, g in pd.DataFrame(scored).groupby("model"):
        n = len(g)
        k = int((g["correct"] == "True").sum())
        lo, hi = clopper_pearson(k, n)
        out.append({"model": model, "n": n, "correct": k, "accuracy": k / n,
                    "ci_low": lo, "ci_high": hi,
                    "malformed": int(((g["malformed"].isin(["True", "true"])).sum()))})
    _all = pd.DataFrame(scored)
    n, k = len(_all), int((_all["correct"] == "True").sum())
    lo, hi = clopper_pearson(k, n)
    out.append({"model": "POOLED", "n": n, "correct": k, "accuracy": k / n,
                "ci_low": lo, "ci_high": hi, "malformed": 0})
    df = pd.DataFrame(out).sort_values("model")

    print(f"raw rows={len(rows)}  scored={len(scored)}  malformed={len(malformed)}  API errors={len(errored)}")
    for _, r in df.iterrows():
        print(f"  {r['model']:<24} {int(r['correct']):>3}/{int(r['n']):<3} = {100 * r['accuracy']:5.1f}%  "
              f"exact 95% CI [{100 * r['ci_low']:5.1f}, {100 * r['ci_high']:5.1f}]")
    if len(malformed):
        print("  malformed responses:", [(r["model"], r["profile_id"], r["raw_response"][:40]) for r in malformed[:5]])
    if len(errored):
        print("  API errors:", [(r["model"], r["error"][:60]) for r in errored[:5]])

    df.to_csv(CAN / "comp_summary.csv", index=False)
    print(f"\nwrote {CAN / 'comp_summary.csv'}")

    # LaTeX rows for Table 3 (data-quality table in the supplement)
    print("\n--- LaTeX-ready (percent, one decimal) ---")
    for _, r in df.iterrows():
        if r["model"] == "POOLED":
            continue
        print(f"{r['model']} & {100 * r['accuracy']:.1f} \\\\")


if __name__ == "__main__":
    main()
