#!/usr/bin/env python3
"""Guard: Table 12 (H1 across the nine- and fifteen-endpoint samples) must match the canonical run.

Regenerate the reference with `analyze_canonical_h1_samples.py` (writes
outputs/canonical/h1_sample_comparison.csv), then this script checks that every row of the
supplement's Table 12 and the numbers quoted in its note agree with that file.

Exit 0 = consistent; 1 = a reported number contradicts the canonical output.
"""
import csv
import re
import sys
from pathlib import Path

import pandas as pd

HOME = Path(__file__).resolve().parent
REF = HOME / "outputs" / "canonical" / "h1_sample_comparison.csv"
SUPP = Path.home() / "Documents" / "value-sensitivity-llm-audit" / "supplementary.tex"

# supplement row label -> (predictor, sample) in the reference file
ROWS = {
    "PVOC composite, \\(N=9\\), fifteen-endpoint standardisation": ("PVOC composite", "N=9, fifteen-endpoint standardisation"),
    "PVOC composite, \\(N=9\\), re-standardised within nine": ("PVOC composite", "N=9, re-standardised within nine"),
    "PVOC composite, \\(N=15\\)": ("PVOC composite", "N=15"),
    "Collective obligation, \\(N=9\\)": ("Collective obligation (raw 1-7)", "N=9"),
    "Collective obligation, \\(N=15\\)": ("Collective obligation (raw 1-7)", "N=15"),
    "Leave-CO-out PVOC, \\(N=15\\)": ("Leave-CO-out composite", "N=15"),
}


def num(s):
    return float(s.replace("$-$", "-").replace("$", "").replace("{,}", "").strip())


def main() -> int:
    ref = pd.read_csv(REF).set_index(["predictor", "sample"])
    supp = SUPP.read_text()
    problems = []
    table = supp[supp.index("\\label{tab:h1_expanded_robustness}"):]
    table = table[:table.index("\\end{table}")]

    for label, key in ROWS.items():
        row = next((ln for ln in table.splitlines() if ln.startswith(label)), None)
        if row is None:
            problems.append(f"row not found in Table 12: {label}")
            continue
        cells = [c.strip() for c in row.rstrip("\\\\ ").split("&")]
        b, p, r2 = num(cells[1]), num(cells[2]), num(cells[3])
        want = ref.loc[key]
        # the table prints coefficients to 2 decimals and p/R^2 to 3, so compare at display precision
        for got, w, what, tol in ((b, want["b"], "coefficient", 0.006), (p, want["p"], "p-value", 0.001),
                                  (r2, want["r2"], "R^2", 0.001)):
            if abs(got - w) > tol:
                problems.append(f"{label}: {what} = {got} but canonical = {w:.3f}")
        print(f"  {label:<52} b={b:+7.2f} p={p:.3f} R2={r2:.3f}  (canonical b={want['b']:+.2f} p={want['p']:.3f} R2={want['r2']:.3f})")

    # the note's scale-variant values must match too
    for val, key in ((-4.24, ("PVOC composite", "N=9, fifteen-endpoint standardisation")),
                     (-2.12, ("PVOC composite", "N=9, re-standardised within nine")),
                     (5.01, ("Collective obligation (standardised)", "N=9")),
                     (1.05, ("Collective obligation (standardised)", "N=15"))):
        want = ref.loc[key, "b"]
        if abs(val - want) > 0.005:
            problems.append(f"note quotes {val} but canonical is {want:+.2f} for {key}")
    for phrase in ("fifteen-endpoint standardisation", "re-standardises the five dimensions within the nine-endpoint subset"):
        if phrase not in supp:
            problems.append(f"the standardisation statement {phrase!r} is missing from the supplement")

    # Dimension-specific table (all five dimensions, not only collective obligation)
    dtab = supp[supp.index("\\label{tab:h1_dimension_checks}"):]
    dtab = dtab[:dtab.index("\\end{table}")]
    for d in ("AP", "ST", "CO", "BS", "CA"):
        row = next((ln for ln in dtab.splitlines() if ln.startswith(d + " &")), None)
        if row is None:
            problems.append(f"dimension row missing from the dimension-specific table: {d}")
            continue
        cells = [c.strip() for c in row.rstrip("\\\\ ").split("&")]
        b, se, ci, p, r2 = num(cells[1]), num(cells[2]), cells[3], num(cells[4]), num(cells[5])
        want = ref.loc[(f"{d} (standardised)", "N=15")]
        lo, hi = [num(v) for v in ci.strip("[]").split(",")]
        for got, w, what, tol in ((b, want["b"], "coefficient", 0.006), (se, want["se"], "SE", 0.006),
                                  (lo, want["lo"], "CI lower", 0.006), (hi, want["hi"], "CI upper", 0.006),
                                  (p, want["p"], "p-value", 0.001), (r2, want["r2"], "R^2", 0.001)):
            if abs(got - w) > tol:
                problems.append(f"{d} dimension row: {what} = {got} but canonical = {w:.3f}")
        print(f"  {d:<3} dimension (N=15)  b={b:+7.2f} SE={se:.2f} p={p:.3f} R2={r2:.3f}")
    if "Table~\\ref{tab:h1_dimension_checks}" not in supp:
        problems.append("the supplement never points to the dimension-specific table")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nTable 12 rows and note match the canonical run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
