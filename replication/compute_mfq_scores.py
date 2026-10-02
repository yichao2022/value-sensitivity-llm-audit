#!/usr/bin/env python3
"""
Compute MFQ-30 foundation scores from mfq30_responses_filled.csv.
Outputs:
  outputs/mfq30_scores.csv          — per-model: 5 dimension means + z-scores + MFQ composite
  outputs/mfq30_table_models.csv    — table-ready CSV (sorted by composite)
  outputs/mfq30_table_models.tex    — table-ready LaTeX

Scoring convention (Graham et al. 2011):
  - 5 foundations: Care (CA), Fairness (FA), Loyalty (LO), Authority (AU), Sanctity (SA)
  - 6 items per foundation, mean-scored (relevance + judgment combined; all on 1-7 here)
  - Catch items (CT) excluded from scoring
  - Composite = mean of individualizing (CA, FA) minus mean of binding (LO, AU, SA),
    standardised across models (i.e., a z-scored MFQ "progressivism" index)
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "outputs" / "mfq30_responses_filled.csv"
OUT_SCORES = PROJECT / "outputs" / "mfq30_scores.csv"
OUT_TABLE_CSV = PROJECT / "outputs" / "mfq30_table_models.csv"
OUT_TABLE_TEX = PROJECT / "outputs" / "mfq30_table_models.tex"

FOUNDATIONS = ["CA", "FA", "LO", "AU", "SA"]  # Care, Fairness, Loyalty, Authority, Sanctity
INDIVIDUALIZING = ["CA", "FA"]
BINDING = ["LO", "AU", "SA"]


def load_filled(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("raw_score", "").strip()
            if raw and row.get("score_valid", "").strip() == "1":
                row["_score"] = int(raw)
            else:
                row["_score"] = None
            rows.append(row)
    return rows


def compute_dimension_means(rows: list[dict]) -> dict:
    """Per-model mean for each of the 5 foundations (exclude CT)."""
    scores = defaultdict(list)
    for r in rows:
        dim = r.get("dimension", "")
        if dim in FOUNDATIONS and r["_score"] is not None:
            scores[(r["model"], dim)].append(r["_score"])

    model_dims = defaultdict(dict)
    for (model, dim), vals in scores.items():
        if vals:
            model_dims[model][dim] = round(statistics.mean(vals), 3)
    return model_dims


def compute_z_scores(model_dims: dict) -> dict:
    """Z-score each foundation across models."""
    all_models = sorted(model_dims.keys())
    z = defaultdict(dict)
    for dim in FOUNDATIONS:
        values = [model_dims[m].get(dim) for m in all_models if dim in model_dims[m]]
        if len(values) < 2:
            continue
        mean = statistics.mean(values)
        sd = statistics.stdev(values)
        for m in all_models:
            if dim in model_dims[m]:
                z[m][f"z_{dim}"] = round((model_dims[m][dim] - mean) / sd, 3) if sd > 0 else 0.0
    return z


def compute_composite(model_dims: dict, z_scores: dict) -> dict:
    """MFQ composite = z(individualizing mean) - z(binding mean).
    Standardised across models for comparability with the old PVOC."""
    comp = {}
    for m in model_dims:
        ind = [z_scores[m].get(f"z_{d}", 0) for d in INDIVIDUALIZING]
        bind = [z_scores[m].get(f"z_{d}", 0) for d in BINDING]
        comp[m] = round(statistics.mean(ind) - statistics.mean(bind), 3)
    return comp


def main():
    rows = load_filled(INPUT_CSV)
    if not rows:
        print(f"ERROR: no valid rows in {INPUT_CSV}")
        return

    model_dims = compute_dimension_means(rows)
    z_scores = compute_z_scores(model_dims)
    composite = compute_composite(model_dims, z_scores)

    models = sorted(model_dims.keys())
    print(f"N = {len(models)} models\n")

    # ── stdout table ──
    header = f"{'Model':<25} " + " ".join(f"{d:>6}" for d in FOUNDATIONS) + f"{'Comp':>8}"
    print(header)
    print("-" * len(header))
    for m in sorted(composite, key=composite.get, reverse=True):
        dims = " ".join(f"{model_dims[m].get(d, '-'):>6}" for d in FOUNDATIONS)
        print(f"{m:<25} {dims} {composite[m]:>8.3f}")
    print("-" * len(header))

    # ── CSV scores ──
    fieldnames = ["model"] + FOUNDATIONS + [f"z_{d}" for d in FOUNDATIONS] + ["MFQ_composite"]
    with open(OUT_SCORES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in models:
            row = {"model": m}
            for d in FOUNDATIONS:
                row[d] = model_dims[m].get(d, "")
            for d in FOUNDATIONS:
                row[f"z_{d}"] = z_scores[m].get(f"z_{d}", "")
            row["MFQ_composite"] = composite[m]
            w.writerow(row)
    print(f"\nSaved: {OUT_SCORES}")

    # ── Table CSV (sorted by composite desc) ──
    table_rows = []
    for m in sorted(composite, key=composite.get, reverse=True):
        row = {"Model": m}
        for d in FOUNDATIONS:
            row[d] = model_dims[m].get(d, "")
        row["MFQ_composite"] = composite[m]
        table_rows.append(row)

    with open(OUT_TABLE_CSV, "w", newline="") as f:
        fieldnames = ["Model"] + FOUNDATIONS + ["MFQ_composite"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(table_rows)
    print(f"Saved: {OUT_TABLE_CSV}")

    # ── Table TeX ──
    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Moral Foundations Profiles by Model (MFQ-30, sorted by composite)}",
        r"\label{tab:mfq-profiles}",
        r"\begin{tabular}{l" + "c" * (len(FOUNDATIONS) + 1) + "}",
        r"\toprule",
        r"Model & " + " & ".join(FOUNDATIONS) + r" & MFQ Comp. \\",
        r"\midrule",
    ]
    for row in table_rows:
        vals = " & ".join(str(row.get(d, "")) for d in FOUNDATIONS)
        tex.append(f"{row['Model']} & {vals} & {row['MFQ_composite']} \\\\")
    tex.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    with open(OUT_TABLE_TEX, "w") as f:
        f.write("\n".join(tex) + "\n")
    print(f"Saved: {OUT_TABLE_TEX}")


if __name__ == "__main__":
    main()
