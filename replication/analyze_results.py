#!/usr/bin/env python3
"""
Table 1 Diagnostics — LLM Policy Simulation
Reads outputs/simulation_outputs.csv and generates Table 1 by model:
  N, JSON validity, score compliance, non-refusal rate,
  mean willingness, repetition stability SD,
  Delta SD (burden effect variability), and sign-flip rate.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path
import openpyxl

PROJECT = Path(__file__).resolve().parent
OUTPUT_CSV = PROJECT / "outputs" / "simulation_outputs.csv"


def load_plan_lookup(xlsx_path: Path) -> dict:
    """Return {run_id: {profile_id, burden_level}} mapping from Simulation Plan."""
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb["Simulation Plan"]
    rows = list(ws.iter_rows(values_only=True))
    headers = [str(h) for h in rows[0]]
    lookup = {}
    for row in rows[1:]:
        record = dict(zip(headers, row))
        lookup[str(record["run_id"])] = {
            "profile_id": str(record["profile_id"]),
            "burden_level": str(record["burden_level"]),
        }
    wb.close()
    return lookup


def load_results(path: Path) -> list[dict]:
    """Load simulation output CSV into list of dicts."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for field in ("json_valid", "score_valid", "non_refusal"):
                val = row.get(field, "").strip().lower()
                row[field] = val in ("true", "1", "yes")
            w = row.get("willingness", "").strip()
            row["willingness"] = float(w) if w else None
            rows.append(row)
    return rows


def compute_table1(rows: list[dict]) -> list[dict]:
    """Compute per-model diagnostics including delta-SD and sign-flip rate."""
    # Group by model
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    # For repetition stability: group by (model, profile_id, burden_level)
    stability_groups = defaultdict(list)
    for r in rows:
        key = (r["model"], r.get("profile_id", ""), r.get("burden_level", ""))
        if r["willingness"] is not None:
            stability_groups[key].append(r["willingness"])

    # Pre-compute per-model list of SDs
    model_sds = defaultdict(list)
    for (model, pid, blevel), scores in stability_groups.items():
        if len(scores) >= 2:
            try:
                sd = statistics.stdev(scores)
                model_sds[model].append(sd)
            except statistics.StatisticsError:
                pass

    # ── Delta SD and sign-flip rate ──────────────────────────────
    # For each (model, profile), compute mean willingness under high vs low burden
    # Delta = mean_high - mean_low  (expected negative: higher burden → lower willingness)
    profile_means = defaultdict(list)
    for r in rows:
        if r["willingness"] is not None:
            profile_means[(r["model"], r["profile_id"], r["burden_level"])].append(r["willingness"])

    model_deltas = defaultdict(list)        # list of deltas per model
    model_sign_flips = defaultdict(int)      # count of sign flips (Δ > 0)
    model_delta_count = defaultdict(int)     # total delta comparisons

    for (model, pid, blevel), scores in profile_means.items():
        pass  # We need pairs, iterate differently

    # Build pairs
    by_profile = defaultdict(lambda: defaultdict(dict))
    for (model, pid, blevel), scores in profile_means.items():
        by_profile[(model, pid)][blevel] = statistics.mean(scores)

    for (model, pid), burdens in by_profile.items():
        if "high" in burdens and "low" in burdens:
            delta = burdens["high"] - burdens["low"]
            model_deltas[model].append(delta)
            model_delta_count[model] += 1
            if delta > 0:  # sign flip: higher burden → HIGHER willingness
                model_sign_flips[model] += 1

    table = []
    for model in sorted(by_model.keys()):
        group = by_model[model]
        n = len(group)
        n_json_valid = sum(1 for r in group if r["json_valid"])
        n_score_valid = sum(1 for r in group if r["score_valid"])
        n_non_refusal = sum(1 for r in group if r["non_refusal"])
        scores = [r["willingness"] for r in group if r["willingness"] is not None]

        mean_willingness = statistics.mean(scores) if scores else None
        sds = model_sds.get(model, [])
        mean_stability_sd = statistics.mean(sds) if sds else None

        # Delta SD
        deltas = model_deltas.get(model, [])
        delta_sd = statistics.stdev(deltas) if len(deltas) >= 2 else None

        # Sign-flip rate: Pr(Δ > 0) i.e. higher burden → higher willingness
        total_deltas = model_delta_count.get(model, 0)
        flips = model_sign_flips.get(model, 0)
        sign_flip_rate = round(100 * flips / total_deltas, 1) if total_deltas > 0 else None

        table.append({
            "model": model,
            "N": n,
            "json_valid_pct": round(100 * n_json_valid / n, 1) if n else 0,
            "score_valid_pct": round(100 * n_score_valid / n, 1) if n else 0,
            "non_refusal_pct": round(100 * n_non_refusal / n, 1) if n else 0,
            "mean_willingness": round(mean_willingness, 2) if mean_willingness is not None else "-",
            "stability_sd": round(mean_stability_sd, 2) if mean_stability_sd is not None else "-",
            "delta_sd": round(delta_sd, 2) if delta_sd is not None else "-",
            "sign_flip_pct": sign_flip_rate if sign_flip_rate is not None else "-",
        })

    return table


def print_table(table: list[dict]):
    """Pretty-print Table 1."""
    header = (
        f"{'Model':<30} {'N':>5} {'JSON%':>7} {'Scor%':>7} {'NoRef%':>7} "
        f"{'MeanW':>7} {'StabSD':>7} {'DeltaSD':>8} {'Flip%':>6}"
    )
    sep = "-" * len(header)
    print("\nTable 1: Simulation Diagnostics by Model\n")
    print(sep)
    print(header)
    print(sep)
    for row in table:
        print(
            f"{row['model']:<30} "
            f"{row['N']:>5} "
            f"{row['json_valid_pct']:>7.1f} "
            f"{row['score_valid_pct']:>7.1f} "
            f"{row['non_refusal_pct']:>7.1f} "
            f"{str(row['mean_willingness']):>7} "
            f"{str(row['stability_sd']):>7} "
            f"{str(row['delta_sd']):>8} "
            f"{str(row['sign_flip_pct']):>6}"
        )
    print(sep)

    # Print interpretation note
    print("\nNotes:")
    print("  StabSD  = mean within-(profile×burden) SD across 5 repetitions (lower = more deterministic)")
    print("  DeltaSD = SD of (mean_high_burden − mean_low_burden) across 27 profiles")
    print("  Flip%   = % of profiles where higher burden → higher willingness (expected: ~0%)")


def export_csv(table: list[dict], path: Path):
    """Export Table 1 to CSV."""
    fields = [
        "model", "N", "json_valid_pct", "score_valid_pct", "non_refusal_pct",
        "mean_willingness", "stability_sd", "delta_sd", "sign_flip_pct",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(f"\nTable 1 saved to: {path}")


def main():
    if not OUTPUT_CSV.exists():
        print(f"ERROR: {OUTPUT_CSV} not found. Run run_simulation.py first.")
        sys.exit(1)

    rows = load_results(OUTPUT_CSV)
    print(f"Loaded {len(rows)} result rows from {OUTPUT_CSV}")

    # Build run_id → profile/burden lookup from plan
    plan_xlsx = PROJECT / "data" / "llm_experiment_inputs.xlsx"
    plan_lookup = load_plan_lookup(plan_xlsx)
    # Attach profile_id and burden_level to each row
    for r in rows:
        rid = r["run_id"]
        if rid in plan_lookup:
            r["profile_id"] = plan_lookup[rid]["profile_id"]
            r["burden_level"] = plan_lookup[rid]["burden_level"]

    table = compute_table1(rows)
    print_table(table)

    export_csv(table, PROJECT / "outputs" / "table1_diagnostics.csv")


if __name__ == "__main__":
    import sys
    main()
