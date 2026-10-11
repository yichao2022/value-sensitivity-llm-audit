#!/usr/bin/env python3
"""H2 pooled Pair-FE analysis with FROZEN strict frame coding.

This script performs the corrected H2 analysis using pooled pair fixed effects
with STRICT coding rules (frozen as of 2026-10-10):
- Access barriers: ONLY explicit resource constraints (cost, time, geography)
- EXCLUDED: paperwork, document, proof of, long wait, waiting list (too broad)
- One fixed effect per model-profile pair (405 pairs)
- Frame coefficients identified from within-pair variation
- Standard errors clustered at the pair level
- Benjamini-Hochberg correction for 2 frames

Frames: access_barriers, coercive_backlash (collective_responsibility excluded)
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HOME = Path(__file__).resolve().parent
CAN = HOME / "outputs" / "canonical"

# FROZEN STRICT: only 2 frames with sufficient variation
FRAMES = ["access_barriers", "coercive_backlash"]

# FROZEN STRICT frame coding rules (2026-10-10)
# Key change: REMOVED paperwork, document, proof of, long wait, waiting list
FRAME_RULES = {
    "access_barriers": [
        # Financial barriers
        r"cannot\s+afford", r"can'?t\s+afford",
        r"miss\s+work", r"miss\w*\s+(her|the)\s+job",
        r"lose\s+(income|wages|pay)", r"lost\s+wages?",
        r"unpaid\s+(leave|time)",
        r"out\s*-\s*of\s*pocket", r"copay", r"deductible",
        # Care/work barriers
        r"childcare", r"child\s+care", r"babysit",
        r"no\s+(paid\s+)?sick\s+leave", r"flexible\s+hours",
        # Geographic/transport barriers
        r"transport", r"commute",
        r"rural", r"remote\s+area",
        # Insurance
        r"insurance", r"uninsured", r"underinsured",
        # Language
        r"language\s+barrier", r"translation", r"interpreter",
        # EXCLUDED (too broad, no explicit resource link):
        # - paperwork, document, proof of
        # - long wait, waiting list, appointment availability
        # - travel time, travel cost (ambiguous)
    ],
    "coercive_backlash": [
        r"government\s+overreach", r"government\s+control",
        r"force", r"coerce", r"mandate", r"compel",
        r"pressure", r"arm\s*-\s*twist",
        r"violate", r"infringe", r"encroach",
        r"my\s+body", r"my\s+choice", r"my\s+decision",
        r"medical\s+freedom", r"health\s+freedom",
        r"parental\s+rights", r"parent\s*'?\s+decision",
        r"trust\s+(issues|problem)", r"distrust",
        r"experiment\w*", r"guinea\s+pig", r"test\s+subject",
        r"unknown\s+(side\s*)?effects?", r"long\s*-\s*term\s+effects?",
        r"rushed", r"too\s+fast", r"not\s+enough\s+(time|research|data)",
        r"big\s+pharma", r"profit", r"money\s+grab",
        r"corrupt", r"corruption", r"bribe",
        r"conspiracy", r"cover\s*-\s*up", r"hidden\s+agenda",
        r"punitive", r"penalty", r"lose\s+(benefits|rights)",
    ],
}


def code(text):
    """Apply frame coding rules to text."""
    text_lower = text.lower()
    return {f: any(re.search(p, text_lower) for p in rules) for f, rules in FRAME_RULES.items()}


def load_data():
    """Load and prepare H2 data."""
    # Load simulation data
    sim_rows = []
    with open(CAN / "sim_raw.csv") as f:
        for r in csv.DictReader(f):
            if r.get("non_refusal") == "True" and r.get("json_valid") == "True":
                sim_rows.append(r)
    
    # Group by model, profile_id
    pairs = defaultdict(lambda: {"low": [], "high": []})
    for r in sim_rows:
        key = (r["model"], r["profile_id"])
        burden = r["burden_level"]
        if burden in ("low", "high"):
            pairs[key][burden].append(r)
    
    # Load PVC data
    pvc = {}
    with open(CAN / "table2_orientation_profiles.csv") as f:
        for r in csv.DictReader(f):
            pvc[r["Model"]] = float(r["PVC"])
    
    # Build rows
    rows = []
    for (model, profile_id), pair in pairs.items():
        if len(pair["low"]) == 0 or len(pair["high"]) == 0:
            continue
        
        low_by_rep = {int(r.get("repetition", 0)): float(r["willingness"]) for r in pair["low"]}
        
        for high_row in pair["high"]:
            rep = int(high_row.get("repetition", 0))
            if rep not in low_by_rep:
                continue
            
            lo = low_by_rep[rep]
            hi = float(high_row["willingness"])
            rationale = (high_row.get("rationale") or "").strip()
            
            coded = code(rationale)
            rows.append({
                "model": model,
                "profile_id": profile_id,
                "repetition": rep,
                "delta": lo - hi,
                "PVC": pvc.get(model),
                "rationale": rationale,
                "pair": f"{model}|{profile_id}",
                **coded
            })
    
    df = pd.DataFrame([r for r in rows if r["PVC"] is not None])
    df = df.rename(columns={"PVC": "PVOC"})
    return df


def compute_prevalence(df):
    """Compute frame prevalence statistics."""
    stats = {}
    for frame in FRAMES:
        n_present = df[frame].sum()
        n_total = len(df)
        stats[frame] = {
            "n_present": int(n_present),
            "n_total": n_total,
            "prevalence_pct": 100.0 * n_present / n_total if n_total > 0 else 0
        }
    return stats


def run_pooled_pairfe(df, frames):
    """Run pooled pair-FE regression with cluster-robust SEs."""
    results = []
    
    for frame in frames:
        formula = f"delta ~ {frame} + C(pair)"
        
        try:
            m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["pair"]})
            
            if frame in m.params.index:
                ci = m.conf_int().loc[frame]
                results.append({
                    "frame": frame,
                    "b": m.params[frame],
                    "se": m.bse[frame],
                    "p": m.pvalues[frame],
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "n_obs": len(df),
                    "n_pairs": df["pair"].nunique(),
                })
            else:
                frame_col = f"{frame}[T.True]"
                if frame_col in m.params.index:
                    ci = m.conf_int().loc[frame_col]
                    results.append({
                        "frame": frame,
                        "b": m.params[frame_col],
                        "se": m.bse[frame_col],
                        "p": m.pvalues[frame_col],
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                        "n_obs": len(df),
                        "n_pairs": df["pair"].nunique(),
                    })
        except Exception as e:
            print(f"Warning: failed for frame {frame}: {e}")
            continue
    
    return pd.DataFrame(results)


def run_lomo(df, frames):
    """Leave-one-model-out robustness check."""
    rows = []
    models = df["model"].unique()
    
    for left_out in models:
        sub = df[df["model"] != left_out].copy()
        for frame in frames:
            formula = f"delta ~ {frame} + C(pair)"
            try:
                m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["pair"]})
                if frame in m.params.index:
                    rows.append({
                        "left_out": left_out,
                        "frame": frame,
                        "b": m.params[frame],
                        "p": m.pvalues[frame]
                    })
                else:
                    frame_col = f"{frame}[T.True]"
                    if frame_col in m.params.index:
                        rows.append({
                            "left_out": left_out,
                            "frame": frame,
                            "b": m.params[frame_col],
                            "p": m.pvalues[frame_col]
                        })
            except:
                continue
    
    return pd.DataFrame(rows)


def run_lopo(df, frames):
    """Leave-one-profile-out robustness check."""
    rows = []
    profiles = df["profile_id"].unique()
    
    for left_out in profiles:
        sub = df[df["profile_id"] != left_out].copy()
        for frame in frames:
            formula = f"delta ~ {frame} + C(pair)"
            try:
                m = smf.ols(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["pair"]})
                if frame in m.params.index:
                    rows.append({
                        "left_out": left_out,
                        "frame": frame,
                        "b": m.params[frame],
                        "p": m.pvalues[frame]
                    })
                else:
                    frame_col = f"{frame}[T.True]"
                    if frame_col in m.params.index:
                        rows.append({
                            "left_out": left_out,
                            "frame": frame,
                            "b": m.params[frame_col],
                            "p": m.pvalues[frame_col]
                        })
            except:
                continue
    
    return pd.DataFrame(rows)


def compute_diagnostics(df):
    """Compute within-pair diagnostics."""
    rows = []
    
    for frame in FRAMES:
        n_no_var = 0
        for pair in df["pair"].unique():
            sub = df[df["pair"] == pair]
            if len(sub) > 0:
                var = sub[frame].var()
                if var == 0 or pd.isna(var):
                    n_no_var += 1
        
        rows.append({
            "variable": frame,
            "pairs_with_no_within_variation": n_no_var,
            "pairs": df["pair"].nunique()
        })
    
    n_no_var_delta = 0
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair]
        if len(sub) > 0:
            var = sub["delta"].var()
            if var == 0 or pd.isna(var):
                n_no_var_delta += 1
    
    rows.append({
        "variable": "delta",
        "pairs_with_no_within_variation": n_no_var_delta,
        "pairs": df["pair"].nunique()
    })
    
    return pd.DataFrame(rows)


def main():
    """Run frozen H2 analysis."""
    print("=" * 60)
    print("H2 Analysis (FROZEN STRICT Coding: 2026-10-10)")
    print("=" * 60)
    print("\nCoding rules:")
    print("  Access barriers: cost, time, geography, insurance, language")
    print("  EXCLUDED: paperwork, document, proof of, long wait, waiting list")
    
    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} rows from {df['model'].nunique()} models")
    print(f"Pairs: {df['pair'].nunique()}")
    
    # Compute prevalence
    print("\n" + "=" * 60)
    print("FRAME PREVALENCE (FROZEN RULES)")
    print("=" * 60)
    prev_stats = compute_prevalence(df)
    for frame, stats in prev_stats.items():
        print(f"\n{frame}:")
        print(f"  Present: {stats['n_present']}/{stats['n_total']} ({stats['prevalence_pct']:.1f}%)")
    
    # Run pooled pair-FE regression
    print("\n" + "=" * 60)
    print("Running pooled pair-FE regression with cluster-robust SEs...")
    print("=" * 60)
    results = run_pooled_pairfe(df, FRAMES)
    
    if len(results) == 0:
        print("ERROR: No results generated. Check data and model specification.")
        return 1
    
    # Apply Benjamini-Hochberg correction
    results["p_bh"] = multipletests(results["p"], method="fdr_bh")[1]
    
    # Display results
    print("\n" + "=" * 60)
    print("FROZEN H2 Results (Pooled Pair-FE)")
    print("=" * 60)
    for _, r in results.iterrows():
        sig = "***" if r["p_bh"] < 0.001 else "**" if r["p_bh"] < 0.01 else "*" if r["p_bh"] < 0.05 else ""
        print(f"\n{r['frame']}:")
        print(f"  b = {r['b']:+.4f}")
        print(f"  SE = {r['se']:.4f}")
        print(f"  95% CI = [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        print(f"  p = {r['p']:.4f}")
        print(f"  p (BH corrected) = {r['p_bh']:.4f} {sig}")
        print(f"  N = {r['n_obs']} observations, {r['n_pairs']} pairs")
    
    # Save main results
    out_file = CAN / "h2_pairfe_frozen.csv"
    results.to_csv(out_file, index=False)
    print(f"\n✓ Main results saved to {out_file}")
    
    # Save prevalence
    prev_df = pd.DataFrame([
        {"frame": f, "n_present": s["n_present"], "n_total": s["n_total"], 
         "prevalence_pct": s["prevalence_pct"]}
        for f, s in prev_stats.items()
    ])
    prev_file = CAN / "h2_prevalence_frozen.csv"
    prev_df.to_csv(prev_file, index=False)
    print(f"✓ Prevalence saved to {prev_file}")
    
    # Run LOMO
    print("\n" + "=" * 60)
    print("Running LOMO (Leave-One-Model-Out)")
    print("=" * 60)
    lomo_results = run_lomo(df, FRAMES)
    lomo_file = CAN / "h2_pairfe_lomo_frozen.csv"
    lomo_results.to_csv(lomo_file, index=False)
    print(f"✓ LOMO results saved to {lomo_file} ({len(lomo_results)} rows)")
    
    # Run LOPO
    print("\n" + "=" * 60)
    print("Running LOPO (Leave-One-Profile-Out)")
    print("=" * 60)
    lopo_results = run_lopo(df, FRAMES)
    lopo_file = CAN / "h2_pairfe_lopo_frozen.csv"
    lopo_results.to_csv(lopo_file, index=False)
    print(f"✓ LOPO results saved to {lopo_file} ({len(lopo_results)} rows)")
    
    # Compute diagnostics
    print("\n" + "=" * 60)
    print("Computing Diagnostics")
    print("=" * 60)
    diag = compute_diagnostics(df)
    diag_file = CAN / "h2_pairfe_diagnostics_frozen.csv"
    diag.to_csv(diag_file, index=False)
    print(f"✓ Diagnostics saved to {diag_file}")
    print("\nDiagnostics:")
    for _, r in diag.iterrows():
        has_var = r['pairs'] - r['pairs_with_no_within_variation']
        print(f"  {r['variable']}: {r['pairs_with_no_within_variation']}/{r['pairs']} pairs no variation, {has_var} pairs have variation")
    
    print("\n" + "=" * 60)
    print("✓ H2 frozen analysis complete")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
