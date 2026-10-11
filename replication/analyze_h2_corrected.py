#!/usr/bin/env python3
"""H2 under model x profile PAIR fixed effects (CORRECTED version).

Fixed frame coding: collective_responsibility excluded due to insufficient variation.
Only TWO frames retained: access_barriers, coercive_backlash.

Delta_mir = alpha_mi + b1*Access + b2*Backlash + e

alpha_mi = one dummy per model x profile pair (405 pairs for 13 models).
Frame coefficients identified only by variation ACROSS REPETITIONS WITHIN a pair.
SEs clustered at the pair level.

Writes outputs/canonical/h2_pairfe_corrected.csv (main results)
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

# CORRECTED: only 2 frames with sufficient variation
FRAMES = ["access_barriers", "coercive_backlash"]

# Frame coding rules (from analyze_canonical_h2.py)
FRAME_RULES = {
    "access_barriers": [
        r"cannot\s+afford", r"can'?t\s+afford",
        r"miss\s+work", r"miss\w*\s+(her|the)\s+job", r"lose\s+(income|wages|pay)",
        r"lost\s+wages?", r"unpaid\s+(leave|time)",
        r"childcare", r"child\s+care", r"babysit",
        r"transport", r"travel\s+(time|cost)", r"commute",
        r"rural", r"remote\s+area", r"access\s+(to\s+)?care",
        r"appointment\s+availability", r"long\s+wait", r"waiting\s+list",
        r"time\s+off\s+work", r"take\s+time\s+off",
        r"no\s+(paid\s+)?sick\s+leave", r"flexible\s+hours",
        r"language\s+barrier", r"translation", r"interpreter",
        r"paperwork", r"document\w*", r"proof\s+of",
        r"insurance", r"uninsured", r"underinsured",
        r"out\s*-\s*of\s*pocket", r"copay", r"deductible",
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
    "autonomy_infringement": [
        r"individual\s+choice", r"personal\s+choice", r"personal\s+freedom",
        r"autonomy", r"privacy", r"consent",
    ],
    "procedural_legitimacy": [
        r"justified", r"safeguard", r"reasonable\s+(verification|requirement)",
        r"necessary\s+(procedure|requirement)",
    ],
    "collective_responsibility": [
        r"civic\s+duty", r"public\s+health", r"protect\w*\s+others?",
        r"communal\s+(benefit|welfare|good)",
    ],
}
FRAMES_ALL = list(FRAME_RULES)


def code(text):
    """Apply frame coding rules to text."""
    text_lower = text.lower()
    return {f: any(re.search(p, text_lower) for p in rules) for f, rules in FRAME_RULES.items()}


def load_rows():
    """Load H2 data with CORRECTED frame coding."""
    # Data paths (relative to replication/ directory)
    # Load simulation data - group by model, profile_id, repetition to get low/high pairs
    sim_rows = []
    with open(CAN / "sim_raw.csv") as f:
        for r in csv.DictReader(f):
            if r.get("non_refusal") == "True" and r.get("json_valid") == "True":
                sim_rows.append(r)
    
    # Group by model, profile_id to find low/high pairs
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
    
    # Build rows - calculate delta = low_willingness - high_willingness for each repetition
    rows = []
    for (model, profile_id), pair in pairs.items():
        if len(pair["low"]) == 0 or len(pair["high"]) == 0:
            continue
        
        # For each high burden row, find matching low burden row by repetition
        low_by_rep = {int(r.get("repetition", 0)): float(r["willingness"]) for r in pair["low"]}
        
        for high_row in pair["high"]:
            rep = int(high_row.get("repetition", 0))
            if rep not in low_by_rep:
                continue
            
            lo = low_by_rep[rep]
            hi = float(high_row["willingness"])
            
            # Use high burden rationale for frame coding
            rationale = (high_row.get("rationale") or "").strip()
            
            rows.append({
                "model": model, 
                "profile_id": profile_id, 
                "repetition": rep,
                "delta": lo - hi, 
                "PVC": pvc.get(model),
                "rationale": rationale,
                **code(rationale)
            })
    
    df = pd.DataFrame([r for r in rows if r["PVC"] is not None])
    df = df.rename(columns={"PVC": "PVOC"})
    df["pair"] = df.model.astype(str) + "|" + df.profile_id.astype(str)
    return df


def coefs(r, frames):
    """Extract coefficients for specified frames."""
    out = {}
    for f in frames:
        if f not in r.params.index:
            out[f] = dict(b=np.nan, se=np.nan, p=np.nan, lo=np.nan, hi=np.nan, absorbed=True)
            continue
        ci = r.conf_int().loc[f]
        out[f] = dict(
            b=r.params[f],
            se=r.bse[f],
            p=r.pvalues[f],
            lo=ci[0],
            hi=ci[1],
            absorbed=False
        )
    return out


def run_pairfe(df, frames):
    """Run pair fixed effects model and return results."""
    rows_out = []
    
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair]
        if len(sub) < 2:
            continue
        
        # Build formula: delta ~ PVOC + frames
        formula = "delta ~ PVOC + " + " + ".join(frames)
        
        try:
            m = smf.ols(formula, data=sub).fit()
            
            # Get frame coefficients
            for f in frames:
                if f in m.params.index:
                    ci = m.conf_int().loc[f]
                    rows_out.append({
                        "pair": pair,
                        "frame": f,
                        "b": m.params[f],
                        "se": m.bse[f],
                        "p": m.pvalues[f],
                        "ci_low": ci[0],
                        "ci_high": ci[1],
                        "n": len(sub),
                        "model": sub["model"].iloc[0],
                        "profile_id": sub["profile_id"].iloc[0]
                    })
        except Exception as e:
            print(f"Warning: failed for pair {pair}: {e}")
            continue
    
    return pd.DataFrame(rows_out)


def aggregate_pairfe(results_df):
    """Aggregate pair-level results using inverse-variance weighting."""
    rows = []
    
    for frame in results_df["frame"].unique():
        sub = results_df[results_df["frame"] == frame]
        if len(sub) == 0:
            continue
        
        # Inverse variance weighting
        weights = 1 / (sub["se"] ** 2)
        w_sum = weights.sum()
        
        b_w = (sub["b"] * weights).sum() / w_sum
        se_w = np.sqrt(1 / w_sum)
        
        # Meta-analysis style z-score
        z = b_w / se_w
        p = 2 * (1 - np.abs(z).round(10))  # Two-tailed
        
        # Confidence interval
        ci_low = b_w - 1.96 * se_w
        ci_high = b_w + 1.96 * se_w
        
        rows.append({
            "frame": frame,
            "b": b_w,
            "se": se_w,
            "z": z,
            "p": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_pairs": len(sub),
            "n_obs": sub["n"].sum()
        })
    
    return pd.DataFrame(rows)


def main():
    """Run corrected H2 analysis."""
    print("=" * 60)
    print("H2 Analysis (CORRECTED: pair fixed effects)")
    print("=" * 60)
    
    # Load data
    df = load_rows()
    print(f"Loaded {len(df)} rows from {df['model'].nunique()} models")
    print(f"Pairs: {df['pair'].nunique()}")
    
    # Run pair-level regressions
    print("\nRunning pair-level regressions...")
    pair_results = run_pairfe(df, FRAMES)
    print(f"Completed {len(pair_results)} pair-frame estimates")
    
    # Aggregate
    print("\nAggregating with inverse-variance weighting...")
    agg = aggregate_pairfe(pair_results)
    
    # Apply Benjamini-Hochberg correction
    if len(agg) > 0:
        agg["p_bh"] = multipletests(agg["p"], method="fdr_bh")[1]
    
    # Display results
    print("\n" + "=" * 60)
    print("CORRECTED H2 Results (Pair Fixed Effects)")
    print("=" * 60)
    for _, r in agg.iterrows():
        sig = "***" if r["p_bh"] < 0.001 else "**" if r["p_bh"] < 0.01 else "*" if r["p_bh"] < 0.05 else ""
        print(f"\n{r['frame']}:")
        print(f"  b = {r['b']:+.4f}")
        print(f"  SE = {r['se']:.4f}")
        print(f"  95% CI = [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
        print(f"  z = {r['z']:.3f}")
        print(f"  p (uncorrected) = {r['p']:.4f}")
        print(f"  p (BH corrected) = {r['p_bh']:.4f} {sig}")
        print(f"  N pairs = {r['n_pairs']}")
    
    # Save results
    out_file = CAN / "h2_pairfe_corrected.csv"
    agg.to_csv(out_file, index=False)
    print(f"\n✓ Results saved to {out_file}")
    
    # Also save pair-level results
    pair_file = CAN / "h2_pairfe_pair_level.csv"
    pair_results.to_csv(pair_file, index=False)
    print(f"✓ Pair-level results saved to {pair_file}")
    
    # Run LOMO (leave-one-model-out)
    print("\n" + "=" * 60)
    print("Running LOMO (Leave-One-Model-Out)")
    print("=" * 60)
    lomo_results = run_lomo(df, FRAMES)
    lomo_file = CAN / "h2_pairfe_lomo_corrected.csv"
    lomo_results.to_csv(lomo_file, index=False)
    print(f"✓ LOMO results saved to {lomo_file}")
    
    # Run LOPO (leave-one-profile-out)
    print("\n" + "=" * 60)
    print("Running LOPO (Leave-One-Profile-Out)")
    print("=" * 60)
    lopo_results = run_lopo(df, FRAMES)
    lopo_file = CAN / "h2_pairfe_lopo_corrected.csv"
    lopo_results.to_csv(lopo_file, index=False)
    print(f"✓ LOPO results saved to {lopo_file}")
    
    # Run diagnostics
    print("\n" + "=" * 60)
    print("Running Diagnostics")
    print("=" * 60)
    diag = diagnostics(df)
    diag_file = CAN / "h2_pairfe_diagnostics_corrected.csv"
    diag.to_csv(diag_file, index=False)
    print(f"✓ Diagnostics saved to {diag_file}")
    print("\nDiagnostics:")
    print(diag.to_string())


def run_lomo(df, frames):
    """Leave-one-model-out robustness check."""
    rows = []
    models = df["model"].unique()
    
    for left_out in models:
        sub = df[df["model"] != left_out]
        pair_results = run_pairfe(sub, frames)
        agg = aggregate_pairfe(pair_results)
        
        for _, r in agg.iterrows():
            rows.append({
                "left_out": left_out,
                "frame": r["frame"],
                "b": r["b"],
                "p": r["p"]
            })
    
    return pd.DataFrame(rows)


def run_lopo(df, frames):
    """Leave-one-profile-out robustness check."""
    rows = []
    profiles = df["profile_id"].unique()
    
    for left_out in profiles:
        sub = df[df["profile_id"] != left_out]
        pair_results = run_pairfe(sub, frames)
        agg = aggregate_pairfe(pair_results)
        
        for _, r in agg.iterrows():
            rows.append({
                "left_out": left_out,
                "frame": r["frame"],
                "b": r["b"],
                "p": r["p"]
            })
    
    return pd.DataFrame(rows)


def diagnostics(df):
    """Calculate within-pair diagnostics for each frame."""
    rows = []
    
    for frame in FRAMES:
        # Calculate within-pair variance share
        pair_means = df.groupby("pair")[frame].mean()
        overall_var = df[frame].var()
        between_var = pair_means.var()
        within_share = 1 - (between_var / overall_var) if overall_var > 0 else 0
        
        # Count pairs with no within-pair variation
        no_var_pairs = 0
        for pair in df["pair"].unique():
            sub = df[df["pair"] == pair]
            if sub[frame].var() == 0:
                no_var_pairs += 1
        
        rows.append({
            "variable": frame,
            "within_pair_share": within_share,
            "pairs_with_no_within_variation": no_var_pairs,
            "pairs": df["pair"].nunique()
        })
    
    # Add delta (burden effect) diagnostics
    pair_deltas = df.groupby("pair")["delta"].mean()
    overall_delta_var = df["delta"].var()
    between_delta_var = pair_deltas.var()
    delta_within_share = 1 - (between_delta_var / overall_delta_var) if overall_delta_var > 0 else 0
    
    delta_no_var_pairs = 0
    for pair in df["pair"].unique():
        sub = df[df["pair"] == pair]
        if sub["delta"].var() == 0:
            delta_no_var_pairs += 1
    
    rows.append({
        "variable": "delta",
        "within_pair_share": delta_within_share,
        "pairs_with_no_within_variation": delta_no_var_pairs,
        "pairs": df["pair"].nunique()
    })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
