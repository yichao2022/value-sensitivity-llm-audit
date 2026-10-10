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
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HOME = Path(__file__).resolve().parent
CAN = HOME / "outputs" / "canonical"

# CORRECTED: only 2 frames with sufficient variation
FRAMES = ["access_barriers", "coercive_backlash"]


def load_rows():
    """Load H2 data with CORRECTED frame coding."""
    spec = importlib.util.spec_from_file_location("h2", HOME / "analyze_canonical_h2.py")
    h2 = importlib.util.module_from_spec(spec)
    sys.modules["h2"] = h2
    spec.loader.exec_module(h2)
    df = pd.DataFrame(h2.load()).rename(columns={"PVC": "PVOC"})
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


def fit_pair_fe(df, cluster_se=True):
    """LSDV: pair dummies + frames."""
    formula = "delta ~ C(pair) + " + " + ".join(FRAMES)
    r = smf.ols(formula, data=df)
    return r.fit(cov_type="cluster", cov_kwds={"groups": df["pair"]}) if cluster_se else r.fit()


def fit_two_way(df):
    """Two-way FE: model + profile."""
    formula = "delta ~ C(model) + C(profile_id) + " + " + ".join(FRAMES)
    r = smf.ols(formula, data=df)
    return r.fit(cov_type="cluster", cov_kwds={"groups": df["pair"]})


def diagnostics(df):
    """Within-pair variation diagnostics."""
    rows = []
    check_vars = FRAMES + ["delta"]
    for f in check_vars:
        grp = df.groupby("pair")[f]
        within = grp.transform("mean")
        wv = ((df[f] - within) ** 2).mean()
        tv = ((df[f] - df[f].mean()) ** 2).mean()
        n_const = int((grp.nunique() == 1).sum())
        rows.append({
            "variable": f, 
            "within_pair_share": wv / tv if tv > 0 else 0,
            "pairs_with_no_within_variation": n_const, 
            "pairs": df["pair"].nunique()
        })
    return pd.DataFrame(rows)


def leave_one_model_out(df):
    """LOMO diagnostics under pair-FE specification."""
    models = df["model"].unique()
    results = []
    for m in models:
        d = df[df["model"] != m].copy()
        try:
            r = fit_pair_fe(d)
            for f in FRAMES:
                results.append({
                    "left_out": m,
                    "frame": f,
                    "b": r.params.get(f, np.nan),
                    "p": r.pvalues.get(f, np.nan)
                })
        except Exception as e:
            for f in FRAMES:
                results.append({
                    "left_out": m,
                    "frame": f,
                    "b": np.nan,
                    "p": np.nan,
                    "error": str(e)
                })
    return pd.DataFrame(results)


def leave_one_profile_out(df):
    """LOPO diagnostics under pair-FE specification."""
    profiles = df["profile_id"].unique()
    results = []
    for p in profiles:
        d = df[df["profile_id"] != p].copy()
        try:
            r = fit_pair_fe(d)
            for f in FRAMES:
                results.append({
                    "left_out": p,
                    "frame": f,
                    "b": r.params.get(f, np.nan),
                    "p": r.pvalues.get(f, np.nan)
                })
        except Exception as e:
            for f in FRAMES:
                results.append({
                    "left_out": p,
                    "frame": f,
                    "b": np.nan,
                    "p": np.nan,
                    "error": str(e)
                })
    return pd.DataFrame(results)


if __name__ == "__main__":
    print("=" * 70)
    print("H2 Analysis: Model x Profile Pair Fixed Effects (CORRECTED)")
    print("=" * 70)
    
    # Load data
    df = load_rows()
    print(f"\nSample: N = {len(df)}, {df.model.nunique()} models, "
          f"{df.profile_id.nunique()} profiles, {df['pair'].nunique()} pairs")
    
    # Diagnostics
    dd = diagnostics(df)
    print("\n=== Identification Diagnostics ===")
    for _, r in dd.iterrows():
        print(f"  {r.variable:<25} within-pair share = {r.within_pair_share:.3f}   "
              f"no variation: {int(r.pairs_with_no_within_variation)}/{int(r.pairs)}")
    
    # Main regression (pair FE)
    r_pair = fit_pair_fe(df)
    v_pair = coefs(r_pair, FRAMES)
    
    # Two-way FE for comparison
    r_two = fit_two_way(df)
    v_two = coefs(r_two, FRAMES)
    
    # BH adjustment
    pvals = [v_pair[f]["p"] for f in FRAMES]
    _, p_bh, _, _ = multipletests(pvals, method="fdr_bh", alpha=0.10)
    
    # Build main results table
    results = []
    for i, f in enumerate(FRAMES):
        results.append({
            "frame": f,
            "b_pairfe": v_pair[f]["b"],
            "se_pairfe": v_pair[f]["se"],
            "p_pairfe": v_pair[f]["p"],
            "lo_pairfe": v_pair[f]["lo"],
            "hi_pairfe": v_pair[f]["hi"],
            "p_bh": p_bh[i],
            "b_twoway": v_two[f]["b"],
            "absorbed": v_pair[f].get("absorbed", False)
        })
    
    df_results = pd.DataFrame(results)
    
    print("\n=== Main Results (Pair FE) ===")
    print(df_results.to_string(index=False))
    
    # Save main results
    CAN.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(CAN / "h2_pairfe_corrected.csv", index=False)
    print(f"\nSaved: {CAN / 'h2_pairfe_corrected.csv'}")
    
    # LOMO
    print("\n=== Running LOMO (Leave-One-Model-Out) ===")
    lomo = leave_one_model_out(df)
    lomo.to_csv(CAN / "h2_pairfe_lomo_corrected.csv", index=False)
    print(f"Saved: {CAN / 'h2_pairfe_lomo_corrected.csv'}")
    
    # LOPO
    print("\n=== Running LOPO (Leave-One-Profile-Out) ===")
    lopo = leave_one_profile_out(df)
    lopo.to_csv(CAN / "h2_pairfe_lopo_corrected.csv", index=False)
    print(f"Saved: {CAN / 'h2_pairfe_lopo_corrected.csv'}")
    
    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)
