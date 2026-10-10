#!/usr/bin/env python3
"""H2 under model x profile PAIR fixed effects (the 2026-09-17 proposal).

    Delta_mir = alpha_mi + b1*Access + b2*Collective + b3*Backlash + e

alpha_mi = one dummy per model x profile pair (405 pairs for the 15-model
sample), so the frame coefficients are identified only by variation ACROSS
REPETITIONS WITHIN a pair. SEs clustered at the pair level (405 clusters).

Reports, for the same three retained frames:
  * pair-FE betas, SEs, p-values, CIs, BH-adjusted p (q = 0.10)
  * the two-way FE (model FE + profile FE) estimate for comparison
  * identification diagnostics: within-pair share of variance in each frame
    indicator and in Delta, and how many pairs have any within-pair variation
  * leave-one-model-out (15 fits) under the pair-FE specification
Writes outputs/canonical/h2_pairfe_main.csv and h2_pairfe_lomo.csv.

Single source for frame coding: analyze_canonical_h2.load().
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
FRAMES = ["access_barriers", "collective_responsibility", "coercive_backlash"]


def load_rows():
    spec = importlib.util.spec_from_file_location("h2", HOME / "analyze_canonical_h2.py")
    h2 = importlib.util.module_from_spec(spec)
    sys.modules["h2"] = h2
    spec.loader.exec_module(h2)
    df = pd.DataFrame(h2.load()).rename(columns={"PVC": "PVOC"})
    df["pair"] = df.model.astype(str) + "|" + df.profile_id.astype(str)
    return df


def coefs(r):
    out = {}
    for f in FRAMES:
        assert f in r.params.index, f"missing {f}"
        ci = r.conf_int().loc[f]
        out[f] = dict(b=r.params[f], se=r.bse[f], p=r.pvalues[f], lo=ci[0], hi=ci[1])
    return out


def fit_pair_fe(df, cluster_se=True):
    """LSDV: pair dummies + frames. Identical coefficients to within-pair demeaning."""
    r = smf.ols("delta ~ C(pair) + " + " + ".join(FRAMES), data=df)
    return r.fit(cov_type="cluster", cov_kwds={"groups": df["pair"]}) if cluster_se else r.fit()


def fit_two_way(df):
    r = smf.ols("delta ~ C(model) + C(profile_id) + " + " + ".join(FRAMES), data=df)
    return r.fit(cov_type="cluster", cov_kwds={"groups": df["pair"]})


def diagnostics(df):
    rows = []
    for f in FRAMES + ["delta"]:
        grp = df.groupby("pair")[f]
        within = grp.transform("mean")
        wv = ((df[f] - within) ** 2).mean()
        tv = ((df[f] - df[f].mean()) ** 2).mean()
        n_const = int((grp.nunique() == 1).sum())
        rows.append({"variable": f, "within_pair_share": wv / tv,
                     "pairs_with_no_within_variation": n_const, "pairs": df["pair"].nunique()})
    return pd.DataFrame(rows)


def demeaned_check(df):
    """Coefficients from within-pair demeaning must equal the LSDV ones."""
    d = df.copy()
    for f in ["delta"] + FRAMES:
        d[f] = d[f] - d.groupby("pair")[f].transform("mean")
    beta = np.linalg.lstsq(d[FRAMES].values, d["delta"].values, rcond=None)[0]
    return dict(zip(FRAMES, beta))


if __name__ == "__main__":
    df = load_rows()
    print(f"H2 sample: N = {len(df)}, {df.model.nunique()} models, {df.profile_id.nunique()} profiles, "
          f"{df['pair'].nunique()} model x profile pairs")

    dd = diagnostics(df)
    print("\n=== identification diagnostics (pair FE absorbs the cell baseline) ===")
    for _, r in dd.iterrows():
        print(f"  {r.variable:<26} within-pair share of variance = {r.within_pair_share:.3f}   "
              f"pairs with no within variation: {int(r.pairs_with_no_within_variation)}/{int(r.pairs)}")

    r_pair = fit_pair_fe(df)
    r_two = fit_two_way(df)
    v_pair, v_two = coefs(r_pair), coefs(r_two)
    chk = demeaned_check(df)

    p_raw = [v_pair[f]["p"] for f in FRAMES]
    _, p_bh, _, _ = multipletests(p_raw, alpha=0.10, method="fdr_bh")

    print("\n=== MAIN: model x profile pair FE ===")
    for f in FRAMES:
        v = v_pair[f]
        print(f"  {f:<26} b={v['b']:+.3f}  SE={v['se']:.3f}  p={v['p']:.4f}  "
              f"CI=[{v['lo']:+.2f}, {v['hi']:+.2f}]   (demeaned check {chk[f]:+.3f})")
    print(f"  N = {int(r_pair.nobs)}   pairs = {df['pair'].nunique()}   "
          f"within R2 = {r_pair.rsquared:.4f}   df_resid = {int(r_pair.df_resid)}")

    print("\n=== comparison: model FE + profile FE ===")
    for f in FRAMES:
        v = v_two[f]
        print(f"  {f:<26} b={v['b']:+.3f}  SE={v['se']:.3f}  p={v['p']:.4f}  CI=[{v['lo']:+.2f}, {v['hi']:+.2f}]")

    print("\n=== BH adjustment (3 frames, q = 0.10), pair-FE specification ===")
    for f, pr, pb in zip(FRAMES, p_raw, p_bh):
        print(f"  {f:<26} raw p={pr:.4f}  BH p={pb:.4f}")

    lomo = []
    for drop in sorted(df.model.unique()):
        v = coefs(fit_pair_fe(df[df.model != drop]))
        lomo.append({"dropped_model": drop, **{f"{f}_{k}": v[f][k] for f in FRAMES for k in ("b", "p")}})
    lomo = pd.DataFrame(lomo)
    print("\n=== leave-one-model-out under pair FE (15 fits) ===")
    for f in FRAMES:
        b, p = lomo[f"{f}_b"], lomo[f"{f}_p"]
        print(f"  {f:<26} min={b.min():+.2f} max={b.max():+.2f}  positive={int((b > 0).sum())}/15  "
              f"p<0.05 in {int((p < 0.05).sum())}/15")

    # leave-one-profile-out under pair FE
    lopo = []
    for drop in sorted(df.profile_id.unique()):
        v = coefs(fit_pair_fe(df[df.profile_id != drop]))
        lopo.append({"dropped_profile": drop, **{f"{f}_{k}": v[f][k] for f in FRAMES for k in ("b", "p")}})
    lopo = pd.DataFrame(lopo)
    print("\n=== leave-one-profile-out under pair FE (27 fits) ===")
    for f in FRAMES:
        b, p = lopo[f"{f}_b"], lopo[f"{f}_p"]
        print(f"  {f:<26} min={b.min():+.2f} max={b.max():+.2f}  positive={int((b > 0).sum())}/27  "
              f"p<0.05 in {int((p < 0.05).sum())}/27")

    main = pd.DataFrame([{"frame": f, "b": v_pair[f]["b"], "se": v_pair[f]["se"], "p": v_pair[f]["p"],
                          "lo": v_pair[f]["lo"], "hi": v_pair[f]["hi"], "p_bh": pb,
                          "b_two_way_fe": v_two[f]["b"], "p_two_way_fe": v_two[f]["p"]}
                         for f, pb in zip(FRAMES, p_bh)])
    main.to_csv(CAN / "h2_pairfe_main.csv", index=False)
    lomo.to_csv(CAN / "h2_pairfe_lomo.csv", index=False)
    lopo.to_csv(CAN / "h2_pairfe_lopo.csv", index=False)
    dd.to_csv(CAN / "h2_pairfe_diagnostics.csv", index=False)
    print(f"\nwrote {CAN}/h2_pairfe_main.csv, h2_pairfe_lomo.csv, h2_pairfe_diagnostics.csv")
