#!/usr/bin/env python3
"""Canonical H2 under the model-FE + profile-FE specification.

Main model (as of the 2026-09-17 specification decision):
    Delta_mir = alpha_m + mu_i + b1*Access + b2*Collective + b3*Backlash + e
with standard errors clustered at the model x profile level. PVOC is NOT in the
H2 model (it belongs to H1); the previous PVOC-adjusted specification is
reported alongside as a robustness comparison.

Also computes leave-one-model-out (15 fits) and leave-one-profile-out (27 fits)
for the new main specification.

Single source for frame coding: analyze_canonical_h2.load().
"""
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HOME = Path(__file__).resolve().parent
OUT = HOME / "outputs"
CAN = OUT / "canonical"

FRAMES = ["access_barriers", "collective_responsibility", "coercive_backlash"]


def load_rows():
    spec = importlib.util.spec_from_file_location("h2", HOME / "analyze_canonical_h2.py")
    h2 = importlib.util.module_from_spec(spec)
    sys.modules["h2"] = h2
    spec.loader.exec_module(h2)
    return pd.DataFrame(h2.load()).rename(columns={"PVC": "PVOC"})


def cluster(df):
    return df.model.astype(str) + "|" + df.profile_id.astype(str)


def fit_fe(df, cluster_se=True):
    form = "delta ~ C(model) + C(profile_id) + " + " + ".join(FRAMES)
    m = smf.ols(form, data=df)
    r = m.fit(cov_type="cluster", cov_kwds={"groups": cluster(df)}) if cluster_se else m.fit()
    return r


def frame_coefs(r):
    out = {}
    for f in FRAMES:
        name = [n for n in r.params.index if n == f or n.startswith(f"{f}[T.")]
        assert len(name) == 1, f"ambiguous coefficient for {f}: {name}"
        ci = r.conf_int().loc[name[0]]
        out[f] = dict(b=r.params[name[0]], se=r.bse[name[0]], p=r.pvalues[name[0]],
                      lo=ci[0], hi=ci[1])
    return out


if __name__ == "__main__":
    df = load_rows()
    print(f"H2 sample: N = {len(df)} observations, {df.model.nunique()} models, "
          f"{df.profile_id.nunique()} profiles")
    print("prevalence:", {f: f"{100*df[f].mean():.1f}%" for f in FRAMES})

    r_fe = fit_fe(df)
    r_pv = smf.ols("delta ~ PVOC + " + " + ".join(FRAMES), data=df).fit(
        cov_type="cluster", cov_kwds={"groups": cluster(df)})

    print("\n=== MAIN (model FE + profile FE + frames), cluster-robust by model x profile ===")
    for f, v in frame_coefs(r_fe).items():
        print(f"  {f:<26} b={v['b']:+.3f}  SE={v['se']:.3f}  p={v['p']:.5f}  CI=[{v['lo']:+.2f}, {v['hi']:+.2f}]")
    print(f"  R2 = {r_fe.rsquared:.3f}   adj R2 = {r_fe.rsquared_adj:.3f}   N = {int(r_fe.nobs)}")

    print("\n=== previous specification (PVOC + frames, no FE), for comparison ===")
    for f, v in frame_coefs(r_pv).items():
        print(f"  {f:<26} b={v['b']:+.3f}  SE={v['se']:.3f}  p={v['p']:.5f}")
    print(f"  PVOC covariate: b={r_pv.params['PVOC']:+.3f} p={r_pv.pvalues['PVOC']:.5f}   R2 = {r_pv.rsquared:.3f}")

    # leave-one-model-out
    lomo = []
    for drop in sorted(df.model.unique()):
        sub = df[df.model != drop]
        v = frame_coefs(fit_fe(sub))
        lomo.append({"dropped_model": drop, **{f"{f}_{k}": v[f][k] for f in FRAMES for k in ("b", "p")}})
    lomo = pd.DataFrame(lomo)
    print("\n=== leave-one-model-out (15 fits, each dropping one model) ===")
    for f in FRAMES:
        b = lomo[f"{f}_b"]
        print(f"  {f:<26} min={b.min():+7.2f} max={b.max():+7.2f}  positive={int((b>0).sum())}/15  p<0.05 in {int((lomo[f'{f}_p']<0.05).sum())}/15")

    # leave-one-profile-out
    lopo = []
    for drop in sorted(df.profile_id.unique()):
        sub = df[df.profile_id != drop]
        v = frame_coefs(fit_fe(sub))
        lopo.append({"dropped_profile": drop, **{f"{f}_{k}": v[f][k] for f in FRAMES for k in ("b", "p")}})
    lopo = pd.DataFrame(lopo)
    print("\n=== leave-one-profile-out (27 fits, each dropping one profile) ===")
    for f in FRAMES:
        b = lopo[f"{f}_b"]
        print(f"  {f:<26} min={b.min():+7.2f} max={b.max():+7.2f}  positive={int((b>0).sum())}/27  p<0.05 in {int((lopo[f'{f}_p']<0.05).sum())}/27")

    from statsmodels.stats.multitest import multipletests
    coefs = frame_coefs(r_fe)
    p_raw = [coefs[f]["p"] for f in FRAMES]
    reject, p_bh, _, _ = multipletests(p_raw, alpha=0.10, method="fdr_bh")
    print("\n=== Benjamini-Hochberg adjustment (3 retained frames, q = 0.10) ===")
    for f, pr, pb, rj in zip(FRAMES, p_raw, p_bh, reject):
        print(f"  {f:<26} raw p={pr:.5f}  BH-adjusted p={pb:.5f}  significant={bool(rj)}")
    main = pd.DataFrame([{"frame": f, "prevalence": df[f].mean(),
                          **{k: coefs[f][k] for k in ("b", "se", "p", "lo", "hi")}, "p_bh": pb}
                         for f, pb in zip(FRAMES, p_bh)])
    main.to_csv(CAN / "h2_fe_main.csv", index=False)
    lomo.to_csv(CAN / "h2_fe_lomo.csv", index=False)
    lopo.to_csv(CAN / "h2_fe_lopo.csv", index=False)
    print(f"\nwrote {CAN/'h2_fe_main.csv'}, h2_fe_lomo.csv, h2_fe_lopo.csv")
