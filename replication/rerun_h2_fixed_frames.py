#!/usr/bin/env python3
"""H2 pair-FE regression, before vs after the collective_responsibility regex fix.

Loads rows exactly like analyze_canonical_h2_pairfe.py (via analyze_canonical_h2.load()),
then swaps in the FIXED collective flag from outputs/narrative_coding_fixed.csv.
Reports, for both codings, the pair-FE spec used in the paper (delta ~ C(pair) + frames).
"""
import csv
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

spec = importlib.util.spec_from_file_location("h2", HOME / "analyze_canonical_h2.py")
h2 = importlib.util.module_from_spec(spec)
sys.modules["h2"] = h2
spec.loader.exec_module(h2)

df = pd.DataFrame(h2.load()).rename(columns={"PVC": "PVOC"})
df["pair"] = df.model.astype(str) + "|" + df.profile_id.astype(str)

# fixed collective flag keyed by (model, profile_id, repetition)
fixed = {}
for r in csv.DictReader(open(HOME / "outputs" / "narrative_coding_fixed.csv", newline="")):
    fixed[(r["model"], r["profile_id"], r["repetition"])] = int(r["collective_responsibility"])

df["collective_fixed"] = df.apply(
    lambda r: fixed.get((r["model"], str(r["profile_id"]), str(r["repetition"])), r["collective_responsibility"]), axis=1)

# sanity: fixed coding must agree with the re-code script's own 1-hit result
assert df["collective_fixed"].sum() == 1, f"expected 1 fixed hit, got {df['collective_fixed'].sum()}"
assert df["collective_responsibility"].sum() == df["collective_responsibility"].sum()  # keep old

def fit(df_, collective_col):
    r = smf.ols(f"delta ~ C(pair) + access_barriers + {collective_col} + coercive_backlash", data=df_)
    return r.fit(cov_type="cluster", cov_kwds={"groups": df_["pair"]})

def row(f, r):
    if f not in r.params.index:
        return dict(frame=f, b=np.nan, se=np.nan, p=np.nan, lo=np.nan, hi=np.nan, absorbed=True)
    ci = r.conf_int().loc[f]
    return dict(frame=f, b=r.params[f], se=r.bse[f], p=r.pvalues[f], lo=ci[0], hi=ci[1], absorbed=False)

def report(df_, col, label):
    r = fit(df_, col)
    rows = [row(f, r) for f in FRAMES]
    for x in rows:
        if x["absorbed"]:
            x["p_bh"] = np.nan
    p_raw = [x["p"] for x in rows if not x["absorbed"]]
    p_bh = multipletests(p_raw, alpha=0.10, method="fdr_bh")[1] if p_raw else []
    it = iter(p_bh)
    for x in rows:
        if not x["absorbed"]:
            x["p_bh"] = next(it)
    print(f"\n=== {label} ===")
    print(f"N={len(df_)}, pairs={df_['pair'].nunique()}, R2={r.rsquared:.4f}")
    for x in rows:
        if x["absorbed"]:
            print(f"  {x['frame']:<26} DROPPED (no within-pair variation / zero column)")
        else:
            print(f"  {x['frame']:<26} b={x['b']:+.4f}  SE={x['se']:.4f}  p={x['p']:.4f}  BH={x['p_bh']:.4f}  CI=[{x['lo']:+.2f}, {x['hi']:+.2f}]")
    return rows

before = report(df, "collective_responsibility", "BEFORE fix (bare public\\s+health in collective)")
after = report(df, "collective_fixed", "AFTER fix (collective ~ 0)")

print("\n=== delta (after - before) ===")
for b, a in zip(before, after):
    print(f"  {b['frame']:<26} b: {b['b']:+.4f} -> {a['b']:+.4f}   (p {b['p']:.4f} -> {a['p']:.4f})")

# prevalence recap
n = len(df)
for label, col in [("orig", "collective_responsibility"), ("fixed", "collective_fixed")]:
    print(f"collective prevalence {label}: {df[col].sum()} / {n} ({100*df[col].sum()/n:.2f}%)")

pd.DataFrame(before).to_csv(CAN / "h2_pairfe_beforefix.csv", index=False)
pd.DataFrame(after).to_csv(CAN / "h2_pairfe_afterfix.csv", index=False)
print("\nwrote h2_pairfe_beforefix.csv / h2_pairfe_afterfix.csv")
