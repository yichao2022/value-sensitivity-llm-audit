#!/usr/bin/env python3
"""H3 robustness: model x profile PAIR fixed effects instead of model FE + profile FE.

    Delta_mir = alpha_mi + sum_k b_k Frame_k + e        (frame is ASSIGNED, not observed)

H3's inferential target is the within-model-profile assigned-frame contrast, so the
pair-FE specification is the one that matches the stated target. Frames are randomized
within each pair, so identification is by design rather than by accident.

Both specifications are reported with the SAME standard errors (clustered at the
model x profile pair, 108 clusters) so that the only difference is the fixed-effect
structure. Writes outputs/canonical/h3_pairfe_robustness.csv.
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HOME = Path(__file__).resolve().parent
CAN = HOME / "outputs" / "canonical"
FRAMES = ["autonomy", "collective", "equity"]  # vs neutral reference


def delta_frame():
    rows = [r for r in csv.DictReader(open(CAN / "h3_raw.csv"))
            if all(r.get(k, "").lower() in ("true", "1")
                   for k in ("json_valid", "score_valid", "non_refusal"))]
    cell = ("model", "frame", "profile_id", "repetition", "burden_level")
    dedup = {tuple(r[k] for k in cell): r for r in rows}          # last row wins
    by_key = defaultdict(dict)
    for r in dedup.values():
        by_key[(r["model"], r["frame"], r["profile_id"], r["repetition"])][r["burden_level"]] = float(r["willingness"])
    out = [{"model": m, "frame": f, "profile_id": p, "repetition": rep,
            "delta": lo["low"] - lo["high"]}
           for (m, f, p, rep), lo in by_key.items() if "low" in lo and "high" in lo]
    df = pd.DataFrame(out)
    df["pair"] = df.model.astype(str) + "|" + df.profile_id.astype(str)
    df["frame"] = pd.Categorical(df.frame, categories=["neutral"] + FRAMES)
    return df


def coefs(r):
    return {f: dict(b=r.params[f"C(frame)[T.{f}]"], se=r.bse[f"C(frame)[T.{f}]"],
                    p=r.pvalues[f"C(frame)[T.{f}]"], lo=r.conf_int().loc[f"C(frame)[T.{f}]"][0],
                    hi=r.conf_int().loc[f"C(frame)[T.{f}]"][1]) for f in FRAMES}


if __name__ == "__main__":
    df = delta_frame()
    g = df.groupby("pair")["frame"]
    print(f"H3: N = {len(df)} assigned-frame deltas, {df.model.nunique()} models, "
          f"{df.profile_id.nunique()} profiles, {df['pair'].nunique()} pairs "
          f"({len(df) / df['pair'].nunique():.0f} obs per pair)")
    print(f"within-pair share of variance in the assigned frame indicator = "
          f"{((df.groupby('pair')['frame'].transform(lambda s: s != s.mode()[0])) ** 1).mean():.3f} "
          f"(= 1 - modal share within pair; frames are randomized by design)")

    two = smf.ols("delta ~ C(model) + C(profile_id) + C(frame)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair"]})
    pair = smf.ols("delta ~ C(pair) + C(frame)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["pair"]})

    v2, vp = coefs(two), coefs(pair)
    print("\n=== pooled H3, same clustered SEs, only the FE structure differs ===")
    print(f"{'contrast':<16} {'model+profile FE':>22} {'pair FE':>22}")
    for f in FRAMES:
        print(f"{f + ' vs neutral':<16} "
              f"{v2[f]['b']:+7.2f} [{v2[f]['lo']:+6.2f},{v2[f]['hi']:+6.2f}] p={v2[f]['p']:.4f}   "
              f"{vp[f]['b']:+7.2f} [{vp[f]['lo']:+6.2f},{vp[f]['hi']:+6.2f}] p={vp[f]['p']:.4f}")
    print(f"  N = {int(pair.nobs)}  pairs = {df['pair'].nunique()}  "
          f"R2: two-way {two.rsquared:.3f} -> pair FE {pair.rsquared:.3f}")

    out = pd.DataFrame([{"contrast": f, "b_two_way": v2[f]["b"], "p_two_way": v2[f]["p"],
                         "b_pair_fe": vp[f]["b"], "se_pair_fe": vp[f]["se"], "p_pair_fe": vp[f]["p"],
                         "lo_pair_fe": vp[f]["lo"], "hi_pair_fe": vp[f]["hi"]} for f in FRAMES])
    out.to_csv(CAN / "h3_pairfe_robustness.csv", index=False)
    print(f"\nwrote {CAN}/h3_pairfe_robustness.csv")
