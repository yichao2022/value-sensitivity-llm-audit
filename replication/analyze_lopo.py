#!/usr/bin/env python3
"""Leave-one-profile-out (LOPO) robustness for H1, H2 and H3.

Single source: outputs/canonical/* (the canonical rerun reported in the paper).
For each of the 27 synthetic profiles, drop that profile from the analysis and
re-estimate:
  H1  model-level bivariate regression of mean burden effect on PVOC
  H2  frame-adjusted model (access barriers, collective responsibility, coercive
      backlash + PVOC covariate), cluster-robust by model x profile
  H3  pooled frame-manipulation model (model + profile fixed effects), cluster-robust
Writes outputs/lopo_robustness.csv and prints sign-stability summaries.

Self-check: the full-sample H3 fit must reproduce the published pooled
coefficients (4.96 / -10.88 / 4.84) to 0.01.
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

# ---------- loaders -------------------------------------------------------
def load_h1():
    """Per-model mean burden effect by profile (cell-deduplicated)."""
    cells = {}
    for r in csv.DictReader(open(CAN / "sim_raw.csv", newline="")):
        cells[(r["model"], r["profile_id"], r["burden_level"], r["repetition"])] = r
    ok = lambda r: (not (r.get("error") or "").strip()
                    and str(r.get("json_valid")).lower() == "true"
                    and str(r.get("score_valid")).lower() == "true"
                    and str(r.get("non_refusal")).lower() == "true"
                    and (r.get("willingness") or "").strip())
    rows = []
    for (m, pid, burden, rep), r in cells.items():
        if burden != "low" or not ok(r):
            continue
        hi = cells.get((m, pid, "high", rep))
        if hi is None or not ok(hi):
            continue
        rows.append({"model": m, "profile_id": pid,
                     "delta": float(r["willingness"]) - float(hi["willingness"])})
    rows2 = list(csv.DictReader(open(CAN / "table2_orientation_profiles.csv", newline="")))
    key = "PVOC" if "PVOC" in rows2[0] else "PVC"
    pvoc = {r["Model"]: float(r[key]) for r in rows2}
    df = pd.DataFrame(rows)
    df["PVOC"] = df["model"].map(pvoc)
    return df


def load_h2():
    """Reuse the canonical H2 loader so the coding rules stay identical."""
    spec = importlib.util.spec_from_file_location("h2", HOME / "analyze_canonical_h2.py")
    h2 = importlib.util.module_from_spec(spec)
    sys.modules["h2"] = h2
    spec.loader.exec_module(h2)
    df = pd.DataFrame(h2.load()).rename(columns={"PVC": "PVOC"})
    return df


def load_h3():
    df = pd.read_csv(OUT / "frame_delta_analysis.csv")
    return df.dropna(subset=["delta"])


# ---------- estimators ----------------------------------------------------
FRAMES_H2 = ["access_barriers", "collective_responsibility", "coercive_backlash"]
FRAMES_H3 = ["autonomy", "collective", "equity"]


def fit_h1(df, drop_profile=None):
    d = df if drop_profile is None else df[df.profile_id != drop_profile]
    m = d.groupby(["model", "PVOC"], as_index=False)["delta"].mean()
    r = smf.ols("delta ~ PVOC", data=m).fit()
    return {"beta1": r.params["PVOC"], "p": r.pvalues["PVOC"], "r2": r.rsquared, "n": int(r.nobs)}


def fit_h2(df, drop_profile=None):
    """Canonical H2 specification (2026-09-17): model + profile fixed effects,
    frame indicators only; PVOC belongs to H1 and does not enter this model."""
    d = df if drop_profile is None else df[df.profile_id != drop_profile]
    form = "delta ~ C(model) + C(profile_id) + " + " + ".join(FRAMES_H2)
    r = smf.ols(form, data=d).fit(cov_type="cluster",
                                  cov_kwds={"groups": d.model.astype(str) + "|" + d.profile_id.astype(str)})
    out = {}
    for f in FRAMES_H2:
        name = [n for n in r.params.index if n == f or n.startswith(f"{f}[T.")]
        assert len(name) == 1, f"ambiguous coefficient for {f}: {name}"
        out[f] = (r.params[name[0]], r.pvalues[name[0]])
    return out | {"n": int(r.nobs)}


def fit_h3(df, drop_profile=None):
    d = df if drop_profile is None else df[df.profile_id != drop_profile]
    r = smf.ols('delta ~ C(model) + C(profile_id) + C(frame, Treatment(reference="neutral"))', data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d.model.astype(str) + "|" + d.profile_id.astype(str)})
    def get(f):
        name = [n for n in r.params.index if n.endswith(f"[T.{f}]")]
        assert len(name) == 1, f"ambiguous coefficient for {f}: {name}"
        return r.params[name[0]], r.pvalues[name[0]]
    return {f: get(f) for f in FRAMES_H3} | {"n": int(r.nobs)}


if __name__ == "__main__":
    h1, h2, h3 = load_h1(), load_h2(), load_h3()
    profiles = sorted(h3.profile_id.unique())
    print(f"H1 rows {len(h1)} | H2 rows {len(h2)} | H3 rows {len(h3)} | profiles {len(profiles)}")

    base_h3 = fit_h3(h3)
    published = {"autonomy": 4.96, "collective": -10.88, "equity": 4.84}
    for f, exp in published.items():
        got = base_h3[f][0]
        assert abs(got - exp) < 0.01, f"H3 spec mismatch on {f}: {got:.3f} vs published {exp}"
        print(f"  self-check H3 {f}: {got:+.3f} (published {exp:+.2f}) OK")

    recs = []
    for p in profiles:
        a, b, c = fit_h1(h1, p), fit_h2(h2, p), fit_h3(h3, p)
        rec = {"dropped_profile": p, "h1_beta1": a["beta1"], "h1_p": a["p"], "h1_n": a["n"]}
        for f in FRAMES_H2:
            rec[f"h2_{f}"] = b[f][0]; rec[f"h2_{f}_p"] = b[f][1]
        for f in FRAMES_H3:
            rec[f"h3_{f}"] = c[f][0]; rec[f"h3_{f}_p"] = c[f][1]
        recs.append(rec)
    res = pd.DataFrame(recs)
    res.to_csv(OUT / "lopo_robustness.csv", index=False)

    print("\n=== leave-one-profile-out summaries (27 fits) ===")
    def line(label, col, pcol):
        v, pv = res[col], res[pcol]
        print(f"  {label:<28} min {v.min():+7.2f}  max {v.max():+7.2f}  |  positive {int((v>0).sum()):2d}/27"
              f"  |  p<0.05 in {int((pv<0.05).sum()):2d}/27")
    line("H1 PVOC slope (beta1)", "h1_beta1", "h1_p")
    for f in FRAMES_H2:
        line(f"H2 {f}", f"h2_{f}", f"h2_{f}_p")
    for f in FRAMES_H3:
        line(f"H3 {f}", f"h3_{f}", f"h3_{f}_p")
    print("\nwrote outputs/lopo_robustness.csv")
