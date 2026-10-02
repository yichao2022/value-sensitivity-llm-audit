#!/usr/bin/env python3
"""
Canonical Table 2 (PVOC profiles) + H1 (orientation -> burden effect).

Sources (single truth):
  outputs/canonical/orientation_raw.csv   — 20 ratings per model, in item_order
  data/orientation_responses.csv          — item_order -> dimension, reverse_coded
  outputs/canonical/table3_burden_effects.csv — model-level mean burden effect

PVC = z(AP) - z(ST) - z(CO) + z(BS) + z(CA), z taken across models (same as
compute_orientation_scores.py, so Table 2 stays comparable).
"""
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import statsmodels.api as sm

P = Path(__file__).resolve().parent
OUT = P / "outputs" / "canonical"
DIMS = ["AP", "ST", "CO", "BS", "CA"]


def item_meta():
    """item_order -> (dimension, reverse_coded)"""
    meta = {}
    for r in csv.DictReader(open(P / "data/orientation_responses.csv", newline="")):
        o = int(r["item_order"])
        if o not in meta:
            meta[o] = (r["dimension"], int(r["reverse_coded"] or 0))
    return meta


def profiles():
    meta = item_meta()
    out = {}
    for r in csv.DictReader(open(OUT / "orientation_raw.csv", newline="")):
        scores = json.loads(r["scores"]) if r["scores"] else []
        if len(scores) != len(meta):
            print(f"  skip {r['model']}: {len(scores)}/{len(meta)} ratings")
            continue
        per_dim = {d: [] for d in DIMS}
        for o, s in sorted(zip(meta, scores)):  # item_order ascending
            dim, rev = meta[o]
            per_dim[dim].append(8 - s if rev else s)
        out[r["model"]] = {d: round(statistics.mean(v), 2) for d, v in per_dim.items()}
    return out


def pvc_scores(prof):
    z = {}
    for d in DIMS:
        vals = [prof[m][d] for m in prof]
        mu, sd = statistics.mean(vals), statistics.stdev(vals)
        for m in prof:
            z.setdefault(m, {})[d] = (prof[m][d] - mu) / sd
    return {m: round(z[m]["AP"] - z[m]["ST"] - z[m]["CO"] + z[m]["BS"] + z[m]["CA"], 3) for m in prof}


def fit(y, X, names):
    res = sm.OLS(np.asarray(y, float), sm.add_constant(np.asarray(X, float))).fit()
    rows = []
    for i, n in enumerate(["(Intercept)"] + names):
        ci = res.conf_int()[i]
        rows.append({"predictor": n, "coef": round(res.params[i], 3), "se": round(res.bse[i], 3),
                     "t": round(res.tvalues[i], 3), "p": res.pvalues[i],
                     "ci_low": round(ci[0], 3), "ci_high": round(ci[1], 3)})
    rows.append({"predictor": "__R2__", "coef": round(res.rsquared, 4), "se": round(res.rsquared_adj, 4),
                 "t": None, "p": None, "ci_low": None, "ci_high": None})
    return rows, res


if __name__ == "__main__":
    prof = profiles()
    pvc = pvc_scores(prof)
    burden = {r["Model"]: float(r["Mean_Delta"])
              for r in csv.DictReader(open(OUT / "table3_burden_effects.csv", newline=""))}
    models = sorted(set(prof) & set(burden))
    print(f"N = {len(models)} models\n")

    t2 = [{"Model": m, **{d: prof[m][d] for d in DIMS}, "PVC": pvc[m]} for m in models]
    t2.sort(key=lambda r: -r["PVC"])
    print(f"{'Model':<24}" + "".join(f"{d:>7}" for d in DIMS) + f"{'PVC':>8}")
    for r in t2:
        print(f"{r['Model']:<24}" + "".join(f"{r[d]:>7}" for d in DIMS) + f"{r['PVC']:>8.3f}")
    with open(OUT / "table2_orientation_profiles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Model"] + DIMS + ["PVC"]); w.writeheader(); w.writerows(t2)

    y = [burden[m] for m in models]
    m_delta = [statistics.mean(y), statistics.stdev(y)]
    print(f"\nmean burden effect across models: {m_delta[0]:.2f} (SD {m_delta[1]:.2f})")

    print("\n--- H1: mean_delta ~ PVC ---")
    rows, res = fit(y, [pvc[m] for m in models], ["PVC"])
    for r in rows:
        if r["predictor"] == "__R2__":
            print(f"  R2={r['coef']}  adjR2={r['se']}")
        else:
            print(f"  {r['predictor']:<10} b={r['coef']:<8} se={r['se']:<7} p={r['p']:.4f} CI=[{r['ci_low']}, {r['ci_high']}]")
            print(f"  beta_std = {r['coef'] * statistics.stdev([pvc[m] for m in models]) / m_delta[1]:.3f}")

    print("\n--- dimension-specific (each dimension alone) ---")
    for d in DIMS:
        rows, res = fit(y, [prof[m][d] for m in models], [d])
        b = rows[1]
        print(f"  {d}: b={b['coef']:<8} se={b['se']:<7} p={b['p']:.4f}  R2={rows[-1]['coef']}")

    with open(OUT / "h1_orientation_effect.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "predictor", "coef", "se", "t", "p", "ci_low", "ci_high"])
        w.writeheader()
        for r in fit(y, [pvc[m] for m in models], ["PVC"])[0]:
            w.writerow({"model": "PVC model", **r})
        for d in DIMS:
            for r in fit(y, [prof[m][d] for m in models], [d])[0]:
                w.writerow({"model": d, **r})
    print("\nwrote", OUT / "table2_orientation_profiles.csv", "and h1_orientation_effect.csv")

    assert len(models) == 15, f"need 15 models with both orientation and burden data, got {len(models)}"
    assert all(1 <= prof[m][d] <= 7 for m in models for d in DIMS), "score out of 1-7 range"
    print("self-check ok")
