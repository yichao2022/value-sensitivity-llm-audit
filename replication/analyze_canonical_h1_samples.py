#!/usr/bin/env python3
"""Regenerate Table 12 (H1 robustness across the nine- and fifteen-endpoint samples).

Single source: canonical orientation responses + canonical model-level burden effects.
Reproduces every row of the supplement's Table 12 and, additionally, the same rows on an
explicitly stated standardisation:

  * nine-endpoint rows keep the FIFTEEN-endpoint standardisation of the five dimensions
    (this is what the manuscript reports as the "-4.24" rows), and
  * the identical rows re-standardised WITHIN the nine endpoints, because the fitted
    slope is scale-dependent.

Also reports the analytic 80% minimum detectable slope (t_{0.975} + z_{0.80}) * SE on the
standardised-composite scale, and the empirical SD of the composite (which is not 1, since
the composite is not rescaled after aggregation).

Writes outputs/canonical/h1_sample_comparison.csv.
"""
import csv
import json
import statistics
from pathlib import Path

import statsmodels.api as sm

P = Path(__file__).resolve().parent
CAN = P / "outputs" / "canonical"
DIMS = ["AP", "ST", "CO", "BS", "CA"]
SIGNS = {"AP": +1, "ST": -1, "CO": -1, "BS": +1, "CA": +1}
NINE_API = ["GPT-4.1", "GPT-4.1 mini", "GPT-4o", "Claude Sonnet 4.6", "Claude Haiku",
            "Claude Opus 4.8", "Gemini 2.5 Pro", "Gemini 2.5 Flash", "DeepSeek-V3.2"]


def dimension_scores():
    meta = {}
    for r in csv.DictReader(open(P / "data/orientation_responses.csv", newline="")):
        o = int(r["item_order"])
        if o not in meta:
            meta[o] = (r["dimension"], int(r["reverse_coded"] or 0))
    prof = {}
    for r in csv.DictReader(open(CAN / "orientation_raw.csv", newline="")):
        scores = json.loads(r["scores"]) if r["scores"] else []
        if len(scores) != len(meta):
            continue
        per = {d: [] for d in DIMS}
        for o, s in sorted(zip(meta, scores)):
            dim, rev = meta[o]
            per[dim].append(8 - s if rev else s)
        prof[r["model"]] = {d: round(statistics.mean(v), 2) for d, v in per.items()}
    return prof


def standardise(prof, base):
    z = {}
    for d in DIMS:
        vals = [prof[m][d] for m in base]
        mu, sd = statistics.mean(vals), statistics.stdev(vals)
        for m in base:
            z.setdefault(m, {})[d] = (prof[m][d] - mu) / sd
    return z


def composite(z, m, dims=DIMS):
    return sum(SIGNS[d] * z[m][d] for d in dims)


def reg(x, y):
    r = sm.OLS(y, sm.add_constant(x)).fit()
    ci = r.conf_int()[1]
    return dict(b=r.params[1], se=r.bse[1], p=r.pvalues[1], r2=r.rsquared,
                lo=ci[0], hi=ci[1], n=int(r.nobs))


def mde(r):
    from scipy import stats as st
    t = st.t.ppf(0.975, r["n"] - 2) + st.norm.ppf(0.80)
    return t * r["se"]


if __name__ == "__main__":
    prof = dimension_scores()
    burden = {r["Model"]: float(r["Mean_Delta"])
              for r in csv.DictReader(open(CAN / "table3_burden_effects.csv", newline=""))}
    models = sorted(set(prof) & set(burden))
    nine = [m for m in NINE_API if m in prof and m in burden]
    z15, z9 = standardise(prof, models), standardise(prof, nine)

    print(f"sample sizes: fifteen = {len(models)}, nine official-API endpoints = {len(nine)}")
    print(f"SD of the fifteen-endpoint composite: {statistics.stdev([composite(z15, m) for m in models]):.3f} "
          f"(not 1: the composite is not rescaled after aggregation)")

    rows = []
    specs = [
        ("PVOC composite", "N=9, fifteen-endpoint standardisation", [composite(z15, m) for m in nine], [burden[m] for m in nine]),
        ("PVOC composite", "N=9, re-standardised within nine", [composite(z9, m) for m in nine], [burden[m] for m in nine]),
        ("PVOC composite", "N=15", [composite(z15, m) for m in models], [burden[m] for m in models]),
        ("Collective obligation (raw 1-7)", "N=9", [prof[m]["CO"] for m in nine], [burden[m] for m in nine]),
        ("Collective obligation (raw 1-7)", "N=15", [prof[m]["CO"] for m in models], [burden[m] for m in models]),
        ("Collective obligation (standardised)", "N=9", [z9[m]["CO"] for m in nine], [burden[m] for m in nine]),
        ("Collective obligation (standardised)", "N=15", [z15[m]["CO"] for m in models], [burden[m] for m in models]),
        ("Leave-CO-out composite", "N=15", [composite(z15, m, [d for d in DIMS if d != "CO"]) for m in models], [burden[m] for m in models]),
    ]
    # Methods claim: "We also fit each orientation dimension separately as a diagnostic check."
    # All five dimensions must therefore appear in the supplement, not only CO.
    for d in DIMS:
        specs.append((f"{d} (standardised)", "N=15", [z15[m][d] for m in models], [burden[m] for m in models]))
    for predictor, sample, x, y in specs:
        r = reg(x, y)
        r["mde80"] = mde(r)
        rows.append({"predictor": predictor, "sample": sample, **r})
        print(f"  {predictor:<34} {sample:<34} b={r['b']:+7.2f} [{r['lo']:+6.2f},{r['hi']:+6.2f}] "
              f"p={r['p']:.3f} R2={r['r2']:.3f} MDE80={r['mde80']:.1f}")

    out = CAN / "h1_sample_comparison.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")
