#!/usr/bin/env python3
"""
Table 1 (diagnostics) + Table 3 (burden effects) from the canonical run.

Single source of truth: outputs/canonical/sim_raw.csv (cell-deduped, keep last).
Same statistic definitions as the legacy analyze_results.py so the paper tables
can be swapped 1:1; only the input file and the dedupe step are new.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

P = Path(__file__).resolve().parent
RAW = P / "outputs" / "canonical" / "sim_raw.csv"
OUT1 = P / "outputs" / "canonical" / "table1_diagnostics.csv"
OUT3 = P / "outputs" / "canonical" / "table3_burden_effects.csv"

TRUTHY = ("true", "1", "yes")


def load_deduped(path: Path) -> list[dict]:
    """One row per (model, profile, burden, repetition) — last write wins (raw file is append-only)."""
    cells = {}
    for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
        cells[(r["model"], r["profile_id"], r["burden_level"], r["repetition"])] = r
    rows = []
    for r in cells.values():
        for f in ("json_valid", "score_valid", "non_refusal"):
            r[f] = str(r.get(f, "")).strip().lower() in TRUTHY
        w = (r.get("willingness") or "").strip()
        try:
            r["_w"] = float(w)
        except ValueError:
            r["_w"] = None
        rows.append(r)
    return rows


def table1(rows):
    by_model = defaultdict(list)
    stab = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
        if r["_w"] is not None:
            stab[(r["model"], r["profile_id"], r["burden_level"])].append(r["_w"])
    model_sds = defaultdict(list)
    for (m, _, _), vals in stab.items():
        if len(vals) >= 2:
            model_sds[m].append(statistics.stdev(vals))
    # recompute per-profile burden means properly
    pm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["_w"] is not None:
            pm[(r["model"], r["profile_id"])][r["burden_level"]].append(r["_w"])
    deltas = defaultdict(list)
    flips = defaultdict(int)
    for (m, _), bl in pm.items():
        if "high" in bl and "low" in bl:
            d = statistics.mean(bl["high"]) - statistics.mean(bl["low"])
            deltas[m].append(d)
            if d > 0:
                flips[m] += 1
    out = []
    for m in sorted(by_model):
        g = by_model[m]
        n = len(g)
        sc = [r["_w"] for r in g if r["_w"] is not None]
        ds = deltas[m]
        out.append({
            "model": m, "N": n,
            "json_valid_pct": round(100 * sum(r["json_valid"] for r in g) / n, 1),
            "score_valid_pct": round(100 * sum(r["score_valid"] for r in g) / n, 1),
            "non_refusal_pct": round(100 * sum(r["non_refusal"] for r in g) / n, 1),
            "mean_willingness": round(statistics.mean(sc), 2) if sc else "-",
            "stability_sd": round(statistics.mean(model_sds[m]), 2) if model_sds[m] else "-",
            "delta_sd": round(statistics.stdev(ds), 2) if len(ds) >= 2 else "-",
            "sign_flip_pct": round(100 * flips[m] / len(ds), 1) if ds else "-",
        })
    return out


def table3(rows):
    pm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["_w"] is not None:
            pm[(r["model"], r["profile_id"])][r["burden_level"]].append(r["_w"])
    acc = defaultdict(lambda: {"low": [], "high": []})
    for (m, _), bl in pm.items():
        if "low" in bl and "high" in bl:
            acc[m]["low"].append(statistics.mean(bl["low"]))
            acc[m]["high"].append(statistics.mean(bl["high"]))
    out = []
    for m in sorted(acc):
        lo, hi = acc[m]["low"], acc[m]["high"]
        d = [a - b for a, b in zip(lo, hi)]
        sd = statistics.stdev(d) if len(d) >= 2 else 0.0
        se = sd / (len(d) ** 0.5) if d else 0.0
        n = len(d)
        md = statistics.mean(d)
        out.append({
            "Model": m,
            "Mean_Low": round(statistics.mean(lo), 2),
            "Mean_High": round(statistics.mean(hi), 2),
            "Mean_Delta": round(md, 2),
            "SD_Delta": round(sd, 2),
            "SE_Delta": round(se, 2),
            "CI_Low": round(md - 1.96 * se, 2),
            "CI_High": round(md + 1.96 * se, 2),
            "N": n,
        })
    return out


def write(rows, path, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}  ({len(rows)} models)")


if __name__ == "__main__":
    rows = load_deduped(RAW)
    t1 = table1(rows)
    t3 = table3(rows)
    print("\nTable 1 — diagnostics")
    print(f"{'model':<24}{'N':>5}{'JSON%':>7}{'Score%':>8}{'NoRef%':>8}{'MeanW':>8}{'StabSD':>8}{'DeltaSD':>9}{'Flip%':>7}")
    for r in t1:
        print(f"{r['model']:<24}{r['N']:>5}{r['json_valid_pct']:>7}{r['score_valid_pct']:>8}{r['non_refusal_pct']:>8}"
              f"{str(r['mean_willingness']):>8}{str(r['stability_sd']):>8}{str(r['delta_sd']):>9}{str(r['sign_flip_pct']):>7}")
    print("\nTable 3 — burden effects")
    for r in t3:
        print(f"{r['Model']:<24} low={r['Mean_Low']:<6} high={r['Mean_High']:<6} delta={r['Mean_Delta']:<7}"
              f" [{r['CI_Low']}, {r['CI_High']}] N={r['N']}")
    write(t1, OUT1, ["model", "N", "json_valid_pct", "score_valid_pct", "non_refusal_pct",
                     "mean_willingness", "stability_sd", "delta_sd", "sign_flip_pct"])
    write(t3, OUT3, ["Model", "Mean_Low", "Mean_High", "Mean_Delta", "SD_Delta", "SE_Delta",
                     "CI_Low", "CI_High", "N"])

    assert len(t1) == 15, f"expected 15 models, got {len(t1)}"
    assert all(r["N"] >= 260 for r in t1), "model with < 260 usable cells"
    print("\nself-check ok: 15 models, all with >=260 cells")
