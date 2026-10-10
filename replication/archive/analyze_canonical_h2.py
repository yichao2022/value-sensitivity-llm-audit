#!/usr/bin/env python3
"""
Canonical H2: narrative frame -> burden effect.

Sources: outputs/canonical/sim_raw.csv (cell-deduped), outputs/canonical/table2_orientation_profiles.csv (PVC).
Frame rules are copied verbatim from build_h2_pipeline.py so the codebook is unchanged;
only the input data and the inference (cluster-robust SE at model x profile) are canonical.
"""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm

P = Path(__file__).resolve().parent
OUT = P / "outputs" / "canonical"

FRAME_RULES = {
    "autonomy_infringement": [
        r"individual\s+choice", r"personal\s+choice", r"personal\s+freedom",
        r"autonomy", r"privacy", r"consent", r"personal\s+discretion",
        r"right\s+to\s+choose", r"individual\s+liberty", r"personal\s+decision",
        r"freedom\s+to\s+decide", r"procedural\s+dignity",
        r"limit\w*\s+(individual|personal|her)\s+(choice|freedom|autonomy)",
        r"infringe\w*\s+on\s+(individual|personal)",
    ],
    "procedural_legitimacy": [
        r"justified", r"safeguard", r"reasonable\s+(verification|requirement|check|procedure)",
        r"necessary\s+(procedure|requirement|check|verification|step)",
        r"eligibility", r"fraud\s+prevention", r"legitimate\s+(reason|need|purpose|public)",
        r"proper\s+(procedure|verification|documentation)", r"due\s+process",
        r"accountability", r"transparency", r"oversight",
        r"prevent\s+(fraud|abuse|misuse)", r"fair\s+(procedure|process|system)",
    ],
    "collective_responsibility": [
        r"civic\s+duty", r"public\s+health", r"protect\w*\s+others?",
        r"communal\s+(benefit|welfare|good|health)", r"vulnerable\s+group",
        r"collective\s+(welfare|benefit|good|responsibility|health)",
        r"common\s+good", r"social\s+responsibility", r"greater\s+good",
        r"protect\w*\s+the\s+public", r"community\s+(health|benefit|welfare)",
        r"herd\s+immunity", r"protect\w*\s+(the\s+)?vulnerable",
    ],
    "access_barriers": [
        r"cannot\s+afford", r"can'?t\s+afford",
        r"miss\s+work", r"miss\w*\s+(her|the)\s+job", r"lose\s+(income|wages|pay)",
        r"lost\s+wages?", r"unpaid\s+(leave|time)",
        r"childcare", r"child\s+care", r"babysit",
        r"no\s+(regular\s+)?(doctor|provider|clinic|healthcare)",
        r"lack\s+of\s+(access|transportation|childcare|flexibility|resource)",
        r"low\s+income", r"limited\s+(income|resource|flexibility|means)",
        r"cannot\s+(comply|attend|go|make\s+it|manage|participate)",
        r"(unable|cannot)\s+to\s+(take\s+time|get\s+time|find\s+time)",
        r"no\s+(way|means)\s+to",
        r"disadvantage", r"unequal\s+(burden|access|impact)",
        r"too\s+(difficult|hard|expensive|costly|burdensome)\s+for\s+(her|someone|a\s+person)",
        r"(financial|economic)\s+(hardship|strain|barrier|constraint)",
        r"(single|working)\s+(parent|mother)",
        r"hourly\s+(wage|worker|job)", r"no\s+paid\s+(leave|time|sick)",
        r"precarious\s+(work|employment|job|income)",
    ],
    "coercive_backlash": [
        r"resentment", r"\bdistrust\b", r"\bresistance\b", r"backlash",
        r"reduced\s+legitimacy", r"negative\s+reaction", r"\bmandate\b",
        r"\bpenalt\w+", r"\benforcement\b", r"undermine\w*\s+trust",
        r"backfire", r"antagonize", r"\balienate\b", r"counterproductive",
        r"coerci\w+", r"\bfine\w*", r"\bpunish\w+", r"resent\w+",
        r"push\s*back", r"non\s*compliance", r"rebelli\w+",
    ],
}
FRAMES_ALL = list(FRAME_RULES)


def code(text):
    t = text.lower()
    return {f: int(any(re.search(p, t) for p in rules)) for f, rules in FRAME_RULES.items()}


def load():
    cells = {}
    for r in csv.DictReader(open(OUT / "sim_raw.csv", newline="")):
        cells[(r["model"], r["profile_id"], r["burden_level"], r["repetition"])] = r
    pvc = {r["Model"]: float(r["PVC"])
           for r in csv.DictReader(open(OUT / "table2_orientation_profiles.csv", newline=""))}
    ok = lambda r: (not (r.get("error") or "").strip()
                    and str(r.get("json_valid")).lower() == "true"
                    and str(r.get("score_valid")).lower() == "true"
                    and str(r.get("non_refusal")).lower() == "true"
                    and (r.get("willingness") or "").strip())
    w = {(k[0], k[1], k[3], k[2]): float(r["willingness"]) for k, r in cells.items() if ok(r)}
    rows = []
    for (m, pid, burden, rep), r in cells.items():
        if burden != "high" or not ok(r):
            continue
        lo = w.get((m, pid, rep, "low"))
        if lo is None:
            continue
        rows.append({"model": m, "profile_id": pid, "repetition": rep,
                     "delta": lo - float(r["willingness"]), "PVC": pvc.get(m),
                     "rationale": (r.get("rationale") or "").strip(),
                     **code(r.get("rationale", ""))})
    return [r for r in rows if r["PVC"] is not None]


def fit(rows, frames, cluster=True):
    y = np.array([r["delta"] for r in rows], float)
    X = sm.add_constant(np.column_stack([[r["PVC"] for r in rows]] + [[r[f] for r in rows] for f in frames]))
    m = sm.OLS(y, X).fit()
    if cluster:
        groups = np.array([f"{r['model']}|{r['profile_id']}" for r in rows])
        m = m.get_robustcov_results(cov_type="cluster", groups=groups)
    names = ["(Intercept)", "PVC"] + frames
    return m, names


if __name__ == "__main__":
    rows = load()
    n = len(rows)
    print(f"H2 rows (high-burden, matched low, valid): {n}   models: {len({r['model'] for r in rows})}")

    prev = {f: 100 * sum(r[f] for r in rows) / n for f in FRAMES_ALL}
    print("\nprevalence:")
    for f in FRAMES_ALL:
        print(f"  {f:<26} {prev[f]:5.1f}%{'   [excluded]' if not (1 <= prev[f] <= 99) else ''}")
    retained = [f for f in FRAMES_ALL if 1 <= prev[f] <= 99]

    m, names = fit(rows, retained)
    p = m.pvalues if hasattr(m, "pvalues") else None
    print(f"\nfull model (clustered at model x profile): R2={m.rsquared:.4f}  adj={m.rsquared_adj:.4f}  N={n}")
    for i, nm in enumerate(names):
        print(f"  {nm:<26} b={m.params[i]:+8.4f}  se={m.bse[i]:.3f}  p={m.pvalues[i]:.4f}")

    m_plain, _ = fit(rows, retained, cluster=False)
    print("  (plain OLS SEs:", " ".join(f"{nm}={m_plain.bse[i]:.3f}" for i, nm in enumerate(names)), ")")

    print("\nleave-one-model-out (sign stability of frame coefficients):")
    stable = defaultdict(list)
    for drop in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] != drop]
        mm, nn = fit(sub, retained)
        for f in retained:
            i = nn.index(f)
            stable[f].append(mm.params[i] > 0)
    for f in retained:
        pos = sum(stable[f])
        print(f"  {f:<26} positive in {pos}/{len(stable[f])} LOMO samples")

    with open(OUT / "h2_frame_decomposition.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "prevalence_pct", "b", "se_clustered", "p", "ci_low", "ci_high", "lomo_positive"])
        w.writeheader()
        ci = m.conf_int()
        for f in retained:
            i = names.index(f)
            w.writerow({"frame": f, "prevalence_pct": round(prev[f], 1), "b": round(m.params[i], 4),
                        "se_clustered": round(m.bse[i], 4), "p": float(m.pvalues[i]),
                        "ci_low": round(ci[i][0], 3), "ci_high": round(ci[i][1], 3),
                        "lomo_positive": f"{sum(stable[f])}/{len(stable[f])}"})
        w.writerow({"frame": "PVC", "prevalence_pct": "", "b": round(m.params[1], 4),
                    "se_clustered": round(m.bse[1], 4), "p": float(m.pvalues[1]),
                    "ci_low": round(ci[1][0], 3), "ci_high": round(ci[1][1], 3), "lomo_positive": ""})
    print("\nwrote", OUT / "h2_frame_decomposition.csv")
    assert n > 1500, f"H2 sample too small: {n}"
    assert all(r["delta"] is not None for r in rows)
    print("self-check ok")
