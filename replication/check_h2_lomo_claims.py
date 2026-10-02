#!/usr/bin/env python3
"""Guard: H2 numbers in the manuscript must match the canonical runs.

Checks three places against the analysis outputs:
  1. Table 2 in main.tex — pair-FE coefficient, BH-adjusted p, and the two-way
     fixed-effects comparison coefficient — vs outputs/canonical/h2_pairfe_main.csv
     (and the two-way values cross-checked against h2_fe_main.csv).
  2. The table note's reported unadjusted/BH two-way p-values vs h2_fe_main.csv.
  3. The leave-one-out range table in supplementary.tex vs h2_pairfe_lomo.csv and
     h2_pairfe_lopo.csv (ranges and the "0 of 15 / 0 of 27" significance counts).

Exit 0 = consistent; 1 = at least one reported number contradicts the canonical output.
"""
import re
import sys
from pathlib import Path

import pandas as pd

HOME = Path(__file__).resolve().parent
CAN = HOME / "outputs" / "canonical"
PAPER = Path.home() / "Documents" / "value-sensitivity-llm-audit"
FRAMES = {"access_barriers": "Access barriers",
          "collective_responsibility": "Collective responsibility",
          "coercive_backlash": "Coercive backlash"}


def close(a, b, tol=0.006):
    return abs(float(a) - float(b)) < tol


def main() -> int:
    pair = pd.read_csv(CAN / "h2_pairfe_main.csv").set_index("frame")
    two = pd.read_csv(CAN / "h2_fe_main.csv").set_index("frame")
    lomo = pd.read_csv(CAN / "h2_pairfe_lomo.csv")
    lopo = pd.read_csv(CAN / "h2_pairfe_lopo.csv")
    tex = (PAPER / "main.tex").read_text()
    supp = (PAPER / "supplementary.tex").read_text()
    problems = []

    for key, label in FRAMES.items():
        row = next((ln for ln in tex.splitlines() if ln.startswith(label + " &")), None)
        if row is None:
            problems.append(f"{label}: Table 2 row not found")
            continue
        cells = [c.strip() for c in row.rstrip("\\ ").split("&")]
        b_pair, se, ci, p_bh, b_two = cells[2], cells[3], cells[4], cells[5], cells[6]
        for got, want, what in ((b_pair, pair.loc[key, "b"], "pair-FE b"),
                                (p_bh, pair.loc[key, "p_bh"], "pair-FE BH p"),
                                (b_two, two.loc[key, "b"], "two-way b")):
            if not close(got, want):
                problems.append(f"{label}: Table 2 {what} = {got} but canonical = {float(want):.3f}")
        if not close(float(ci.strip("[]").split(",")[0].replace("\\(-", "-").replace("\\(+", "").replace("\\)", "")),
                     pair.loc[key, "lo"], 0.02):
            problems.append(f"{label}: Table 2 CI lower does not match the canonical interval")
        print(f"  {label:<24} pair-FE b={float(b_pair):+.2f} BH p={float(p_bh):.2f} "
              f"| two-way b={float(b_two):+.2f} (canonical {two.loc[key,'b']:+.2f}, "
              f"BH p={two.loc[key,'p_bh']:.3f})")

    # the secondary specification's raw and BH-adjusted p-values must be reported somewhere
    both = tex + supp
    for key in FRAMES:
        raw, bh = two.loc[key, "p"], two.loc[key, "p_bh"]
        for val, what in ((raw, "unadjusted"), (bh, "BH-adjusted")):
            if f"{val:.3f}" not in both:
                problems.append(f"{key}: {what} two-way p ({val:.3f}) is not reported in either document")

    # leave-one-out ranges in the supplement
    lines = supp.splitlines()
    start = next(i for i, ln in enumerate(lines) if "tab:h2_pairfe_leaveout" in ln)
    block = lines[start:start + 12]
    for key, label in FRAMES.items():
        row = next((ln for ln in block if ln.startswith(label + " &")), None)
        if row is None:
            problems.append(f"{label}: leave-one-out row not found after tab:h2_pairfe_leaveout")
            continue
        lo_b, hi_b = lomo[f"{key}_b"].min(), lomo[f"{key}_b"].max()
        lo_p, hi_p = lopo[f"{key}_b"].min(), lopo[f"{key}_b"].max()
        for span, want_lo, want_hi in ((row, lo_b, hi_b),):
            nums = re.findall(r"([+-]\d\.\d\d)", span)
            if len(nums) < 4 or not (close(nums[0], want_lo, 0.011) and close(nums[1], want_hi, 0.011)
                                     and close(nums[2], lo_p, 0.011) and close(nums[3], hi_p, 0.011)):
                problems.append(f"{label}: LOMO/LOPO ranges {nums[:4]} vs canonical "
                                f"[{lo_b:+.2f},{hi_b:+.2f}] / [{lo_p:+.2f},{hi_p:+.2f}]")
        n_lomo_sig = int((lomo[f"{key}_p"] < 0.05).sum())
        n_lopo_sig = int((lopo[f"{key}_p"] < 0.05).sum())
        if n_lomo_sig or n_lopo_sig:
            problems.append(f"{label}: canonical leave-one-out has significant fits "
                            f"(LOMO {n_lomo_sig}/15, LOPO {n_lopo_sig}/27) but the supplement reports 0")
        print(f"  {label:<24} LOMO [{lo_b:+.2f},{hi_b:+.2f}] sig {n_lomo_sig}/15  "
              f"LOPO [{lo_p:+.2f},{hi_p:+.2f}] sig {n_lopo_sig}/27")

    # ---- prose claims: main text counts and the leave-one-out paragraph in the appendix
    n_lomo_pos = int((lomo["access_barriers_b"] > 0).sum())
    n_lopo_pos = int((lopo["access_barriers_b"] > 0).sum())
    for phrase, name, where in ((f"{n_lomo_pos} of 15 leave-one-model-out", "main", tex),
                                (f"{n_lopo_pos} of 27 leave-one-profile-out", "main", tex),
                                (f"{n_lomo_pos} of 15 LOMO", "supplement", supp),
                                (f"{n_lopo_pos} of 27 LOPO", "supplement", supp)):
        if phrase not in where:
            problems.append(f"{name}: missing/broken sign count {phrase!r}")
    stale = ("sign-stable in all fifteen leave-one-model-out",
             "positive and significant in all 27 fits",
             "p<0.05 throughout")
    for phrase in stale:
        for name, doc in (("main", tex), ("supplement", supp)):
            if phrase in doc:
                problems.append(f"stale two-way-FE wording still present in {name}: {phrase!r}")
    # the appendix paragraph's ranges must equal the canonical pair-FE ranges
    anchor = "For H2 under the primary pair-fixed-effects specification"
    if anchor in supp:
        seg = supp[supp.index(anchor):][:600]
        nums = [float(x) for x in re.findall(r"([+-]\d\.\d\d)", seg)]
        want = [lopo["access_barriers_b"].min(), lopo["access_barriers_b"].max(),
                lopo["collective_responsibility_b"].min(), lopo["collective_responsibility_b"].max(),
                lopo["coercive_backlash_b"].min(), lopo["coercive_backlash_b"].max()]
        if len(nums) < 6 or any(not close(g, w, 0.006) for g, w in zip(nums[:6], want)):
            problems.append(f"appendix leave-one-out prose ranges {nums[:6]} vs canonical {[round(w,2) for w in want]}")
    else:
        problems.append("the appendix paragraph on pair-FE leave-one-out fits is missing")

    # ---- appendix pointers and the retired secondary-specification leave-one-out block
    if "Leave-one-model-out under the secondary specification" in supp:
        problems.append("supplement still contains the bare heading for a secondary-specification LOMO block")
    if "leave-one-out variants of both specifications" in supp:
        problems.append("Appendix G still promises leave-one-out variants of both specifications")
    if "Leave-one-out ranges for the primary specification" not in tex:
        problems.append("main text does not point at the primary-specification leave-one-out ranges")
    n_leaveout_tables = supp.count("\\label{tab:h2_pairfe_leaveout}")
    if n_leaveout_tables != 1:
        problems.append(f"expected exactly one leave-one-out table in the supplement, found {n_leaveout_tables}")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nH2 numbers in main.tex and supplementary.tex match the canonical runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
