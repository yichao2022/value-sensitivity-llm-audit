#!/usr/bin/env python3
"""Re-run H2 narrative coding with the 'public health center' false-positive fixed.

Original rule: the bare "public health" pattern fired collective_responsibility on every
rationale that mentioned a "public health center" (the vaccine site) — a collocation
with no collective-responsibility meaning. Fixed rule excludes that collocation.

Recomputes prevalence on the same 2,019 high-burden valid rationales and prints
the before/after for every frame. Writes narrative_coding_fixed.csv.
"""
import csv
import re
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs"

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
        r"civic\s+duty", r"protect\w*\s+others?",
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
# The only change from the original: the bare r"public\s+health" rule is dropped.
FIXED_COLLECTIVE = FRAME_RULES["collective_responsibility"]

# Original build_h2_pipeline.py had a bare r"public\s+health" in collective_responsibility.
# On this corpus it fired 1,495× on "trust in public health" (trust in authorities — not
# collective responsibility) and 81× on "public health center" (the vaccination site).
# Reconstruct the buggy rule for the before column; the fixed rule drops it entirely
# and keeps only genuine collective-responsibility phrases (protect others, herd
# immunity, common good, civic duty, vulnerable groups, community welfare, ...).
ORIGINAL = {**FRAME_RULES, "collective_responsibility": FRAME_RULES["collective_responsibility"] + [r"public\s+health"]}
FIXED = FRAME_RULES

FRAMES = list(FRAME_RULES)


def code(text, rules):
    t = text.lower()
    return {f: 1 if any(re.search(p, t) for p in rules[f]) else 0 for f in FRAMES}


def main():
    rows = [r for r in csv.DictReader(open(OUT / "canonical" / "sim_raw.csv"))
            if r["burden_level"] == "high" and r["json_valid"].lower() == "true"
            and r["score_valid"].lower() == "true" and r["non_refusal"].lower() == "true"
            and (r.get("rationale") or "").strip()]
    n = len(rows)
    print(f"high-burden valid rationales: {n}\n")

    pre = Counter(); post = Counter()
    changed = Counter()
    for r in rows:
        text = r["rationale"]
        a = code(text, ORIGINAL)
        b = code(text, FIXED)
        for f in FRAMES:
            pre[f] += a[f]; post[f] += b[f]
            if a[f] != b[f]:
                changed[f] += 1

    print(f"{'frame':<26} {'orig':>6} {'fixed':>6} {'changed':>8}")
    for f in FRAMES:
        print(f"{f:<26} {pre[f]:>5} ({100*pre[f]/n:4.1f}%) {post[f]:>5} ({100*post[f]/n:4.1f}%) {changed[f]:>8}")

    # write fixed narrative coding for downstream reuse
    fields = ["model", "profile_id", "repetition", "rationale", *FRAMES]
    with open(OUT / "narrative_coding_fixed.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            frames = code(r["rationale"], FIXED)
            w.writerow({"model": r["model"], "profile_id": r["profile_id"],
                        "repetition": r["repetition"], "rationale": r["rationale"], **frames})
    print(f"\nwrote {OUT / 'narrative_coding_fixed.csv'}")

    # show what fired collective under the fixed rule (if anything)
    fired = set()
    for r in rows:
        if any(re.search(p, r["rationale"].lower()) for p in FIXED_COLLECTIVE):
            fired.add(r["rationale"][:150])
    print(f"\ncollective under FIXED rule: {post['collective_responsibility']} ({100*post['collective_responsibility']/n:.1f}%)")
    for s in list(fired)[:5]:
        print("   -", s)


if __name__ == "__main__":
    main()
