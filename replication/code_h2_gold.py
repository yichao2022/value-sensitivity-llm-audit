#!/usr/bin/env python3
"""One-rationale-at-a-time coding CLI for the blinded H2 gold-coding sheet.

Fills outputs/h2_gold_coding_sheet.csv (the copied key file is never read, so the coding
stays blind to the algorithm's labels). Saves after every item, so quitting is safe and
re-running resumes at the first blank row.

    python3 code_h2_gold.py                 # code the real sheet
    python3 code_h2_gold.py --sheet X.csv   # code a copy (used by the self-test)

Keys:  digits of the frames that apply (e.g. 14 = access + autonomy), 'n' = none of the
five, Enter = skip for now, 'b' = go back one, '?' = show the frame definitions, 'q' = save and quit.
"""
import argparse
import csv
import sys
import textwrap
from pathlib import Path

FRAMES = ["access_barriers", "collective_responsibility", "coercive_backlash",
          "autonomy_infringement", "procedural_legitimacy"]
COLS = [f"human_{f}" for f in FRAMES]
MENU = "\n".join(f"  {i + 1} = {f.replace('_', ' ')}" for i, f in enumerate(FRAMES))
DEFS = f"""Frames (code only what the rationale itself says; several may apply):
{MENU}
  n = none of the five apply
Do not open outputs/h2_gold_coding_key.csv while coding."""


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def save(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def parse(text):
    """-> dict of column -> '0'/'1', or None for skip, or 'q'/'b'."""
    t = text.strip().lower()
    if t == "q":
        return "q"
    if t == "b":
        return "b"
    if not t:
        return None
    if t == "n":
        return {c: "0" for c in COLS}
    digits = {d for d in t if d.isdigit()}
    if not digits or any(int(d) < 1 or int(d) > len(FRAMES) for d in digits):
        return "invalid"
    return {c: ("1" if str(i + 1) in digits else "0") for i, c in enumerate(COLS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=str(Path(__file__).resolve().parent /
                                           "outputs" / "h2_gold_coding_sheet.csv"))
    a = ap.parse_args()
    path = Path(a.sheet)
    fields, rows = load(path)
    print(DEFS + "\n")
    i = next((n for n, r in enumerate(rows) if not (r[COLS[0]] or "").strip()), len(rows))
    while i < len(rows):
        r = rows[i]
        print(f"--- {i + 1}/{len(rows)}  {r['rationale_id']} ---")
        print(textwrap.fill(r["rationale_text"], 100))
        try:
            ans = input("frames [1-5, n, b, ?, q]: ")
        except (EOFError, KeyboardInterrupt):
            print("\nsaved.")
            break
        if ans.strip() == "?":
            print(DEFS + "\n")
            continue
        p = parse(ans)
        if p == "q":
            print("saved.")
            break
        if p == "b":
            if i:
                i -= 1
                rows[i].update({c: "" for c in COLS})  # re-code the previous item
            continue
        if p in (None, "invalid"):
            print("  (invalid: use digits 1-5, 'n', 'b', '?', Enter to skip, 'q' to quit)")
            continue
        rows[i].update(p)
        save(path, fields, rows)
        i += 1
    coded = sum(1 for r in rows if (r[COLS[0]] or "").strip())
    print(f"\n{coded}/{len(rows)} rationales coded -> {path}")
    if coded == len(rows):
        print("all done — now run: python3 score_h2_gold_coding.py")


if __name__ == "__main__":
    sys.exit(main())
