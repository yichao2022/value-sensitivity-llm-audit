#!/usr/bin/env python3
"""Independent frame coding of the 200 H2 audit rationales with TypeSafe Jev.

Same five frames as build_h2_pipeline.py's regex rules, but asked as typed Noul
questions instead of keyword matching. Output keeps the raw probabilities so
calibration can be checked separately from the 0/1 threshold.

    python3 jev_h2_coding.py           # needs TYPESAFE_API_KEY in .env
"""
import csv
import json
from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import Noul, TypeSafeClient

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / ".env")
OUT = PROJECT / "outputs/h2_jev_coding.csv"

FRAMES = {
    "access_barriers": "The rationale attributes reduced willingness to access barriers: cost, time, work inflexibility, childcare, travel distance, or socioeconomic constraints that make vaccination harder to obtain",
    "collective_responsibility": "The rationale frames vaccination as a collective responsibility: protecting others, public health, community benefit, or civic duty",
    "coercive_backlash": "The rationale describes backlash, resentment, distrust, resistance, or reduced legitimacy arising from mandates, penalties, or enforcement",
    "autonomy_infringement": "The rationale invokes infringement of personal autonomy, freedom of choice, privacy, consent, or the right to decide for oneself",
    "procedural_legitimacy": "The rationale treats the procedure itself as legitimate, justified, reasonable, necessary, or a fair safeguard (eligibility checks, fraud prevention, due process)",
}


def main():
    sheet = list(csv.DictReader(open(PROJECT / "outputs/h2_gold_coding_sheet.csv")))
    client = TypeSafeClient()
    questions = {f: Noul(instructions=text) for f, text in FRAMES.items()}
    rows = []
    for i, r in enumerate(sheet, 1):
        ans = client.system_one(state=r["rationale_text"], questions=questions).answers
        row = {"rationale_id": r["rationale_id"]}
        row.update({f: round(ans[f].noul, 4) for f in FRAMES})
        rows.append(row)
        if i % 50 == 0:
            print(f"  {i}/{len(sheet)}")
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["rationale_id"] + list(FRAMES))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
