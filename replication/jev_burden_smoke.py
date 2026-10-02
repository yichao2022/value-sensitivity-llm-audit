#!/usr/bin/env python3
"""Run the vaccine-willingness judgment cells through TypeSafe Jev.

Same (profile x burden-scenario) grid as run_expansion_burden.py, but asks typed
questions instead of requesting JSON text from a chat model.

Usage:
    TYPESAFE_API_KEY=ts-... python3 jev_burden_smoke.py [n_cells]
"""
import csv
import sys
from pathlib import Path

import openpyxl
from dotenv import load_dotenv
from typesafe_sdk import Noul, Score, TypeSafeClient

PROJECT = Path(__file__).resolve().parent
load_dotenv(PROJECT / ".env")
OUT = PROJECT / "outputs/jev_burden_smoke.csv"

WILLINGNESS = [
    "Very unwilling: would not vaccinate",
    "Unwilling: leans against",
    "Uncertain: genuinely mixed",
    "Willing: leans toward",
    "Very willing: would vaccinate",
]


def load_cells():
    wb = openpyxl.load_workbook(PROJECT / "data/llm_experiment_inputs.xlsx", read_only=True)
    profiles = []
    scenarios = []
    for sheet, sink in (("Profiles", profiles), ("Burden Scenarios", scenarios)):
        rows = list(wb[sheet].iter_rows(values_only=True))
        sink.extend(dict(zip([str(h) for h in rows[0]], r)) for r in rows[1:])
    wb.close()
    return profiles, scenarios


def build_questions(profile, scenario):
    state = (
        f"Person profile:\n{profile['profile_text']}\n\n"
        f"Scenario:\n{scenario['scenario_text']}"
    )
    questions = {
        "willingness": Score(
            instructions="How willing this person would be to proceed with vaccination under this scenario",
            criteria=WILLINGNESS,
        ),
        "access_barrier": Noul(
            instructions="The scenario imposes a substantial logistical, travel, or time-cost barrier to being vaccinated",
        ),
        "coercion": Noul(
            instructions="The scenario relies on pressure, penalties, or mandates rather than voluntary choice",
        ),
    }
    return state, questions


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    profiles, scenarios = load_cells()
    cells = [(p, s) for p in profiles for s in scenarios][:n]
    client = TypeSafeClient()
    rows = []
    for profile, scenario in cells:
        state, questions = build_questions(profile, scenario)
        answers = client.system_one(state=state, questions=questions).answers
        rows.append({
            "profile_id": profile["profile_id"],
            "burden_level": scenario["burden_level"],
            "willingness": answers["willingness"].score,
            "willingness_conf": answers["willingness"].confidence,
            "access_barrier": answers["access_barrier"].noul,
            "coercion": answers["coercion"].noul,
        })
        print(rows[-1])
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}")


def _self_check():
    profiles, scenarios = load_cells()
    assert len(profiles) == 27, len(profiles)
    assert len(scenarios) == 2, len(scenarios)
    state, qs = build_questions(profiles[0], scenarios[0])
    assert set(qs) == {"willingness", "access_barrier", "coercion"}
    assert len(qs["willingness"].criteria) == 5
    assert profiles[0]["profile_text"][:20] in state and scenarios[0]["scenario_text"][:20] in state
    print("self-check ok", len(profiles), "profiles x", len(scenarios), "scenarios")


if __name__ == "__main__":
    if "--check" in sys.argv:
        _self_check()
    else:
        main()
