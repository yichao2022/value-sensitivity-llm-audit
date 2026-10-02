#!/usr/bin/env python3
"""Guard: the model/route table in the supplement must match the executed plan.

Appendix A of supplementary.tex lists each endpoint's provider, access route and
identifier. Those are facts about what actually ran, so they are checked against the
frozen execution plan (canonical/plans/plan_sim.csv). Catches the class of error where
prose says "the first wave used official provider APIs" while the table (correctly)
shows a gateway route.

Checks:
  * same set of thirteen/twenty endpoint rows in both (labels normalized)
  * route column consistent with the plan (Official API ~ openai/anthropic/gemini/deepseek,
    OpenRouter ~ openrouter, DashScope API ~ dashscope)
  * prose in the appendix does not classify routes by wave
Exit 0 = consistent.
"""
import csv
import re
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
PLAN = HOME / "canonical" / "plans" / "plan_sim.csv"
SUPP = Path.home() / "Documents" / "value-sensitivity-llm-audit" / "supplementary.tex"

ROUTE_LABEL = {"openai": "Official API", "anthropic": "Official API", "gemini": "Official API",
               "deepseek": "Official API", "openrouter": "OpenRouter", "dashscope": "DashScope API"}


def main() -> int:
    plan_rows = list(csv.DictReader(open(PLAN, newline="")))
    plan = {}
    for r in plan_rows:
        plan[r["model"]] = (r["route"], r["requested_model_id"])
    supp = SUPP.read_text()
    problems = []

    if re.search(r"first wave|second wave", supp):
        problems.append("supplementary.tex still classifies access routes by wave (first/second wave)")

    body = supp[supp.index("\\begin{longtable}"):supp.index("\\end{longtable}")]

    seen = set()
    for line in body.splitlines():
        m = re.match(r"^(?P<model>[^&]+?) & (?P<provider>[^&]+?) & (?P<route>[^&]+?) &", line)
        if not m:
            continue
        model = m.group("model").strip()
        route = m.group("route").strip()
        if model.startswith("\\textbf"):
            continue
        # the plan uses short internal labels ("Claude Haiku"); the table uses the full
        # product name ("Claude Haiku 4.5"). Match on the longest plan label that prefixes the row.
        key = max((k for k in plan if model.lower().startswith(k.lower())), key=len, default=None)
        if key is None:
            key = next((k for k in plan if k.lower().startswith(model.lower())), None)
        seen.add(key if key else model)
        if key is None:
            problems.append(f"table row {model!r} is not in the executed plan")
            continue
        want = ROUTE_LABEL[plan[key][0]]
        if route != want:
            problems.append(f"{key}: table route {route!r} but the executed plan used {want!r} "
                            f"({plan[key][0]}, {plan[key][1]})")
    missing = set(plan) - seen
    if missing:
        problems.append(f"endpoints in the executed plan missing from the table: {sorted(missing)}")

    # prose route counts must match the plan
    n_official = sum(1 for v in plan.values() if ROUTE_LABEL[v[0]] == "Official API")
    n_router = sum(1 for v in plan.values() if ROUTE_LABEL[v[0]] == "OpenRouter")
    n_dash = sum(1 for v in plan.values() if ROUTE_LABEL[v[0]] == "DashScope API")
    for count, words in ((n_official, "nine endpoints were reached through official"),
                         (n_router, "five through the OpenRouter gateway"),
                         (n_dash, "one through the Alibaba DashScope API")):
        if count and words.split()[0] not in ("nine", "five", "one") or True:
            word = {9: "nine", 5: "five", 1: "one"}.get(count, str(count))
            expected = words.replace(words.split()[0], word, 1)
            if expected not in supp:
                problems.append(f"expected prose '{expected}' (count {count}) not found in the supplement")

    print(f"  executed plan: {len(plan)} endpoints - {n_official} official API, "
          f"{n_router} OpenRouter, {n_dash} DashScope")
    print(f"  supplement table rows matched: {len(seen)}/{len(plan)}")
    if problems:
        print("\nFAIL:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nAppendix A model routes match the executed plan.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
