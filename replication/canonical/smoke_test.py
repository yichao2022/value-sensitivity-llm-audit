"""Smoke test: exercise the real call+log path end to end on a few cells (6 sim + 4 h3 + 1 orient).
Writes to outputs/canonical/_smoke_*.csv so it never pollutes the canonical run files."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import protocol as P
import run_canonical as R

models = R.load_models()
profiles, scenarios = R.inputs()
OUT = R.OUT

sim_models = [m for m in models if m["model"] in ("GPT-4.1", "Llama 3.1 8B Instruct")]
plan = R.build_plan_sim(sim_models, profiles[:2], scenarios)[:6]
p = R.write_plan(plan, R.PLANS / "_smoke_plan_sim.csv")
print("smoke sim plan:", len(plan), "->", p)

R.run_block(plan, OUT / "_smoke_sim.csv",
            lambda r: P.system_prompt("neutral"),
            lambda r: P.USER_TEMPLATE.format(profile_text=r["profile_text"],
                                             scenario_text=r["scenario_text"]),
            R.parse_score, "smoke-sim")

h3_models = [m for m in models if m["model"] == "GPT-4.1"]
hp = R.build_plan_h3(h3_models, profiles[:1], scenarios)[:4]
print("smoke h3 plan:", len(hp), "frames:", [r["frame"] for r in hp], "levels:", [r["burden_level"] for r in hp])
R.run_block(hp, OUT / "_smoke_h3.csv",
            lambda r: P.system_prompt(r["frame"]),
            lambda r: P.USER_TEMPLATE.format(profile_text=r["profile_text"],
                                             scenario_text=r["scenario_text"]),
            R.parse_score, "smoke-h3")
