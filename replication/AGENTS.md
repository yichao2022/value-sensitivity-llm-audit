# AGENTS.md — LLM Policy Simulation

## Project structure

```
data/                          Input files (.xlsx, .csv)
outputs/                       All generated results (.csv, .tex, .png)
run_simulation.py              Main simulation pipeline (Table 1)
run_orientation.py             Policy-value orientation battery
run_h1_regression.py           H1 regression + scatterplot
build_h2_pipeline.py           H2 narrative coding + mediation
build_h3_plan.py               H3 frame manipulation plan
run_h3_simulation.py           H3 API runner
analyze_results.py             Table 1 diagnostics
compute_orientation_scores.py  Table 2 orientation scores
analyze_h3.py                  H3 analysis + Table 6
```

## Running

```bash
# Single pipeline
python3 run_simulation.py      # → outputs/simulation_outputs.csv
python3 analyze_results.py     # → Table 1

# Full workflow
python3 run_simulation.py && python3 analyze_results.py
python3 run_orientation.py && python3 compute_orientation_scores.py
python3 run_h1_regression.py
python3 build_h2_pipeline.py
python3 build_h3_plan.py && python3 run_h3_simulation.py && python3 analyze_h3.py
```

## API keys

Copy `.env.example` → `.env` and fill in keys. `.env` is gitignored.
