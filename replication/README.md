# Replication package — Auditing value sensitivity in LLM-assisted health policy simulation

Code and derived data for the manuscript *A Framework for Auditing Value Sensitivity in
Large Language Model–Assisted Health Policy Simulation*. All results were generated from the
**canonical rerun** in `outputs/canonical/`; the paper's H1/H2/H3 numbers, tables and figures all
trace back to the files listed below.

- Data collection window: **16–17 September 2026 (UTC)** (15 LLM endpoints, five repetitions per cell).
- MFQ-30 orientation battery: **32 items** (Part 1 relevance + Part 2 judgment), **0–5 scale**, administered in **fixed blocked order** (CA→FA→LO→AU→SA). See `canonical_mfq/` for scripts and raw responses.
- License: **MIT** (see `LICENSE`). Reuse, modification and redistribution are permitted with attribution.
  Raw model responses remain subject to the terms of service of the providers that generated them.
- No human participants were involved; all inputs are synthetic profiles plus model-generated text.

## MFQ-30 Canonical Battery (H1)

The moral-foundations orientation battery uses the **MFQ-30** (Graham et al., 2011) with a two-part 0–5 administration:

| Component | Description |
|-----------|-------------|
| **Items** | 30 items (6 per foundation: Care, Fairness, Loyalty, Authority, Sanctity) |
| **Scale** | 0–5 (Part 1: relevance 0=not relevant, 5=extremely relevant; Part 2: agreement 0=strongly disagree, 5=strongly agree) |
| **Order** | Fixed blocked: CA01–CA06 → FA07–FA12 → LO13–LO18 → AU19–AU24 → SA25–SA30 |
| **System prompt** | `"You are completing a moral foundations questionnaire. Output only valid JSON."` |
| **Models** | 15 endpoints (GPT-4.1, GPT-4.1 mini, GPT-4o, Claude Sonnet 4.6, Claude Haiku 4.5, Claude Opus 4.8, Gemini 2.5 Pro, Gemini 2.5 Flash, DeepSeek-V3.2, Kimi K2.5, Mistral Large, Command R+, Llama 3.1 70B, Llama 3.1 8B, Qwen3.6-72B) |

**Files in `canonical_mfq/`:**
- `run_mfq30_v4.py` — API runner for the 15-model battery
- `compute_mfq30_composites_v2.py` — Composite scoring (Z-standardized per foundation, sum: CA+FA−LO−AU−SA)
- `mfq30_FINAL_15models_v4.csv` — Raw responses (15 models × 30 items)
- `mfq30_canonical_final.csv` — Final composites + H1 regression data

**Composite formula:**
```
MFQ_m = Z(Care)_m + Z(Fairness)_m − Z(Loyalty)_m − Z(Authority)_m − Z(Sanctity)_m
```
where each Z is standardized across the 15 endpoints.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy statsmodels matplotlib python-dotenv requests
cp .env.example .env    # add provider API keys (never commit .env)
```

Analysis scripts were run with Python 3.14 (`/Library/Frameworks/Python.framework/Versions/3.14/bin/python3`
on the author's machine); any Python ≥3.10 with the packages above is sufficient to re-run the analysis
from the archived responses.

## Pipeline

```
run_canonical.py (sim | h3 | orient)   →  outputs/canonical/*_raw.csv      raw API responses
        ↓
analyze_canonical_tables.py            →  outputs/canonical/table1_diagnostics.csv, table3_burden_effects.csv
analyze_canonical_orientation.py       →  outputs/canonical/table2_orientation_profiles.csv  (+ H1)
analyze_h3.py                          →  outputs/h3_clustered_results.csv, h3_model_specific_full_4models.csv,
                                          h3_mean_delta_by_model_frame.csv, frame_delta_analysis.csv
analyze_canonical_h2_pairfe.py         →  outputs/canonical/h2_pairfe_main.csv (+ lomo/lopo/diagnostics); H2 PRIMARY spec
canonical/run_comprehension.py            →  outputs/canonical/comp_raw.csv (auxiliary burden-classification diagnostic, 15 x 27 x 2 = 810 calls, 2026-09-17)
analyze_comprehension.py                 →  outputs/canonical/comp_summary.csv (per-model + pooled accuracy, exact 95% CI; 100% for all models)
analyze_canonical_h1_samples.py     →  outputs/canonical/h1_sample_comparison.csv (H1 sample comparison, Table S13; also the five dimension-specific fits of Table S14)          →  outputs/canonical/h1_sample_comparison.csv (Table 12 rows, both standardisations, analytic MDE80)
check_table12_claims.py                  →  guard: Table 12 rows/note vs h1_sample_comparison.csv
check_model_routes.py                    →  guard: Appendix A model table vs the executed plan (canonical/plans/plan_sim.csv)
analyze_canonical_h2_fe.py             →  outputs/canonical/h2_fe_main.csv (+ h2_fe_lomo.csv, h2_fe_lopo.csv); H2 SECONDARY spec (two-way FE)
                                          H2 main model: model FE + profile FE + frame indicators, no PVOC
analyze_canonical_h2.py                →  outputs/canonical/h2_frame_decomposition.csv  (frame-coding loader; PVOC-adjusted comparison spec)
analyze_lopo.py                        →  outputs/lopo_robustness.csv       (leave-one-profile-out)
analyze_h3_small_sample.py             →  outputs/h3_small_sample_inference.csv (CR1 / corrected / wild-bootstrap p)
analyze_h3_pairfe.py                   →  outputs/canonical/h3_pairfe_robustness.csv (pooled H3 under model x profile pair FE)
plot_h1_canonical.py                   →  output.png                        (manuscript Figure 2)
plot_fig3_forest.py                    →  fig3_h3_forest.pdf               (manuscript Figure 3)
```

## Script → manuscript item map

| Manuscript item | Script | Output file |
|---|---|---|
| Figure 1 (diagnostic framework) | `fig1_conceptual.py` (manuscript repo) | `fig1_conceptual.pdf` |
| Table 1 (PVOC, low/high willingness, burden effect) | `analyze_canonical_orientation.py`, `analyze_canonical_tables.py` | `outputs/canonical/table2_orientation_profiles.csv`, `table3_burden_effects.csv` |
| Figure 2 (PVOC vs burden effect) | `plot_h1_canonical.py` | `output.png` |
| Table 12 (H1 nine- vs fifteen-endpoint) | `analyze_canonical_h1_samples.py` | `outputs/canonical/h1_sample_comparison.csv`; PVOC scoring rule documented in Appendix B |
| Table 2 (H2 frame–outcome associations) | `analyze_canonical_h2_pairfe.py` (PRIMARY: model x profile pair FE) ← `analyze_canonical_h2.py` (frame coding) | `outputs/canonical/h2_pairfe_main.csv`; secondary two-way FE in `h2_fe_main.csv` |
| Figure 3 (H3 forest plot) | `plot_fig3_forest.py` ← `analyze_h3.py` | `fig3_h3_forest.pdf` ← `outputs/h3_clustered_results.csv` |
| Supplement Table S1 (data-quality diagnostics) | `analyze_canonical_tables.py` | `outputs/canonical/table1_diagnostics.csv` |
| Supplement H1 robustness (9 vs 15 endpoints, leave-CO-out) | `analyze_canonical_orientation.py` | `outputs/canonical/table2_orientation_profiles.csv` |
| Supplement H2 leave-one-out + identification diagnostics | `analyze_canonical_h2_pairfe.py` (primary) and `analyze_canonical_h2_fe.py` (secondary) | `outputs/canonical/h2_pairfe_lomo.csv`, `h2_pairfe_lopo.csv`, `h2_pairfe_diagnostics.csv`, `h2_fe_lomo.csv` |
| Supplement leave-one-profile-out (H1/H2/H3) | `analyze_lopo.py` | `outputs/lopo_robustness.csv` |
| Supplement small-sample inference (H3, 27 clusters) | `analyze_h3_small_sample.py` | `outputs/h3_small_sample_inference.csv` |
| Supplement H3 pair-FE robustness | `analyze_h3_pairfe.py` | `outputs/canonical/h3_pairfe_robustness.csv` |
| H2 specification note | `analyze_canonical_h2_pairfe.py` | PVOC enters H1 only; H2 primary spec = model x profile pair FE (α_mi), identified from within-pair repetition variation; two-way FE kept as secondary |
| Supplement frame-manipulation prompts + model-specific H3 | `analyze_h3.py` | `outputs/h3_model_specific_full_4models.csv`, `h3_mean_delta_by_model_frame.csv` |
| Prompt templates and SHA256 hashes | `canonical/run_canonical.py` | prompts embedded in the runner; hashes re-derivable |
| Narrative frame regular expressions | `build_h2_pipeline.py` (`FRAME_RULES`) and the canonical coder | — |

## Reproducing the reported analyses

```bash
# 1. Re-run the API experiments (requires keys; ~4,000 calls for sim, ~2,200 for H3, 15 for orientation)
python3 canonical/run_canonical.py sim
python3 canonical/run_canonical.py h3
python3 canonical/run_canonical.py orient

# 2. Regenerate every table / figure number reported in the manuscript
python3 analyze_canonical_tables.py
python3 analyze_canonical_orientation.py
python3 analyze_canonical_h2.py
python3 analyze_h3.py
python3 analyze_lopo.py
python3 analyze_h3_small_sample.py
python3 analyze_canonical_h2_pairfe.py
python3 analyze_h3_pairfe.py
python3 check_h2_lomo_claims.py   # guard: manuscript H2 numbers vs canonical outputs
python3 plot_h1_canonical.py
python3 plot_fig3_forest.py
```

To reproduce from the archived responses only (no API keys needed), start at step 2 — the raw
response files in `outputs/canonical/` are the frozen inputs behind every number in the paper.

## Notes on the data layout

- `outputs/canonical/*_raw.csv` — one row per API call (request metadata, SHA256 of the prompts,
  raw response, validity flags, willingness, rationale, UTC timestamp). These are append-only
  accumulation files: cell-level de-duplication (keep the last valid row per cell) is applied inside
  the analysis scripts, matching what the manuscript reports.
- `outputs/canonical/models.csv` — model identifiers, providers, access routes.
- `outputs/expanded15/` — **superseded** first-wave (June 2026) outputs kept only for provenance; not
  used by any number in the current manuscript.
- `outputs/h3_four_model/` — superseded interim H3 outputs; not used.
- Unrelated scratch folders (`camphoto-cli/`, `colibri/`, `demosaic/`) are not part of this study.

## Auxiliary comprehension diagnostic (Appendix E)

`python3 canonical/run_comprehension.py plan|run` fields the burden-classification prompt (system-prompt SHA256 `4bf68c6c…`, plan SHA256 `96777194…`) over 15 endpoints x 27 profiles x 2 conditions = 810 one-word calls, appending to `outputs/canonical/comp_raw.csv` (plan: `canonical/plans/plan_comp.csv`). Run it with `env -u DASHSCOPE_API_KEY` so the shell's older DashScope key cannot shadow `.env`. `analyze_comprehension.py` writes `outputs/canonical/comp_summary.csv`. It is a separate administration from the frozen canonical run and is labelled as such in the manuscript.

`check_table12_claims.py` now also verifies Table S14 (all five dimension-specific regressions, not only collective obligation) against `h1_sample_comparison.csv`.

## Blinded H2 gold coding (manual audit upgrade)

`prepare_h2_gold_sample.py` writes the blinded 200-rationale sheet + separate key (seed 20260917, stratified by endpoint; re-running reproduces the same 200 ids). `code_h2_gold.py` walks the sheet one rationale at a time (digits of the frames that apply, `n` = none, `b` = back, `?` = definitions, `q` = save & quit; content is saved after every item and a re-run resumes at the first blank row). The key file is never read, so the coding stays blind to the algorithm's labels. `score_h2_gold_coding.py` then reports per-frame precision, recall, F1, exact agreement and Cohen's kappa. Self-test: pipe `printf '1\nn\n?\nb\n34\nq\n'` into it against a copy via `--sheet`.
