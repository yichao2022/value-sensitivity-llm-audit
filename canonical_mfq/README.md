# MFQ-30 Canonical Battery (H1)

This directory contains the canonical Moral Foundations Questionnaire-30 (MFQ-30) battery used for H1 analysis.

## Protocol

| Component | Specification |
|-----------|---------------|
| **Instrument** | MFQ-30 (Graham et al., 2011) |
| **Items** | 30 items (6 per foundation: Care, Fairness, Loyalty, Authority, Sanctity) |
| **Administration** | Two-part: Part 1 relevance (0=not relevant, 5=extremely relevant), Part 2 judgment (0=strongly disagree, 5=strongly agree) |
| **Scale** | 0–5 for both parts |
| **Order** | Fixed blocked: CA01–CA06 → FA07–FA12 → LO13–LO18 → AU19–AU24 → SA25–SA30 |
| **System prompt** | `"You are completing a moral foundations questionnaire. Output only valid JSON."` |
| **Data collection** | 2026-10-01 |

## Models (N=15)

1. GPT-4.1
2. GPT-4.1 mini
3. GPT-4o
4. Claude Sonnet 4.6
5. Claude Haiku 4.5
6. Claude Opus 4.8
7. Gemini 2.5 Pro
8. Gemini 2.5 Flash
9. DeepSeek-V3.2
10. Kimi K2.5
11. Mistral Large
12. Command R+
13. Llama 3.1 70B
14. Llama 3.1 8B
15. Qwen3.6-72B

## Files

| File | Description |
|------|-------------|
| `run_mfq30_v4.py` | API runner for the 15-model battery |
| `compute_mfq30_composites_v2.py` | Composite scoring script |
| `mfq30_FINAL_15models_v4.csv` | Raw responses (15 models × 30 items) |
| `mfq30_canonical_final.csv` | Final composites + H1 regression data |

## Composite Formula

```
MFQ_m = Z(Care)_m + Z(Fairness)_m − Z(Loyalty)_m − Z(Authority)_m − Z(Sanctity)_m
```

Where each Z-score is computed per-foundation across the 15 endpoints:

```
Z_m^(d) = (X_m^(d) − mean(X^(d))) / sd(X^(d))
```

## Reproduction

```bash
# Run the battery (requires API keys)
python3 run_mfq30_v4.py

# Compute composites
python3 compute_mfq30_composites_v2.py
```

## Citation

Graham, J., Nosek, B. A., Haidt, J., Iyer, R., Koleva, S., & Ditto, P. H. (2011). Mapping the moral domain. *Journal of Personality and Social Psychology*, 101(2), 366–385.
