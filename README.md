# A Framework for Auditing Value Sensitivity in LLM-Assisted Health Policy Simulation

Manuscript source files for *A Framework for Auditing Value Sensitivity in Large Language Model–Assisted Health Policy Simulation*.

## Files

| File | Description |
|------|-------------|
| `main.tex` | Main manuscript (LaTeX source) |
| `main.pdf` | Compiled manuscript |
| `manuscript_full.pdf` | Copy of main.pdf |
| `supplementary-content.tex` | Supplementary materials |
| `supplementary.pdf` | Compiled supplementary |
| `references.bib` | Bibliography |
| `fig1_conceptual.{py,pdf,png}` | Figure 1 (framework diagram) |
| `fig3_h3_forest.pdf` | Figure 3 (forest plot) |
| `canonical_mfq/` | **MFQ-30 canonical battery** (H1 orientation measurement) |

## MFQ-30 Canonical Battery

See `canonical_mfq/` for the complete moral foundations orientation protocol:

- **Instrument**: MFQ-30 (Graham et al., 2011)
- **Administration**: Two-part 0–5 scale (relevance + judgment)
- **Order**: Fixed blocked (CA→FA→LO→AU→SA)
- **Models**: 15 LLM endpoints
- **Data**: Raw responses + computed composites

## Compilation

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## License

MIT License — see LICENSE file.

## Citation

> Jin, Y., & Kim, D. (2026). A framework for auditing value sensitivity in large language model–assisted health policy simulation. *Value in Health*.
