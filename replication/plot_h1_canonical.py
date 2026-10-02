#!/usr/bin/env python3
"""Regenerate the H1 scatter (paper fig: output.png) from canonical outputs (MFQ version).

Label collision is resolved by measuring actual text-box extents via the renderer and
iteratively pushing overlapping labels downward — deterministic, no hand-tuned offsets.
"""
import csv
from pathlib import Path
import numpy as np, statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = Path("/Users/cary/Documents/New project/outputs")
OUT = Path("/Users/cary/Documents/value-sensitivity-llm-audit/output.png")

# MFQ composite (15 models, computed from MFQ-30 fielding)
mfq = {r["model"]: float(r["mfq_composite"]) for r in csv.DictReader(open(P / "mfq30_scores.csv", newline=""))}
# Canonical burden deltas (paper Table 3)
burd = {r["Model"]: float(r["Mean_Delta"]) for r in csv.DictReader(open(P / "canonical/table3_burden_effects.csv", newline=""))}
# Model names already mapped in mfq30_scores.csv

models = sorted(set(mfq) & set(burd))
X = np.array([mfq[m] for m in models]); DELTA = np.array([burd[m] for m in models])
r1 = sm.OLS(DELTA, sm.add_constant(X)).fit()

def label(m):
    return m.replace(" Instruct", "")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X, DELTA, color="steelblue", s=70, zorder=5)

anns = [ax.annotate(label(m), (X[i], DELTA[i]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8) for i, m in enumerate(models)]

def overlap(b1, b2):
    return not (b1.x1 <= b2.x0 or b2.x1 <= b1.x0 or b1.y1 <= b2.y0 or b2.y1 <= b1.y0)

fig.canvas.draw(); r = fig.canvas.get_renderer()
boxes = [a.get_window_extent(r) for a in anns]
moved, iters = True, 0
while moved and iters < 20:
    moved, iters = False, iters + 1
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            if overlap(boxes[i], boxes[j]):
                x0, y0 = anns[j].xyann
                anns[j].set_position((x0, y0 - 12))
                moved = True
    fig.canvas.draw(); boxes = [a.get_window_extent(r) for a in anns]

residual = sum(1 for i in range(len(anns)) for j in range(i + 1, len(anns)) if overlap(boxes[i], boxes[j]))
assert residual == 0, f"{residual} label overlaps remain"

xr = np.linspace(X.min() - 1, X.max() + 1, 100)
ax.plot(xr, r1.predict(sm.add_constant(xr)), color="darkred", linewidth=1.5, linestyle="--",
        label=rf"$\beta$={r1.params[1]:.2f}, $R^2$={r1.rsquared:.2f}")
ax.set_xlabel("MFQ Composite (z(individualizing) − z(binding))", fontsize=11)
ax.set_ylabel(r"Mean Burden Effect ($\Delta$ willingness, 0–100)", fontsize=11)
ax.set_title("H1: MFQ-30 Composite and Administrative-Burden Effects", fontsize=13)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(OUT, dpi=150); plt.close()
print(f"wrote {OUT}  beta={r1.params[1]:+.3f} R2={r1.rsquared:.3f} N={len(models)} (0 overlaps)")
