#!/usr/bin/env python3
"""Regenerate Figure 2: H1 scatter plot with selective label placement."""
import csv
from pathlib import Path
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = Path(__file__).resolve().parent
OUT = HOME.parent / "output.png"

# Load data
data = list(csv.DictReader(open(HOME / "outputs" / "h1_mfq_results_table.csv", newline="")))
models = [r["Model"] for r in data]
X = np.array([float(r["MFQ_Composite"]) for r in data])
DELTA = np.array([float(r["Mean_Delta"]) for r in data])

# Regression
r1 = sm.OLS(DELTA, sm.add_constant(X)).fit()

# Only label extreme/leverage points
LABELS = {
    "Llama 3.1 8B Instruct": "Llama 8B",
    "Qwen3.7 Plus": "Qwen",
    "Gemini 2.5 Pro": "Gemini Pro", 
    "GPT-4.1": "GPT-4.1",
    "GPT-4o": "GPT-4o",
}

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X, DELTA, color="steelblue", s=70, zorder=5)

# Add selective labels
for i, m in enumerate(models):
    if m in LABELS:
        offset = (8, 8) if m == "Llama 3.1 8B Instruct" else (8, -12)
        ax.annotate(LABELS[m], (X[i], DELTA[i]), 
                    textcoords="offset points", xytext=offset, 
                    fontsize=9, ha='left', fontweight='bold')

# Regression line
xr = np.linspace(X.min() - 0.3, X.max() + 0.3, 100)
ax.plot(xr, r1.predict(sm.add_constant(xr)), color="darkred", linewidth=1.5, 
        linestyle="--", label=rf"$\beta$={r1.params[1]:.2f}, $R^2$={r1.rsquared:.2f}")

ax.set_xlabel("MFQ Composite (individualizing minus binding foundations)", fontsize=11)
ax.set_ylabel(r"Mean Burden Effect ($\Delta$ willingness, 0–100)", fontsize=11)
ax.set_title("H1: MFQ Composite and Administrative-Burden Effects", fontsize=13)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches='tight')
plt.close()

print(f"Saved {OUT}")
print(f"  beta={r1.params[1]:+.3f}, R2={r1.rsquared:.3f}, N={len(models)}")
