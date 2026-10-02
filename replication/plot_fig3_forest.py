#!/usr/bin/env python3
"""Main Figure 3: forest plot of within-model frame-manipulation effects.

Single source: outputs/h3_clustered_results.csv (cluster-robust, canonical H3).
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/Users/cary/Documents/value-sensitivity-llm-audit/fig3_h3_forest.pdf")
rows = list(csv.DictReader(open("/Users/cary/Documents/New project/outputs/h3_clustered_results.csv", newline="")))
contrasts = [("autonomy", "Autonomy vs. neutral"), ("collective", "Collective obligation vs. neutral"),
             ("equity", "Equity/access vs. neutral")]
order = ["Pooled", "GPT-4.1", "Llama 3.1 70B Instruct", "Qwen3.7 Plus", "Mistral Large"]
by = {r["Model"]: r for r in rows}

fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
ys = list(range(len(order)))[::-1]
for ax, (key, title) in zip(axes, contrasts):
    for y, m in zip(ys, order):
        r = by[m]
        c = float(r[f"{key}_coef"]); lo = float(r[f"{key}_ci_low"]); hi = float(r[f"{key}_ci_high"])
        pooled = m == "Pooled"
        ax.errorbar(c, y, xerr=[[c - lo], [hi - c]], fmt="D" if pooled else "o",
                    color="black" if pooled else "steelblue", markersize=6 if pooled else 5,
                    elinewidth=1.4, capsize=2.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Shift in burden effect (0--100 scale)", fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    ax.tick_params(labelsize=8)
axes[0].set_yticks(ys)
axes[0].set_yticklabels(["Pooled (4-model average)"] + [m.replace(" Instruct", "") for m in order[1:]], fontsize=9)
plt.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
for key, _ in contrasts:
    print(" ", key, {m: (round(float(by[m][f'{key}_coef']), 2)) for m in order})
