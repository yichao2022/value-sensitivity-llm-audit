#!/usr/bin/env python3
"""Fix Table 1 and regenerate Figure 2 with correct Z-score data."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load canonical data
df = pd.read_csv('/Users/cary/Documents/value-sensitivity-llm-audit/replication/outputs/canonical_h1_table.csv')

# Sort by final_mfq_score descending
df = df.sort_values('final_mfq_score', ascending=False)

# Generate Table 1 LaTeX
print("=== Table 1 LaTeX ===")
for _, row in df.iterrows():
    model = row['model']
    if 'Instruct' in model:
        model = model.replace(' Instruct', '')
    print(f"{model} & {row['final_mfq_score']:.2f} & ... & ... & {row['delta']:.2f} & [...] \\\\")

# Generate Figure 2
fig, ax = plt.subplots(figsize=(9, 6))
X = df['final_mfq_score'].values
Y = df['delta'].values

ax.scatter(X, Y, color="steelblue", s=70, zorder=5)

# Selective labels for extremes
labels_to_show = {
    'Gemini 2.5 Pro': (8, 8),
    'Gemini 2.5 Flash': (8, 8),
    'GPT-4.1': (8, -12),
    'Llama 3.1 8B Instruct': (-8, 8),
    'Mistral Large': (-8, -12),
}

for i, row in df.iterrows():
    if row['model'] in labels_to_show:
        offset = labels_to_show[row['model']]
        ax.annotate(row['model'].replace(' Instruct', ''), 
                   (X[i], Y[i]), 
                   textcoords="offset points", 
                   xytext=offset, 
                   fontsize=8, ha='center')

# Regression line
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const).fit()
xr = np.linspace(X.min() - 0.5, X.max() + 0.5, 100)
ax.plot(xr, model.predict(sm.add_constant(xr)), color="darkred", 
        linewidth=1.5, linestyle="--",
        label=rf"$\beta$={model.params[1]:.2f}, $R^2$={model.rsquared:.2f}")

ax.set_xlabel("MFQ Composite (standardized individualizing minus binding foundations)", fontsize=11)
ax.set_ylabel(r"Mean Burden Effect ($\Delta$ willingness, 0–100)", fontsize=11)
ax.set_title("H1: MFQ Composite and Administrative-Burden Effects", fontsize=13)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('/Users/cary/Documents/value-sensitivity-llm-audit/output.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"\n=== Figure 2 Generated ===")
print(f"β = {model.params[1]:.2f}")
print(f"R² = {model.rsquared:.2f}")
print(f"p = {model.pvalues[1]:.3f}")
