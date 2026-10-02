#!/usr/bin/env python3
"""
H1 (MFQ version, Plan B): Orientation-effect regression using MFQ-30 composite.
mean_delta ~ MFQ_composite
mean_delta ~ MFQ_composite + mean_low
Also reports per-foundation univariate diagnostics (CA/FA/LO/AU/SA).

Data:
  outputs/mfq30_scores.csv          (14 models; DeepSeek missing until recharge)
  outputs/expanded15/table3_burden_effects_15models.csv
"""

import csv
import numpy as np
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "outputs"

# ── Load & merge data ────────────────────────────────────────────

def load_csv(path, key_col="model"):
    data = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            model = row.get(key_col, row.get("Model", ""))
            data[model] = {k.lower(): (float(v) if v else None)
                          for k, v in row.items() if k not in (key_col, "Model")}
    return data

mfq_data = load_csv(OUT_DIR / "mfq30_scores.csv")
# Canonical burden data — the paper's official source (matches main.tex Table 3)
burden_data = load_csv(OUT_DIR / "canonical" / "table3_burden_effects.csv")

# 模型名归一化：MFQ fielding 用 Qwen3.6-72B，burden 表用 Qwen3.7 Plus（同一模型）
MODEL_ALIASES = {"Qwen3.6-72B Instruct": "Qwen3.7 Plus"}
mfq_data = {MODEL_ALIASES.get(m, m): v for m, v in mfq_data.items()}

models = sorted(set(mfq_data.keys()) & set(burden_data.keys()))
print(f"N = {len(models)} models (MFQ ∩ burden)")
print(f"Missing MFQ (not run): {sorted(set(burden_data) - set(mfq_data)) or 'none'}")
print(f"Missing burden: {sorted(set(mfq_data) - set(burden_data)) or 'none'}\n")

MFQ_COMP = np.array([mfq_data[m]["mfq_composite"] for m in models])
MEAN_DELTA = np.array([burden_data[m]["mean_delta"] for m in models])
MEAN_LOW = np.array([burden_data[m]["mean_low"] for m in models])

FOUNDATIONS = ["ca", "fa", "lo", "au", "sa"]
found_data = {d: np.array([mfq_data[m].get(d, np.nan) for m in models]) for d in FOUNDATIONS}

# ── OLS helper ───────────────────────────────────────────────────

def ols_table(model_name, y, X_vars, var_names):
    X = sm.add_constant(np.column_stack(X_vars))
    res = sm.OLS(y, X).fit()
    rows = []
    all_names = ["(Intercept)"] + var_names
    pvals = res.pvalues
    params = res.params
    bse = res.bse
    tvals = res.tvalues
    ci = res.conf_int()
    for i, name in enumerate(all_names):
        p_val = pvals[i]
        rows.append({
            "model": model_name,
            "predictor": name,
            "coef": round(params[i], 4),
            "se": round(bse[i], 4),
            "t": round(tvals[i], 4),
            "p": round(p_val, 4) if p_val >= 0.001 else 0.0,
            "ci_low": round(ci[i][0], 4),
            "ci_high": round(ci[i][1], 4),
            "r2": round(res.rsquared, 4),
            "adj_r2": round(res.rsquared_adj, 4),
            "n": int(res.nobs),
        })
    return rows, res

# ── Model 1: mean_delta ~ MFQ composite ──────────────────────────

print("=== Model 1: mean_delta ~ MFQ_composite ===")
t1, r1 = ols_table("Model 1", MEAN_DELTA, [MFQ_COMP], ["MFQ_composite"])
for row in t1:
    print(f"  {row['predictor']:<16} β={row['coef']:>8.4f}  SE={row['se']:.4f}  t={row['t']:.4f}  p={row['p']:.4f}  CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
print(f"  R²={t1[0]['r2']:.4f}  adjR²={t1[0]['adj_r2']:.4f}  N={t1[0]['n']}\n")

# ── Model 2: mean_delta ~ MFQ composite + mean_low ───────────────

print("=== Model 2: mean_delta ~ MFQ_composite + mean_low ===")
t2, r2 = ols_table("Model 2", MEAN_DELTA, [MFQ_COMP, MEAN_LOW], ["MFQ_composite", "mean_low"])
for row in t2:
    print(f"  {row['predictor']:<16} β={row['coef']:>8.4f}  SE={row['se']:.4f}  t={row['t']:.4f}  p={row['p']:.4f}  CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
print(f"  R²={t2[0]['r2']:.4f}  adjR²={t2[0]['adj_r2']:.4f}  N={t2[0]['n']}\n")

# ── Per-foundation univariate diagnostics ────────────────────────

print("=== Per-foundation univariate: mean_delta ~ foundation ===")
diag_rows = []
for d in FOUNDATIONS:
    valid = ~np.isnan(found_data[d])
    y = MEAN_DELTA[valid]
    x = found_data[d][valid]
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    beta, se, t, p = res.params[1], res.bse[1], res.tvalues[1], res.pvalues[1]
    ci = res.conf_int()[1]
    p_str = "<.001" if p < 0.001 else f"{p:.4f}"
    print(f"  {d:<4} β={beta:>8.4f}  SE={se:.4f}  t={t:.4f}  p={p_str}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]  R²={res.rsquared:.4f}  N={int(res.nobs)}")
    diag_rows.append({
        "foundation": d, "coef": round(beta, 4), "se": round(se, 4),
        "t": round(t, 4), "p": round(p, 4) if p >= 0.001 else 0.0,
        "ci_low": round(ci[0], 4), "ci_high": round(ci[1], 4),
        "r2": round(res.rsquared, 4), "n": int(res.nobs),
    })

# ── Save Table 4 CSV ─────────────────────────────────────────────

all_rows = t1 + t2 + diag_rows
fields = ["model", "predictor", "foundation", "coef", "se", "t", "p", "ci_low", "ci_high", "r2", "adj_r2", "n"]
out_rows = []
for row in t1 + t2:
    out_rows.append({**row, "foundation": ""})
for row in diag_rows:
    out_rows.append({"model": "Diagnostic", "predictor": "", **row,
                     "adj_r2": "", "p": row["p"]})

with open(OUT_DIR / "table4_h1_regression_mfq.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out_rows)
print(f"\nSaved: outputs/table4_h1_regression_mfq.csv")

# ── Save Table 4 TeX ─────────────────────────────────────────────

tex = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{H1 (MFQ): Orientation-Effect Regression (OLS, N=14)}",
    r"\label{tab:h1-regression-mfq}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"& Predictor & $\beta$ & SE & $t$ & $p$ & 95\% CI \\",
    r"\midrule",
]
for i, rows in enumerate([t1, t2]):
    if i > 0:
        tex.append(r"\addlinespace")
    for row in rows:
        p_str = ".000" if row["p"] == 0 else (f"{row['p']:.4f}".lstrip("0") if row["p"] >= 0.001 else "<.001")
        tex.append(
            f"{row['model']} & {row['predictor']} & {row['coef']:.3f} & {row['se']:.3f} & "
            f"{row['t']:.3f} & {p_str} & [{row['ci_low']:.3f}, {row['ci_high']:.3f}] \\\\"
        )
    tex.append(
        f"\\multicolumn{{7}}{{l}}{{R$^2$ = {rows[0]['r2']:.3f}, "
        f"adj. R$^2$ = {rows[0]['adj_r2']:.3f}, N = {int(rows[0]['n'])}}} \\\\"
    )
tex.extend([
    r"\midrule",
    r"\multicolumn{7}{l}{\textit{Per-foundation univariate diagnostics}} \\",
])
for row in diag_rows:
    p_str = ".000" if row["p"] == 0 else (f"{row['p']:.4f}".lstrip("0") if row["p"] >= 0.001 else "<.001")
    tex.append(
        f"& {row['foundation']} & {row['coef']:.3f} & {row['se']:.3f} & "
        f"{row['t']:.3f} & {p_str} & [{row['ci_low']:.3f}, {row['ci_high']:.3f}] \\\\"
    )
tex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

with open(OUT_DIR / "table4_h1_regression_mfq.tex", "w") as f:
    f.write("\n".join(tex) + "\n")
print("Saved: outputs/table4_h1_regression_mfq.tex")

# ── Scatterplot ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(MFQ_COMP, MEAN_DELTA, color="steelblue", s=70, zorder=5)

for i, m in enumerate(models):
    label = m.replace(" Instruct", "").replace(" 70B", "")
    ax.annotate(label, (MFQ_COMP[i], MEAN_DELTA[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

x_range = np.linspace(MFQ_COMP.min() - 1, MFQ_COMP.max() + 1, 100)
X_pred = sm.add_constant(x_range)
y_pred = r1.predict(X_pred)
ax.plot(x_range, y_pred, color="darkred", linewidth=1.5, linestyle="--",
        label=f"Model 1: $\\beta$={r1.params[1]:.2f}, $R^2$={r1.rsquared:.2f}")

ax.set_xlabel("MFQ Composite (z(individualizing) - z(binding))", fontsize=11)
ax.set_ylabel("Mean Burden Effect ($\\Delta$ willingness)", fontsize=11)
ax.set_title("H1 (MFQ): Orientation-Effect Regression", fontsize=13)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_h1_mfq_delta.png", dpi=150)
plt.close()
print("Saved: outputs/fig_h1_mfq_delta.png")
