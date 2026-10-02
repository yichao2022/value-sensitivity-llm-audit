#!/usr/bin/env python3
"""
H1: Orientation-effect regression.
mean_delta ~ PVC
mean_delta ~ PVC + mean_low
N=9 → reports coefficients, SE, t, p, 95% CI, R², adj R².
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
            data[model] = {k: float(v) if v else None
                          for k, v in row.items() if k not in (key_col, "Model")}
    return data

pvc_data = load_csv(OUT_DIR / "table2_orientation_profiles.csv")
burden_data = load_csv(OUT_DIR / "table3_burden_effects.csv")

models = sorted(set(pvc_data.keys()) & set(burden_data.keys()))
print(f"N = {len(models)}")

PVC = np.array([pvc_data[m]["PVC"] for m in models])
MEAN_DELTA = np.array([burden_data[m]["mean_delta"] for m in models])
MEAN_LOW = np.array([burden_data[m]["mean_low"] for m in models])

# ── OLS helper ───────────────────────────────────────────────────

def ols_table(model_name, y, X_vars, var_names):
    """Fit OLS and return list of row dicts + fitted result."""
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

# ── Model 1 ──────────────────────────────────────────────────────

print("\n=== Model 1: mean_delta ~ PVC ===")
t1, r1 = ols_table("Model 1", MEAN_DELTA, [PVC], ["PVC"])
for row in t1:
    print(f"  {row['predictor']:<14} β={row['coef']:>8.4f}  SE={row['se']:.4f}  t={row['t']:.4f}  p={row['p']:.4f}  CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
print(f"  R²={t1[0]['r2']:.4f}  adjR²={t1[0]['adj_r2']:.4f}  N={t1[0]['n']}")

# ── Model 2 ──────────────────────────────────────────────────────

print("\n=== Model 2: mean_delta ~ PVC + mean_low ===")
t2, r2 = ols_table("Model 2", MEAN_DELTA, [PVC, MEAN_LOW], ["PVC", "mean_low"])
for row in t2:
    print(f"  {row['predictor']:<14} β={row['coef']:>8.4f}  SE={row['se']:.4f}  t={row['t']:.4f}  p={row['p']:.4f}  CI=[{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
print(f"  R²={t2[0]['r2']:.4f}  adjR²={t2[0]['adj_r2']:.4f}  N={t2[0]['n']}")

# ── Save Table 4 CSV ─────────────────────────────────────────────

all_rows = t1 + t2
fields = ["model","predictor","coef","se","t","p","ci_low","ci_high","r2","adj_r2","n"]
with open(OUT_DIR / "table4_h1_regression.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(all_rows)
print(f"\nSaved: outputs/table4_h1_regression.csv")

# ── Save Table 4 TeX ─────────────────────────────────────────────

tex = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{H1: Orientation-Effect Regression (OLS)}",
    r"\label{tab:h1-regression}",
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
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

with open(OUT_DIR / "table4_h1_regression.tex", "w") as f:
    f.write("\n".join(tex) + "\n")
print("Saved: outputs/table4_h1_regression.tex")

# ── Scatterplot ──────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(PVC, MEAN_DELTA, color="steelblue", s=70, zorder=5)

for i, m in enumerate(models):
    label = m.replace(" Instruct", "").replace(" 70B", "")
    ax.annotate(label, (PVC[i], MEAN_DELTA[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

x_range = np.linspace(PVC.min() - 1, PVC.max() + 1, 100)
X_pred = sm.add_constant(x_range)
y_pred = r1.predict(X_pred)
ax.plot(x_range, y_pred, color="darkred", linewidth=1.5, linestyle="--",
        label=f"Model 1: $\\beta$={r1.params[1]:.2f}, $R^2$={r1.rsquared:.2f}")

ax.set_xlabel("PVC (Policy-Value Configuration)", fontsize=11)
ax.set_ylabel("Mean Burden Effect ($\\Delta$ willingness)", fontsize=11)
ax.set_title("H1: Orientation-Effect Regression", fontsize=13)
ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig_h1_pvc_delta.png", dpi=150)
plt.close()
print("Saved: outputs/fig_h1_pvc_delta.png")
