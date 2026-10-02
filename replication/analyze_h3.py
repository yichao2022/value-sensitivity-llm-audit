#!/usr/bin/env python3
"""H3 Analysis: frame_delta_analysis + delta ~ model + profile + frame with Holm adjustment."""

import csv, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs"
# Use canonical data
INPUT_CSV = Path("outputs/canonical/h3_raw.csv")

if not INPUT_CSV.exists():
    print(f"ERROR: {INPUT_CSV} not found. Run canonical simulation first.")
    exit(1)

# ═══════════════════════════════════════════════════════════════════
# 1. Build frame_delta_analysis.csv
# ═══════════════════════════════════════════════════════════════════

rows = []
with open(INPUT_CSV) as f:
    for row in csv.DictReader(f):
        if (row.get('json_valid','').lower() in ('true','1')
            and row.get('score_valid','').lower() in ('true','1')
            and row.get('non_refusal','').lower() in ('true','1')):
            rows.append(row)

print(f"Loaded {len(rows)} valid rows from {INPUT_CSV}")

# Canonical cell = (model, frame, profile, repetition, burden). h3_raw.csv accumulates
# appended runs (Mistral was run twice; 27 GPT-4.1 rows were re-run after HTTP 429s), so
# keep the LAST row per cell -- that is the row the run order designates as final.
_CELL = ('model', 'frame', 'profile_id', 'repetition', 'burden_level')
_dedup = {tuple(r[k] for k in _CELL): r for r in rows}
print(f"Deduped to {len(_dedup)} canonical cells (dropped {len(rows) - len(_dedup)} superseded rows)")
rows = list(_dedup.values())

# Pair low/high by (model, frame, profile_id, repetition)
by_key = defaultdict(dict)
for r in rows:
    key = (r['model'], r['frame'], r['profile_id'], r['repetition'])
    by_key[key][r['burden_level']] = float(r['willingness'])

delta_rows = []
for (model, frame, pid, rep), burdens in sorted(by_key.items()):
    if 'low' in burdens and 'high' in burdens:
        delta_rows.append({
            'model': model, 'frame': frame, 'profile_id': pid,
            'repetition': rep,
            'w_low': burdens['low'], 'w_high': burdens['high'],
            'delta': round(burdens['low'] - burdens['high'], 2),
        })

with open(OUT / "frame_delta_analysis.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=['model','frame','profile_id','repetition',
                                       'w_low','w_high','delta'])
    w.writeheader()
    w.writerows(delta_rows)
print(f"frame_delta_analysis.csv: {len(delta_rows)} rows")

# ═══════════════════════════════════════════════════════════════════
# 2. Run H3 regression: delta ~ C(model) + C(profile_id) + C(frame)
# ═══════════════════════════════════════════════════════════════════

import pandas as pd
df = pd.DataFrame(delta_rows)
df['delta'] = df['delta'].astype(float)

# Set reference categories
df['frame'] = pd.Categorical(df['frame'], categories=['neutral','autonomy','collective','equity'], ordered=False)
df['model'] = pd.Categorical(df['model'])
df['profile_id'] = pd.Categorical(df['profile_id'])

# Fit
formula = "delta ~ C(model) + C(profile_id) + C(frame)"
mod = smf.ols(formula, data=df).fit()

print(f"\n=== H3: delta ~ model + profile + frame ===\n")
print(f"R² = {mod.rsquared:.4f}, adj R² = {mod.rsquared_adj:.4f}, N = {len(df)}")

# Extract frame coefficients (vs neutral)
frame_vars = [v for v in mod.params.index if 'C(frame)' in v and 'T.' in v]
frame_coefs = {}
print("\nFrame effects (ref = neutral):")
for v in frame_vars:
    frame_name = v.split('T.')[1].rstrip(']')
    coef = mod.params[v]
    se = mod.bse[v]
    t = mod.tvalues[v]
    p = mod.pvalues[v]
    ci = mod.conf_int().loc[v]
    print(f"  {frame_name:<14} β={coef:+.4f}  SE={se:.4f}  t={t:.3f}  p={p:.4f}  95%CI=[{ci[0]:.4f},{ci[1]:.4f}]")
    frame_coefs[frame_name] = {'coef':coef,'se':se,'t':t,'p':p,'ci_low':ci[0],'ci_high':ci[1]}

# ═══════════════════════════════════════════════════════════════════
# 3. Holm-adjusted p-values for frame effects
# ═══════════════════════════════════════════════════════════════════

print("\n--- Holm-Bonferroni adjustment (3 frame contrasts) ---")
frame_pvals = [(v.split('T.')[1].rstrip(']'), mod.pvalues[v]) for v in frame_vars]
frame_pvals.sort(key=lambda x: x[1])  # sort by p ascending
n_tests = len(frame_pvals)
holm_adj = {}
for rank, (name, p) in enumerate(frame_pvals, 1):
    adj_p = min(p * (n_tests - rank + 1), 1.0)
    holm_adj[name] = adj_p
    sig = "***" if adj_p < 0.001 else ("**" if adj_p < 0.01 else ("*" if adj_p < 0.05 else "ns"))
    print(f"  {name:<14} raw p={p:.4f}  Holm p={adj_p:.4f}  {sig}")

for name in frame_coefs:
    frame_coefs[name]['holm_p'] = holm_adj.get(name, 1.0)

# ═══════════════════════════════════════════════════════════════════
# 4. Save Table 6
# ═══════════════════════════════════════════════════════════════════

with open(OUT / "table6_h3_frame_manipulation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=['frame','coef','se','t','p','holm_p','ci_low','ci_high'])
    w.writeheader()
    for name in ['autonomy','collective','equity']:
        fc = frame_coefs[name]
        w.writerow({'frame':name,'coef':round(fc['coef'],4),'se':round(fc['se'],4),
                    't':round(fc['t'],4),'p':round(fc['p'],4),
                    'holm_p':round(fc['holm_p'],4),
                    'ci_low':round(fc['ci_low'],4),'ci_high':round(fc['ci_high'],4)})
print(f"\nSaved: outputs/table6_h3_frame_manipulation.csv")

# TeX
tex = [
    r"\begin{table}[ht]", r"\centering",
    r"\caption{H3: Within-Model Frame Manipulation Effects (ref = neutral)}",
    r"\label{tab:h3-frame-manipulation}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Frame & $\beta$ & SE & $t$ & $p$ & Holm $p$ & 95\% CI \\",
    r"\midrule",
]
for name in ['autonomy','collective','equity']:
    fc = frame_coefs[name]
    tex.append(
        f"{name} & {fc['coef']:.3f} & {fc['se']:.3f} & {fc['t']:.3f} & "
        f"{fc['p']:.4f} & {fc['holm_p']:.4f} & [{fc['ci_low']:.3f}, {fc['ci_high']:.3f}] \\\\"
    )
tex.append(r"\midrule")
tex.append(f"\\multicolumn{{7}}{{l}}{{R$^2$ = {mod.rsquared:.3f}, adj. R$^2$ = {mod.rsquared_adj:.3f}, N = {len(df)}}} \\\\")
tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

with open(OUT / "table6_h3_frame_manipulation.tex", "w") as f:
    f.write("\n".join(tex) + "\n")
print("Saved: outputs/table6_h3_frame_manipulation.tex")

# ═══════════════════════════════════════════════════════════════════
# 5. Mean delta by model × frame
# ═══════════════════════════════════════════════════════════════════

print("\n--- Mean delta by model × frame ---")
summary = df.groupby(['model','frame'])['delta'].agg(['mean','std','count']).round(2)
print(summary.to_string())

summary.to_csv(OUT / "h3_mean_delta_by_model_frame.csv")
print(f"\nSaved: outputs/h3_mean_delta_by_model_frame.csv")

# ═══════════════════════════════════════════════════════════════════
# 6. Figure: frame effects
# ═══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Coefficient plot
ax = axes[0]
names = ['autonomy','collective','equity']
coefs = [frame_coefs[n]['coef'] for n in names]
cis_low = [frame_coefs[n]['ci_low'] for n in names]
cis_high = [frame_coefs[n]['ci_high'] for n in names]
errors_low = [c - l for c, l in zip(coefs, cis_low)]
errors_high = [h - c for c, h in zip(coefs, cis_high)]
colors = ['#2196F3','#4CAF50','#FF9800']
y_pos = range(len(names))

ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')
ax.barh(y_pos, coefs, color=colors, edgecolor='white', height=0.5)
ax.errorbar(coefs, y_pos, xerr=[errors_low, errors_high], fmt='none',
            ecolor='black', capsize=3, linewidth=1.2)
# Significance stars
for i, name in enumerate(names):
    hp = frame_coefs[name]['holm_p']
    star = '***' if hp < 0.001 else ('**' if hp < 0.01 else ('*' if hp < 0.05 else ''))
    x_pos = coefs[i] + (0.5 if coefs[i] >= 0 else -0.5)
    ax.text(x_pos, i, star, va='center', fontsize=12, fontweight='bold')
ax.set_yticks(list(y_pos))
ax.set_yticklabels([n.capitalize() for n in names])
ax.set_xlabel("Δ Delta (vs Neutral frame)", fontsize=11)
ax.set_title("A. Frame Manipulation Effects", fontsize=12, fontweight='bold')
ax.invert_yaxis()

# Panel B: By-model mean delta
ax2 = axes[1]
summary_means = df.groupby(['model','frame'])['delta'].mean().unstack()
frame_order = ['neutral','autonomy','collective','equity']
summary_means = summary_means[frame_order]
x = np.arange(len(summary_means.index))
width = 0.2
for i, frame in enumerate(frame_order):
    ax2.bar(x + i*width, summary_means[frame], width, label=frame.capitalize(),
            color=colors[i] if i < 3 else '#9E9E9E')
ax2.set_xticks(x + width*1.5)
ax2.set_xticklabels([m.replace(' Instruct','').replace(' 70B','') for m in summary_means.index], fontsize=9)
ax2.set_ylabel("Mean Delta", fontsize=11)
ax2.set_title("B. Mean Delta by Model × Frame", fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, ncol=4, loc='lower left')

plt.tight_layout()
fig.savefig(OUT / "fig_h3_frame_effects.png", dpi=150)
plt.close()
print("Saved: outputs/fig_h3_frame_effects.png")

# ═══════════════════════════════════════════════════════════════════
# 7. Four-model subset analysis (GPT-4.1, Llama 3.1 70B, Qwen3.6-72B, Mistral Large)
# ═══════════════════════════════════════════════════════════════════

FOUR_MODELS = ['GPT-4.1', 'Llama 3.1 70B Instruct', 'Qwen3.7 Plus', 'Mistral Large']
df4 = df[df['model'].isin(FOUR_MODELS)].copy()

print(f"\n=== Four-model subset (N={len(df4)}) ===")

# Run model-specific regressions
model_specific = []
for model in FOUR_MODELS:
    mdf = df4[df4['model'] == model]
    if len(mdf) < 10:
        continue
    try:
        mod_m = smf.ols("delta ~ C(profile_id) + C(frame)", data=mdf).fit()
        row = {'Model': model, 'R2': mod_m.rsquared, 'Adj_R2': mod_m.rsquared_adj, 'N': len(mdf)}
        for frame in ['autonomy', 'collective', 'equity']:
            var = f"C(frame)[T.{frame}]"
            if var in mod_m.params.index:
                row[f'{frame}_coef'] = mod_m.params[var]
                row[f'{frame}_se'] = mod_m.bse[var]
                row[f'{frame}_p'] = mod_m.pvalues[var]
                ci = mod_m.conf_int().loc[var]
                row[f'{frame}_ci_low'] = ci[0]
                row[f'{frame}_ci_high'] = ci[1]
            else:
                row[f'{frame}_coef'] = row[f'{frame}_se'] = row[f'{frame}_p'] = row[f'{frame}_ci_low'] = row[f'{frame}_ci_high'] = float('nan')
        model_specific.append(row)
    except Exception as e:
        print(f"  Warning: {model} failed: {e}")

# Pooled 4-model
mod_pooled = smf.ols("delta ~ C(model) + C(profile_id) + C(frame)", data=df4).fit()
pooled_row = {'Model': 'Pooled', 'R2': mod_pooled.rsquared, 'Adj_R2': mod_pooled.rsquared_adj, 'N': len(df4)}
for frame in ['autonomy', 'collective', 'equity']:
    var = f"C(frame)[T.{frame}]"
    if var in mod_pooled.params.index:
        pooled_row[f'{frame}_coef'] = mod_pooled.params[var]
        pooled_row[f'{frame}_se'] = mod_pooled.bse[var]
        pooled_row[f'{frame}_p'] = mod_pooled.pvalues[var]
        ci = mod_pooled.conf_int().loc[var]
        pooled_row[f'{frame}_ci_low'] = ci[0]
        pooled_row[f'{frame}_ci_high'] = ci[1]
    else:
        pooled_row[f'{frame}_coef'] = pooled_row[f'{frame}_se'] = pooled_row[f'{frame}_p'] = pooled_row[f'{frame}_ci_low'] = pooled_row[f'{frame}_ci_high'] = float('nan')
model_specific.append(pooled_row)

# Save model-specific full CSV
import csv
with open(OUT / "h3_model_specific_full_4models.csv", "w", newline="") as f:
    fieldnames = ['Model'] + [f"{f}_{s}" for f in ['autonomy','collective','equity'] for s in ['coef','se','p','ci_low','ci_high']] + ['R2','Adj_R2','N']
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(model_specific)
print("Saved: outputs/h3_model_specific_full_4models.csv")

# Print summary
for row in model_specific:
    print(f"\n{row['Model']}: R²={row['R2']:.3f}, N={row['N']}")
    for frame in ['autonomy','collective','equity']:
        c = row.get(f'{frame}_coef', float('nan'))
        ci_l = row.get(f'{frame}_ci_low', float('nan'))
        ci_h = row.get(f'{frame}_ci_high', float('nan'))
        print(f"  {frame}: β={c:+.2f} [{ci_l:.2f}, {ci_h:.2f}]")

print("\n=== Done ===")

# ═══════════════════════════════════════════════════════════════════
# 8. Cluster-robust inference (SEs clustered at model x profile)
#    Main-text H3 reports clustered SEs because the 27 profiles are the
#    controlled test cases and queries within a profile are repeated measures.
# ═══════════════════════════════════════════════════════════════════

df['cluster'] = df['model'].astype(str) + '|' + df['profile_id'].astype(str)
df4['cluster'] = df4['model'].astype(str) + '|' + df4['profile_id'].astype(str)
print(f"\n=== Cluster-robust H3 (clusters = model x profile, n={df4['cluster'].nunique()}) ===")


def _clustered(fit, data, label):
    r = fit.get_robustcov_results(cov_type='cluster', groups=data['cluster'])
    out = {}
    for frame in ['autonomy', 'collective', 'equity']:
        var = f"C(frame)[T.{frame}]"
        i = list(fit.params.index).index(var)
        ci = r.conf_int()[i]
        out[frame] = {'coef': fit.params[var], 'se': r.bse[i], 'p': r.pvalues[i],
                      'ci_low': ci[0], 'ci_high': ci[1]}
    # Holm over the 3 frame contrasts
    order = sorted(out, key=lambda k: out[k]['p'])
    for rank, name in enumerate(order, 1):
        out[name]['holm_p'] = min(out[name]['p'] * (3 - rank + 1), 1.0)
    print(f"\n{label}: N={int(fit.nobs)}, clusters={data['cluster'].nunique()}, "
          f"R2={fit.rsquared:.4f}, adj R2={fit.rsquared_adj:.4f}")
    for frame in ['autonomy', 'collective', 'equity']:
        c = out[frame]
        print(f"  {frame:<11} beta={c['coef']:+.4f}  SE={c['se']:.4f}  "
              f"p={c['p']:.4g}  Holm p={c['holm_p']:.4g}  95%CI=[{c['ci_low']:.3f},{c['ci_high']:.3f}]")
    return out


pooled_cl = _clustered(mod_pooled, df4.assign(cluster=df4['model'].astype(str) + '|' + df4['profile_id'].astype(str)), 'Pooled four-model')

rows_cl = [{'Model': 'Pooled', 'N': int(mod_pooled.nobs),
            'clusters': df4['cluster'].nunique(),
            'R2': mod_pooled.rsquared, 'Adj_R2': mod_pooled.rsquared_adj,
            **{f"{k}_{s}": pooled_cl[k][s] for k in pooled_cl for s in ('coef', 'se', 'p', 'holm_p', 'ci_low', 'ci_high')}}]
for model in FOUR_MODELS:
    mdf = df4[df4['model'] == model].copy()
    if len(mdf) < 10:
        continue
    fit = smf.ols("delta ~ C(profile_id) + C(frame)", data=mdf).fit()
    cl = _clustered(fit, mdf.assign(cluster=model + '|' + mdf['profile_id'].astype(str)), model)
    rows_cl.append({'Model': model, 'N': int(fit.nobs), 'clusters': mdf['profile_id'].nunique(),
                    'R2': fit.rsquared, 'Adj_R2': fit.rsquared_adj,
                    **{f"{k}_{s}": cl[k][s] for k in cl for s in ('coef', 'se', 'p', 'holm_p', 'ci_low', 'ci_high')}})

with open(OUT / "h3_clustered_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_cl[0].keys()))
    w.writeheader()
    w.writerows(rows_cl)
print("\nSaved: outputs/h3_clustered_results.csv")

