#!/usr/bin/env python3
"""
H3 Analysis — Combined (3 original models + Mistral Large).
Pooled regression: delta ~ C(model) + C(profile_id) + C(frame)
Per-model frame effects for comparison.
"""
import csv, numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from collections import defaultdict
from pathlib import Path
from scipy.stats import ttest_ind

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs"

# ═══════════════════════════════════════════════════════════════════
# 1. Load and combine both datasets
# ═══════════════════════════════════════════════════════════════════

# Original 3 models (from frame_manipulation_outputs.csv)
orig_path = OUT / "frame_manipulation_outputs.csv"
mistral_path = OUT / "expanded15" / "h3_mistral_large_raw.csv"

rows_orig = []
with open(orig_path) as f:
    for row in csv.DictReader(f):
        if (row.get('json_valid','').lower() in ('true','1')
            and row.get('score_valid','').lower() in ('true','1')
            and row.get('non_refusal','').lower() in ('true','1')):
            rows_orig.append(row)

# Mistral Large
rows_mistral = []
with open(mistral_path) as f:
    for row in csv.DictReader(f):
        if (row.get('json_valid','').lower() in ('true','1')
            and row.get('score_valid','').lower() in ('true','1')
            and row.get('non_refusal','').lower() in ('true','1')):
            rows_mistral.append(row)

print(f"Original 3 models: {len(rows_orig)} valid rows")
print(f"Mistral Large: {len(rows_mistral)} valid rows")

# Combine
all_rows = rows_orig + rows_mistral
print(f"Combined: {len(all_rows)} rows, {len(set(r['model'] for r in all_rows))} models")

# ═══════════════════════════════════════════════════════════════════
# 2. Build delta pairs
# ═══════════════════════════════════════════════════════════════════

by_key = defaultdict(dict)
for r in all_rows:
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

print(f"Delta pairs: {len(delta_rows)}")
print(f"  Models: {sorted(set(r['model'] for r in delta_rows))}")
print(f"  Profiles: {len(set(r['profile_id'] for r in delta_rows))}")
print(f"  Per model: {[(m, sum(1 for r in delta_rows if r['model']==m)) for m in sorted(set(r['model'] for r in delta_rows))]}")

df = pd.DataFrame(delta_rows)
df['delta'] = df['delta'].astype(float)

# ═══════════════════════════════════════════════════════════════════
# 3. Per-model frame effects (for comparison)
# ═══════════════════════════════════════════════════════════════════

print("\n═══ Per-Model Frame Effects ═══")
print(f"{'Model':<28} {'Frame':<12} {'Mean Δ':>8} {'SD':>6} {'n':>4}")
print("-"*62)

per_model_effects = {}
for model in sorted(df['model'].unique()):
    sub = df[df['model'] == model]
    mfx = sub.groupby('frame')['delta'].agg(['mean', 'std', 'count'])
    per_model_effects[model] = mfx
    for frame in ['neutral', 'autonomy', 'collective', 'equity']:
        r = mfx.loc[frame]
        print(f"  {model:<26} {frame:<12} {r['mean']:8.2f} {r['std']:6.2f} {r['count']:4}")

# ═══════════════════════════════════════════════════════════════════
# 4. Pooled regression: delta ~ C(model) + C(profile_id) + C(frame)
# ═══════════════════════════════════════════════════════════════════

df['frame'] = pd.Categorical(df['frame'], categories=['neutral','autonomy','collective','equity'], ordered=False)
df['model'] = pd.Categorical(df['model'])
df['profile_id'] = pd.Categorical(df['profile_id'])

formula = "delta ~ C(model) + C(profile_id) + C(frame)"
mod = smf.ols(formula, data=df).fit()

print(f"\n═══ Pooled H3 (4 models): delta ~ model + profile + frame ═══")
print(f"R² = {mod.rsquared:.4f}, adj R² = {mod.rsquared_adj:.4f}, N = {len(df)}")

frame_vars = [v for v in mod.params.index if 'C(frame)' in v and 'T.' in v]
frame_coefs = {}
print(f"\nFrame effects (ref = neutral):")
for v in frame_vars:
    frame_name = v.split('T.')[1].rstrip(']')
    coef = mod.params[v]
    se = mod.bse[v]
    t_val = mod.tvalues[v]
    p = mod.pvalues[v]
    ci = mod.conf_int().loc[v]
    print(f"  {frame_name:<14} β={coef:+7.3f}  SE={se:.4f}  t={t_val:6.3f}  p={p:.4f}  95%CI=[{ci[0]:.3f},{ci[1]:.3f}]")
    frame_coefs[frame_name] = {'coef':coef,'se':se,'t':t_val,'p':p,'ci_low':ci[0],'ci_high':ci[1]}

# Holm adjustment
frame_pvals = [(v.split('T.')[1].rstrip(']'), mod.pvalues[v]) for v in frame_vars]
frame_pvals.sort(key=lambda x: x[1])
n_tests = len(frame_pvals)
for rank, (name, p) in enumerate(frame_pvals, 1):
    adj_p = min(p * (n_tests - rank + 1), 1.0)
    frame_coefs[name]['holm_p'] = adj_p
    sig = "***" if adj_p < 0.001 else ("**" if adj_p < 0.01 else ("*" if adj_p < 0.05 else "ns"))
    print(f"  {name:<14} Holm p={adj_p:.4f}  {sig}")

# ═══════════════════════════════════════════════════════════════════
# 5. Save combined Table 6
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
print(f"\nSaved: table6_h3_frame_manipulation.csv")

# TeX
model_names = sorted(df['model'].unique())
model_count = len(model_names)
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
        f"  {name} & {fc['coef']:.3f} & {fc['se']:.3f} & {fc['t']:.3f} & "
        f"{fc['p']:.4f} & {fc['holm_p']:.4f} & [{fc['ci_low']:.3f}, {fc['ci_high']:.3f}] \\\\"
    )
tex.append(r"\midrule")
tex.append(f"  \\multicolumn{{7}}{{l}}{{N = {len(df)} paired deltas, "
           f"{model_count} models ({', '.join(model_names)}), "
           f"27 profiles, adj. $R^2$ = {mod.rsquared_adj:.3f}}} \\\\")
tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

with open(OUT / "table6_h3_frame_manipulation.tex", "w") as f:
    f.write("\n".join(tex) + "\n")
print("Saved: table6_h3_frame_manipulation.tex")

# ═══════════════════════════════════════════════════════════════════
# 6. Per-model sensitivity: which models show significant framing?
# ═══════════════════════════════════════════════════════════════════

print("\n═══ Per-model framing sensitivity tests ═══")
print(f"{'Model':<28} {'ANOVA F':>8} {'ANOVA p':>8} {'autonomy p':>10} {'collective p':>12} {'equity p':>10}")
print("-"*70)

for model in sorted(df['model'].unique()):
    sub = df[df['model'] == model]
    from scipy.stats import f_oneway
    groups = [sub[sub['frame']==f]['delta'].values for f in ['neutral','autonomy','collective','equity']]
    f_stat, p_val = f_oneway(*groups)
    neutral = sub[sub['frame']=='neutral']['delta']
    p_auto = ttest_ind(neutral, sub[sub['frame']=='autonomy']['delta'], equal_var=False).pvalue
    p_coll = ttest_ind(neutral, sub[sub['frame']=='collective']['delta'], equal_var=False).pvalue
    p_equi = ttest_ind(neutral, sub[sub['frame']=='equity']['delta'], equal_var=False).pvalue
    print(f"  {model:<26} {f_stat:8.2f} {p_val:8.4f} {p_auto:10.4f} {p_coll:12.4f} {p_equi:10.4f}")

# ═══════════════════════════════════════════════════════════════════
# 7. Mean delta by model × frame (extended comparison)
# ═══════════════════════════════════════════════════════════════════

summary = df.groupby(['model','frame'])['delta'].agg(['mean','std','count']).round(2)
summary.to_csv(OUT / "h3_mean_delta_by_model_frame.csv")
print(f"\nSaved: h3_mean_delta_by_model_frame.csv")
print(summary.to_string())

print("\n=== Done ===")
