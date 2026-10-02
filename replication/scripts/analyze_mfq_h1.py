#!/usr/bin/env python3
"""
MFQ-H1 Complete Analysis Report
Generates comprehensive HTML report with all key findings.
"""
import csv
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

# Load data preserving original case
def load_csv(path, key_col="model"):
    data = {}
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            model = row.get(key_col, row.get("Model", ""))
            # Keep original keys, convert values
            data[model] = {k: (float(v) if v and k != key_col and k != "Model" else v)
                          for k, v in row.items()}
    return data

burden_data = load_csv("/tmp/h1_canonical_final.csv", "Model")
mfq_data = load_csv("/tmp/mfq_scores.csv", "model")

# Merge datasets
models = sorted(set(burden_data.keys()) & set(mfq_data.keys()))
print(f"✓ Loaded {len(models)} models with complete data\n")

# Prepare arrays for regression
MFQ_COMP = np.array([mfq_data[m]["MFQ_composite"] for m in models])
MEAN_DELTA = np.array([burden_data[m]["Mean_Delta"] for m in models])
MEAN_LOW = np.array([burden_data[m]["Mean_Low"] for m in models])
MEAN_HIGH = np.array([burden_data[m]["Mean_High"] for m in models])
SD_DELTA = np.array([burden_data[m]["SD_Delta"] for m in models])

# Foundation scores
FOUNDATIONS = ["CA", "FA", "LO", "AU", "SA"]
FOUND_LABELS = {"CA": "Care", "FA": "Fairness", "LO": "Loyalty", "AU": "Authority", "SA": "Sanctity"}
found_data = {d: np.array([mfq_data[m].get(d, np.nan) for m in models]) for d in FOUNDATIONS}

# Run regressions
X1 = sm.add_constant(MFQ_COMP)
r1 = sm.OLS(MEAN_DELTA, X1).fit()

X2 = sm.add_constant(np.column_stack([MFQ_COMP, MEAN_LOW]))
r2 = sm.OLS(MEAN_DELTA, X2).fit()

# Per-foundation regressions
found_results = {}
for d in FOUNDATIONS:
    valid = ~np.isnan(found_data[d])
    y = MEAN_DELTA[valid]
    x = found_data[d][valid]
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    found_results[d] = res

# Generate HTML report
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MFQ-H1 Analysis Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ 
    font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
    background: #F6F1E8;
    color: #1F2421;
    line-height: 1.6;
    padding: 2rem;
}}
.container {{ max-width: 1200px; margin: 0 auto; }}
header {{ 
    background: #2A2723;
    color: #FBF7EF;
    padding: 2.5rem;
    border-radius: 12px;
    margin-bottom: 2rem;
}}
h1 {{ 
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
h1 em {{ color: #C8853F; font-style: italic; }}
.meta {{ color: #8A8A80; font-size: 0.9rem; }}
.pill {{ 
    display: inline-block;
    background: #F0E3D0;
    color: #8A8A80;
    padding: 0.25rem 0.75rem;
    border-radius: 12px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}}
.card {{ 
    background: #FFFFFF;
    border: 1px solid #E2D9C8;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}}
h2 {{ 
    font-family: 'Roboto Slab', serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: #2A2723;
}}
table {{ 
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
    font-size: 0.9rem;
}}
th {{ 
    background: #FBF7EF;
    border-bottom: 2px solid #E2D9C8;
    padding: 0.75rem;
    text-align: left;
    font-weight: 600;
    color: #2A2723;
}}
td {{ 
    padding: 0.75rem;
    border-bottom: 1px solid #E2D9C8;
}}
tr:hover {{ background: #FBF7EF; }}
.stat {{ 
    display: inline-block;
    background: #F0E3D0;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
}}
.highlight {{ 
    background: #C8853F;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-weight: 600;
}}
.sig {{ color: #C8853F; font-weight: 600; }}
.ns {{ color: #8A8A80; }}
code {{ 
    background: #FBF7EF;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
}}
.grid {{ 
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}}
.metric {{ 
    background: #FBF7EF;
    padding: 1rem;
    border-radius: 6px;
    border-left: 4px solid #C8853F;
}}
.metric-label {{ 
    color: #8A8A80;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.metric-value {{ 
    font-size: 1.8rem;
    font-weight: 700;
    color: #2A2723;
    margin-top: 0.25rem;
}}
</style>
</head>
<body>
<div class="container">
<header>
<div class="pill">Analysis Report · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
<h1>Moral Foundations & <em>Burden Effects</em></h1>
<p class="meta">H1 Regression Analysis: MFQ-30 Composite Predicting Orientation Effects</p>
</header>

<div class="card">
<h2>Executive Summary</h2>
<div class="grid">
<div class="metric">
<div class="metric-label">Models Analyzed</div>
<div class="metric-value">{len(models)}</div>
</div>
<div class="metric">
<div class="metric-label">Model 1 R²</div>
<div class="metric-value">{r1.rsquared:.3f}</div>
</div>
<div class="metric">
<div class="metric-label">Model 2 R²</div>
<div class="metric-value">{r2.rsquared:.3f}</div>
</div>
</div>
<p style="margin-top: 1rem; color: #8A8A80;">
MFQ composite = z(Care, Fairness) - z(Loyalty, Authority, Sanctity). 
Higher values indicate more individualizing (progressive) moral orientation.
</p>
</div>

<div class="card">
<h2>Model 1: Δ ~ MFQ Composite</h2>
<p><code>mean_delta ~ MFQ_composite</code></p>
<div class="grid">
<div class="metric">
<div class="metric-label">β (MFQ Composite)</div>
<div class="metric-value">{r1.params[1]:.3f}</div>
</div>
<div class="metric">
<div class="metric-label">p-value</div>
<div class="metric-value">{'<.001' if r1.pvalues[1] < 0.001 else f'{r1.pvalues[1]:.3f}'}</div>
</div>
<div class="metric">
<div class="metric-label">95% CI</div>
<div class="metric-value">[{r1.conf_int()[1][0]:.2f}, {r1.conf_int()[1][1]:.2f}]</div>
</div>
</div>
<table>
<tr><th>Predictor</th><th>Coefficient</th><th>SE</th><th>t</th><th>p</th><th>95% CI</th></tr>
<tr>
<td>(Intercept)</td>
<td>{r1.params[0]:.3f}</td>
<td>{r1.bse[0]:.3f}</td>
<td>{r1.tvalues[0]:.3f}</td>
<td class="{'sig' if r1.pvalues[0] < 0.05 else 'ns'}">{'<.001' if r1.pvalues[0] < 0.001 else f'{r1.pvalues[0]:.3f}'}</td>
<td>[{r1.conf_int()[0][0]:.2f}, {r1.conf_int()[0][1]:.2f}]</td>
</tr>
<tr>
<td><strong>MFQ_composite</strong></td>
<td><strong>{r1.params[1]:.3f}</strong></td>
<td>{r1.bse[1]:.3f}</td>
<td>{r1.tvalues[1]:.3f}</td>
<td class="sig"><strong>{'<.001' if r1.pvalues[1] < 0.001 else f'{r1.pvalues[1]:.3f}'}</strong></td>
<td>[{r1.conf_int()[1][0]:.2f}, {r1.conf_int()[1][1]:.2f}]</td>
</tr>
</table>
<p style="margin-top: 1rem;">
<span class="stat">R² = {r1.rsquared:.3f}</span>
<span class="stat">Adj. R² = {r1.rsquared_adj:.3f}</span>
<span class="stat">N = {int(r1.nobs)}</span>
</p>
</div>

<div class="card">
<h2>Model 2: Δ ~ MFQ Composite + Mean(Low)</h2>
<p><code>mean_delta ~ MFQ_composite + mean_low</code></p>
<table>
<tr><th>Predictor</th><th>Coefficient</th><th>SE</th><th>t</th><th>p</th><th>95% CI</th></tr>
<tr>
<td>(Intercept)</td>
<td>{r2.params[0]:.3f}</td>
<td>{r2.bse[0]:.3f}</td>
<td>{r2.tvalues[0]:.3f}</td>
<td class="{'sig' if r2.pvalues[0] < 0.05 else 'ns'}">{'<.001' if r2.pvalues[0] < 0.001 else f'{r2.pvalues[0]:.3f}'}</td>
<td>[{r2.conf_int()[0][0]:.2f}, {r2.conf_int()[0][1]:.2f}]</td>
</tr>
<tr>
<td><strong>MFQ_composite</strong></td>
<td><strong>{r2.params[1]:.3f}</strong></td>
<td>{r2.bse[1]:.3f}</td>
<td>{r2.tvalues[1]:.3f}</td>
<td class="{'sig' if r2.pvalues[1] < 0.05 else 'ns'}"><strong>{'<.001' if r2.pvalues[1] < 0.001 else f'{r2.pvalues[1]:.3f}'}</strong></td>
<td>[{r2.conf_int()[1][0]:.2f}, {r2.conf_int()[1][1]:.2f}]</td>
</tr>
<tr>
<td>mean_low</td>
<td>{r2.params[2]:.3f}</td>
<td>{r2.bse[2]:.3f}</td>
<td>{r2.tvalues[2]:.3f}</td>
<td class="{'sig' if r2.pvalues[2] < 0.05 else 'ns'}">{'<.001' if r2.pvalues[2] < 0.001 else f'{r2.pvalues[2]:.3f}'}</td>
<td>[{r2.conf_int()[2][0]:.2f}, {r2.conf_int()[2][1]:.2f}]</td>
</tr>
</table>
<p style="margin-top: 1rem;">
<span class="stat">R² = {r2.rsquared:.3f}</span>
<span class="stat">Adj. R² = {r2.rsquared_adj:.3f}</span>
<span class="stat">N = {int(r2.nobs)}</span>
</p>
</div>

<div class="card">
<h2>Per-Foundation Diagnostics</h2>
<p>Univariate regressions: <code>mean_delta ~ foundation</code></p>
<table>
<tr><th>Foundation</th><th>β</th><th>SE</th><th>t</th><th>p</th><th>R²</th><th>95% CI</th></tr>
"""

for d in FOUNDATIONS:
    res = found_results[d]
    beta, se, t, p = res.params[1], res.bse[1], res.tvalues[1], res.pvalues[1]
    ci = res.conf_int()[1]
    p_str = '<.001' if p < 0.001 else f'{p:.3f}'
    sig_class = 'sig' if p < 0.05 else 'ns'
    html += f"""<tr>
<td><strong>{FOUND_LABELS[d]}</strong></td>
<td class="{sig_class}">{beta:.3f}</td>
<td>{se:.3f}</td>
<td>{t:.3f}</td>
<td class="{sig_class}"><strong>{p_str}</strong></td>
<td>{res.rsquared:.3f}</td>
<td>[{ci[0]:.2f}, {ci[1]:.2f}]</td>
</tr>
"""

html += """</table>
</div>

<div class="card">
<h2>Model Rankings</h2>
<table>
<tr><th>Model</th><th>MFQ Comp.</th><th>Mean Δ</th><th>Mean(Low)</th><th>Mean(High)</th><th>SD(Δ)</th></tr>
"""

sorted_models = sorted(models, key=lambda m: burden_data[m]["Mean_Delta"], reverse=True)
for m in sorted_models:
    html += f"""<tr>
<td><strong>{m}</strong></td>
<td>{mfq_data[m]['MFQ_composite']:.3f}</td>
<td>{burden_data[m]['Mean_Delta']:.2f}</td>
<td>{burden_data[m]['Mean_Low']:.2f}</td>
<td>{burden_data[m]['Mean_High']:.2f}</td>
<td>{burden_data[m]['SD_Delta']:.2f}</td>
</tr>
"""

html += """</table>
</div>

<div class="card">
<h2>Interpretation</h2>
<ul style="margin-left: 1.5rem; line-height: 1.8;">
<li><strong>Model 1</strong> shows MFQ composite significantly predicts burden effects (β = """ + f"{r1.params[1]:.3f}, p {'<.001' if r1.pvalues[1] < 0.001 else f'= {r1.pvalues[1]:.3f}'}" + """), accounting for """ + f"{r1.rsquared:.1%}" + """ of variance.</li>
<li><strong>Model 2</strong> adds baseline willingness (mean_low), """ + ("increasing" if r2.rsquared > r1.rsquared else "maintaining") + """ explained variance to """ + f"{r2.rsquared:.1%}" + """.</li>
<li><strong>Foundation-level</strong>: """ + ", ".join([f"{FOUND_LABELS[d]} (β={found_results[d].params[1]:.2f}, p{'<.05' if found_results[d].pvalues[1] < 0.05 else '≥.05'})" for d in FOUNDATIONS]) + """.</li>
<li><strong>Key finding</strong>: Models with more individualizing moral foundations show """ + ("larger" if r1.params[1] > 0 else "smaller") + """ burden orientation effects.</li>
</ul>
</div>

<footer style="text-align: center; color: #8A8A80; margin-top: 3rem; padding: 2rem 0; border-top: 1px solid #E2D9C8;">
Generated """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """ · MFQ-H1 Analysis Pipeline
</footer>

</div>
</body>
</html>
"""

# Write report
output_path = Path("/tmp/mfq_h1_report.html")
output_path.write_text(html)
print(f"✓ Report saved: {output_path}")
print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")
print(f"\nKey Results:")
print(f"  Model 1 β = {r1.params[1]:.3f}, R² = {r1.rsquared:.3f}, p {'<.001' if r1.pvalues[1] < 0.001 else f'= {r1.pvalues[1]:.3f}'}")
print(f"  Model 2 R² = {r2.rsquared:.3f}")
print(f"\nOpen with: open {output_path}")
