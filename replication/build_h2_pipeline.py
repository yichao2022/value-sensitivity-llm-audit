#!/usr/bin/env python3
"""
H2 Pipeline: Narrative coding → merge → mediation-style regressions.
Outputs: narrative_coding.csv, h2_analysis.csv, table5_h2_mediation_style.csv/.tex
"""

import csv, re, openpyxl
from collections import defaultdict
from pathlib import Path
import numpy as np
import statsmodels.api as sm

PROJECT = Path(__file__).resolve().parent
OUT = PROJECT / "outputs"

# ═══════════════════════════════════════════════════════════════════
# Step 1: Narrative coding of high-burden rationales
# ═══════════════════════════════════════════════════════════════════

# Keyword rules (case-insensitive, word-boundary-aware where useful)
FRAME_RULES = {
    "autonomy_infringement": [
        r"individual\s+choice", r"personal\s+choice", r"personal\s+freedom",
        r"autonomy", r"privacy", r"consent", r"personal\s+discretion",
        r"right\s+to\s+choose", r"individual\s+liberty", r"personal\s+decision",
        r"freedom\s+to\s+decide", r"procedural\s+dignity",
        r"limit\w*\s+(individual|personal|her)\s+(choice|freedom|autonomy)",
        r"infringe\w*\s+on\s+(individual|personal)",
    ],
    "procedural_legitimacy": [
        r"justified", r"safeguard", r"reasonable\s+(verification|requirement|check|procedure)",
        r"necessary\s+(procedure|requirement|check|verification|step)",
        r"eligibility", r"fraud\s+prevention", r"legitimate\s+(reason|need|purpose|public)",
        r"proper\s+(procedure|verification|documentation)", r"due\s+process",
        r"accountability", r"transparency", r"oversight",
        r"prevent\s+(fraud|abuse|misuse)", r"fair\s+(procedure|process|system)",
    ],
    "collective_responsibility": [
        r"civic\s+duty", r"public\s+health", r"protect\w*\s+others?",
        r"communal\s+(benefit|welfare|good|health)", r"vulnerable\s+group",
        r"collective\s+(welfare|benefit|good|responsibility|health)",
        r"common\s+good", r"social\s+responsibility", r"greater\s+good",
        r"protect\w*\s+the\s+public", r"community\s+(health|benefit|welfare)",
        r"herd\s+immunity", r"protect\w*\s+(the\s+)?vulnerable",
    ],
    "access_barriers": [
        # Explicit linkage of burden to resources/constraints/vulnerability
        r"cannot\s+afford", r"can'?t\s+afford",
        r"miss\s+work", r"miss\w*\s+(her|the)\s+job", r"lose\s+(income|wages|pay)",
        r"lost\s+wages?", r"unpaid\s+(leave|time)",
        r"childcare", r"child\s+care", r"babysit",
        r"no\s+(regular\s+)?(doctor|provider|clinic|healthcare)",
        r"lack\s+of\s+(access|transportation|childcare|flexibility|resource)",
        r"low\s+income", r"limited\s+(income|resource|flexibility|means)",
        r"cannot\s+(comply|attend|go|make\s+it|manage|participate)",
        r"(unable|cannot)\s+to\s+(take\s+time|get\s+time|find\s+time)",
        r"no\s+(way|means)\s+to",
        r"disadvantage", r"unequal\s+(burden|access|impact)",
        r"too\s+(difficult|hard|expensive|costly|burdensome)\s+for\s+(her|someone|a\s+person)",
        r"(financial|economic)\s+(hardship|strain|barrier|constraint)",
        r"(single|working)\s+(parent|mother)",
        r"hourly\s+(wage|worker|job)", r"no\s+paid\s+(leave|time|sick)",
        r"precarious\s+(work|employment|job|income)",
    ],
    "coercive_backlash": [
        r"resentment", r"\bdistrust\b", r"\bresistance\b", r"backlash",
        r"reduced\s+legitimacy", r"negative\s+reaction", r"\bmandate\b",
        r"\bpenalt\w+", r"\benforcement\b", r"undermine\w*\s+trust",
        r"backfire", r"antagonize", r"\balienate\b", r"counterproductive",
        r"coerci\w+", r"\bfine\w*", r"\bpunish\w+", r"resent\w+",
        r"push\s*back", r"non\s*compliance", r"rebelli\w+",
    ],
}

def code_rationale(text: str) -> dict:
    """Apply all frame rules to a single rationale text. Returns {frame: 0|1}."""
    text_lower = text.lower()
    scores = {}
    for frame, patterns in FRAME_RULES.items():
        scores[frame] = 1 if any(re.search(p, text_lower) for p in patterns) else 0
    return scores

# Load plan for run_id -> profile_id, burden_level
wb = openpyxl.load_workbook(PROJECT / "data" / "llm_experiment_inputs.xlsx", read_only=True)
ws = wb['Simulation Plan']
headers = [str(h) for h in next(ws.iter_rows(values_only=True))]
plan_lookup = {}
for row in ws.iter_rows(values_only=True):
    r = dict(zip(headers, row))
    plan_lookup[str(r['run_id'])] = {
        'profile_id': str(r['profile_id']),
        'burden_level': str(r['burden_level']),
        'repetition': str(r['repetition']),
    }
wb.close()

# Extract high-burden rationales
narrative_rows = []
with open(OUT / "simulation_outputs.csv") as f:
    for row in csv.DictReader(f):
        rid = row['run_id']
        p = plan_lookup.get(rid, {})
        if (p.get('burden_level') == 'high'
            and row.get('json_valid','').lower() in ('true','1')
            and row.get('score_valid','').lower() in ('true','1')
            and row.get('non_refusal','').lower() in ('true','1')):
            frames = code_rationale(row.get('rationale', ''))
            narrative_rows.append({
                'model': row['model'],
                'provider': row['provider'],
                'deployment': row.get('deployment',''),
                'profile_id': p['profile_id'],
                'repetition': p['repetition'],
                'rationale_high': row.get('rationale',''),
                **frames,
            })

# Save narrative_coding.csv
nc_fields = ['model','provider','deployment','profile_id','repetition','rationale_high',
             'autonomy_infringement','procedural_legitimacy','collective_responsibility',
             'access_barriers','coercive_backlash']
with open(OUT / "narrative_coding.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=nc_fields)
    w.writeheader()
    w.writerows(narrative_rows)

print(f"narrative_coding.csv: {len(narrative_rows)} rows")
# Frame prevalence
for frame in ['autonomy_infringement','procedural_legitimacy','collective_responsibility',
              'access_barriers','coercive_backlash']:
    count = sum(1 for r in narrative_rows if r[frame] == 1)
    print(f"  {frame}: {count} ({100*count/len(narrative_rows):.1f}%)")

# ═══════════════════════════════════════════════════════════════════
# Step 2: Merge into h2_analysis.csv
# ═══════════════════════════════════════════════════════════════════

# Load delta_analysis
delta = {}
with open(OUT / "delta_analysis.csv") as f:
    for row in csv.DictReader(f):
        key = (row['model'], row['profile_id'], row['repetition'])
        delta[key] = float(row['delta'])

# Load PVC
pvc = {}
with open(OUT / "table2_orientation_profiles.csv") as f:
    for row in csv.DictReader(f):
        pvc[row['Model']] = float(row['PVC'])

# Merge
h2_rows = []
for nr in narrative_rows:
    key = (nr['model'], nr['profile_id'], nr['repetition'])
    if key in delta:
        h2_rows.append({
            'model': nr['model'],
            'profile_id': nr['profile_id'],
            'repetition': nr['repetition'],
            'delta': delta[key],
            'PVC': pvc.get(nr['model'], None),
            'autonomy_infringement': nr['autonomy_infringement'],
            'procedural_legitimacy': nr['procedural_legitimacy'],
            'collective_responsibility': nr['collective_responsibility'],
            'access_barriers': nr['access_barriers'],
            'coercive_backlash': nr['coercive_backlash'],
        })

h2_fields = ['model','profile_id','repetition','delta','PVC',
             'autonomy_infringement','procedural_legitimacy',
             'collective_responsibility','access_barriers','coercive_backlash']
with open(OUT / "h2_analysis.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=h2_fields)
    w.writeheader()
    w.writerows(h2_rows)
print(f"\nh2_analysis.csv: {len(h2_rows)} rows")

# ═══════════════════════════════════════════════════════════════════
# Step 3: H2 mediation-style regressions
# ═══════════════════════════════════════════════════════════════════

ALL_FRAMES = ['autonomy_infringement','procedural_legitimacy',
              'collective_responsibility','access_barriers','coercive_backlash']

# Compute prevalence and filter frames with <1% or >99%
frame_prevalence = {}
for f in ALL_FRAMES:
    count = sum(1 for r in h2_rows if r[f] == 1)
    pct = 100 * count / len(h2_rows)
    frame_prevalence[f] = pct

print("\n--- Frame prevalence (post-recode) ---")
for f in ALL_FRAMES:
    pct = frame_prevalence[f]
    flag = ""
    if pct < 1 or pct > 99:
        flag = " [EXCLUDED from regression]"
    print(f"  {f:<28} {pct:.1f}%{flag}")

# Filter: only frames with 1% <= prevalence <= 99%
FRAMES = [f for f in ALL_FRAMES if 1 <= frame_prevalence[f] <= 99]
EXCLUDED = [f for f in ALL_FRAMES if f not in FRAMES]
print(f"\nFrames in regression: {FRAMES}")
print(f"Frames excluded: {EXCLUDED} (reported descriptively only)")

# Prepare arrays
delta_arr = np.array([r['delta'] for r in h2_rows])
pvc_arr = np.array([r['PVC'] for r in h2_rows])
frame_arrs = {f: np.array([r[f] for r in h2_rows]) for f in FRAMES}

print(f"\n=== H2 Mediation-Style Analysis (N={len(h2_rows)}) ===\n")

# Total effect: delta ~ PVC (from H1, but now on individual level)
X_tot = sm.add_constant(pvc_arr)
tot = sm.OLS(delta_arr, X_tot).fit()
c_total = tot.params[1]
c_total_se = tot.bse[1]
c_total_p = tot.pvalues[1]
print(f"Total effect (delta ~ PVC): c = {c_total:.4f}, SE = {c_total_se:.4f}, p = {c_total_p:.4f}")

# a paths: each frame ~ PVC
a_results = {}
print("\n--- a paths: Frame_f ~ PVC ---")
for f in FRAMES:
    X_a = sm.add_constant(pvc_arr)
    a_mod = sm.OLS(frame_arrs[f], X_a).fit()
    a_results[f] = {
        'a_coef': a_mod.params[1], 'a_se': a_mod.bse[1],
        'a_t': a_mod.tvalues[1], 'a_p': a_mod.pvalues[1],
    }
    print(f"  {f:<28} a = {a_results[f]['a_coef']:+.4f}  p = {a_results[f]['a_p']:.4f}")

# Full model: delta ~ PVC + all frames
X_full = sm.add_constant(np.column_stack([pvc_arr] + [frame_arrs[f] for f in FRAMES]))
full_mod = sm.OLS(delta_arr, X_full).fit()

b_names = ['PVC'] + FRAMES
print("\n--- Full model: delta ~ PVC + all frames ---")
print(f"  R² = {full_mod.rsquared:.4f}, adj R² = {full_mod.rsquared_adj:.4f}")
for i, name in enumerate(b_names):
    idx = i + 1  # skip const
    print(f"  {name:<28} coef = {full_mod.params[idx]:+.4f}  p = {full_mod.pvalues[idx]:.4f}")

c_prime = full_mod.params[1]
c_prime_p = full_mod.pvalues[1]
print(f"\n  Direct effect c' = {c_prime:.4f} (p = {c_prime_p:.4f})")
print(f"  Total effect c   = {c_total:.4f} (p = {c_total_p:.4f})")
print(f"  Δ = c - c'       = {c_total - c_prime:.4f}")

# Indirect effects
print("\n--- Indirect effects (a × b) ---")
table5 = []
for i, f in enumerate(FRAMES):
    a = a_results[f]['a_coef']
    a_p = a_results[f]['a_p']
    b = full_mod.params[i + 2]  # +2 because const(0) + PVC(1) + frame(i+2)
    b_p = full_mod.pvalues[i + 2]
    indirect = a * b
    print(f"  {f:<28} a={a:+.4f} (p={a_p:.4f})  b={b:+.4f} (p={b_p:.4f})  a×b={indirect:+.4f}")
    table5.append({
        'frame': f,
        'a_coef': round(a, 4), 'a_p': round(a_p, 4),
        'b_coef': round(b, 4), 'b_p': round(b_p, 4),
        'indirect': round(indirect, 4),
    })

# ═══════════════════════════════════════════════════════════════════
# Step 4: Save Table 5
# ═══════════════════════════════════════════════════════════════════

# CSV
with open(OUT / "table5_h2_mediation_style.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=['frame','a_coef','a_p','b_coef','b_p','indirect'])
    w.writeheader()
    w.writerows(table5)
print(f"\nSaved: outputs/table5_h2_mediation_style.csv")

# TeX
tex = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\caption{H2: Mediation-Style Analysis — Interpretive Frames}",
    r"\label{tab:h2-mediation}",
    r"\begin{tabular}{lcccccc}",
    r"\toprule",
    r"Frame & $a$ & $p_a$ & $b$ & $p_b$ & $a \times b$ \\",
    r"\midrule",
]
for row in table5:
    if row.get('note'):
        tex.append(
            f"{row['frame'].replace('_',' ')} & {row['a_coef']:.4f} & {row['a_p']:.4f} & "
            f"\multicolumn{{3}}{{l}}{{{row['note']}}} \\\\"
        )
    else:
        tex.append(
            f"{row['frame'].replace('_',' ')} & {row['a_coef']:.4f} & {row['a_p']:.4f} & "
            f"{row['b_coef']:.4f} & {row['b_p']:.4f} & {row['indirect']:.4f} \\\\"
        )
tex.append(r"\midrule")
tex.append(
    f"Total effect $c$ & \\multicolumn{{5}}{{l}}{{ {c_total:.4f} ($p$ = {c_total_p:.4f}) }} \\\\"
)
tex.append(
    f"Direct effect $c'$ & \\multicolumn{{5}}{{l}}{{ {c_prime:.4f} ($p$ = {c_prime_p:.4f}) }} \\\\"
)
tex.append(
    f"$\\Delta = c - c'$ & \\multicolumn{{5}}{{l}}{{ {c_total - c_prime:.4f} }} \\\\"
)
tex.append(
    f"Full model $R^2$ & \\multicolumn{{5}}{{l}}{{ {full_mod.rsquared:.4f} (adj. {full_mod.rsquared_adj:.4f}) }} \\\\"
)
tex.extend([
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
])

with open(OUT / "table5_h2_mediation_style.tex", "w") as f:
    f.write("\n".join(tex) + "\n")
print("Saved: outputs/table5_h2_mediation_style.tex")

print("\n=== Done ===")
