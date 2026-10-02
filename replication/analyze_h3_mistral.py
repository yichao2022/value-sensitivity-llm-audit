#!/usr/bin/env python3
"""H3 Mistral Large frame effect analysis."""
import pandas as pd
import numpy as np
from scipy.stats import f_oneway, ttest_ind

df = pd.read_csv("outputs/expanded15/h3_mistral_large_raw.csv", dtype_backend="numpy_nullable")
print(f"Raw rows: {len(df)}")

# Clean
df = df[df["profile_id"] != "5"].copy()
df = df.drop_duplicates(subset=["profile_id", "frame", "burden_level", "repetition"], keep="first")
df["willingness"] = pd.to_numeric(df["willingness"], errors="coerce")
print(f"After cleaning: {len(df)} rows, {df['profile_id'].nunique()} profiles")

# Means per (profile, frame, burden)
means = df.groupby(["profile_id", "frame", "burden_level"])["willingness"].agg(["mean", "std", "count"]).reset_index()

# Delta = low - high per (profile, frame)
delta = means.pivot_table(index=["profile_id", "frame"], columns="burden_level", values="mean").reset_index()
delta["burden_delta"] = delta["low"] - delta["high"]

# Frame-conditioned means
print()
print("=== Frame-conditioned Burden Deltas (Mistral Large) ===")
frame_effects = delta.groupby("frame")["burden_delta"].agg(["mean", "std", "count"]).reset_index()
for _, r in frame_effects.iterrows():
    print(f"  {r['frame']:<15} Delta = {r['mean']:6.2f} (SD = {r['std']:5.2f}, n = {r['count']})")

# ANOVA
groups = [delta[delta["frame"] == f]["burden_delta"].values for f in ["neutral", "autonomy", "collective", "equity"]]
f_stat, p_val = f_oneway(*groups)
print(f"\nANOVA: F = {f_stat:.4f}, p = {p_val:.4f}")

# Pairwise vs neutral
neutral = delta[delta["frame"] == "neutral"]["burden_delta"]
print()
print("=== Pairwise t-tests (Welch) vs neutral ===")
for frame in ["autonomy", "collective", "equity"]:
    f_vals = delta[delta["frame"] == frame]["burden_delta"]
    t, p = ttest_ind(neutral, f_vals, equal_var=False)
    cohens_d = (f_vals.mean() - neutral.mean()) / np.sqrt((f_vals.var() + neutral.var()) / 2)
    print(f"  {frame:<12} vs neutral: t = {t:+6.3f}, p = {p:.4f}, d = {cohens_d:+6.3f}")

# Save
delta.to_csv("outputs/expanded15/h3_mistral_large_delta.csv", index=False)
frame_effects.to_csv("outputs/expanded15/h3_mistral_large_frame_effects.csv", index=False)
print("\nSaved.")
