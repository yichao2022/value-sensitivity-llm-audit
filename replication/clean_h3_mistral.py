#!/usr/bin/env python3
"""Clean the corrupted h3_mistral_large_raw.csv — dedup, remove strays, verify integrity."""
import pandas as pd
import csv
from pathlib import Path

fpath = Path("outputs/expanded15/h3_mistral_large_raw.csv")

# Read as raw strings to inspect corruption
raw_rows = list(csv.DictReader(open(fpath)))
print(f"Raw rows: {len(raw_rows)}")

# Check for column-shift corruption: rows where burden_level is not 'low' or 'high'
bad = []
good = []
for r in raw_rows:
    bl = r.get("burden_level", "")
    if bl not in ("low", "high"):
        bad.append(r)
    else:
        good.append(r)

print(f"Corrupted rows (bad burden_level): {len(bad)}")
if bad:
    print(f"  Samples:")
    for b in bad[:3]:
        print(f"    profile={b.get('profile_id')}, frame={b.get('frame')}, burden={b.get('burden_level')}, rep={b.get('repetition')}")

# Also check for non-numeric willingness values
df = pd.DataFrame(good)
df["willingness"] = pd.to_numeric(df["willingness"], errors="coerce")
null_will = df["willingness"].isna().sum()
print(f"Rows with null willingness: {null_will}")

df = df.dropna(subset=["willingness"])
df = df[df["profile_id"] != "5"]  # stray profile

# Deduplicate by (profile_id, frame, burden_level, repetition) - keep first
clean = df.drop_duplicates(subset=["profile_id", "frame", "burden_level", "repetition"], keep="first")
print(f"\nAfter cleaning: {len(clean)} rows (expected {27*4*2*5} = {27*4*2*5})")
print(f"Profiles: {clean['profile_id'].nunique()}")
print(f"Frames: {sorted(clean['frame'].unique())}")
print(f"Burden levels: {sorted(clean['burden_level'].unique())}")
print(f"Reps: {sorted(clean['repetition'].unique())}")

# Check per-profile completeness
counts = clean.groupby("profile_id").size()
incomplete = counts[counts < 40]
if len(incomplete) > 0:
    print(f"\nIncomplete profiles: {incomplete.to_dict()}")
else:
    print(f"\nAll 27 profiles have exactly 40 rows each.")

# Save
clean.to_csv(fpath, index=False)
print(f"\nSaved: {len(clean)} clean rows to {fpath}")
