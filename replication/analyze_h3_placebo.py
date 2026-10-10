#!/usr/bin/env python3
"""Parse H3 placebo willingness scores and summarize them by model."""
import re
from pathlib import Path

import pandas as pd

HOME = Path(__file__).resolve().parent
RESULTS = HOME / "outputs" / "h3_placebo_results.csv"
PARSED = HOME / "outputs" / "h3_placebo_parsed.csv"

# Score sits near the word "willingness". A stated range (70–85) becomes its midpoint.
_WINDOW = 700
_PATTERNS = [
    ("range", r"(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\(on a 0\s*[-–—]\s*100 scale\)"),
    ("scale", r"(\d{1,3})\s*\(on a 0\s*[-–—]\s*100 scale\)"),
    ("label", r"\(0\s*[-–—]\s*100\)\s*[:*\s]*(\d{1,3})\b"),
    ("json", r"""willingness["']?\s*[:=]\s*(\d{1,3})\b"""),
    ("out", r"(\d{1,3})\s*out of\s*100\b"),
    ("slash", r"(\d{1,3})\s*/\s*100\b"),
    ("eq", r"willingness\s*(?:score|estimate)?\s*[=:]\s*\**\s*(\d{1,3})\b"),
    ("of", r"willingness to proceed with vaccination of(?: around)?\s*(\d{1,3})\b"),
]


def extract_willingness(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    raw = str(text).strip()
    if not raw or raw.lower() == "nan":
        return None
    starts = [m.start() for m in re.finditer(r"willingness", raw[:2500], re.I)]
    if not starts:
        starts = [0]
    for start in starts:
        window = raw[start:start + _WINDOW]
        for name, pattern in _PATTERNS:
            match = re.search(pattern, window, re.I)
            if not match:
                continue
            if name == "range":
                low, high = int(match.group(1)), int(match.group(2))
                if 0 <= low <= 100 and 0 <= high <= 100 and high > low:
                    return round((low + high) / 2)
                continue
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score
    return None


def _self_check():
    cases = [
        ("Estimated willingness to proceed with vaccination: 65 (on a 0-100 scale)", 65),
        ("willingness to proceed with vaccination at **72 out of 100**.", 72),
        ("willingness to proceed with vaccination is estimated at **28/100**", 28),
        ("Willingness to proceed with vaccination (0-100):** **35**", 35),
        ("Willingness to proceed with vaccination: 70/100", 70),
        ("Willingness to proceed with vaccination: 70–85 (on a 0–100 scale)", 78),
        ('{"willingness": 65}', 65),
        ("predict a willingness to proceed with vaccination of 80.", 80),
        ("willingness to proceed with vaccination of around 40.", 40),
        ("Willingness: 40", 40),
        ("", None),
    ]
    for text, expected in cases:
        got = extract_willingness(text)
        assert got == expected, (text, got, expected)


def main():
    _self_check()
    df = pd.read_csv(RESULTS)
    n_raw = len(df)
    # A later append wrote a shorter row, so 75 Qwen replies landed in profile_id
    # and duplicated a run_id that already has the full response.
    df = df.assign(_raw_len=df["raw_response"].fillna("").astype(str).str.len())
    df = df.sort_values("_raw_len", ascending=False).drop_duplicates("run_id", keep="first")
    df = df.drop(columns="_raw_len")

    df["willingness"] = df["raw_response"].map(extract_willingness)
    has_text = df["raw_response"].fillna("").astype(str).str.strip().ne("")
    has_error = df["error"].fillna("").astype(str).str.strip().ne("")

    print(f"Rows in file: {n_raw} ({n_raw - len(df)} duplicate run_ids dropped)")
    print(f"Unique runs: {len(df)}")
    print(f"Parsed: {int(df['willingness'].notna().sum())}/{len(df)} "
          f"({df['willingness'].notna().mean():.1%})")
    print(f"Parsed among responses with text: "
          f"{int(df.loc[has_text, 'willingness'].notna().sum())}/{int(has_text.sum())}")

    print("\n=== By model ===")
    for model, sub in df.groupby("model", sort=False):
        parsed = sub["willingness"].notna()
        text = sub["raw_response"].fillna("").astype(str).str.strip().ne("")
        errors = sub.loc[sub["error"].fillna("").astype(str).str.strip().ne(""), "error"]
        print(f"{model}: {int(parsed.sum())}/{len(sub)} parsed "
              f"({parsed.mean():.1%}), text {int(text.sum())}/{len(sub)}")
        if parsed.any():
            scores = sub.loc[parsed, "willingness"]
            print(f"  mean {scores.mean():.2f}  sd {scores.std():.2f}  "
                  f"median {scores.median():.1f}  min {scores.min():.0f}  max {scores.max():.0f}")
        if len(errors):
            reason = errors.astype(str).str.slice(0, 110).value_counts().index[0]
            print(f"  api errors {len(errors)}/{len(sub)}: {reason}")

    valid = df[df["willingness"].notna()]
    print("\n=== Parsed rows ===")
    print(f"Mean {valid['willingness'].mean():.2f}  "
          f"median {valid['willingness'].median():.1f}  "
          f"sd {valid['willingness'].std():.2f}  "
          f"min {valid['willingness'].min():.0f}  max {valid['willingness'].max():.0f}")
    if has_error.any():
        print(f"Unparsed because the API returned no text: {int((has_error & ~has_text).sum())}")
    valid.to_csv(PARSED, index=False)
    print(f"Saved {len(valid)} rows to {PARSED}")


if __name__ == "__main__":
    main()
