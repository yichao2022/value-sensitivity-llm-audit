#!/usr/bin/env python3
"""
Compute MFQ-30 composite scores - handles duplicate entries.
"""

import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "outputs" / "mfq30_FINAL_15models_v5.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_composites_v2.csv"

DIMENSION_MAP = {
    "CA": "Care",
    "FA": "Fairness", 
    "LO": "Loyalty",
    "AU": "Authority",
    "SA": "Sanctity",
}

def load_scores():
    """Load raw scores from CSV, keeping last occurrence for duplicates."""
    scores = defaultdict(dict)
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            item_id = row["item_id"]
            score = row["raw_score"]
            if score and score.strip():
                try:
                    scores[model][item_id] = float(score)
                except ValueError:
                    pass
    return scores

def compute_composite(scores):
    """Compute MFQ composite."""
    dims = defaultdict(list)
    for item_id, score in scores.items():
        prefix = item_id[:2]
        dimension = DIMENSION_MAP.get(prefix)
        if dimension:
            dims[dimension].append(score)
    
    if not dims:
        return None
    
    dim_means = {}
    for dim, vals in dims.items():
        if vals:
            dim_means[dim] = np.mean(vals)
    
    required = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"]
    if not all(d in dim_means for d in required):
        missing = [d for d in required if d not in dim_means]
        print(f"    Missing dimensions: {missing}")
        return None
    
    all_scores = [s for vals in dims.values() for s in vals]
    if len(all_scores) < 2:
        return None
    
    mean_all = np.mean(all_scores)
    sd_all = np.std(all_scores, ddof=1)
    
    if sd_all == 0:
        return None
    
    z_scores = {d: (m - mean_all) / sd_all for d, m in dim_means.items()}
    
    composite = (z_scores["Care"] + z_scores["Fairness"] - 
                 z_scores["Loyalty"] - z_scores["Authority"] - z_scores["Sanctity"])
    
    return {
        "composite": composite,
        "care_mean": dim_means["Care"],
        "fairness_mean": dim_means["Fairness"],
        "loyalty_mean": dim_means["Loyalty"],
        "authority_mean": dim_means["Authority"],
        "sanctity_mean": dim_means["Sanctity"],
        "care_z": z_scores["Care"],
        "fairness_z": z_scores["Fairness"],
        "loyalty_z": z_scores["Loyalty"],
        "authority_z": z_scores["Authority"],
        "sanctity_z": z_scores["Sanctity"],
    }

def main():
    print("=" * 60)
    print("Computing MFQ-30 Composite Scores (v2)")
    print("=" * 60)
    
    scores = load_scores()
    print(f"\nLoaded scores for {len(scores)} models")
    
    results = []
    for model, model_scores in sorted(scores.items()):
        print(f"\n{model} ({len(model_scores)} items):")
        composite = compute_composite(model_scores)
        if composite:
            print(f"  Composite: {composite['composite']:.3f}")
            results.append({
                "model": model,
                "composite_score": round(composite['composite'], 4),
                "care_mean": round(composite['care_mean'], 2),
                "fairness_mean": round(composite['fairness_mean'], 2),
                "loyalty_mean": round(composite['loyalty_mean'], 2),
                "authority_mean": round(composite['authority_mean'], 2),
                "sanctity_mean": round(composite['sanctity_mean'], 2),
            })
        else:
            print(f"  ✗ Could not compute composite")
    
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "composite_score", "care_mean", "fairness_mean",
            "loyalty_mean", "authority_mean", "sanctity_mean"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*60}")
    print(f"Done. {len(results)} models → {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
