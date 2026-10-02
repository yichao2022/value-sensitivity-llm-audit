#!/usr/bin/env python3
"""
Compute MFQ-30 composite scores from canonical two-part 0-5 data.
Formula: Z(Care) + Z(Fairness) - Z(Loyalty) - Z(Authority) - Z(Sanctity)
"""

import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_responses_filled_v4.csv"
OUTPUT_CSV = PROJECT / "outputs" / "mfq30_canonical_composites.csv"

# Dimension mapping (item_id prefix → dimension)
DIMENSION_MAP = {
    "CA": "Care",
    "FA": "Fairness",
    "LO": "Loyalty", 
    "AU": "Authority",
    "SA": "Sanctity",
}

def load_scores():
    """Load raw scores from CSV."""
    scores = defaultdict(lambda: defaultdict(dict))
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
    """Compute MFQ composite: Z(Care) + Z(Fairness) - Z(Loyalty) - Z(Authority) - Z(Sanctity)"""
    # Group by dimension
    dims = defaultdict(list)
    for item_id, score in scores.items():
        prefix = item_id[:2]
        dimension = DIMENSION_MAP.get(prefix)
        if dimension:
            dims[dimension].append(score)
    
    if not dims:
        return None
    
    # Compute mean for each dimension
    dim_means = {}
    for dim, vals in dims.items():
        if vals:
            dim_means[dim] = np.mean(vals)
    
    # Need all 5 dimensions
    required = ["Care", "Fairness", "Loyalty", "Authority", "Sanctity"]
    if not all(d in dim_means for d in required):
        return None
    
    # Z-score standardization (using sample SD)
    all_scores = [s for vals in dims.values() for s in vals]
    if len(all_scores) < 2:
        return None
    
    mean_all = np.mean(all_scores)
    sd_all = np.std(all_scores, ddof=1)  # sample SD
    
    if sd_all == 0:
        return None
    
    z_scores = {d: (m - mean_all) / sd_all for d, m in dim_means.items()}
    
    # Composite: Z(Care) + Z(Fairness) - Z(Loyalty) - Z(Authority) - Z(Sanctity)
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
    print("Computing MFQ-30 Composite Scores (Canonical 0-5)")
    print("=" * 60)
    
    scores = load_scores()
    print(f"\nLoaded scores for {len(scores)} models")
    
    results = []
    for model, model_scores in sorted(scores.items()):
        print(f"\n{model}:")
        composite = compute_composite(model_scores)
        if composite:
            print(f"  Composite: {composite['composite']:.3f}")
            print(f"  Care: {composite['care_mean']:.2f} (Z={composite['care_z']:.3f})")
            print(f"  Fairness: {composite['fairness_mean']:.2f} (Z={composite['fairness_z']:.3f})")
            print(f"  Loyalty: {composite['loyalty_mean']:.2f} (Z={composite['loyalty_z']:.3f})")
            print(f"  Authority: {composite['authority_mean']:.2f} (Z={composite['authority_z']:.3f})")
            print(f"  Sanctity: {composite['sanctity_mean']:.2f} (Z={composite['sanctity_z']:.3f})")
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
            print(f"  ✗ Could not compute composite (missing data)")
    
    # Write output
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
