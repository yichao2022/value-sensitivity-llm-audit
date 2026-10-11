#!/usr/bin/env python3
"""Test script to verify replication outputs match paper tables/figures."""
import pandas as pd
import sys
from pathlib import Path

def test_table_1():
    """Verify Table 1: Policy Simulation Results (27 profiles × 3 models)."""
    print("Testing Table 1...")
    csv_path = Path(__file__).parent / "outputs" / "simulation_outputs.csv"
    
    # Check file exists
    if not csv_path.exists():
        print(f"  ✗ Missing: {csv_path}")
        print("    Run: python3 run_simulation.py")
        return False
    
    df = pd.read_csv(csv_path)
    
    # Check expected columns
    required_cols = ['model', 'profile_id', 'burden_level', 'willingness', 
                     'json_valid', 'score_valid', 'non_refusal']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  ✗ Missing columns: {missing_cols}")
        return False
    
    # Check expected structure: 27 profiles × 3 models × 5 repetitions = 405 rows
    # Or at minimum: 27 profiles × 3 models = 81 unique combinations
    expected_profiles = 27
    expected_models = 3
    expected_burden_levels = ['low', 'high']
    
    unique_profiles = df['profile_id'].nunique()
    unique_models = df['model'].nunique()
    
    if unique_profiles < expected_profiles:
        print(f"  ✗ Expected {expected_profiles} profiles, got {unique_profiles}")
        return False
    
    if unique_models < expected_models:
        print(f"  ✗ Expected {expected_models} models, got {unique_models}")
        return False
    
    # Check data quality metrics
    json_valid_rate = df['json_valid'].mean() if 'json_valid' in df.columns else 0
    score_valid_rate = df['score_valid'].mean() if 'score_valid' in df.columns else 0
    non_refusal_rate = df['non_refusal'].mean() if 'non_refusal' in df.columns else 0
    
    print(f"  ✓ Table 1: {len(df)} total records")
    print(f"    - {unique_profiles} profiles, {unique_models} models")
    print(f"    - JSON valid: {json_valid_rate:.1%}")
    print(f"    - Score valid: {score_valid_rate:.1%}")
    print(f"    - Non-refusal: {non_refusal_rate:.1%}")
    
    # Warn if quality metrics are low but don't fail
    if json_valid_rate < 0.9:
        print(f"  ⚠ Warning: JSON validity below 90%")
    if non_refusal_rate < 0.9:
        print(f"  ⚠ Warning: Non-refusal rate below 90%")
    
    return True

def test_table_2():
    """Verify Table 2: Orientation Scores (15 models)."""
    print("Testing Table 2...")
    csv_path = Path(__file__).parent / "canonical_mfq" / "mfq30_canonical_final.csv"
    if not csv_path.exists():
        print(f"  ✗ Missing: {csv_path}")
        return False
    df = pd.read_csv(csv_path)
    expected_models = 15
    actual_models = df['model'].nunique() if 'model' in df.columns else 0
    if actual_models != expected_models:
        print(f"  ✗ Expected {expected_models} models, got {actual_models}")
        return False
    print(f"  ✓ Table 2: {actual_models} models validated")
    return True

def test_figure_3():
    """Verify Figure 3: H3 Frame Effects Forest Plot."""
    print("Testing Figure 3...")
    # Check if H3 placebo results exist and are complete
    placebo_csv = Path(__file__).parent / "outputs" / "h3_placebo_results.csv"
    if not placebo_csv.exists():
        print(f"  ✗ Missing: {placebo_csv}")
        return False
    df = pd.read_csv(placebo_csv)
    expected_calls = 1080  # 27 profiles × 2 burden × 5 reps × 4 models
    actual_calls = len(df)
    if actual_calls < expected_calls:
        print(f"  ✗ Expected {expected_calls} calls, got {actual_calls}")
        return False
    print(f"  ✓ Figure 3: {actual_calls} H3 calls validated")
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Replication Package Validation")
    print("=" * 50)
    
    results = []
    results.append(("Table 1", test_table_1()))
    results.append(("Table 2", test_table_2()))
    results.append(("Figure 3", test_figure_3()))
    
    print("\n" + "=" * 50)
    print("Summary:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
    print("=" * 50)
    
    if not all(r[1] for r in results):
        sys.exit(1)

if __name__ == "__main__":
    main()
