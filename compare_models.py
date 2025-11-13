#!/usr/bin/env python3
"""
Standalone script to compare models from existing CV runs.

Usage:
    python compare_models.py <ann_run_dir> <gbdt_run_dir>
    python compare_models.py --latest
    python compare_models.py --all
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

from src.baselines.compare_models import compare_models


def find_latest_runs(artifacts_dir: str = "artifacts/cv") -> tuple:
    """Find the latest ANN and GBDT run directories."""
    cv_dir = Path(artifacts_dir)
    if not cv_dir.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return None, None

    # Find directories with summary.json (ANN runs)
    ann_runs = []
    gbdt_runs = []

    for run_dir in cv_dir.iterdir():
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.json"
        gbdt_summary_path = run_dir / "gbdt_summary.json"

        if summary_path.exists():
            ann_runs.append((run_dir.stat().st_mtime, str(run_dir)))
        if gbdt_summary_path.exists():
            gbdt_runs.append((run_dir.stat().st_mtime, str(run_dir)))

    # Get latest by modification time
    ann_run_dir = max(ann_runs, key=lambda x: x[0])[1] if ann_runs else None
    gbdt_run_dir = max(gbdt_runs, key=lambda x: x[0])[1] if gbdt_runs else None

    # If no ANN runs but have GBDT runs, use GBDT as main run
    if not ann_run_dir and gbdt_run_dir:
        ann_run_dir = gbdt_run_dir

    return ann_run_dir, gbdt_run_dir


def compare_all_runs(artifacts_dir: str = "artifacts/cv"):
    """Compare all matching ANN and GBDT runs."""
    cv_dir = Path(artifacts_dir)
    if not cv_dir.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return
    
    # Find all runs with both summaries
    matching_runs = []
    
    for run_dir in cv_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        summary_path = run_dir / "summary.json"
        gbdt_summary_path = run_dir / "gbdt_summary.json"
        
        if summary_path.exists() and gbdt_summary_path.exists():
            matching_runs.append(str(run_dir))
    
    if not matching_runs:
        print("❌ No matching runs found with both ANN and GBDT results")
        return
    
    print(f"📊 Found {len(matching_runs)} matching runs")
    print("=" * 70)
    
    for run_dir in sorted(matching_runs):
        print(f"\n📁 Run: {Path(run_dir).name}")
        print("-" * 70)
        comparison_df = compare_models(
            ann_run_dir=run_dir,
            xgb_run_dir=run_dir,
            lgb_run_dir=run_dir
        )
        print(comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = os.path.join(run_dir, "model_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\n✅ Saved to: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ANN, XGBoost, and LightGBM models from CV runs"
    )
    parser.add_argument(
        "ann_run_dir",
        nargs="?",
        help="Path to ANN CV run directory"
    )
    parser.add_argument(
        "gbdt_run_dir",
        nargs="?",
        help="Path to GBDT CV run directory (can be same as ANN if both in one dir)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Compare latest runs automatically"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Compare all matching runs"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Path to save comparison CSV (default: <ann_run_dir>/model_comparison.csv)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_runs()
        return
    
    if args.latest:
        ann_run_dir, gbdt_run_dir = find_latest_runs()
        if not ann_run_dir and not gbdt_run_dir:
            print("❌ Could not find any runs")
            print("   Make sure you have run training (e.g., python3 train_gbdt_only.py)")
            sys.exit(1)
        elif not gbdt_run_dir:
            print("⚠️  Warning: No GBDT runs found, only ANN runs")
        elif not ann_run_dir:
            print("ℹ️  Note: Only GBDT runs found (XGBoost + LightGBM comparison)")
    elif args.ann_run_dir:
        ann_run_dir = args.ann_run_dir
        gbdt_run_dir = args.gbdt_run_dir or args.ann_run_dir
    else:
        parser.print_help()
        sys.exit(1)
    
    print("📊 Model Comparison")
    print("=" * 70)
    print(f"ANN Run:  {ann_run_dir}")
    print(f"GBDT Run: {gbdt_run_dir}")
    print("=" * 70)
    
    comparison_df = compare_models(
        ann_run_dir=ann_run_dir,
        xgb_run_dir=gbdt_run_dir,
        lgb_run_dir=gbdt_run_dir
    )
    
    if comparison_df.empty:
        print("❌ No model results found. Check that the run directories are correct.")
        sys.exit(1)
    
    print("\n")
    print(comparison_df.to_string(index=False))
    print("\n" + "=" * 70)
    
    # Save comparison
    save_path = args.save or os.path.join(ann_run_dir, "model_comparison.csv")
    comparison_df.to_csv(save_path, index=False)
    print(f"✅ Comparison saved to: {save_path}")


if __name__ == "__main__":
    main()

