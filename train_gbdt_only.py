#!/usr/bin/env python3
"""
Train GBDT models only (XGBoost and LightGBM) - Works on macOS!

This script trains XGBoost and LightGBM with 5-fold cross-validation
WITHOUT requiring TensorFlow, so it works perfectly on macOS.

Usage:
    python3 train_gbdt_only.py

    Or with custom parameters:
    python3 train_gbdt_only.py --folds 10 --seed 123
"""

import argparse
from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
from src.baselines.xgb_lgb import run_gbdt_cv
from src.baselines.compare_models import compare_models
import os


def main():
    parser = argparse.ArgumentParser(
        description="Train XGBoost and LightGBM models (no TensorFlow required)"
    )
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/sample.csv',
        help='Path to data file (default: data/sample.csv)'
    )

    args = parser.parse_args()

    print("🚀 Training GBDT Models (XGBoost + LightGBM)")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Folds: {args.folds}")
    print(f"  • Seed: {args.seed}")
    print(f"  • Data: {args.data}")
    print("=" * 70)
    print()

    # Step 1: Load data
    print('📊 Step 1: Loading data...')
    loader = DataLoader('configs/schema.yaml')
    df = loader.load_data(args.data)
    print(f'✅ Data loaded: {len(df)} samples, {len(df.columns)} columns')
    print()

    # Step 2: Preprocess
    print('🔧 Step 2: Preprocessing...')
    pipeline = create_lightweight_pipeline()
    pipeline.fit_on(df)
    X, y = pipeline.transform(df)
    print(f'✅ Preprocessing complete: X={X.shape}, y={y.shape}')
    print()

    # Step 3: Train GBDT models
    print(f'🏋️  Step 3: Training XGBoost and LightGBM ({args.folds}-fold CV)...')
    print('This may take 2-5 minutes depending on your machine...')
    print()

    results = run_gbdt_cv(
        X, y,
        k_folds=args.folds,
        seed=args.seed,
        models=['xgb', 'lgb']
    )

    print()
    print('✅ GBDT training complete!')
    print()

    # Step 4: Display results
    print('=' * 70)
    print('📊 TRAINING RESULTS')
    print('=' * 70)
    print()

    xgb_metrics = results['results']['xgb']['calibrated_metrics']
    lgb_metrics = results['results']['lgb']['calibrated_metrics']

    print('XGBoost Metrics:')
    print(f"  • ROC-AUC:  {xgb_metrics['roc_auc']:.4f}")
    print(f"  • PR-AUC:   {xgb_metrics['pr_auc']:.4f}")
    print(f"  • Brier:    {xgb_metrics['brier']:.4f}")
    print(f"  • ECE:      {xgb_metrics['ece']:.4f}")
    print(f"  • Accuracy: {xgb_metrics['accuracy']:.4f}")
    print()

    print('LightGBM Metrics:')
    print(f"  • ROC-AUC:  {lgb_metrics['roc_auc']:.4f}")
    print(f"  • PR-AUC:   {lgb_metrics['pr_auc']:.4f}")
    print(f"  • Brier:    {lgb_metrics['brier']:.4f}")
    print(f"  • ECE:      {lgb_metrics['ece']:.4f}")
    print(f"  • Accuracy: {lgb_metrics['accuracy']:.4f}")
    print()

    # Step 5: Model comparison
    print('=' * 70)
    print('📊 MODEL COMPARISON')
    print('=' * 70)

    comparison_df = compare_models(
        xgb_run_dir=results['run_dir'],
        lgb_run_dir=results['run_dir']
    )

    print()
    print(comparison_df.to_string(index=False))
    print()

    # Step 6: Save comparison
    comparison_path = os.path.join(results['run_dir'], 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    print('=' * 70)
    print(f'✅ Comparison saved to: {comparison_path}')
    print(f'📁 All artifacts saved to: {results["run_dir"]}')
    print()
    print('🎉 Training pipeline completed successfully!')
    print()
    print('💡 Next steps:')
    print('   • View results: cat', comparison_path)
    print('   • Compare with other runs: python3 compare_models.py --latest')
    print('   • Export best model: make export')
    print('=' * 70)


if __name__ == "__main__":
    main()
