#!/usr/bin/env python3
"""
Train PyTorch ANN with cross-validation - Works on macOS!

This script trains a PyTorch ANN with 5-fold cross-validation.
No TensorFlow required, uses Apple Silicon GPU acceleration.

Usage:
    python3 train_pytorch_ann.py

    Or with custom parameters:
    python3 train_pytorch_ann.py --folds 10 --epochs 100 --seed 123
"""

import argparse
import os
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
from src.models.pytorch_ann import create_pytorch_ann
from src.metrics.eval import compute_all_metrics
from src.metrics.calibration import select_best_calibrator


def run_pytorch_ann_cv(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    seed: int = 42,
    hidden_layers: list = [128, 64, 32],
    dropout_rate: float = 0.3,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    epochs: int = 50,
    early_stopping_patience: int = 10,
    weight_decay: float = 0.001,
    verbose: int = 1
):
    """
    Run PyTorch ANN with K-fold cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        k_folds: Number of CV folds
        seed: Random seed
        hidden_layers: Hidden layer sizes
        dropout_rate: Dropout rate
        learning_rate: Learning rate
        batch_size: Batch size
        epochs: Maximum epochs
        early_stopping_patience: Early stopping patience
        weight_decay: L2 regularization
        verbose: Verbosity level

    Returns:
        Dictionary with results and metrics
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"artifacts/cv/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 Output directory: {output_dir}")
    print()

    # Setup K-fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    # Storage for results
    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    oof_indices = []

    # Training loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"{'='*70}")
        print(f"Fold {fold + 1}/{k_folds}")
        print(f"{'='*70}")

        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        print(f"Train: {X_train.shape}, Val: {X_val.shape}")

        # Create and train model
        start_time = time.time()

        model = create_pytorch_ann(
            input_dim=X_train.shape[1],
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            weight_decay=weight_decay,
            verbose=0 if verbose == 0 else 1,
            random_seed=seed + fold
        )

        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=verbose)

        training_time = time.time() - start_time

        # Predict on validation set
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        oof_predictions[val_idx] = y_pred_proba
        oof_indices.extend(val_idx.tolist())

        # Compute metrics
        metrics = compute_all_metrics(y_val, y_pred_proba)
        metrics['fold'] = fold
        metrics['training_time'] = training_time
        fold_metrics.append(metrics)

        # Print fold results
        print(f"\nFold {fold + 1} Results:")
        print(f"  • ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"  • PR-AUC:   {metrics['pr_auc']:.4f}")
        print(f"  • Brier:    {metrics['brier']:.4f}")
        print(f"  • ECE:      {metrics['ece']:.4f}")
        print(f"  • Accuracy: {metrics['accuracy']:.4f}")
        print(f"  • Time:     {training_time:.2f}s")
        print()

    # Compute mean metrics across folds
    mean_metrics = {
        'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in fold_metrics]),
        'brier': np.mean([m['brier'] for m in fold_metrics]),
        'ece': np.mean([m['ece'] for m in fold_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics])
    }

    mean_training_time = np.mean([m['training_time'] for m in fold_metrics])
    total_training_time = sum([m['training_time'] for m in fold_metrics])

    # Calibrate predictions
    print(f"{'='*70}")
    print("Calibrating predictions...")
    print(f"{'='*70}")

    best_method, best_calibrator, calibration_brier_scores = select_best_calibrator(
        y, oof_predictions
    )

    if best_calibrator is not None:
        calibrated_probs = best_calibrator.predict_proba(oof_predictions)
    else:
        calibrated_probs = oof_predictions

    calibrated_metrics = compute_all_metrics(y, calibrated_probs)

    print(f"\nCalibration Results:")
    print(f"  • Best method: {best_method}")
    print(f"  • Uncalibrated Brier: {calibration_brier_scores['uncalibrated']:.4f}")
    print(f"  • Calibrated Brier:   {calibration_brier_scores[best_method]:.4f}")
    print(f"  • Calibrated ROC-AUC: {calibrated_metrics['roc_auc']:.4f}")
    print(f"  • Calibrated ECE:     {calibrated_metrics['ece']:.4f}")
    print()

    # Save results
    results = {
        'model': 'PyTorch_ANN',
        'run_dir': output_dir,
        'oof_path': os.path.join(output_dir, 'oof_pytorch_ann.csv'),
        'fold_metrics': fold_metrics,
        'mean_metrics': mean_metrics,
        'calibrated_metrics': calibrated_metrics,
        'calibration_method': best_method,
        'calibration_metrics': calibration_brier_scores,
        'mean_training_time': mean_training_time,
        'total_training_time': total_training_time,
        'config': {
            'k_folds': k_folds,
            'seed': seed,
            'hidden_layers': hidden_layers,
            'dropout_rate': dropout_rate,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
            'early_stopping_patience': early_stopping_patience,
            'weight_decay': weight_decay
        }
    }

    # Save summary
    summary_path = os.path.join(output_dir, 'pytorch_ann_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save OOF predictions
    import pandas as pd
    oof_df = pd.DataFrame({
        'index': oof_indices,
        'true_label': y[oof_indices],
        'predicted_proba': oof_predictions[oof_indices],
        'calibrated_proba': calibrated_probs[oof_indices]
    })
    oof_df.to_csv(results['oof_path'], index=False)

    print(f"✅ Results saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train PyTorch ANN with cross-validation (no TensorFlow required)"
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
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Maximum epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    args = parser.parse_args()

    print("🚀 Training PyTorch ANN with Cross-Validation")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  • Folds: {args.folds}")
    print(f"  • Seed: {args.seed}")
    print(f"  • Data: {args.data}")
    print(f"  • Epochs: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.learning_rate}")
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

    # Step 3: Train PyTorch ANN
    print(f'🏋️  Step 3: Training PyTorch ANN ({args.folds}-fold CV)...')
    print('This uses Apple Silicon GPU acceleration!')
    print()

    results = run_pytorch_ann_cv(
        X, y,
        k_folds=args.folds,
        seed=args.seed,
        hidden_layers=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=10,
        weight_decay=0.001,
        verbose=1
    )

    print()
    print('✅ PyTorch ANN training complete!')
    print()

    # Step 4: Display results
    print('=' * 70)
    print('📊 TRAINING RESULTS')
    print('=' * 70)
    print()

    metrics = results['calibrated_metrics']

    print('PyTorch ANN Metrics:')
    print(f"  • ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  • PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"  • Brier:    {metrics['brier']:.4f}")
    print(f"  • ECE:      {metrics['ece']:.4f}")
    print(f"  • Accuracy: {metrics['accuracy']:.4f}")
    print(f"  • Calibration: {results['calibration_method']}")
    print(f"  • Avg training time: {results['mean_training_time']:.2f}s/fold")
    print()

    print('=' * 70)
    print(f'✅ All artifacts saved to: {results["run_dir"]}')
    print()
    print('🎉 Training pipeline completed successfully!')
    print()
    print('💡 Next steps:')
    print(f'   • View results: cat {results["run_dir"]}/pytorch_ann_summary.json')
    print(f'   • View predictions: head {results["oof_path"]}')
    print('   • Compare with GBDT: python3 train_gbdt_only.py')
    print('=' * 70)


if __name__ == "__main__":
    main()
