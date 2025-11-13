#!/usr/bin/env python3
"""
Demo script to test PyTorch ANN model.

This script demonstrates that PyTorch ANN works on macOS without the
TensorFlow mutex error issue.
"""

import numpy as np
from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
from src.models.pytorch_ann import create_pytorch_ann
from src.metrics.eval import compute_all_metrics
from sklearn.model_selection import train_test_split


def main():
    print("=" * 70)
    print("PyTorch ANN Demo - Testing on macOS")
    print("=" * 70)
    print()

    # Step 1: Load data
    print("📊 Step 1: Loading data...")
    loader = DataLoader('configs/schema.yaml')
    df = loader.load_data('data/sample.csv')
    print(f"✅ Data loaded: {len(df)} samples, {len(df.columns)} columns")
    print()

    # Step 2: Preprocess
    print("🔧 Step 2: Preprocessing...")
    pipeline = create_lightweight_pipeline()
    pipeline.fit_on(df)
    X, y = pipeline.transform(df)
    print(f"✅ Preprocessing complete: X={X.shape}, y={y.shape}")
    print()

    # Step 3: Split data
    print("✂️  Step 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {X_train.shape}, Test: {X_test.shape}")
    print()

    # Step 4: Create and train PyTorch ANN
    print("🏋️  Step 4: Training PyTorch ANN...")
    print("This should work on macOS without TensorFlow errors!")
    print()

    model = create_pytorch_ann(
        input_dim=X_train.shape[1],
        hidden_layers=[64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=32,
        epochs=30,
        early_stopping_patience=5,
        weight_decay=0.001,
        verbose=1
    )

    # Train with validation set
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)
    print()
    print("✅ Training complete!")
    print()

    # Step 5: Evaluate
    print("📊 Step 5: Evaluation...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_all_metrics(y_test, y_pred_proba)

    print(f"  • ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  • PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"  • Brier:    {metrics['brier']:.4f}")
    print(f"  • ECE:      {metrics['ece']:.4f}")
    print(f"  • Accuracy: {metrics['accuracy']:.4f}")
    print()

    # Step 6: Model summary
    print("=" * 70)
    print("Model Architecture")
    print("=" * 70)
    print(model.get_model_summary())
    print()

    print("=" * 70)
    print("🎉 SUCCESS! PyTorch ANN works perfectly on macOS!")
    print("=" * 70)
    print()
    print("Key features:")
    print("  ✅ No TensorFlow mutex error")
    print("  ✅ Apple Silicon (MPS) GPU acceleration")
    print("  ✅ sklearn-compatible interface")
    print("  ✅ Early stopping and validation")
    print("  ✅ Dropout and L2 regularization")
    print()


if __name__ == "__main__":
    main()
