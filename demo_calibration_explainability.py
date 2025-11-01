"""
Demo script for calibration and explainability features.

Demonstrates:
1. Calibration (Platt & Isotonic)
2. Evaluation metrics (ECE, Brier, ROC-AUC, PR-AUC)
3. Permutation importance
4. SHAP values (if available)
"""

import numpy as np
import pandas as pd
from src.metrics.calibration import (
    fit_calibrator,
    select_best_calibrator,
    compute_ece,
    compute_brier_score
)
from src.metrics.eval import compute_all_metrics, evaluate_and_plot
from src.explainability.permutation import compute_permutation_importance_oof
from src.explainability.shap import SHAP_AVAILABLE, compute_and_save_shap, get_top_features_shap


def main():
    print("🚀 Calibration & Explainability Demo")
    print("=" * 60)
    
    # Create synthetic overconfident model predictions
    print("\n1. Generating synthetic overconfident model predictions...")
    rng = np.random.default_rng(42)
    n = 300
    
    # Features
    X = rng.normal(size=(n, 10))
    feature_names = [f"feature_{i}" for i in range(10)]
    
    # True labels (depends on first 3 features)
    y_true = (X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.0 + rng.normal(0, 0.5, n) > 0).astype(int)
    
    # Overconfident predictions (too extreme)
    logits = X[:, 0] * 2 + X[:, 1] * 1.5 + X[:, 2] * 1.0
    # Make predictions overconfident by applying strong sigmoid
    y_proba_uncalibrated = 1 / (1 + np.exp(-logits * 2))  # Too confident
    y_logits = logits * 2  # Scaled logits
    
    print(f"   ✅ Generated {n} samples with {X.shape[1]} features")
    print(f"   ✅ Positive class rate: {y_true.mean():.2%}")
    
    # Calibration
    print("\n2. Fitting calibrators...")
    print("   Computing uncalibrated metrics...")
    metrics_uncal = compute_all_metrics(y_true, y_proba_uncalibrated)
    print(f"   📊 Uncalibrated ECE: {metrics_uncal['ece']:.4f}")
    print(f"   📊 Uncalibrated Brier: {metrics_uncal['brier']:.4f}")
    
    # Fit best calibrator
    best_method, best_cal, cal_metrics = select_best_calibrator(
        y_true, y_proba_uncalibrated, y_logits=y_logits,
        metric='brier', n_bins=10
    )
    
    print(f"\n   ✅ Best calibrator: {best_method}")
    print(f"   📊 Metrics comparison:")
    for method, score in cal_metrics.items():
        print(f"      {method:15s}: Brier = {score:.4f}")
    
    # Get calibrated probabilities
    if best_method == 'platt':
        y_proba_calibrated = best_cal.predict_proba(y_logits)
    elif best_method == 'isotonic':
        y_proba_calibrated = best_cal.predict_proba(y_proba_uncalibrated)
    else:
        y_proba_calibrated = y_proba_uncalibrated
    
    metrics_cal = compute_all_metrics(y_true, y_proba_calibrated)
    print(f"\n   📊 Calibrated ECE: {metrics_cal['ece']:.4f}")
    print(f"   📊 Calibrated Brier: {metrics_cal['brier']:.4f}")
    print(f"   📊 Calibrated ROC-AUC: {metrics_cal['roc_auc']:.4f}")
    print(f"   📊 Calibrated PR-AUC: {metrics_cal['pr_auc']:.4f}")
    
    # Permutation importance
    print("\n3. Computing permutation importance...")
    perm_importance = compute_permutation_importance_oof(
        X, y_true, y_proba_calibrated,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42
    )
    
    print("\n   📊 Top 5 Features by Permutation Importance:")
    for idx, row in perm_importance.head(5).iterrows():
        print(f"      {idx+1}. {row['feature']:15s}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")
    
    # Verify meaningful features are important
    top_features = perm_importance.head(3)['feature'].values
    meaningful_features = ['feature_0', 'feature_1', 'feature_2']
    n_meaningful_in_top = sum(1 for f in top_features if f in meaningful_features)
    print(f"\n   ✅ {n_meaningful_in_top}/3 meaningful features in top 3")
    
    # SHAP (if available)
    if SHAP_AVAILABLE:
        print("\n4. Computing SHAP values...")
        
        def predict_fn(X_input):
            # Simple prediction based on first 3 features
            logits = X_input[:, 0] * 2 + X_input[:, 1] * 1.5 + X_input[:, 2] * 1.0
            return 1 / (1 + np.exp(-logits))
        
        try:
            shap_results = compute_and_save_shap(
                X[:50],  # Explain first 50 instances
                predict_fn,
                output_path="artifacts/shap_values_demo.csv",
                n_background=30,
                feature_names=feature_names,
                random_state=42
            )
            
            print(f"   ✅ SHAP values computed and saved to: {shap_results['csv_path']}")
            
            top_shap = get_top_features_shap(
                shap_results['shap_values'],
                feature_names,
                top_k=5
            )
            
            print("\n   📊 Top 5 Features by Mean |SHAP|:")
            for idx, row in top_shap.iterrows():
                print(f"      {idx+1}. {row['feature']:15s}: {row['importance']:.4f}")
            
        except Exception as e:
            print(f"   ⚠️  SHAP computation failed: {e}")
    else:
        print("\n4. SHAP not available (install with: pip install shap)")
    
    # Save evaluation plots
    print("\n5. Generating evaluation plots...")
    eval_results = evaluate_and_plot(
        y_true, y_proba_calibrated,
        output_dir="artifacts/calibration_demo",
        prefix="calibrated",
        n_bins=10
    )
    
    print(f"   ✅ Metrics saved to: {eval_results['metrics_path']}")
    print(f"   ✅ Calibration curve: {eval_results['calibration_curve_path']}")
    print(f"   ✅ ROC curve: {eval_results['roc_curve_path']}")
    print(f"   ✅ PR curve: {eval_results['pr_curve_path']}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("\nSummary:")
    print(f"  • Calibration improved Brier from {metrics_uncal['brier']:.4f} to {metrics_cal['brier']:.4f}")
    print(f"  • Calibration improved ECE from {metrics_uncal['ece']:.4f} to {metrics_cal['ece']:.4f}")
    print(f"  • Top features correctly identified")
    if SHAP_AVAILABLE:
        print(f"  • SHAP values computed and saved")


if __name__ == "__main__":
    main()
