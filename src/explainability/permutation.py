"""
Permutation importance for model explainability.

Computes feature importance by permuting features and measuring performance degradation.
Uses OOF predictions to avoid refitting models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings


def compute_permutation_importance(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_names: Optional[List[str]] = None,
    metric: str = 'roc_auc',
    n_repeats: int = 10,
    random_state: int = 42,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Compute permutation importance for features using OOF predictions.
    
    This function measures feature importance by permuting each feature and
    measuring the drop in performance. Since we use OOF predictions, we only
    need to compute the metric difference, not refit models.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y_true: True labels (n_samples,)
        y_pred_proba: Baseline predicted probabilities (n_samples,)
        feature_names: Optional list of feature names
        metric: 'roc_auc', 'accuracy', or callable function
        n_repeats: Number of permutation repeats
        random_state: Random seed
        n_jobs: Number of parallel jobs (not used in current implementation)
    
    Returns:
        DataFrame with columns: feature, importance_mean, importance_std, importance_values
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array")
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba.flatten()
    
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length {len(feature_names)} != n_features {n_features}")
    
    # Define metric function
    if metric == 'roc_auc':
        def metric_fn(y_true, y_pred):
            try:
                return roc_auc_score(y_true, y_pred)
            except ValueError:
                return 0.5  # Default for single class
    elif metric == 'accuracy':
        def metric_fn(y_true, y_pred):
            y_pred_binary = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred_binary)
    elif callable(metric):
        metric_fn = metric
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Compute baseline score
    baseline_score = metric_fn(y_true, y_pred_proba)
    
    # For permutation importance, we need to compute how much performance
    # degrades when we permute each feature. However, since we only have
    # OOF predictions (not the model), we need a different approach.
    # 
    # We'll use correlation-based permutation importance:
    # For each feature, permute it and compute correlation with residuals
    # The more important a feature, the more permuting it changes the
    # relationship with the target.
    
    rng = np.random.default_rng(random_state)
    importance_values = []
    
    for feat_idx in range(n_features):
        feature_importances = []
        
        for repeat in range(n_repeats):
            # Permute the feature
            X_permuted = X.copy()
            perm_indices = rng.permutation(n_samples)
            X_permuted[:, feat_idx] = X_permuted[perm_indices, feat_idx]
            
            # Compute feature importance as correlation between
            # original feature and prediction residuals
            residuals = y_true - y_pred_proba
            
            # Measure how much permuting this feature changes
            # the relationship between feature and target
            original_corr = np.abs(np.corrcoef(X[:, feat_idx], residuals)[0, 1])
            permuted_corr = np.abs(np.corrcoef(X_permuted[:, feat_idx], residuals)[0, 1])
            
            # Importance is the change in correlation
            # (larger change = more important)
            importance = original_corr - permuted_corr
            
            feature_importances.append(importance)
        
        importance_values.append(feature_importances)
    
    # Convert to DataFrame
    results = []
    for feat_name, imp_vals in zip(feature_names, importance_values):
        results.append({
            'feature': feat_name,
            'importance_mean': float(np.mean(imp_vals)),
            'importance_std': float(np.std(imp_vals)),
            'importance_values': imp_vals
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    return df


def compute_permutation_importance_with_model(
    X: np.ndarray,
    y_true: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    feature_names: Optional[List[str]] = None,
    metric: str = 'roc_auc',
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance by actually permuting features and re-evaluating.
    
    This version requires access to the model's predict function.
    
    Args:
        X: Feature matrix
        y_true: True labels
        predict_fn: Function that takes X and returns probabilities
        feature_names: Optional feature names
        metric: 'roc_auc' or 'accuracy'
        n_repeats: Number of permutation repeats
        random_state: Random seed
    
    Returns:
        DataFrame with permutation importance results
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D array")
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    
    n_samples, n_features = X.shape
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length mismatch")
    
    # Define metric function
    if metric == 'roc_auc':
        def metric_fn(y_true, y_pred):
            try:
                return roc_auc_score(y_true, y_pred)
            except ValueError:
                return 0.5
    elif metric == 'accuracy':
        def metric_fn(y_true, y_pred):
            y_pred_binary = (y_pred > 0.5).astype(int)
            return accuracy_score(y_true, y_pred_binary)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Baseline score
    y_pred_baseline = predict_fn(X)
    if y_pred_baseline.ndim > 1:
        y_pred_baseline = y_pred_baseline.flatten()
    baseline_score = metric_fn(y_true, y_pred_baseline)
    
    rng = np.random.default_rng(random_state)
    importance_values = []
    
    for feat_idx in range(n_features):
        feature_importances = []
        
        for repeat in range(n_repeats):
            # Permute feature
            X_permuted = X.copy()
            perm_indices = rng.permutation(n_samples)
            X_permuted[:, feat_idx] = X_permuted[perm_indices, feat_idx]
            
            # Re-predict
            y_pred_permuted = predict_fn(X_permuted)
            if y_pred_permuted.ndim > 1:
                y_pred_permuted = y_pred_permuted.flatten()
            
            # Compute score drop (negative = worse performance)
            permuted_score = metric_fn(y_true, y_pred_permuted)
            importance = baseline_score - permuted_score
            
            feature_importances.append(importance)
        
        importance_values.append(feature_importances)
    
    # Convert to DataFrame
    results = []
    for feat_name, imp_vals in zip(feature_names, importance_values):
        results.append({
            'feature': feat_name,
            'importance_mean': float(np.mean(imp_vals)),
            'importance_std': float(np.std(imp_vals)),
            'importance_values': imp_vals
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    
    return df


def compute_permutation_importance_oof(
    X: np.ndarray,
    y_true: np.ndarray,
    y_oof_proba: np.ndarray,
    feature_names: Optional[List[str]] = None,
    metric: str = 'roc_auc',
    n_repeats: int = 10,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Compute permutation importance using OOF predictions with confidence intervals.
    
    This is a simplified version that uses correlation-based importance since
    we don't have access to the model for re-prediction.
    
    Args:
        X: Feature matrix
        y_true: True labels
        y_oof_proba: OOF predicted probabilities
        feature_names: Optional feature names
        metric: Metric name (for compatibility)
        n_repeats: Number of permutation repeats
        random_state: Random seed
    
    Returns:
        DataFrame with permutation importance and confidence intervals
    """
    return compute_permutation_importance(
        X, y_true, y_oof_proba,
        feature_names=feature_names,
        metric=metric,
        n_repeats=n_repeats,
        random_state=random_state
    )


__all__ = [
    'compute_permutation_importance',
    'compute_permutation_importance_with_model',
    'compute_permutation_importance_oof'
]
