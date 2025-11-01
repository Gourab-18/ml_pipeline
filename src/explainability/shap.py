"""
SHAP (SHapley Additive exPlanations) for model interpretability.

Provides SHAP explainer for tabular models with sampled background data.
Saves per-instance SHAP values to CSV for visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any, Union
import warnings
import os

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


def sample_background_data(
    X: np.ndarray,
    n_samples: int = 100,
    random_state: int = 42
) -> np.ndarray:
    """
    Sample background data for SHAP explainer.
    
    Args:
        X: Full feature matrix
        n_samples: Number of samples to use as background
        random_state: Random seed
    
    Returns:
        Sampled background data
    """
    rng = np.random.default_rng(random_state)
    n_total = X.shape[0]
    
    if n_samples >= n_total:
        return X.copy()
    
    indices = rng.choice(n_total, size=n_samples, replace=False)
    return X[indices]


def compute_shap_values(
    X: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    background_data: Optional[np.ndarray] = None,
    n_background: int = 100,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    explainer_type: str = 'kernel'
) -> Union[np.ndarray, Dict[str, Any]]:
    """
    Compute SHAP values for instances.
    
    Args:
        X: Feature matrix to explain (n_samples, n_features)
        predict_fn: Function that takes X and returns probabilities
        background_data: Optional pre-sampled background data
        n_background: Number of background samples if not provided
        feature_names: Optional feature names
        explainer_type: 'kernel' or 'linear' (future: 'tree' for tree models)
        random_state: Random seed
    
    Returns:
        SHAP values array (n_samples, n_features) or dict with explainer info
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is not installed. Install with: pip install shap")
    
    if background_data is None:
        # Use all available data or sample
        # In practice, we'd sample from training data
        # For now, use provided X
        background_data = sample_background_data(X, n_samples=n_background, random_state=random_state)
    
    # Wrap predict function to ensure correct output shape
    def wrapped_predict(X_input):
        pred = predict_fn(X_input)
        if pred.ndim > 1:
            pred = pred[:, 0]  # Take first column if multi-output
        return pred
    
    if explainer_type == 'kernel':
        explainer = shap.KernelExplainer(wrapped_predict, background_data)
        shap_values = explainer.shap_values(X, nsamples=min(100, X.shape[0]))
    else:
        raise ValueError(f"Unsupported explainer_type: {explainer_type}")
    
    # Ensure 2D array
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Take first output for binary classification
    
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    
    return shap_values


def save_shap_values(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    output_path: str = "shap_values.csv",
    instance_ids: Optional[np.ndarray] = None
) -> str:
    """
    Save SHAP values to CSV.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        X: Original feature matrix
        feature_names: Optional feature names
        output_path: Path to save CSV
        instance_ids: Optional instance IDs
    
    Returns:
        Path to saved file
    """
    n_samples, n_features = shap_values.shape
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if len(feature_names) != n_features:
        raise ValueError(f"feature_names length mismatch: {len(feature_names)} != {n_features}")
    
    # Create DataFrame
    data = {}
    if instance_ids is not None:
        data['instance_id'] = instance_ids
    else:
        data['instance_id'] = np.arange(n_samples)
    
    # Add SHAP values
    for i, feat_name in enumerate(feature_names):
        data[f"shap_{feat_name}"] = shap_values[:, i]
    
    # Add feature values
    for i, feat_name in enumerate(feature_names):
        data[f"value_{feat_name}"] = X[:, i]
    
    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return output_path


def compute_and_save_shap(
    X: np.ndarray,
    predict_fn: Callable[[np.ndarray], np.ndarray],
    output_path: str = "shap_values.csv",
    background_data: Optional[np.ndarray] = None,
    n_background: int = 100,
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    explainer_type: str = 'kernel',
    instance_ids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute SHAP values and save to CSV.
    
    Args:
        X: Feature matrix to explain
        predict_fn: Prediction function
        output_path: Path to save CSV
        background_data: Optional background data
        n_background: Number of background samples
        feature_names: Optional feature names
        random_state: Random seed
        explainer_type: Type of SHAP explainer
        instance_ids: Optional instance IDs
    
    Returns:
        Dictionary with SHAP values and metadata
    """
    shap_values = compute_shap_values(
        X, predict_fn,
        background_data=background_data,
        n_background=n_background,
        feature_names=feature_names,
        random_state=random_state,
        explainer_type=explainer_type
    )
    
    saved_path = save_shap_values(
        shap_values, X,
        feature_names=feature_names,
        output_path=output_path,
        instance_ids=instance_ids
    )
    
    return {
        'shap_values': shap_values,
        'csv_path': saved_path,
        'feature_names': feature_names or [f"feature_{i}" for i in range(X.shape[1])],
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }


def get_top_features_shap(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_k: int = 10,
    aggregation: str = 'mean'
) -> pd.DataFrame:
    """
    Get top K most important features by mean absolute SHAP value.
    
    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: Feature names
        top_k: Number of top features to return
        aggregation: 'mean' or 'abs_mean'
    
    Returns:
        DataFrame with top features and their importance
    """
    if aggregation == 'mean':
        importance = np.mean(shap_values, axis=0)
    elif aggregation == 'abs_mean':
        importance = np.mean(np.abs(shap_values), axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    df = df.sort_values('importance', key=abs, ascending=False).head(top_k)
    return df.reset_index(drop=True)


__all__ = [
    'compute_shap_values',
    'save_shap_values',
    'compute_and_save_shap',
    'get_top_features_shap',
    'sample_background_data',
    'SHAP_AVAILABLE'
]
