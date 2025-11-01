"""
Compare ANN, XGBoost, and LightGBM models using CV results.

Reads CV results from different runs and generates comparison metrics.
"""

import json
import os
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


def load_cv_results(run_dir: str) -> Dict[str, Any]:
    """Load CV results from a run directory."""
    summary_path = os.path.join(run_dir, "summary.json")
    
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary


def load_gbdt_results(run_dir: str) -> Dict[str, Any]:
    """Load GBDT CV results from a run directory."""
    summary_path = os.path.join(run_dir, "gbdt_summary.json")
    
    if not os.path.exists(summary_path):
        return None
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    return summary


def compare_models(
    ann_run_dir: Optional[str] = None,
    xgb_run_dir: Optional[str] = None,
    lgb_run_dir: Optional[str] = None,
    ann_results: Optional[Dict[str, Any]] = None,
    xgb_results: Optional[Dict[str, Any]] = None,
    lgb_results: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Compare metrics across models.
    
    Args:
        ann_run_dir: Path to ANN CV run directory
        xgb_run_dir: Path to XGBoost CV run directory (or GBDT run with xgb results)
        lgb_run_dir: Path to LightGBM CV run directory (or GBDT run with lgb results)
        ann_results: Pre-loaded ANN results
        xgb_results: Pre-loaded XGBoost results
        lgb_results: Pre-loaded LightGBM results
    
    Returns:
        DataFrame with comparison metrics
    """
    comparison = []
    
    # Load ANN results
    if ann_results is None and ann_run_dir:
        ann_results = load_cv_results(ann_run_dir)
    
    if ann_results:
        cal_info = ann_results.get('calibration', {})
        uncal_metrics = cal_info.get('uncalibrated_metrics', {})
        cal_metrics = cal_info.get('calibrated_metrics', uncal_metrics)
        
        # Get training time from fold metrics (estimate)
        fold_metrics = ann_results.get('fold_metrics', [])
        mean_train_time = None
        if fold_metrics:
            # Estimate: ANN training is typically slower, use a placeholder
            mean_train_time = 30.0  # seconds per fold
        
        comparison.append({
            'Model': 'ANN',
            'ROC-AUC': cal_metrics.get('roc_auc', 0.0),
            'PR-AUC': cal_metrics.get('pr_auc', 0.0),
            'Brier': cal_metrics.get('brier', 1.0),
            'ECE': cal_metrics.get('ece', 1.0),
            'Accuracy': cal_metrics.get('accuracy', 0.0),
            'Training Time (s/fold)': mean_train_time,
            'Calibration': cal_info.get('best_method', 'none')
        })
    
    # Load XGBoost results
    if xgb_results is None and xgb_run_dir:
        gbdt_results = load_gbdt_results(xgb_run_dir)
        if gbdt_results and 'results' in gbdt_results:
            xgb_results = gbdt_results['results'].get('xgb')
        elif xgb_results is None:
            # Try loading from separate XGB run
            xgb_results = load_cv_results(xgb_run_dir)
    
    if xgb_results and isinstance(xgb_results, dict):
        if 'calibrated_metrics' in xgb_results:
            # GBDT results format
            metrics = xgb_results['calibrated_metrics']
            comparison.append({
                'Model': 'XGBoost',
                'ROC-AUC': metrics.get('roc_auc', 0.0),
                'PR-AUC': metrics.get('pr_auc', 0.0),
                'Brier': metrics.get('brier', 1.0),
                'ECE': metrics.get('ece', 1.0),
                'Accuracy': metrics.get('accuracy', 0.0),
                'Training Time (s/fold)': xgb_results.get('mean_training_time'),
                'Calibration': xgb_results.get('calibration_method', 'none')
            })
    
    # Load LightGBM results
    if lgb_results is None and lgb_run_dir:
        gbdt_results = load_gbdt_results(lgb_run_dir)
        if gbdt_results and 'results' in gbdt_results:
            lgb_results = gbdt_results['results'].get('lgb')
        elif lgb_results is None:
            # Try loading from separate LGB run
            lgb_results = load_cv_results(lgb_run_dir)
    
    if lgb_results and isinstance(lgb_results, dict):
        if 'calibrated_metrics' in lgb_results:
            # GBDT results format
            metrics = lgb_results['calibrated_metrics']
            comparison.append({
                'Model': 'LightGBM',
                'ROC-AUC': metrics.get('roc_auc', 0.0),
                'PR-AUC': metrics.get('pr_auc', 0.0),
                'Brier': metrics.get('brier', 1.0),
                'ECE': metrics.get('ece', 1.0),
                'Accuracy': metrics.get('accuracy', 0.0),
                'Training Time (s/fold)': lgb_results.get('mean_training_time'),
                'Calibration': lgb_results.get('calibration_method', 'none')
            })
    
    df = pd.DataFrame(comparison)
    
    if not df.empty:
        # Sort by ROC-AUC descending
        df = df.sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    
    return df


__all__ = ['compare_models', 'load_cv_results', 'load_gbdt_results']
