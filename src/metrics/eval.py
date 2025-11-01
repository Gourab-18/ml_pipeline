"""
Evaluation metrics and calibration curve plotting.

Computes ROC-AUC, PR-AUC, Brier score, ECE, and saves calibration curves.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import os
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)
from sklearn.calibration import calibration_curve

from .calibration import compute_ece, compute_brier_score


def compute_all_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for binary classification.
    
    Args:
        y_true: True binary labels (shape: (n_samples,))
        y_proba: Predicted probabilities (shape: (n_samples,))
        n_bins: Number of bins for ECE computation
    
    Returns:
        Dictionary of metrics
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_proba.ndim > 1:
        y_proba = y_proba.flatten()
    
    metrics = {}
    
    # ROC-AUC
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        metrics['roc_auc'] = 0.0  # Single class case
    
    # PR-AUC (Average Precision)
    try:
        metrics['pr_auc'] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        metrics['pr_auc'] = 0.0
    
    # Brier score
    metrics['brier'] = compute_brier_score(y_true, y_proba)
    
    # ECE
    metrics['ece'] = compute_ece(y_true, y_proba, n_bins=n_bins)
    
    # Accuracy at 0.5 threshold
    y_pred = (y_proba > 0.5).astype(int)
    metrics['accuracy'] = float(np.mean(y_pred == y_true))
    
    return metrics


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    name: str = "Model",
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        name: Name for legend
        n_bins: Number of bins
        save_path: Optional path to save figure
    
    Returns:
        Tuple of (figure, axes)
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_proba.ndim > 1:
        y_proba = y_proba.flatten()
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives, 's-', label=name)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(f'Calibration Curve ({name})', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    name: str = "Model",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        name: Name for legend
        save_path: Optional path to save figure
    
    Returns:
        Tuple of (figure, axes)
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_proba.ndim > 1:
        y_proba = y_proba.flatten()
    
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        # Single class case
        fpr = np.array([0, 1])
        tpr = np.array([0, 1])
        roc_auc = 0.0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    name: str = "Model",
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        name: Name for legend
        save_path: Optional path to save figure
    
    Returns:
        Tuple of (figure, axes)
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_proba.ndim > 1:
        y_proba = y_proba.flatten()
    
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        precision = np.array([1, 0])
        recall = np.array([0, 1])
        pr_auc = 0.0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, lw=2, label=f'{name} (AP = {pr_auc:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def evaluate_and_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_dir: str,
    prefix: str = "eval",
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute all metrics and generate all plots.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities
        output_dir: Directory to save plots and metrics
        prefix: Prefix for output files
        n_bins: Number of bins for ECE and calibration curve
    
    Returns:
        Dictionary with metrics and paths to saved plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Compute metrics
    metrics = compute_all_metrics(y_true, y_proba, n_bins=n_bins)
    
    # Save metrics to JSON
    import json
    metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    cal_path = os.path.join(output_dir, f"{prefix}_calibration_curve.png")
    roc_path = os.path.join(output_dir, f"{prefix}_roc_curve.png")
    pr_path = os.path.join(output_dir, f"{prefix}_pr_curve.png")
    
    plot_calibration_curve(y_true, y_proba, name=prefix, n_bins=n_bins, save_path=cal_path)
    plt.close()
    
    plot_roc_curve(y_true, y_proba, name=prefix, save_path=roc_path)
    plt.close()
    
    plot_pr_curve(y_true, y_proba, name=prefix, save_path=pr_path)
    plt.close()
    
    return {
        'metrics': metrics,
        'metrics_path': metrics_path,
        'calibration_curve_path': cal_path,
        'roc_curve_path': roc_path,
        'pr_curve_path': pr_path
    }


__all__ = [
    'compute_all_metrics',
    'plot_calibration_curve',
    'plot_roc_curve',
    'plot_pr_curve',
    'evaluate_and_plot'
]
