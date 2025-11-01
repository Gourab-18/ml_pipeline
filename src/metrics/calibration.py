"""
Calibration methods for probability predictions.

Implements:
- Platt scaling (LogisticRegression on logits)
- Isotonic regression
- Expected Calibration Error (ECE)
- Brier score
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import pickle
import os


class PlattScaler:
    """Platt scaling: fit a logistic regression on logits to calibrate probabilities."""
    
    def __init__(self):
        self.calibrator = LogisticRegression(max_iter=1000, solver='lbfgs')
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> 'PlattScaler':
        """
        Fit Platt scaler on logits and true labels.
        
        Args:
            logits: Raw logit predictions (shape: (n_samples,))
            y_true: True binary labels (shape: (n_samples,))
        
        Returns:
            Self for method chaining
        """
        if logits.ndim > 1:
            logits = logits.flatten()
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        
        self.calibrator.fit(logits.reshape(-1, 1), y_true)
        self.is_fitted = True
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities from logits.
        
        Args:
            logits: Raw logit predictions (shape: (n_samples,))
        
        Returns:
            Calibrated probabilities (shape: (n_samples,))
        """
        if not self.is_fitted:
            raise ValueError("PlattScaler must be fitted before predict_proba")
        
        if logits.ndim > 1:
            logits = logits.flatten()
        
        proba = self.calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        return proba
    
    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'PlattScaler':
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class IsotonicCalibrator:
    """Isotonic regression calibrator for probability calibration."""
    
    def __init__(self, out_of_bounds: str = 'clip'):
        """
        Args:
            out_of_bounds: How to handle out-of-bounds predictions. Options: 'clip', 'nan', 'raise'
        """
        self.calibrator = IsotonicRegression(out_of_bounds=out_of_bounds)
        self.out_of_bounds = out_of_bounds
        self.is_fitted = False
    
    def fit(self, probas: np.ndarray, y_true: np.ndarray) -> 'IsotonicCalibrator':
        """
        Fit isotonic calibrator on probabilities and true labels.
        
        Args:
            probas: Raw probability predictions (shape: (n_samples,))
            y_true: True binary labels (shape: (n_samples,))
        
        Returns:
            Self for method chaining
        """
        if probas.ndim > 1:
            probas = probas.flatten()
        if y_true.ndim > 1:
            y_true = y_true.flatten()
        
        self.calibrator.fit(probas, y_true)
        self.is_fitted = True
        return self
    
    def predict_proba(self, probas: np.ndarray) -> np.ndarray:
        """
        Predict calibrated probabilities from raw probabilities.
        
        Args:
            probas: Raw probability predictions (shape: (n_samples,))
        
        Returns:
            Calibrated probabilities (shape: (n_samples,))
        """
        if not self.is_fitted:
            raise ValueError("IsotonicCalibrator must be fitted before predict_proba")
        
        if probas.ndim > 1:
            probas = probas.flatten()
        
        return self.calibrator.predict(probas)
    
    def save(self, path: str) -> None:
        """Save calibrator to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'IsotonicCalibrator':
        """Load calibrator from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def compute_ece(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well-calibrated probabilities are by binning predictions
    and comparing mean predicted probability to mean actual frequency in each bin.
    
    Args:
        y_true: True binary labels (shape: (n_samples,))
        y_proba: Predicted probabilities (shape: (n_samples,))
        n_bins: Number of bins for calibration evaluation
    
    Returns:
        ECE score (lower is better)
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_proba.ndim > 1:
        y_proba = y_proba.flatten()
    
    # Get calibration curve data
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    # Assign samples to bins
    bin_indices = np.digitize(y_proba, bin_boundaries[1:])
    
    # Compute ECE: weighted average of absolute differences
    ece = 0.0
    for i in range(n_bins):
        mask = (bin_indices == i + 1)
        if mask.sum() > 0:
            bin_weight = mask.sum() / len(y_proba)
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_proba[mask].mean()
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).
    
    Args:
        y_true: True binary labels (shape: (n_samples,))
        y_proba: Predicted probabilities (shape: (n_samples,))
    
    Returns:
        Brier score (lower is better)
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    if y_proba.ndim > 1:
        y_proba = y_proba.flatten()
    
    return float(np.mean((y_proba - y_true) ** 2))


def fit_calibrator(
    y_true: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    y_logits: Optional[np.ndarray] = None,
    method: str = 'platt',
    out_of_bounds: str = 'clip'
) -> Tuple[Any, np.ndarray]:
    """
    Fit a calibrator and return calibrated probabilities.
    
    Args:
        y_true: True binary labels
        y_proba: Raw probability predictions (for isotonic)
        y_logits: Raw logit predictions (for Platt)
        method: 'platt' or 'isotonic'
        out_of_bounds: For isotonic, how to handle out-of-bounds predictions
    
    Returns:
        Tuple of (calibrator, calibrated_probabilities)
    """
    if method == 'platt':
        if y_logits is None:
            raise ValueError("y_logits required for Platt scaling")
        calibrator = PlattScaler()
        calibrator.fit(y_logits, y_true)
        calibrated_proba = calibrator.predict_proba(y_logits)
    
    elif method == 'isotonic':
        if y_proba is None:
            raise ValueError("y_proba required for isotonic calibration")
        calibrator = IsotonicCalibrator(out_of_bounds=out_of_bounds)
        calibrator.fit(y_proba, y_true)
        calibrated_proba = calibrator.predict_proba(y_proba)
    
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    return calibrator, calibrated_proba


def select_best_calibrator(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_logits: Optional[np.ndarray] = None,
    metric: str = 'brier',
    n_bins: int = 10
) -> Tuple[str, Any, Dict[str, float]]:
    """
    Fit both calibrators and select the one with the best metric.
    
    Args:
        y_true: True binary labels
        y_proba: Raw probability predictions
        y_logits: Raw logit predictions (optional, for Platt)
        metric: 'brier' or 'ece'
        n_bins: Number of bins for ECE computation
    
    Returns:
        Tuple of (best_method, best_calibrator, metrics_dict)
    """
    metrics = {}
    
    # Baseline (uncalibrated)
    if metric == 'brier':
        baseline_score = compute_brier_score(y_true, y_proba)
    else:
        baseline_score = compute_ece(y_true, y_proba, n_bins=n_bins)
    metrics['uncalibrated'] = baseline_score
    
    # Try Platt scaling if logits available
    if y_logits is not None:
        platt_cal, platt_proba = fit_calibrator(y_true, y_logits=y_logits, method='platt')
        if metric == 'brier':
            platt_score = compute_brier_score(y_true, platt_proba)
        else:
            platt_score = compute_ece(y_true, platt_proba, n_bins=n_bins)
        metrics['platt'] = platt_score
    else:
        platt_cal = None
        platt_score = float('inf')
    
    # Try Isotonic
    iso_cal, iso_proba = fit_calibrator(y_true, y_proba=y_proba, method='isotonic')
    if metric == 'brier':
        iso_score = compute_brier_score(y_true, iso_proba)
    else:
        iso_score = compute_ece(y_true, iso_proba, n_bins=n_bins)
    metrics['isotonic'] = iso_score
    
    # Select best
    if platt_cal is not None and platt_score < iso_score and platt_score < baseline_score:
        return 'platt', platt_cal, metrics
    elif iso_score < baseline_score:
        return 'isotonic', iso_cal, metrics
    else:
        return 'uncalibrated', None, metrics


__all__ = [
    'PlattScaler',
    'IsotonicCalibrator',
    'compute_ece',
    'compute_brier_score',
    'fit_calibrator',
    'select_best_calibrator'
]
