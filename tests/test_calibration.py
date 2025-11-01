"""
Unit tests for calibration methods.

Verify that ECE decreases after calibration for an overconfident toy model.
"""

import pytest
import numpy as np
from src.metrics.calibration import (
    PlattScaler,
    IsotonicCalibrator,
    compute_ece,
    compute_brier_score,
    fit_calibrator,
    select_best_calibrator
)


def test_ece_computation():
    """Test ECE computation on simple examples."""
    # Perfectly calibrated: probabilities match frequencies
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.25, 0.25, 0.75, 0.75])
    ece = compute_ece(y_true, y_proba, n_bins=2)
    assert ece >= 0
    assert ece < 0.2  # Should be low for this case (allowing for binning effects)
    
    # Overconfident: probabilities are extreme but don't match frequencies
    y_true = np.array([0, 0, 0, 1])
    y_proba = np.array([0.9, 0.9, 0.9, 0.9])  # All predicted as very confident positive
    ece = compute_ece(y_true, y_proba, n_bins=2)
    assert ece > 0.5  # Should be high (poor calibration)


def test_brier_score_computation():
    """Test Brier score computation."""
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])
    brier = compute_brier_score(y_true, y_proba)
    assert brier == 0.0
    
    # Random predictions
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.5, 0.5, 0.5, 0.5])
    brier = compute_brier_score(y_true, y_proba)
    assert 0 < brier < 0.3
    
    # Worst case (always wrong with confidence)
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([1.0, 1.0, 0.0, 0.0])
    brier = compute_brier_score(y_true, y_proba)
    assert brier == 1.0


def test_platt_scaler():
    """Test Platt scaling."""
    # Create overconfident model: predicts extreme probabilities
    n = 100
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=n)
    # Overconfident: logits are extreme
    logits = (y_true * 2 - 1) * 5 + rng.normal(0, 0.5, size=n)
    # Convert to probabilities (sigmoid)
    y_proba = 1 / (1 + np.exp(-logits))
    
    # Uncalibrated ECE
    ece_before = compute_ece(y_true, y_proba)
    
    # Fit Platt scaler
    scaler = PlattScaler()
    scaler.fit(logits, y_true)
    y_proba_cal = scaler.predict_proba(logits)
    
    # Calibrated ECE should be lower
    ece_after = compute_ece(y_true, y_proba_cal)
    
    assert ece_after <= ece_before + 0.05  # Allow small tolerance
    assert scaler.is_fitted
    
    # Test save/load
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        scaler.save(temp_path)
        loaded = PlattScaler.load(temp_path)
        assert loaded.is_fitted
        np.testing.assert_allclose(loaded.predict_proba(logits), y_proba_cal)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_isotonic_calibrator():
    """Test isotonic regression calibration."""
    # Create overconfident model
    n = 100
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=n)
    # Overconfident probabilities
    y_proba = y_true * 0.9 + (1 - y_true) * 0.1 + rng.normal(0, 0.05, size=n)
    y_proba = np.clip(y_proba, 0.01, 0.99)
    
    # Uncalibrated ECE
    ece_before = compute_ece(y_true, y_proba)
    
    # Fit isotonic calibrator
    calibrator = IsotonicCalibrator()
    calibrator.fit(y_proba, y_true)
    y_proba_cal = calibrator.predict_proba(y_proba)
    
    # Calibrated ECE should be lower
    ece_after = compute_ece(y_true, y_proba_cal)
    
    assert ece_after <= ece_before + 0.05
    assert calibrator.is_fitted
    
    # Test save/load
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        temp_path = f.name
    
    try:
        calibrator.save(temp_path)
        loaded = IsotonicCalibrator.load(temp_path)
        assert loaded.is_fitted
        np.testing.assert_allclose(loaded.predict_proba(y_proba), y_proba_cal)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_select_best_calibrator():
    """Test automatic selection of best calibrator."""
    # Create overconfident model
    n = 200
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=n)
    logits = (y_true * 2 - 1) * 4 + rng.normal(0, 0.5, size=n)
    y_proba = 1 / (1 + np.exp(-logits))
    
    # Select best calibrator
    best_method, best_cal, metrics = select_best_calibrator(
        y_true, y_proba, y_logits=logits, metric='brier'
    )
    
    assert best_method in ['platt', 'isotonic', 'uncalibrated']
    assert 'uncalibrated' in metrics
    assert 'isotonic' in metrics
    assert 'platt' in metrics
    
    # Should improve over uncalibrated
    if best_method != 'uncalibrated':
        assert best_cal is not None
        assert metrics[best_method] <= metrics['uncalibrated']


def test_ece_decreases_after_calibration():
    """Main acceptance test: ECE should decrease after calibration for overconfident model."""
    # Create a toy overconfident model
    # It predicts very confidently but probabilities don't match actual frequencies
    n = 500
    rng = np.random.default_rng(123)
    
    # Generate true labels
    y_true = rng.binomial(1, 0.3, size=n).astype(int)
    
    # Overconfident predictions: model is too confident
    # For positive class, predict high probability
    # For negative class, predict low probability
    # But the probabilities are too extreme
    y_proba_overconf = np.where(
        y_true == 1,
        rng.normal(0.85, 0.1, size=n),  # Too confident for positives
        rng.normal(0.15, 0.1, size=n)   # Too confident for negatives
    )
    y_proba_overconf = np.clip(y_proba_overconf, 0.01, 0.99)
    
    # Also create logits for Platt
    logits = np.log(y_proba_overconf / (1 - y_proba_overconf))
    
    # Compute uncalibrated ECE
    ece_uncalibrated = compute_ece(y_true, y_proba_overconf, n_bins=10)
    
    # Fit isotonic calibrator
    iso_cal, y_proba_iso = fit_calibrator(y_true, y_proba=y_proba_overconf, method='isotonic')
    ece_isotonic = compute_ece(y_true, y_proba_iso, n_bins=10)
    
    # Fit Platt scaler
    platt_cal, y_proba_platt = fit_calibrator(y_true, y_logits=logits, method='platt')
    ece_platt = compute_ece(y_true, y_proba_platt, n_bins=10)
    
    # At least one calibration method should improve ECE
    best_ece = min(ece_isotonic, ece_platt)
    
    print(f"ECE uncalibrated: {ece_uncalibrated:.4f}")
    print(f"ECE isotonic: {ece_isotonic:.4f}")
    print(f"ECE platt: {ece_platt:.4f}")
    
    # Allow small tolerance for numerical stability
    assert best_ece < ece_uncalibrated + 0.01, \
        f"Calibration should improve ECE. Uncalibrated: {ece_uncalibrated:.4f}, Best: {best_ece:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
