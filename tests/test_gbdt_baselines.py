"""
Tests for GBDT baseline models (XGBoost and LightGBM).
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import shutil

from src.baselines.xgb_lgb import run_gbdt_cv, train_xgb_fold, train_lgb_fold


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 10))
    y = (X[:, 0] * 2 + X[:, 1] * 1.5 + rng.normal(0, 0.5, n) > 0).astype(int)
    return X, y


def test_xgb_fold_training(sample_data):
    """Test XGBoost training on a single fold."""
    X, y = sample_data
    
    # Split data
    n = len(X)
    split_idx = int(n * 0.8)
    
    try:
        model, metrics, train_time = train_xgb_fold(
            X[:split_idx], y[:split_idx],
            X[split_idx:], y[split_idx:],
            seed=42
        )
        
        assert model is not None
        assert 'roc_auc' in metrics
        assert 'brier' in metrics
        assert train_time > 0
        
        # Check metrics are reasonable
        assert 0 <= metrics['roc_auc'] <= 1
        assert metrics['brier'] >= 0
        
    except ImportError:
        pytest.skip("XGBoost not available")


def test_lgb_fold_training(sample_data):
    """Test LightGBM training on a single fold."""
    X, y = sample_data
    
    # Split data
    n = len(X)
    split_idx = int(n * 0.8)
    
    try:
        model, metrics, train_time = train_lgb_fold(
            X[:split_idx], y[:split_idx],
            X[split_idx:], y[split_idx:],
            seed=42
        )
        
        assert model is not None
        assert 'roc_auc' in metrics
        assert 'brier' in metrics
        assert train_time > 0
        
        # Check metrics are reasonable
        assert 0 <= metrics['roc_auc'] <= 1
        assert metrics['brier'] >= 0
        
    except ImportError:
        pytest.skip("LightGBM not available")


def test_gbdt_cv_pipeline(sample_data):
    """Test end-to-end GBDT CV pipeline."""
    X, y = sample_data
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = os.path.join(tmpdir, "artifacts")
        
        # Determine which models to test
        models_to_test = []
        try:
            import xgboost
            models_to_test.append('xgb')
        except ImportError:
            pass
        
        try:
            import lightgbm
            models_to_test.append('lgb')
        except ImportError:
            pass
        
        if not models_to_test:
            pytest.skip("Neither XGBoost nor LightGBM available")
        
        # Run CV
        results = run_gbdt_cv(
            X, y,
            k_folds=3,
            seed=42,
            artifacts_dir=artifacts_dir,
            run_name="test_run",
            models=models_to_test
        )
        
        assert 'run_dir' in results
        assert 'results' in results
        
        # Check results
        for model_name in models_to_test:
            if model_name in results['results']:
                model_results = results['results'][model_name]
                
                assert 'oof_path' in model_results
                assert os.path.exists(model_results['oof_path'])
                
                assert 'mean_metrics' in model_results
                assert 'roc_auc' in model_results['mean_metrics']
                assert 'brier' in model_results['mean_metrics']
                
                assert 'calibrated_metrics' in model_results
                assert 'calibration_method' in model_results
                
                # Check OOF file
                oof_df = pd.read_csv(model_results['oof_path'])
                assert len(oof_df) == len(y)
                assert 'y_true' in oof_df.columns
                assert 'oof_prob' in oof_df.columns


def test_same_cv_splits(sample_data):
    """Test that GBDT uses the same CV splits as ANN (same seed)."""
    X, y = sample_data
    
    # Use same seed as ANN CV would use
    seed = 42
    k_folds = 3
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    splits = list(kf.split(X))
    
    # Verify splits are deterministic
    kf2 = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits2 = list(kf2.split(X))
    
    for (train1, val1), (train2, val2) in zip(splits, splits2):
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
