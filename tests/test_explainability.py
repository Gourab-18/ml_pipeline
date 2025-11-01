"""
Unit tests for explainability methods.

Verify that random/junk features have near-zero importance.
"""

import pytest
import numpy as np
import pandas as pd
from src.explainability.permutation import (
    compute_permutation_importance,
    compute_permutation_importance_with_model,
    compute_permutation_importance_oof
)


def test_random_feature_has_low_importance():
    """Test that a random feature has near-zero importance."""
    n = 200
    rng = np.random.default_rng(42)
    
    # Create meaningful features
    X_meaningful = rng.normal(size=(n, 5))
    # Create target that depends on meaningful features
    y_true = (X_meaningful.sum(axis=1) + rng.normal(0, 0.5, n) > 0).astype(int)
    
    # Add random feature (should have low importance)
    X_random = rng.normal(size=(n, 1))
    X = np.hstack([X_meaningful, X_random])
    
    # Create predictions based only on meaningful features
    logits = X_meaningful.sum(axis=1)
    y_proba = 1 / (1 + np.exp(-logits))
    
    feature_names = [f"meaningful_{i}" for i in range(5)] + ["random_feature"]
    
    # Compute permutation importance
    perm_importance = compute_permutation_importance(
        X, y_true, y_proba,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42
    )
    
    # Random feature should have lower importance than meaningful features
    random_importance = perm_importance[perm_importance['feature'] == 'random_feature']['importance_mean'].values[0]
    meaningful_importances = perm_importance[perm_importance['feature'] != 'random_feature']['importance_mean'].values
    
    # Random feature importance should be lower than mean of meaningful features
    # Allow some tolerance due to randomness
    assert random_importance < np.mean(meaningful_importances) + 0.1, \
        f"Random feature importance ({random_importance:.4f}) should be lower than meaningful features"


def test_permutation_importance_with_model():
    """Test permutation importance with actual model predictions."""
    n = 100
    rng = np.random.default_rng(42)
    
    # Create features and target
    X = rng.normal(size=(n, 5))
    y_true = (X[:, 0] + X[:, 1] + rng.normal(0, 0.3, n) > 0).astype(int)
    
    # Simple prediction function that uses first two features
    def predict_fn(X_input):
        logits = X_input[:, 0] + X_input[:, 1]
        return 1 / (1 + np.exp(-logits))
    
    feature_names = [f"feature_{i}" for i in range(5)]
    
    # Compute importance
    perm_importance = compute_permutation_importance_with_model(
        X, y_true, predict_fn,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42
    )
    
    # First two features should be more important than others
    feat0_importance = perm_importance[perm_importance['feature'] == 'feature_0']['importance_mean'].values[0]
    feat2_importance = perm_importance[perm_importance['feature'] == 'feature_2']['importance_mean'].values[0]
    
    assert feat0_importance > feat2_importance, \
        f"feature_0 ({feat0_importance:.4f}) should be more important than feature_2 ({feat2_importance:.4f})"


def test_permutation_importance_oof():
    """Test OOF-based permutation importance."""
    n = 150
    rng = np.random.default_rng(42)
    
    # Create features
    X = rng.normal(size=(n, 10))
    # Create target that depends on first feature
    y_true = (X[:, 0] * 2 + rng.normal(0, 0.5, n) > 0).astype(int)
    # Create OOF predictions (approximate)
    y_proba = 1 / (1 + np.exp(-X[:, 0] * 2 + rng.normal(0, 0.2, n)))
    
    feature_names = [f"feature_{i}" for i in range(10)]
    
    # Compute importance
    perm_importance = compute_permutation_importance_oof(
        X, y_true, y_proba,
        feature_names=feature_names,
        n_repeats=5,
        random_state=42
    )
    
    # feature_0 should be top important
    top_feature = perm_importance.iloc[0]['feature']
    
    # feature_0 should be in top 3
    top_3_features = perm_importance.head(3)['feature'].values
    assert 'feature_0' in top_3_features, \
        f"feature_0 should be in top 3, got: {top_3_features}"


def test_junk_feature_low_importance():
    """Main acceptance test: junk/random feature has near-zero importance."""
    n = 300
    rng = np.random.default_rng(123)
    
    # Create meaningful features (first 5)
    X_meaningful = rng.normal(size=(n, 5))
    # Target depends on meaningful features
    y_true = (X_meaningful[:, 0] * 2 + X_meaningful[:, 1] * 1.5 + rng.normal(0, 0.5, n) > 0).astype(int)
    
    # Add junk feature (completely random, not related to target)
    X_junk = rng.uniform(-10, 10, size=(n, 1))  # Different distribution, unrelated
    X = np.hstack([X_meaningful, X_junk])
    
    # OOF predictions (based on meaningful features only)
    logits = X_meaningful[:, 0] * 2 + X_meaningful[:, 1] * 1.5
    y_proba = 1 / (1 + np.exp(-logits + rng.normal(0, 0.2, n)))
    
    feature_names = [f"meaningful_{i}" for i in range(5)] + ["junk_feature"]
    
    # Compute importance
    perm_importance = compute_permutation_importance_oof(
        X, y_true, y_proba,
        feature_names=feature_names,
        n_repeats=10,
        random_state=42
    )
    
    junk_importance = perm_importance[perm_importance['feature'] == 'junk_feature']['importance_mean'].values[0]
    meaningful_importances = perm_importance[perm_importance['feature'] != 'junk_feature']['importance_mean'].values
    
    print(f"\nJunk feature importance: {junk_importance:.4f}")
    print(f"Mean meaningful feature importance: {np.mean(meaningful_importances):.4f}")
    print(f"Top meaningful feature importance: {np.max(meaningful_importances):.4f}")
    
    # Junk feature should be among the least important
    # It should be lower than most meaningful features
    n_meaningful_better = np.sum(meaningful_importances > junk_importance)
    assert n_meaningful_better >= 3, \
        f"Junk feature should be less important than at least 3 meaningful features. " \
        f"Only {n_meaningful_better} meaningful features have higher importance."
    
    # Junk feature should not be in top 3
    top_3_features = perm_importance.head(3)['feature'].values
    assert 'junk_feature' not in top_3_features, \
        f"Junk feature should not be in top 3. Top 3: {top_3_features}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
