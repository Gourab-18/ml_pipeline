"""
Unit tests for cross-validation orchestration.
Checks OOF shape and reproducibility with a fixed seed.
"""

import os
import json
import numpy as np
import pandas as pd

from src.training.cv import run_kfold_cv


def _make_toy_data(n: int = 128, seed: int = 123):
    rng = np.random.default_rng(seed)
    # Features: 1 numeric, 1 onehot(3), 1 embed(int index)
    # Layout per feature_info below: [num] + [onehot3] + [embed_index]
    num = rng.normal(size=(n, 1))
    onehot_idx = rng.integers(0, 3, size=(n, 1))
    onehot = np.zeros((n, 3))
    onehot[np.arange(n), onehot_idx.flatten()] = 1.0
    embed_idx = rng.integers(0, 10, size=(n, 1))
    X = np.concatenate([num, onehot, embed_idx], axis=1).astype(float)
    # Simple target correlated with num and onehot_idx
    y = (num.flatten() + (onehot_idx.flatten() == 1).astype(float) + rng.normal(scale=0.3, size=n) > 0.0).astype(int)
    # feature_info defines parsing of X
    feature_info = {
        'f_num': {'action': 'scale', 'vocabulary_size': 0},
        'f_oh': {'action': 'onehot', 'vocabulary_size': 3},
        'f_emb': {'action': 'embed', 'vocabulary_size': 10},
    }
    return X, y, feature_info


def test_oof_shape_and_files(tmp_path):
    X, y, feature_info = _make_toy_data(n=96, seed=7)
    run_dir = tmp_path / "cv_artifacts"
    summary = run_kfold_cv(
        X=X,
        y=y,
        feature_info=feature_info,
        k_folds=3,
        seed=123,
        artifacts_dir=str(run_dir),
        run_name="test_run",
        base_params={"epochs": 5, "batch_size": 32},
    )

    # OOF CSV exists and shape matches
    oof_csv = os.path.join(summary["run_dir"], "oof_predictions.csv")
    assert os.path.exists(oof_csv)
    oof_df = pd.read_csv(oof_csv)
    assert len(oof_df) == X.shape[0]
    assert {"id", "y_true", "oof_prob", "oof_logit"}.issubset(oof_df.columns)

    # Summary json exists
    assert os.path.exists(os.path.join(summary["run_dir"], "summary.json"))

    # Per-fold directories and metrics
    for fold in range(3):
        fold_dir = os.path.join(summary["run_dir"], f"fold_{fold}")
        assert os.path.exists(os.path.join(fold_dir, "metrics.json"))
        assert os.path.exists(os.path.join(fold_dir, "model.weights.h5"))


def test_reproducibility_with_seed(tmp_path):
    X, y, feature_info = _make_toy_data(n=64, seed=9)
    run_root = tmp_path / "cv_artifacts"

    s1 = run_kfold_cv(
        X=X, y=y, feature_info=feature_info, k_folds=3, seed=777,
        artifacts_dir=str(run_root), run_name="run1", base_params={"epochs": 3, "batch_size": 32}
    )
    s2 = run_kfold_cv(
        X=X, y=y, feature_info=feature_info, k_folds=3, seed=777,
        artifacts_dir=str(run_root), run_name="run2", base_params={"epochs": 3, "batch_size": 32}
    )

    oof1 = pd.read_csv(os.path.join(s1["run_dir"], "oof_predictions.csv"))
    oof2 = pd.read_csv(os.path.join(s2["run_dir"], "oof_predictions.csv"))
    # With same seed and same data, OOF should be identical in ordering and values
    assert np.allclose(oof1["oof_prob"].values, oof2["oof_prob"].values, atol=1e-6)


