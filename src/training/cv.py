"""
Cross-validation orchestration with optional inner-loop HPO.

Features:
- Outer KFold producing OOF predictions and per-fold artifacts
- Optional inner CV/Random-Grid search across a few hyperparameters
- Reproducibility via random seed
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import KFold

import tensorflow as tf

from src.models.tabular_ann import create_tabular_ann, TabularANN
from src.models.train_utils import (
    create_callbacks,
    train_tabular_ann,
    evaluate_model,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _serialize_normalizer(layer: tf.keras.layers.Layer) -> Dict[str, Any]:
    """Serialize a Keras normalizer layer (config + weights)."""
    try:
        config = layer.get_config()
    except Exception:
        config = {}
    try:
        weights = [w.tolist() for w in layer.get_weights()]
    except Exception:
        weights = []
    return {"class_name": layer.__class__.__name__, "config": config, "weights": weights}


def _save_normalizers(normalizers: Dict[str, tf.keras.layers.Layer], path: str) -> None:
    data = {name: _serialize_normalizer(layer) for name, layer in normalizers.items()}
    _save_json(data, path)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _parameter_grid(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Create cartesian product grid from dict of lists."""
    if not search_space:
        return [{}]
    keys = list(search_space.keys())
    values_lists = [search_space[k] for k in keys]
    grid: List[Dict[str, Any]] = []
    def backtrack(i: int, current: Dict[str, Any]):
        if i == len(keys):
            grid.append(current.copy())
            return
        k = keys[i]
        for v in values_lists[i]:
            current[k] = v
            backtrack(i + 1, current)
            current.pop(k, None)
    backtrack(0, {})
    return grid


def _evaluate_hparams(
    X: np.ndarray,
    y: np.ndarray,
    feature_info: Dict[str, Any],
    base_params: Dict[str, Any],
    search_space: Dict[str, List[Any]],
    inner_splits: int,
    seed: int,
) -> Dict[str, Any]:
    """Very lightweight inner CV over a small grid. Returns best params by mean val accuracy."""
    if not search_space:
        return base_params

    kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    candidates = _parameter_grid(search_space)

    best_score = -1.0
    best_params = base_params.copy()

    for cand in candidates:
        scores: List[float] = []
        for inner_train_idx, inner_val_idx in kf.split(X):
            _set_seed(seed)
            params = {**base_params, **cand}
            model: TabularANN = create_tabular_ann(
                feature_info=feature_info,
                embedding_dims=params.get("embedding_dims"),
                hidden_layers=params.get("hidden_layers", [128, 64, 32]),
                dropout_rate=params.get("dropout_rate", 0.3),
                l2_reg=params.get("l2_reg", 0.001),
                learning_rate=params.get("learning_rate", 0.001),
                random_seed=seed,
                numeric_normalizers=params.get("numeric_normalizers"),
            )

            # quick train
            history = train_tabular_ann(
                model=model,
                X=X[inner_train_idx],
                y=y[inner_train_idx],
                feature_info=feature_info,
                epochs=params.get("epochs", 30),
                batch_size=params.get("batch_size", 64),
                validation_split=0.2,
                callbacks=create_callbacks(patience=5, verbose=0),
                verbose=0,
                random_seed=seed,
            )
            # evaluate on held-out inner validation set
            metrics = evaluate_model(model, X[inner_val_idx], y[inner_val_idx], feature_info)
            scores.append(float(metrics.get("accuracy", 0.0)))

        mean_score = float(np.mean(scores)) if scores else 0.0
        if mean_score > best_score:
            best_score = mean_score
            best_params = {**base_params, **cand}

    return best_params


def run_kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_info: Dict[str, Any],
    k_folds: int = 5,
    seed: int = 42,
    artifacts_dir: str = "artifacts/cv",
    run_name: Optional[str] = None,
    base_params: Optional[Dict[str, Any]] = None,
    hpo_search_space: Optional[Dict[str, List[Any]]] = None,
    hpo_inner_splits: int = 3,
    sample_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Run outer KFold CV with optional inner-loop HPO.

    Saves per-fold artifacts and an OOF predictions CSV.
    Returns a summary dict with paths and metrics.
    """
    _set_seed(seed)

    n_samples = X.shape[0]
    if sample_ids is None:
        sample_ids = np.arange(n_samples)
    if base_params is None:
        base_params = {}

    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, run_name)
    _ensure_dir(run_dir)

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    oof_probs = np.zeros((n_samples, 1), dtype=float)
    oof_logits = np.zeros((n_samples, 1), dtype=float)
    fold_metrics: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
        _ensure_dir(fold_dir)

        # Inner HPO
        chosen_params = base_params
        if hpo_search_space:
            chosen_params = _evaluate_hparams(
                X[train_idx], y[train_idx], feature_info, base_params, hpo_search_space, hpo_inner_splits, seed
            )

        # Build model with chosen params
        model: TabularANN = create_tabular_ann(
            feature_info=feature_info,
            embedding_dims=chosen_params.get("embedding_dims"),
            hidden_layers=chosen_params.get("hidden_layers", [128, 64, 32]),
            dropout_rate=chosen_params.get("dropout_rate", 0.3),
            l2_reg=chosen_params.get("l2_reg", 0.001),
            learning_rate=chosen_params.get("learning_rate", 0.001),
            random_seed=seed,
            numeric_normalizers=chosen_params.get("numeric_normalizers"),
        )

        # Train on fold
        train_summary = train_tabular_ann(
            model=model,
            X=X[train_idx],
            y=y[train_idx],
            feature_info=feature_info,
            epochs=chosen_params.get("epochs", 100),
            batch_size=chosen_params.get("batch_size", 64),
            validation_split=0.2,
            callbacks=create_callbacks(patience=10, verbose=1),
            verbose=1,
            random_seed=seed,
        )

        # Predict on validation fold
        probs = model.predict_proba(X[val_idx]).reshape(-1, 1)
        logits = model.predict_logits(X[val_idx]).reshape(-1, 1)
        oof_probs[val_idx] = probs
        oof_logits[val_idx] = logits

        # Evaluate and save metrics
        metrics = evaluate_model(model, X[val_idx], y[val_idx], feature_info)
        metrics.update({"fold": fold_idx, "n_val": int(len(val_idx))})
        _save_json(metrics, os.path.join(fold_dir, "metrics.json"))

        # Save model weights
        model_path = os.path.join(fold_dir, "model.weights.h5")
        model.model.save_weights(model_path)

        # Save feature_info and chosen params
        _save_json({"feature_info": feature_info, "params": chosen_params, "train_summary": train_summary}, os.path.join(fold_dir, "metadata.json"))

        # Save normalizers if any
        numeric_normalizers = chosen_params.get("numeric_normalizers", {}) or {}
        if numeric_normalizers:
            _save_normalizers(numeric_normalizers, os.path.join(fold_dir, "normalizers.json"))

        fold_metrics.append(metrics)

    # Save OOF predictions CSV
    oof_df = pd.DataFrame({
        "id": sample_ids,
        "y_true": y.astype(int),
        "oof_prob": oof_probs.reshape(-1),
        "oof_logit": oof_logits.reshape(-1),
    })
    oof_csv_path = os.path.join(run_dir, "oof_predictions.csv")
    oof_df.to_csv(oof_csv_path, index=False)

    # Summary
    summary = {
        "run_dir": run_dir,
        "k_folds": k_folds,
        "seed": seed,
        "oof_path": oof_csv_path,
        "fold_metrics": fold_metrics,
        "mean_accuracy": float(np.mean([m.get("accuracy", 0.0) for m in fold_metrics])) if fold_metrics else 0.0,
    }
    _save_json(summary, os.path.join(run_dir, "summary.json"))

    return summary


__all__ = ["run_kfold_cv"]


