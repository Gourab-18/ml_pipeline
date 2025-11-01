"""
XGBoost and LightGBM baseline models using the same CV splits as ANN.

Trains XGBoost and LightGBM on the same KFold splits, produces OOF predictions,
and applies calibration (Platt/Isotonic).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import time
from datetime import datetime
from sklearn.model_selection import KFold

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from src.metrics.calibration import select_best_calibrator
from src.metrics.eval import compute_all_metrics


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    if XGB_AVAILABLE:
        # XGBoost uses its own random seed
        pass


def train_xgb_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Train XGBoost on a single fold.
    
    Returns:
        Tuple of (model, metrics_dict, training_time)
    """
    if not XGB_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")
    
    if params is None:
        params = {}
    
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': seed,
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbosity': 0
    }
    default_params.update(params)
    
    start_time = time.time()
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = xgb.train(
        default_params,
        dtrain,
        num_boost_round=default_params.pop('n_estimators', 100),
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    training_time = time.time() - start_time
    
    # Predict on validation set
    y_pred_proba = model.predict(dval)
    
    # Compute metrics
    metrics = compute_all_metrics(y_val, y_pred_proba)
    
    return model, metrics, training_time


def train_lgb_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    seed: int = 42
) -> Tuple[Any, Dict[str, Any], float]:
    """
    Train LightGBM on a single fold.
    
    Returns:
        Tuple of (model, metrics_dict, training_time)
    """
    if not LGB_AVAILABLE:
        raise ImportError("LightGBM not available. Install with: pip install lightgbm")
    
    if params is None:
        params = {}
    
    default_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': seed
    }
    default_params.update(params)
    
    start_time = time.time()
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        default_params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        valid_names=['val'],
        callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]
    )
    
    training_time = time.time() - start_time
    
    # Predict on validation set
    y_pred_proba = model.predict(X_val)
    
    # Compute metrics
    metrics = compute_all_metrics(y_val, y_pred_proba)
    
    return model, metrics, training_time


def run_gbdt_cv(
    X: np.ndarray,
    y: np.ndarray,
    k_folds: int = 5,
    seed: int = 42,
    artifacts_dir: str = "artifacts/cv",
    run_name: Optional[str] = None,
    xgb_params: Optional[Dict[str, Any]] = None,
    lgb_params: Optional[Dict[str, Any]] = None,
    sample_ids: Optional[np.ndarray] = None,
    models: List[str] = ['xgb', 'lgb']
) -> Dict[str, Any]:
    """
    Run XGBoost and/or LightGBM CV using the same splits as ANN CV.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        k_folds: Number of CV folds
        seed: Random seed (must match ANN CV seed for same splits)
        artifacts_dir: Directory for artifacts
        run_name: Run name (if None, uses timestamp)
        xgb_params: XGBoost hyperparameters
        lgb_params: LightGBM hyperparameters
        sample_ids: Optional sample IDs
        models: List of models to train ['xgb', 'lgb']
    
    Returns:
        Dictionary with OOF predictions, metrics, and calibration info
    """
    _set_seed(seed)
    
    n_samples = X.shape[0]
    if sample_ids is None:
        sample_ids = np.arange(n_samples)
    
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(artifacts_dir, run_name)
    _ensure_dir(run_dir)
    
    # Create same KFold splits as ANN CV
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    
    results = {}
    
    # Train XGBoost
    if 'xgb' in models and XGB_AVAILABLE:
        print("\n" + "="*60)
        print("Training XGBoost baseline...")
        print("="*60)
        
        oof_probs_xgb = np.zeros((n_samples,), dtype=float)
        fold_metrics_xgb = []
        fold_times_xgb = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold_idx + 1}/{k_folds}")
            
            model, metrics, train_time = train_xgb_fold(
                X[train_idx], y[train_idx],
                X[val_idx], y[val_idx],
                params=xgb_params,
                seed=seed
            )
            
            # Predict on validation set
            dval = xgb.DMatrix(X[val_idx])
            y_pred_proba = model.predict(dval)
            oof_probs_xgb[val_idx] = y_pred_proba
            
            metrics['fold'] = fold_idx
            metrics['training_time'] = train_time
            fold_metrics_xgb.append(metrics)
            fold_times_xgb.append(train_time)
            
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}, Brier: {metrics['brier']:.4f}, Time: {train_time:.2f}s")
        
        # Calibrate XGBoost predictions
        print("\nCalibrating XGBoost predictions...")
        # XGBoost doesn't produce logits directly, so we'll use isotonic or platt on probabilities
        best_method_xgb, best_cal_xgb, cal_metrics_xgb = select_best_calibrator(
            y, oof_probs_xgb, y_logits=None,  # No logits for GBDT
            metric='brier',
            n_bins=10
        )
        
        if best_cal_xgb is not None:
            oof_probs_xgb_cal = best_cal_xgb.predict_proba(oof_probs_xgb)
            metrics_xgb_cal = compute_all_metrics(y, oof_probs_xgb_cal)
            print(f"✅ Best calibrator: {best_method_xgb}")
            print(f"   Brier: {cal_metrics_xgb['uncalibrated']:.4f} → {cal_metrics_xgb[best_method_xgb]:.4f}")
        else:
            oof_probs_xgb_cal = oof_probs_xgb
            metrics_xgb_cal = compute_all_metrics(y, oof_probs_xgb)
        
        # Save XGBoost results
        oof_df_xgb = pd.DataFrame({
            'id': sample_ids,
            'y_true': y.astype(int),
            'oof_prob': oof_probs_xgb,
            'oof_prob_calibrated': oof_probs_xgb_cal if best_cal_xgb is not None else oof_probs_xgb
        })
        oof_path_xgb = os.path.join(run_dir, "oof_xgb.csv")
        oof_df_xgb.to_csv(oof_path_xgb, index=False)
        
        results['xgb'] = {
            'oof_path': oof_path_xgb,
            'fold_metrics': fold_metrics_xgb,
            'mean_metrics': {
                'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics_xgb]),
                'pr_auc': np.mean([m['pr_auc'] for m in fold_metrics_xgb]),
                'brier': np.mean([m['brier'] for m in fold_metrics_xgb]),
                'ece': np.mean([m['ece'] for m in fold_metrics_xgb]),
                'accuracy': np.mean([m['accuracy'] for m in fold_metrics_xgb])
            },
            'calibrated_metrics': metrics_xgb_cal,
            'calibration_method': best_method_xgb,
            'calibration_metrics': cal_metrics_xgb,
            'mean_training_time': np.mean(fold_times_xgb),
            'total_training_time': np.sum(fold_times_xgb)
        }
        
        print(f"\n✅ XGBoost CV completed")
        print(f"   Mean ROC-AUC: {results['xgb']['mean_metrics']['roc_auc']:.4f}")
        print(f"   Mean Brier: {results['xgb']['mean_metrics']['brier']:.4f}")
        print(f"   Mean Training Time: {results['xgb']['mean_training_time']:.2f}s")
    
    # Train LightGBM
    if 'lgb' in models and LGB_AVAILABLE:
        print("\n" + "="*60)
        print("Training LightGBM baseline...")
        print("="*60)
        
        oof_probs_lgb = np.zeros((n_samples,), dtype=float)
        fold_metrics_lgb = []
        fold_times_lgb = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold_idx + 1}/{k_folds}")
            
            model, metrics, train_time = train_lgb_fold(
                X[train_idx], y[train_idx],
                X[val_idx], y[val_idx],
                params=lgb_params,
                seed=seed
            )
            
            # Predict on validation set
            y_pred_proba = model.predict(X[val_idx])
            oof_probs_lgb[val_idx] = y_pred_proba
            
            metrics['fold'] = fold_idx
            metrics['training_time'] = train_time
            fold_metrics_lgb.append(metrics)
            fold_times_lgb.append(train_time)
            
            print(f"  ROC-AUC: {metrics['roc_auc']:.4f}, Brier: {metrics['brier']:.4f}, Time: {train_time:.2f}s")
        
        # Calibrate LightGBM predictions
        print("\nCalibrating LightGBM predictions...")
        best_method_lgb, best_cal_lgb, cal_metrics_lgb = select_best_calibrator(
            y, oof_probs_lgb, y_logits=None,
            metric='brier',
            n_bins=10
        )
        
        if best_cal_lgb is not None:
            oof_probs_lgb_cal = best_cal_lgb.predict_proba(oof_probs_lgb)
            metrics_lgb_cal = compute_all_metrics(y, oof_probs_lgb_cal)
            print(f"✅ Best calibrator: {best_method_lgb}")
            print(f"   Brier: {cal_metrics_lgb['uncalibrated']:.4f} → {cal_metrics_lgb[best_method_lgb]:.4f}")
        else:
            oof_probs_lgb_cal = oof_probs_lgb
            metrics_lgb_cal = compute_all_metrics(y, oof_probs_lgb)
        
        # Save LightGBM results
        oof_df_lgb = pd.DataFrame({
            'id': sample_ids,
            'y_true': y.astype(int),
            'oof_prob': oof_probs_lgb,
            'oof_prob_calibrated': oof_probs_lgb_cal if best_cal_lgb is not None else oof_probs_lgb
        })
        oof_path_lgb = os.path.join(run_dir, "oof_lgb.csv")
        oof_df_lgb.to_csv(oof_path_lgb, index=False)
        
        results['lgb'] = {
            'oof_path': oof_path_lgb,
            'fold_metrics': fold_metrics_lgb,
            'mean_metrics': {
                'roc_auc': np.mean([m['roc_auc'] for m in fold_metrics_lgb]),
                'pr_auc': np.mean([m['pr_auc'] for m in fold_metrics_lgb]),
                'brier': np.mean([m['brier'] for m in fold_metrics_lgb]),
                'ece': np.mean([m['ece'] for m in fold_metrics_lgb]),
                'accuracy': np.mean([m['accuracy'] for m in fold_metrics_lgb])
            },
            'calibrated_metrics': metrics_lgb_cal,
            'calibration_method': best_method_lgb,
            'calibration_metrics': cal_metrics_lgb,
            'mean_training_time': np.mean(fold_times_lgb),
            'total_training_time': np.sum(fold_times_lgb)
        }
        
        print(f"\n✅ LightGBM CV completed")
        print(f"   Mean ROC-AUC: {results['lgb']['mean_metrics']['roc_auc']:.4f}")
        print(f"   Mean Brier: {results['lgb']['mean_metrics']['brier']:.4f}")
        print(f"   Mean Training Time: {results['lgb']['mean_training_time']:.2f}s")
    
    # Save summary
    summary_path = os.path.join(run_dir, "gbdt_summary.json")
    _save_json(results, summary_path)
    
    return {
        'run_dir': run_dir,
        'summary_path': summary_path,
        'results': results
    }


__all__ = ['run_gbdt_cv', 'train_xgb_fold', 'train_lgb_fold']
