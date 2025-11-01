"""
Model export utilities.

Exports trained models with:
- TensorFlow SavedModel format
- Serialized calibrator
- JSON metadata (feature schema, preprocessing info)
"""

import os
import json
import pickle
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow import saved_model

from src.models.tabular_ann import TabularANN
from src.metrics.calibration import PlattScaler, IsotonicCalibrator


def export_model(
    model: TabularANN,
    export_dir: str,
    feature_info: Dict[str, Any],
    calibrator: Optional[Any] = None,
    calibrator_method: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    version: str = "1"
) -> Dict[str, str]:
    """
    Export a trained TabularANN model with all artifacts.
    
    Args:
        model: Trained TabularANN model
        export_dir: Directory to export to
        feature_info: Feature information dictionary
        calibrator: Calibrator object (PlattScaler or IsotonicCalibrator)
        calibrator_method: Calibration method name ('platt', 'isotonic', or None)
        metadata: Additional metadata to save
        version: Model version (default: "1")
    
    Returns:
        Dictionary with paths to exported artifacts
    """
    export_path = Path(export_dir) / version
    export_path.mkdir(parents=True, exist_ok=True)
    
    artifacts = {}
    
    # 1. Export TensorFlow SavedModel
    saved_model_path = export_path / "saved_model"
    if model.model is not None:
        model.model.save(str(saved_model_path))
        artifacts['saved_model'] = str(saved_model_path)
        print(f"✅ Saved TensorFlow model to: {saved_model_path}")
    
    # 2. Save calibrator
    if calibrator is not None:
        calibrator_path = export_path / "calibrator.pkl"
        calibrator.save(str(calibrator_path))
        artifacts['calibrator'] = str(calibrator_path)
        print(f"✅ Saved calibrator ({calibrator_method}) to: {calibrator_path}")
    
    # 3. Save feature info and metadata
    metadata_dict = {
        'feature_info': feature_info,
        'calibrator_method': calibrator_method,
        'model_type': 'TabularANN',
        'version': version
    }
    
    if metadata:
        metadata_dict.update(metadata)
    
    metadata_path = export_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_dict, f, indent=2)
    artifacts['metadata'] = str(metadata_path)
    print(f"✅ Saved metadata to: {metadata_path}")
    
    # 4. Create a simple signature/version info
    version_info = {
        'version': version,
        'model_type': 'TabularANN',
        'has_calibrator': calibrator is not None,
        'calibrator_method': calibrator_method
    }
    
    version_path = export_path / "version.json"
    with open(version_path, 'w') as f:
        json.dump(version_info, f, indent=2)
    artifacts['version_info'] = str(version_path)
    
    return artifacts


def load_model(
    export_dir: str,
    version: str = "1"
) -> Dict[str, Any]:
    """
    Load exported model and artifacts.
    
    Args:
        export_dir: Export directory
        version: Model version
    
    Returns:
        Dictionary with loaded model, calibrator, and metadata
    """
    export_path = Path(export_dir) / version
    
    # Load metadata
    metadata_path = export_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load TensorFlow model
    saved_model_path = export_path / "saved_model"
    loaded_model = None
    if saved_model_path.exists():
        loaded_model = tf.keras.models.load_model(str(saved_model_path))
    
    # Load calibrator
    calibrator_path = export_path / "calibrator.pkl"
    calibrator = None
    calibrator_method = metadata.get('calibrator_method')
    
    if calibrator_path.exists() and calibrator_method:
        if calibrator_method == 'platt':
            calibrator = PlattScaler.load(str(calibrator_path))
        elif calibrator_method == 'isotonic':
            calibrator = IsotonicCalibrator.load(str(calibrator_path))
        else:
            # Fallback to pickle
            with open(calibrator_path, 'rb') as f:
                calibrator = pickle.load(f)
    
    return {
        'model': loaded_model,
        'calibrator': calibrator,
        'calibrator_method': calibrator_method,
        'metadata': metadata,
        'feature_info': metadata.get('feature_info', {})
    }


def export_from_cv_run(
    cv_run_dir: str,
    fold_idx: int,
    export_dir: str,
    version: str = "1"
) -> Dict[str, str]:
    """
    Export a model from a CV run (specific fold).
    
    Args:
        cv_run_dir: CV run directory
        fold_idx: Fold index to export
        export_dir: Directory to export to
        version: Model version
    
    Returns:
        Dictionary with paths to exported artifacts
    """
    fold_dir = Path(cv_run_dir) / f"fold_{fold_idx}"
    
    # Load metadata
    metadata_path = fold_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Fold metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        fold_metadata = json.load(f)
    
    feature_info = fold_metadata.get('feature_info', {})
    
    # Load model weights
    model_weights_path = fold_dir / "model.weights.h5"
    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")
    
    # Reconstruct model architecture
    from src.models.tabular_ann import create_tabular_ann
    
    params = fold_metadata.get('params', {})
    model = create_tabular_ann(
        feature_info=feature_info,
        embedding_dims=params.get("embedding_dims"),
        hidden_layers=params.get("hidden_layers", [128, 64, 32]),
        dropout_rate=params.get("dropout_rate", 0.3),
        l2_reg=params.get("l2_reg", 0.001),
        learning_rate=params.get("learning_rate", 0.001),
        random_seed=params.get("random_seed", 42),
        numeric_normalizers=params.get("numeric_normalizers")
    )
    
    # Load weights
    model.model.load_weights(str(model_weights_path))
    
    # Load calibrator from CV run (if available)
    cv_summary_path = Path(cv_run_dir) / "summary.json"
    calibrator = None
    calibrator_method = None
    
    if cv_summary_path.exists():
        with open(cv_summary_path, 'r') as f:
            cv_summary = json.load(f)
        
        cal_info = cv_summary.get('calibration', {})
        calibrator_method = cal_info.get('best_method')
        
        if calibrator_method and calibrator_method != 'uncalibrated':
            calibrator_path = Path(cv_run_dir) / "calibrator.pkl"
            if calibrator_path.exists():
                if calibrator_method == 'platt':
                    calibrator = PlattScaler.load(str(calibrator_path))
                elif calibrator_method == 'isotonic':
                    calibrator = IsotonicCalibrator.load(str(calibrator_path))
    
    # Export
    return export_model(
        model=model,
        export_dir=export_dir,
        feature_info=feature_info,
        calibrator=calibrator,
        calibrator_method=calibrator_method,
        metadata={
            'fold_idx': fold_idx,
            'cv_run_dir': cv_run_dir,
            **fold_metadata
        },
        version=version
    )


__all__ = ['export_model', 'load_model', 'export_from_cv_run']
