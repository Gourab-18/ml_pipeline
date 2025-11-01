"""
FastAPI inference server for TabularANN models.

Provides REST API endpoint for model predictions with calibration support.
TensorFlow is lazy-loaded only when a model is actually loaded.
"""

import os
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Lazy imports - don't import TensorFlow at module level
# from src.models.export import load_model  # Will import on demand
# from src.models.tabular_ann import TabularANN  # Not needed at startup


# Global model state
_model_state: Optional[Dict[str, Any]] = None
_model_path: Optional[str] = None


# Request/Response models
class PredictRequest(BaseModel):
    """Prediction request model."""
    features: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {"feature_0": 1.5, "feature_1": 0.3, "feature_2": 2.1}
                ]
            }
        }


class PredictResponse(BaseModel):
    """Prediction response model."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "probability": 0.75,
                        "probability_calibrated": 0.72,
                        "logit": 1.1,
                        "prediction": 1
                    }
                ]
            }
        }


# FastAPI app
app = FastAPI(
    title="TabularANN Inference API",
    description="REST API for TabularANN model predictions",
    version="1.0.0"
)


def load_model_state(model_dir: str, version: str = "1", model_type: str = "auto"):
    """
    Load model and calibrator from export directory.
    
    Args:
        model_dir: Directory containing model
        version: Model version
        model_type: "tensorflow", "sklearn", or "auto" (detect)
    """
    global _model_state, _model_path
    
    # Try sklearn/joblib first (faster, no TF needed)
    model_path = Path(model_dir) / version / "model.pkl"
    if model_path.exists() and model_type in ["sklearn", "auto"]:
        try:
            # Lazy import joblib only when needed (avoid heavy imports)
            import joblib
            import pickle
            
            print(f"Loading sklearn model from {model_path}...")
            start_time = time.time()
            model = joblib.load(model_path)
            load_time = time.time() - start_time
            
            calibrator_path = Path(model_dir) / version / "calibrator.pkl"
            calibrator = None
            calibrator_method = None
            if calibrator_path.exists():
                calibrator = pickle.load(open(calibrator_path, "rb"))
                calibrator_method = "isotonic"  # Assume isotonic for sklearn
            
            metadata_path = Path(model_dir) / version / "metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            _model_state = {
                'model': model,
                'calibrator': calibrator,
                'calibrator_method': calibrator_method,
                'metadata': metadata,
                'feature_info': metadata.get('feature_info', {}),
                'model_type': 'sklearn'
            }
            _model_path = model_dir
            print(f"✅ Sklearn model loaded in {load_time:.2f}s from: {model_dir}")
            return True
        except Exception as e:
            print(f"⚠️  Sklearn load failed: {e}, trying TensorFlow...")
            # Fall through to TensorFlow
    
    # Try TensorFlow SavedModel (if sklearn failed or model_type="tensorflow")
    if model_type in ["tensorflow", "auto"]:
        try:
            from src.models.export import load_model
            
            _model_state = load_model(model_dir, version=version)
            _model_state['model_type'] = 'tensorflow'
            _model_path = model_dir
            print(f"✅ TensorFlow model loaded from: {model_dir}")
            return True
        except ImportError as e:
            print(f"❌ TensorFlow not available: {e}")
            print("   Install tensorflow-cpu or tensorflow")
            print("   Or use sklearn models (save as .pkl)")
            return False
        except Exception as e:
            print(f"❌ Error loading TensorFlow model: {e}")
            return False
    
    return False


def prepare_inputs(features: List[Dict[str, Any]], feature_info: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Prepare model inputs from feature dictionaries.
    
    Args:
        features: List of feature dictionaries
        feature_info: Feature information from model metadata
    
    Returns:
        Dictionary of input arrays for the model
    """
    n_samples = len(features)
    inputs = {}
    
    # Group features by type
    for feature_name, info in feature_info.items():
        action = info.get('action')
        vocab_size = info.get('vocabulary_size', 0)
        key = f"{feature_name}_input"
        
        if action == 'scale':
            # Numeric feature
            values = np.array([f.get(feature_name, 0.0) for f in features], dtype=float)
            inputs[key] = values.reshape(-1, 1)
        
        elif action in ['onehot', 'embed']:
            if action == 'embed' and vocab_size > 0:
                # Embedding: expect integer indices
                values = np.array([int(f.get(feature_name, 0)) for f in features], dtype=int)
                inputs[key] = values.reshape(-1, 1)
            else:
                # One-hot: expect vector of size vocab_size
                # For simplicity, assume input is already one-hot or integer that we encode
                values = np.array([f.get(feature_name, 0) for f in features])
                if values.ndim == 1:
                    # If single value, assume it's an index - convert to one-hot
                    one_hot = np.zeros((n_samples, vocab_size))
                    for i, val in enumerate(values):
                        idx = int(val) % vocab_size
                        one_hot[i, idx] = 1.0
                    inputs[key] = one_hot
                else:
                    inputs[key] = values
    
    return inputs


@app.on_event("startup")
async def startup_event():
    """Load model on startup (optional, lazy-loads TensorFlow)."""
    # Try to load from environment variable
    model_dir = os.getenv("MODEL_DIR")
    version = os.getenv("MODEL_VERSION", "1")
    
    if model_dir and os.path.exists(model_dir):
        print("Loading model on startup...")
        load_model_state(model_dir, version)
    else:
        if model_dir:
            print(f"⚠️  Model directory not found: {model_dir}")
        else:
            print("ℹ️  No MODEL_DIR set - server starting without model")
        print("   Set MODEL_DIR environment variable or load model via POST /load endpoint")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TabularANN Inference API",
        "status": "ready" if _model_state else "no_model_loaded",
        "model_path": _model_path
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if _model_state else "no_model",
        "model_loaded": _model_state is not None
    }


@app.post("/load")
async def load_model_endpoint(model_dir: str, version: str = "1", model_type: str = "auto"):
    """
    Load model from directory.
    
    Args:
        model_dir: Path to model directory
        version: Model version (default: "1")
        model_type: "auto" (detect), "tensorflow", or "sklearn"
    """
    success = load_model_state(model_dir, version, model_type)
    if success:
        return {
            "status": "loaded",
            "model_dir": model_dir,
            "version": version,
            "model_type": _model_state.get('model_type', 'unknown')
        }
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load model from {model_dir}")


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict endpoint.
    
    Takes a list of feature dictionaries and returns predictions.
    """
    if _model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Use /load endpoint first.")
    
    model = _model_state['model']
    calibrator = _model_state['calibrator']
    calibrator_method = _model_state['calibrator_method']
    feature_info = _model_state['feature_info']
    
    try:
        # Prepare inputs
        inputs = prepare_inputs(request.features, feature_info)
        
        # Predict - handle both TensorFlow and sklearn models
        model_type = _model_state.get('model_type', 'tensorflow')
        
        if model_type == 'sklearn':
            # sklearn model: expects 2D array
            # Convert feature dict to flat array matching feature_info order
            feature_names_ordered = sorted(feature_info.keys())
            X_list = []
            for feat_name in feature_names_ordered:
                key = f"{feat_name}_input"
                if key in inputs:
                    X_list.append(inputs[key])
                else:
                    # Feature not provided, use default 0
                    X_list.append(np.zeros((n_samples, 1)))
            
            X_array = np.hstack(X_list) if len(X_list) > 1 else X_list[0]
            predictions = model.predict_proba(X_array)[:, 1]  # Probability of class 1
            probas = predictions
            logits = np.log(probas / (1 - probas + 1e-10))
        else:
            # TensorFlow model
            predictions = model.predict(inputs, verbose=0)
            
            # Extract probabilities and logits
            if isinstance(predictions, list) and len(predictions) == 2:
                probas = predictions[0].flatten()
                logits = predictions[1].flatten()
            else:
                probas = predictions.flatten()
                logits = np.log(probas / (1 - probas + 1e-10))  # Approximate logits
        
        # Apply calibration if available
        probas_calibrated = probas.copy()
        if calibrator is not None:
            if calibrator_method == 'platt':
                probas_calibrated = calibrator.predict_proba(logits)
            elif calibrator_method == 'isotonic':
                probas_calibrated = calibrator.predict_proba(probas)
        
        # Format response
        results = []
        for i in range(len(request.features)):
            results.append({
                "probability": float(probas[i]),
                "probability_calibrated": float(probas_calibrated[i]),
                "logit": float(logits[i]),
                "prediction": int(probas_calibrated[i] > 0.5)
            })
        
        return PredictResponse(predictions=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model_info")
async def model_info():
    """Get model information."""
    if _model_state is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = _model_state['metadata']
    return {
        "model_type": metadata.get('model_type', 'TabularANN'),
        "version": metadata.get('version', '1'),
        "has_calibrator": _model_state['calibrator'] is not None,
        "calibrator_method": _model_state['calibrator_method'],
        "feature_info": _model_state['feature_info']
    }


def main():
    """Run the server."""
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
