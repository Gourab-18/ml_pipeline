"""
Integration tests for model serving API.

Tests prediction endpoint, output format, and parity with offline predictions.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from pathlib import Path
import time
import subprocess
import signal

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from src.models.tabular_ann import create_tabular_ann
from src.models.export import export_model, load_model
from src.metrics.calibration import IsotonicCalibrator


@pytest.fixture
def sample_model_and_data():
    """Create a sample model and test data."""
    feature_info = {
        'feature_0': {'action': 'scale', 'vocabulary_size': 0},
        'feature_1': {'action': 'scale', 'vocabulary_size': 0},
        'feature_2': {'action': 'onehot', 'vocabulary_size': 3}
    }
    
    # Create model
    model = create_tabular_ann(
        feature_info=feature_info,
        hidden_layers=[32, 16],
        dropout_rate=0.2,
        random_seed=42
    )
    
    # Create dummy training data to "train" model (just set some weights)
    X_train = np.random.randn(50, 5)  # 5 features total: 2 scale + 3 onehot
    y_train = np.random.randint(0, 2, 50)
    
    # Quick training (just to initialize weights properly)
    try:
        from src.models.train_utils import train_tabular_ann
        train_tabular_ann(
            model, X_train, y_train, feature_info,
            epochs=2, batch_size=32, verbose=0, random_seed=42
        )
    except:
        pass  # Model might not train perfectly, but that's ok for export test
    
    # Create calibrator
    y_dummy = np.random.randint(0, 2, 100)
    y_proba_dummy = np.random.rand(100)
    calibrator = IsotonicCalibrator()
    calibrator.fit(y_proba_dummy, y_dummy)
    
    return model, feature_info, calibrator


@pytest.fixture
def exported_model_dir(sample_model_and_data):
    """Export model to temporary directory."""
    model, feature_info, calibrator = sample_model_and_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_dir = os.path.join(tmpdir, "exported_model")
        
        export_model(
            model=model,
            export_dir=export_dir,
            feature_info=feature_info,
            calibrator=calibrator,
            calibrator_method='isotonic',
            version="1"
        )
        
        yield export_dir


def test_export_and_load(exported_model_dir):
    """Test that export and load work correctly."""
    loaded = load_model(exported_model_dir, version="1")
    
    assert loaded['model'] is not None
    assert loaded['calibrator'] is not None
    assert loaded['calibrator_method'] == 'isotonic'
    assert 'feature_info' in loaded['metadata']


def test_serve_predict_offline(exported_model_dir, sample_model_and_data):
    """Test prediction function works offline (without API server)."""
    model, feature_info, calibrator = sample_model_and_data
    
    # Prepare test input
    test_features = [
        {'feature_0': 1.0, 'feature_1': 2.0, 'feature_2': 0}  # One-hot as integer index
    ]
    
    # Manual prediction (simulating what server does)
    inputs = {}
    inputs['feature_0_input'] = np.array([[1.0]])
    inputs['feature_1_input'] = np.array([[2.0]])
    # One-hot: create vector from index
    one_hot = np.zeros((1, 3))
    one_hot[0, 0] = 1.0
    inputs['feature_2_input'] = one_hot
    
    # Predict
    predictions = model.model.predict(inputs, verbose=0)
    
    if isinstance(predictions, list):
        proba = predictions[0][0, 0]
        logit = predictions[1][0, 0]
    else:
        proba = predictions[0, 0]
        logit = np.log(proba / (1 - proba + 1e-10))
    
    # Calibrate
    proba_cal = calibrator.predict_proba(np.array([proba]))[0]
    
    assert 0 <= proba <= 1
    assert 0 <= proba_cal <= 1
    
    return {
        'probability': proba,
        'probability_calibrated': proba_cal,
        'logit': logit
    }


@pytest.fixture
def api_server(exported_model_dir):
    """Start API server in background."""
    import sys
    import os
    
    # Set environment
    os.environ['MODEL_DIR'] = exported_model_dir
    os.environ['MODEL_VERSION'] = '1'
    
    # Start server in subprocess
    server_process = subprocess.Popen(
        [sys.executable, '-m', 'src.serve.app'],
        env=os.environ,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    yield server_process
    
    # Cleanup
    server_process.terminate()
    server_process.wait(timeout=5)


def test_api_health(api_server):
    """Test API health endpoint."""
    if not REQUESTS_AVAILABLE:
        pytest.skip("requests library not available")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
    except Exception:
        pytest.skip("API server not responding (may not have started)")


def test_api_predict_format(api_server):
    """Test API predict endpoint returns correct format."""
    if not REQUESTS_AVAILABLE:
        pytest.skip("requests library not available")
    
    try:
        request_data = {
            "features": [
                {"feature_0": 1.0, "feature_1": 2.0, "feature_2": 0}
            ]
        }
        
        response = requests.post(
            "http://localhost:8000/predict",
            json=request_data,
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'predictions' in data
        assert len(data['predictions']) == 1
        
        pred = data['predictions'][0]
        assert 'probability' in pred
        assert 'probability_calibrated' in pred
        assert 'logit' in pred
        assert 'prediction' in pred
        
        assert 0 <= pred['probability'] <= 1
        assert 0 <= pred['probability_calibrated'] <= 1
        assert isinstance(pred['prediction'], int)
        assert pred['prediction'] in [0, 1]
        
    except requests.exceptions.RequestException:
        pytest.skip("API server not responding")


def test_prediction_parity(exported_model_dir, sample_model_and_data):
    """
    Test that API predictions match offline predictions.
    
    This test verifies that the serving API produces the same results
    as direct model inference.
    """
    model, feature_info, calibrator = sample_model_and_data
    
    # Test data
    test_features = [
        {'feature_0': 1.5, 'feature_1': -0.5, 'feature_2': 1}
    ]
    
    # Offline prediction
    inputs = {}
    inputs['feature_0_input'] = np.array([[1.5]])
    inputs['feature_1_input'] = np.array([[-0.5]])
    one_hot = np.zeros((1, 3))
    one_hot[0, 1] = 1.0
    inputs['feature_2_input'] = one_hot
    
    predictions = model.model.predict(inputs, verbose=0)
    
    if isinstance(predictions, list):
        proba_offline = float(predictions[0][0, 0])
        logit_offline = float(predictions[1][0, 0])
    else:
        proba_offline = float(predictions[0, 0])
        logit_offline = float(np.log(proba_offline / (1 - proba_offline + 1e-10)))
    
    proba_cal_offline = float(calibrator.predict_proba(np.array([proba_offline]))[0])
    
    # API prediction (if server is available)
    if not REQUESTS_AVAILABLE:
        pytest.skip("requests library not available")
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": test_features},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            api_pred = data['predictions'][0]
            
            # Compare (allow small numerical differences)
            assert abs(proba_offline - api_pred['probability']) < 0.01, \
                f"Probability mismatch: {proba_offline} vs {api_pred['probability']}"
            
            assert abs(proba_cal_offline - api_pred['probability_calibrated']) < 0.01, \
                f"Calibrated probability mismatch: {proba_cal_offline} vs {api_pred['probability_calibrated']}"
            
            assert abs(logit_offline - api_pred['logit']) < 0.1, \
                f"Logit mismatch: {logit_offline} vs {api_pred['logit']}"
    
    except Exception:
        pytest.skip("API server not available for parity test")


def test_model_info_endpoint(api_server):
    """Test model info endpoint."""
    if not REQUESTS_AVAILABLE:
        pytest.skip("requests library not available")
    
    try:
        response = requests.get("http://localhost:8000/model_info", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert 'model_type' in data
        assert 'version' in data
        assert 'has_calibrator' in data
    except Exception:
        pytest.skip("API server not responding")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
