# API Server Documentation

Complete guide to all server endpoints and commands.

## Server Information

- **Base URL**: `http://localhost:8000`
- **Framework**: FastAPI
- **Startup Time**: ~0.5 seconds (no TensorFlow at startup)
- **Model Loading**: On-demand (lazy loading)

---

## Starting the Server

### Option 1: Using the script
```bash
./start_server.sh
```

### Option 2: Using uvicorn directly
```bash
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
```

### Option 3: With custom port
```bash
export PORT=8080
uvicorn src.serve.app:app --host 0.0.0.0 --port $PORT
```

### Option 4: With model auto-load
```bash
export MODEL_DIR=artifacts/models/champion
export MODEL_VERSION=1
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### 1. Root Endpoint

**GET** `/`

Get basic server information.

**Request:**
```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "message": "TabularANN Inference API",
  "status": "no_model_loaded",
  "model_path": null
}
```

**Status values:**
- `"no_model_loaded"` - No model loaded yet (normal)
- `"model_loaded"` - Model is loaded and ready

---

### 2. Health Check

**GET** `/health`

Check server health and model status.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Status values:**
- `"healthy"` - Server is running
- `"no_model"` - Server running but no model loaded

---

### 3. Load Model

**POST** `/load`

Load a model from a directory. Supports both TensorFlow and sklearn models.

**Request:**
```bash
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1&model_type=auto"
```

**Parameters:**
- `model_dir` (required): Path to model directory
- `version` (optional, default: "1"): Model version
- `model_type` (optional, default: "auto"): `"auto"`, `"tensorflow"`, or `"sklearn"`

**Response:**
```json
{
  "status": "loaded",
  "model_dir": "artifacts/models/champion",
  "version": "1",
  "model_type": "sklearn"
}
```

**Examples:**

Load sklearn model (fast, no TensorFlow needed):
```bash
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/demo_sklearn&version=1&model_type=sklearn"
```

Load TensorFlow model:
```bash
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1&model_type=tensorflow"
```

Auto-detect model type:
```bash
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1"
```

---

### 4. Make Predictions

**POST** `/predict`

Get predictions from the loaded model.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {
        "feature_0": 1.5,
        "feature_1": -0.3,
        "feature_2": 2.1
      }
    ]
  }'
```

**Request Body:**
```json
{
  "features": [
    {
      "feature_name_1": value1,
      "feature_name_2": value2,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "probability": 0.75,
      "probability_calibrated": 0.72,
      "logit": 1.1,
      "prediction": 1
    }
  ]
}
```

**Response Fields:**
- `probability`: Raw uncalibrated probability (0-1)
- `probability_calibrated`: Calibrated probability (if calibrator available)
- `logit`: Raw logit value
- `prediction`: Binary prediction (0 or 1, threshold=0.5)

**Example: Single prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": 2.0, "feature_2": 0}]}'
```

**Example: Batch predictions**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"feature_0": 1.0, "feature_1": 2.0, "feature_2": 0},
      {"feature_0": 0.5, "feature_1": 1.5, "feature_2": 1},
      {"feature_0": -1.0, "feature_1": 0.0, "feature_2": 2}
    ]
  }'
```

**Error Responses:**

No model loaded:
```json
{
  "detail": "Model not loaded. Use /load endpoint first."
}
```

Invalid features:
```json
{
  "detail": "Prediction error: ..."
}
```

---

### 5. Model Information

**GET** `/model_info`

Get information about the currently loaded model.

**Request:**
```bash
curl http://localhost:8000/model_info
```

**Response (when model loaded):**
```json
{
  "model_type": "sklearn",
  "version": "1",
  "has_calibrator": true,
  "calibrator_method": "isotonic",
  "feature_info": {
    "feature_0": {
      "action": "scale"
    },
    "feature_1": {
      "action": "scale"
    },
    "feature_2": {
      "action": "onehot",
      "vocabulary_size": 3
    }
  }
}
```

**Response (no model loaded):**
```json
{
  "detail": "Model not loaded"
}
```

---

## Complete Workflow Examples

### Example 1: Full Workflow with Sklearn Model

```bash
# 1. Start server (in background)
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000 &

# 2. Check server is running
curl http://localhost:8000/health

# 3. Load sklearn model (fast, ~1-2 seconds)
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/demo_sklearn&version=1&model_type=sklearn"

# 4. Check model info
curl http://localhost:8000/model_info

# 5. Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": -0.5, "feature_2": 0.8}]}'
```

### Example 2: Full Workflow with TensorFlow Model

```bash
# 1. Load TensorFlow model (slower, ~5-10 seconds due to TF init)
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1&model_type=tensorflow"

# 2. Make predictions (fast after loading)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": 2.0}]}'
```

### Example 3: Batch Predictions

```bash
# Make multiple predictions at once
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"feature_0": 1.0, "feature_1": 2.0, "feature_2": 0},
      {"feature_0": 0.5, "feature_1": 1.5, "feature_2": 1},
      {"feature_0": -1.0, "feature_1": 0.0, "feature_2": 2}
    ]
  }'
```

---

## Python Client Examples

### Using requests library

```python
import requests

# Base URL
BASE_URL = "http://localhost:8000"

# 1. Check health
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. Load model
response = requests.post(
    f"{BASE_URL}/load",
    params={
        "model_dir": "artifacts/models/demo_sklearn",
        "version": "1",
        "model_type": "sklearn"
    }
)
print(response.json())

# 3. Make prediction
response = requests.post(
    f"{BASE_URL}/predict",
    json={
        "features": [
            {"feature_0": 1.0, "feature_1": -0.5, "feature_2": 0.8}
        ]
    }
)
result = response.json()
print(f"Probability: {result['predictions'][0]['probability_calibrated']}")
```

### Using http.client

```python
import http.client
import json

conn = http.client.HTTPConnection("localhost", 8000)

# Load model
conn.request("POST", "/load?model_dir=artifacts/models/demo_sklearn&version=1")
response = conn.getresponse()
print(response.read().decode())

# Make prediction
payload = json.dumps({
    "features": [{"feature_0": 1.0, "feature_1": -0.5, "feature_2": 0.8}]
})
headers = {"Content-Type": "application/json"}
conn.request("POST", "/predict", payload, headers)
response = conn.getresponse()
print(response.read().decode())
```

---

## Quick Reference

### Server Management

| Command | Description |
|---------|-------------|
| `./start_server.sh` | Start server using script |
| `uvicorn src.serve.app:app --host 0.0.0.0 --port 8000` | Start server directly |
| `pkill -f "uvicorn src.serve.app"` | Stop server |

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server info |
| GET | `/health` | Health check |
| GET | `/model_info` | Model information |
| POST | `/load` | Load model |
| POST | `/predict` | Make predictions |

### Common curl Commands

```bash
# Check server
curl http://localhost:8000/health

# Load sklearn model
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/demo_sklearn&version=1&model_type=sklearn"

# Load TensorFlow model
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1&model_type=tensorflow"

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": 2.0}]}'

# Get model info
curl http://localhost:8000/model_info
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Auto-load model from this directory | None |
| `MODEL_VERSION` | Model version to load | "1" |
| `PORT` | Server port | 8000 |
| `HOST` | Server host | 0.0.0.0 |

**Example:**
```bash
export MODEL_DIR=artifacts/models/champion
export MODEL_VERSION=1
export PORT=8080
uvicorn src.serve.app:app --host 0.0.0.0 --port $PORT
```

---

## Error Codes

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 400 | Bad request (e.g., failed to load model) |
| 500 | Server error (e.g., prediction failed) |
| 503 | Service unavailable (e.g., no model loaded) |

---

## Model Types Supported

### Sklearn Models
- **Format**: `.pkl` files (joblib)
- **Load Time**: ~1-2 seconds
- **Requirements**: scikit-learn only
- **Best For**: Fast serving, no TensorFlow needed

### TensorFlow Models
- **Format**: SavedModel format
- **Load Time**: ~5-10 seconds (first time, includes TF init)
- **Requirements**: TensorFlow
- **Best For**: ANN models with embeddings

---

## Tips & Best Practices

1. **Start server without model** - Fast startup, load model on-demand
2. **Use sklearn models for serving** - Faster loading, no TF dependency
3. **Batch predictions** - Send multiple samples in one request
4. **Keep model loaded** - Don't reload between requests (model stays in memory)
5. **Use environment variables** - For production deployments

---

## Stopping the Server

### Commands

```bash
# Option 1: Kill by process name (recommended)
pkill -f "uvicorn src.serve.app"

# Option 2: Kill by port
lsof -ti:8000 | xargs kill

# Option 3: Force kill if needed
pkill -9 -f "uvicorn src.serve.app"

# Check if server is running
lsof -ti:8000 && echo "Server is running" || echo "Server not running"
```

### If server is in terminal
- Press `Ctrl+C` to stop gracefully

---

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -ti:8000

# Kill existing server
pkill -f "uvicorn src.serve.app"
```

### Model won't load
- Check model directory exists
- Verify model files are present (`.pkl` or `saved_model/`)
- Check model_type parameter matches your model

### Predictions fail
- Ensure model is loaded first (check `/health`)
- Verify feature names match model's feature_info
- Check feature values are correct types (numbers)

---

## Performance

- **Server startup**: ~0.5 seconds (no TF)
- **Sklearn model load**: ~1-2 seconds
- **TensorFlow model load**: ~5-10 seconds (first time)
- **Prediction latency**: <50ms per sample (after model loaded)

---

## See Also

- `docs/LIGHTWEIGHT_OPTIONS.md` - Lightweight deployment options
- `docs/PYTORCH_VS_TENSORFLOW.md` - Framework comparison
- `start_server.sh` - Server startup script
