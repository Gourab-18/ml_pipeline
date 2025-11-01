# API Server - Quick Reference Card

## 🚀 Starting Server

```bash
# Option 1: Script
./start_server.sh

# Option 2: Direct
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Option 3: With auto-load
export MODEL_DIR=artifacts/models/champion
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000
```

---

## 📡 API Endpoints

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Server Info
```bash
curl http://localhost:8000/
```

### 3. Load Model
```bash
# Sklearn (fast)
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/demo_sklearn&version=1&model_type=sklearn"

# TensorFlow
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1&model_type=tensorflow"
```

### 4. Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": 2.0}]}'
```

### 5. Model Info
```bash
curl http://localhost:8000/model_info
```

---

## 💻 Python Example

```python
import requests

BASE = "http://localhost:8000"

# Load model
requests.post(f"{BASE}/load", params={
    "model_dir": "artifacts/models/demo_sklearn",
    "version": "1"
})

# Predict
response = requests.post(
    f"{BASE}/predict",
    json={"features": [{"feature_0": 1.0, "feature_1": 2.0}]}
)
print(response.json())
```

---

## ⚙️ Environment Variables

```bash
export MODEL_DIR=artifacts/models/champion  # Auto-load on startup
export MODEL_VERSION=1                      # Model version
export PORT=8000                           # Server port
export HOST=0.0.0.0                        # Server host
```

---

## 🛑 Stop Server

```bash
# Option 1: Kill by process name (recommended)
pkill -f "uvicorn src.serve.app"

# Option 2: Kill by port
lsof -ti:8000 | xargs kill

# Option 3: Force kill if needed
pkill -9 -f "uvicorn src.serve.app"

# Check if server is running
lsof -ti:8000 && echo "Running" || echo "Not running"
```

---

## 📊 Response Format

**Prediction Response:**
```json
{
  "predictions": [{
    "probability": 0.75,
    "probability_calibrated": 0.72,
    "logit": 1.1,
    "prediction": 1
  }]
}
```

---

**Full documentation:** See `docs/API_DOCUMENTATION.md`
