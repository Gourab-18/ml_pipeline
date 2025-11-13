# ML Pipeline - Essential Commands

Quick reference for the most important commands in the ML pipeline project.

**Project Status:** ✅ Clean & Optimized (2.4 MB, 99.93% size reduction)
**Last Cleanup:** 2025-01-13 - See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for details

---

## 🚀 Quick Start

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install OpenMP (macOS only - required for XGBoost/LightGBM)
brew install libomp

# 4. Run tests
make test-fast

# 5. Train models
python3 train_gbdt_only.py
```

---

## 📦 Installation

### Basic Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### macOS Specific (Required!)
```bash
# Install OpenMP for XGBoost/LightGBM
brew install libomp

# Verify installation
brew list libomp
```

---

## 🧪 Testing

### Fast Tests (Recommended)
```bash
# Run fast tests (~2 seconds, no TensorFlow)
make test-fast
```

### Full Tests
```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_loader.py -v
pytest tests/test_gbdt_baselines.py -v
```

---

## 🏋️ Training Models

### Train GBDT Models (XGBoost + LightGBM)
```bash
# Train both models with 5-fold CV (recommended)
python3 train_gbdt_only.py

# With custom parameters
python3 train_gbdt_only.py --folds 10 --seed 123
```

### Train PyTorch ANN (Recommended for ANN)
```bash
# Train PyTorch ANN with 5-fold CV (works on macOS!)
python3 train_pytorch_ann.py

# With custom parameters
python3 train_pytorch_ann.py --folds 5 --epochs 50 --batch-size 64 --learning-rate 0.001

# Quick test (3 folds, 20 epochs)
python3 train_pytorch_ann.py --folds 3 --epochs 20
```

### Quick Baseline Training
```bash
# Train quick sklearn baseline (~10 seconds)
python3 demo_cv_sklearn_quick.py
```

### Demo Scripts
```bash
# Test PyTorch ANN (quick demo)
python3 demo_pytorch_ann.py

# Mock ANN (pure NumPy, no frameworks)
python3 src/models/mock_ann.py
```

---

## 📊 Model Evaluation

### View Latest Results
```bash
# List all training runs
ls -lt artifacts/cv/

# View model comparison
cat artifacts/cv/LATEST_RUN/model_comparison.csv

# View detailed metrics
cat artifacts/cv/LATEST_RUN/gbdt_summary.json | python3 -m json.tool
```

### Compare Models
```bash
# Compare models from latest run
python3 compare_models.py --latest

# Compare specific run
python3 compare_models.py artifacts/cv/20251113_013245

# Or simply view existing comparison (faster)
cat artifacts/cv/20251113_013245/model_comparison.csv
```

### View Training Results
```bash
# Pretty print comparison
python3 -c "
import pandas as pd
df = pd.read_csv('artifacts/cv/LATEST_RUN/model_comparison.csv')
print(df.to_string(index=False))
print(f'\nWinner: {df.iloc[0][\"Model\"]} (ROC-AUC: {df.iloc[0][\"ROC-AUC\"]:.4f})')
"
```

---

## 🚀 Model Serving

### Complete Serving Workflow

#### Step 1: Export Model with Pipeline
```bash
# Export trained model with preprocessing pipeline
python3 export_model_with_pipeline.py

# This creates:
# - artifacts/models/champion/1/model.pkl (trained model)
# - artifacts/models/champion/1/pipeline.pkl (preprocessing)
# - artifacts/models/champion/1/metadata.json (model info)
```

#### Step 2: Start API Server
```bash
# Using Makefile
make serve

# Or directly
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# View interactive API documentation
open http://localhost:8000/docs
```

#### Step 3: Load Model into Server
```bash
# Load model with preprocessing pipeline
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1"

# Check if model loaded successfully
curl http://localhost:8000/health
# Response: {"status": "healthy", "model_loaded": true}
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Model Information
```bash
curl http://localhost:8000/model_info
```

#### Make Predictions (Method 1: Using JSON file - Recommended)
```bash
# Create request file
cat > request.json << 'EOF'
{
  "features": [{
    "customer_id": "CUST_001",
    "age": 35,
    "gender": "M",
    "location_country": "USA",
    "subscription_type": "Premium",
    "subscription_duration_days": 180,
    "monthly_revenue": 99.99,
    "login_frequency": 15,
    "feature_usage_count": 5,
    "session_duration_avg": 30.0,
    "page_views_total": 100,
    "support_tickets_count": 0,
    "support_tickets_avg_resolution_days": 0.0,
    "payment_failures_count": 0,
    "days_since_last_payment": 10,
    "email_opens_count": 5,
    "email_clicks_count": 2
  }]
}
EOF

# Make prediction
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @request.json | python3 -m json.tool
```

#### Make Predictions (Method 2: Inline - Single line)
```bash
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features":[{"customer_id":"C001","age":35,"gender":"M","location_country":"USA","subscription_type":"Premium","subscription_duration_days":180,"monthly_revenue":99.99,"login_frequency":15,"feature_usage_count":5,"session_duration_avg":30.0,"page_views_total":100,"support_tickets_count":0,"support_tickets_avg_resolution_days":0.0,"payment_failures_count":0,"days_since_last_payment":10,"email_opens_count":5,"email_clicks_count":2}]}' \
  | python3 -m json.tool
```

**Expected Response:**
```json
{
  "predictions": [{
    "probability": 0.1747,
    "probability_calibrated": 0.1747,
    "logit": -1.5526,
    "prediction": 0
  }]
}
```

### Python Script Testing
```bash
# Test predictions directly with Python (no API needed)
python3 test_prediction.py

# This will:
# - Load the model
# - Test on 5 sample customers
# - Test on a custom customer profile
# - Show detailed prediction results
```

### Stop the Server
```bash
# Find process ID
lsof -i :8000

# Kill the server
kill -9 <PID>

# Or use pkill
pkill -f "uvicorn src.serve.app"
```

---

## 🔧 Troubleshooting

### TensorFlow Mutex Error (macOS)
```bash
# Problem: TensorFlow crashes with mutex lock error
# Cause: TensorFlow 2.20.0 + macOS compatibility issue
# Solution: Use PyTorch ANN instead (recommended)

# Option 1: Use PyTorch ANN (recommended - best performance!)
python3 train_pytorch_ann.py
python3 demo_pytorch_ann.py

# Option 2: Use Mock ANN (pure NumPy, no dependencies)
python3 src/models/mock_ann.py

# Option 3: Use Docker for TensorFlow
docker run -it --rm -v $(pwd):/app python:3.9 bash
cd /app && pip install -r requirements.txt && python3 train_pytorch_ann.py
```

### XGBoost/LightGBM OpenMP Error (macOS)
```bash
# Problem: Library not loaded: libomp.dylib
# Solution: Install OpenMP
brew install libomp

# Verify
brew list libomp
```

### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port
PORT=8080 make serve
```

### Tests Taking Too Long
```bash
# Use fast tests instead of full suite
make test-fast

# Skip TensorFlow tests
pytest tests/ -v -k "not tensorflow"
```

---

## 📁 Important File Locations

```
configs/
  schema.yaml              # Data schema definition
  feature_list.csv         # Feature configurations

data/
  sample.csv              # Sample dataset (500 rows)

src/
  models/
    pytorch_ann.py        # PyTorch ANN (Recommended! Works on macOS)
    tabular_ann.py        # TensorFlow ANN (macOS issue)
    mock_ann.py           # Mock ANN (pure NumPy)
  baselines/
    xgb_lgb.py           # XGBoost & LightGBM
    compare_models.py     # Model comparison
  serve/
    app.py               # FastAPI server

artifacts/
  cv/                    # Cross-validation results
    YYYYMMDD_HHMMSS/    # Timestamped run folders
      gbdt_summary.json       # GBDT training metrics
      pytorch_ann_summary.json # PyTorch ANN metrics
      model_comparison.csv     # Comparison table
      oof_xgb.csv             # Out-of-fold predictions (XGBoost)
      oof_lgb.csv             # Out-of-fold predictions (LightGBM)
      oof_pytorch_ann.csv     # Out-of-fold predictions (PyTorch)
  models/               # Exported models for serving
    champion/1/         # Production-ready model (version 1)
      model.pkl         # Trained model
      pipeline.pkl      # Preprocessing pipeline
      metadata.json     # Model metadata

tests/                         # All test files (37 tests)
train_gbdt_only.py             # GBDT training (XGBoost + LightGBM)
train_pytorch_ann.py           # PyTorch ANN training (recommended)
compare_models.py              # Model comparison tool
export_model_with_pipeline.py # Export model + pipeline for serving
test_prediction.py             # Test predictions with Python
test_request.json              # Sample API request file (curl)
```

---

## ⚡ Common Workflows

### Development Workflow
```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
brew install libomp  # macOS only

# Test
make test-fast

# Train
python3 train_gbdt_only.py

# Compare
python3 compare_models.py --latest
```

### Production Workflow (Complete End-to-End)
```bash
# 1. Train final models
python3 train_gbdt_only.py --folds 10 --seed 42

# 2. Export model with preprocessing pipeline
python3 export_model_with_pipeline.py

# 3. Start API server (in background)
make serve &

# 4. Wait for server to start, then load model
sleep 3
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1"

# 5. Test prediction
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @test_request.json | python3 -m json.tool

# 6. View API documentation
open http://localhost:8000/docs
```


### Train PyTorch ANN
```bash
# Full training with 5-fold CV
python3 train_pytorch_ann.py

# Quick test (3 folds, 20 epochs)
python3 train_pytorch_ann.py --folds 3 --epochs 20

# Custom parameters
python3 train_pytorch_ann.py --epochs 100 --batch-size 128 --learning-rate 0.0005
```

### Demo PyTorch ANN
```bash
# Quick demo to verify it works
python3 demo_pytorch_ann.py
```

### View Results
```bash
# List all runs
ls -lt artifacts/cv/

# View PyTorch ANN summary
cat artifacts/cv/LATEST_RUN/pytorch_ann_summary.json | python3 -m json.tool

# View OOF predictions
head artifacts/cv/LATEST_RUN/oof_pytorch_ann.csv
```


---

## 💡 Key Commands Summary

| Task | Command |
|------|---------|
| **Setup & Testing** | |
| Install dependencies | `pip install -r requirements.txt` |
| Install OpenMP (macOS) | `brew install libomp` |
| Run tests | `make test-fast` |
| **Training** | |
| Train GBDT models | `python3 train_gbdt_only.py` |
| Train PyTorch ANN | `python3 train_pytorch_ann.py` |
| Demo PyTorch ANN | `python3 demo_pytorch_ann.py` |
| **Evaluation** | |
| View results | `cat artifacts/cv/LATEST_RUN/*.csv` |
| Compare models | `python3 compare_models.py --latest` |
| List all runs | `ls -lt artifacts/cv/` |
| **Model Serving** | |
| Export model + pipeline | `python3 export_model_with_pipeline.py` |
| Start API server | `make serve` |
| Load model | `curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion"` |
| Health check | `curl http://localhost:8000/health` |
| Model info | `curl http://localhost:8000/model_info` |
| Make prediction | `curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_request.json` |
| Test with Python | `python3 test_prediction.py` |
| Stop server | `pkill -f "uvicorn src.serve.app"` |

---

## 📊 Model Performance (Expected)

**With Synthetic Data (current):**
- ROC-AUC: 0.60-0.65
- Accuracy: 75-80%

**With Real Customer Data (expected):**
- ROC-AUC: 0.75-0.85
- Accuracy: 80-90%

Current low performance is due to synthetic random data with no real patterns.

---

## ✅ What Works on macOS

✅ **Fully Working:**
- **PyTorch ANN** ( Recommended! Uses Apple Silicon GPU, Best ROC-AUC: 0.6389)
- XGBoost (ROC-AUC: 0.6098)
- LightGBM (ROC-AUC: 0.6297)
- Mock ANN (pure NumPy, no dependencies)
- LogisticRegression
- All preprocessing
- All tests (37/37 passing)
- Model comparison
- API serving

⚠️ **Limited:**
- TensorFlow ANN (mutex error - using PyTorch ANN instead)

---

## 🗑️ Project Cleanup

**Last Cleanup:** 2025-01-13
**Size Reduction:** 3.5 GB → 2.4 MB (99.93% reduction)

**Deleted:**
- Virtual environments (venv/, .venv/)
- Old release backups (releases/v1.0.0/)
- Redundant demo scripts
- Ad-hoc test files
- Temporary documentation
- System/IDE files

**See:** [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for full details.

All functionality preserved - only bloat removed! ✅

---

**Last Updated:** 2025-01-13
**Version:** 2.1 (Cleaned & Optimized)
**Project:** Tabular ML Pipeline for Customer Churn Prediction
