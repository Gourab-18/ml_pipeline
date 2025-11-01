# Lightweight TensorFlow & Serving Options

## Option 1: tensorflow-cpu (Recommended)

**Much lighter than full TensorFlow** - no GPU dependencies, smaller footprint:

```bash
pip install tensorflow-cpu
```

**Benefits:**
- ✅ ~50% smaller than full TensorFlow
- ✅ No CUDA/GPU dependencies
- ✅ Still supports SavedModel format
- ✅ Faster installation
- ⚠️ Still has some initialization overhead (5-10s on macOS)

**When to use:**
- You need TensorFlow models (ANN)
- You don't need GPU acceleration
- You want smaller footprint

---

## Option 2: sklearn-only Serving (Fastest!)

For serving sklearn baselines (XGBoost, LightGBM, LogisticRegression), you can use **pure sklearn** - no TensorFlow needed!

### Create sklearn-serving app

The sklearn models are much faster to load:
- XGBoost/LightGBM: ~100ms load time
- LogisticRegression: ~10ms load time
- No TensorFlow dependency

**Example:**
```python
import pickle
import joblib
from sklearn.linear_model import LogisticRegression

# Load sklearn model (instant)
model = joblib.load("model.pkl")
calibrator = pickle.load(open("calibrator.pkl", "rb"))

# Predict (very fast)
proba = model.predict_proba(X)[:, 1]
proba_cal = calibrator.predict_proba(proba)
```

---

## Option 3: Hybrid Approach

**Current setup (already implemented):**
- Server starts instantly (no TF at startup)
- TensorFlow loads only when ANN model is loaded
- Can also serve sklearn models (just need to add endpoint)

---

## Comparison

| Option | Startup Time | Memory | Model Types | Best For |
|--------|-------------|--------|-------------|----------|
| **Full TensorFlow** | 10-30s | High | ANN | GPU training, production ANN |
| **tensorflow-cpu** | 5-10s | Medium | ANN | CPU inference, smaller footprint |
| **sklearn only** | <1s | Low | XGB/LGB/LR | Fast serving, no ANN needed |
| **Hybrid (current)** | <1s* | Low→High | All | Flexible, lazy loading |

*Starts instantly, TF loads only when model loaded

---

## Quick Start with tensorflow-cpu

```bash
# Install lightweight version
pip uninstall tensorflow  # if installed
pip install tensorflow-cpu

# Server starts instantly, TF loads on demand
uvicorn src.serve.app:app --port 8000
```

---

## Quick Start with sklearn-only Serving

For XGBoost/LightGBM models, you don't need TensorFlow at all:

```python
# Save sklearn model
import joblib
joblib.dump(model, "model.pkl")

# Load and serve (no TF needed)
model = joblib.load("model.pkl")
# Use same FastAPI structure, just different model.load()
```

---

## Recommendation

**For development/testing:**
- Use **tensorflow-cpu** (lighter, still supports ANN)
- Server already lazy-loads it

**For production sklearn models:**
- Use **pure sklearn** serving (no TF needed)
- Can use our existing calibration module (no TF dependency)

**For production ANN models:**
- Use **tensorflow-cpu** or full TensorFlow
- Lazy loading means startup is still fast

---

## Current Server Behavior

✅ **Server starts in <1 second** (no TF at startup)
✅ **TensorFlow loads only when needed** (when model is loaded)
✅ **Works with tensorflow-cpu** (just install it instead of tensorflow)

You're already optimized! Just swap `tensorflow` → `tensorflow-cpu` for lighter installation.
