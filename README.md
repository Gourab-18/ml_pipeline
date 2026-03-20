# Tabular ML Pipeline - Customer Churn Prediction

A Machine Learning pipeline for tabular data with preprocessing, multiple model architectures, cross-validation, and calibration.

---

## 🎯 Project Overview

- **Problem**: Binary classification for customer churn prediction
- **Target**: Predict probability that customer will churn within 30 days
- **Models**: PyTorch ANN, XGBoost, LightGBM
- **Preprocessing**: Fold-safe sklearn-only pipeline (no TensorFlow)

---

## 📁 Project Structure

```
ml_pipeline/
├── src/
│   ├── data/
│   │   ├── loader.py                    # YAML schema-based data loader
│   │   └── generate_sample.py           # Synthetic data generation
│   ├── preprocessing/
│   │   └── lightweight_transformers.py  # Fold-safe sklearn pipeline
│   ├── models/
│   │   └── pytorch_ann.py               # PyTorch ANN (macOS/Apple Silicon)
│   ├── metrics/
│   │   ├── calibration.py               # Platt & Isotonic calibration
│   │   └── eval.py                      # ROC-AUC, PR-AUC, Brier, ECE
│   ├── explainability/
│   │   └── permutation.py               # Permutation feature importance
│   └── baselines/
│       ├── xgb_lgb.py                   # XGBoost & LightGBM training
│       └── compare_models.py            # Model comparison utilities
├── notebooks/
│   ├── 01_eda.ipynb                     # Exploratory data analysis
│   └── generate_plots.py                # Plot generation script
├── tests/
│   ├── test_loader.py
│   ├── test_lightweight_preprocessing.py
│   ├── test_calibration.py
│   ├── test_explainability.py
│   └── test_gbdt_baselines.py
├── docs/                                # Documentation
├── configs/
│   ├── schema.yaml                      # Data schema definition
│   └── feature_list.csv                 # Feature preprocessing decisions
├── data/
│   └── sample.csv                       # Sample dataset
├── train_pytorch_ann.py                 # Train PyTorch ANN with CV
├── train_gbdt_only.py                   # Train XGBoost + LightGBM
├── compare_models.py                    # Compare CV run results
├── demo_pytorch_ann.py                  # Quick ANN demo
└── export_model_with_pipeline.py        # Export trained model
```

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip3 install -r requirements.txt
```

### Run Preprocessing

```bash
python3 -m src.preprocessing.lightweight_transformers
```

### Train Models

```bash
# Train PyTorch ANN with 5-fold CV
python3 train_pytorch_ann.py

# Train XGBoost + LightGBM baselines
python3 train_gbdt_only.py

# Quick demo
python3 demo_pytorch_ann.py
```

### Compare Models

```bash
# Compare latest runs
python3 compare_models.py --latest

# Compare all matching runs
python3 compare_models.py --all

# Compare specific run directories
python3 compare_models.py artifacts/cv/run1 artifacts/cv/run1
```

---

## 🔧 Key Components

### Preprocessing

`LightweightPreprocessingPipeline` (sklearn-only, no TensorFlow):
- `LightweightNumericTransformer` — Imputation + StandardScaler
- `LightweightCategoricalTransformer` — One-hot encoding
- Fold-safe: always fit on train, transform val/test

### Models

**PyTorchANN** (`src/models/pytorch_ann.py`):
- Works on macOS / Apple Silicon (MPS support)
- Configurable hidden layers, dropout, L2
- Sklearn-compatible interface

**XGBoost / LightGBM** (`src/baselines/xgb_lgb.py`):
- Same CV splits as ANN for fair comparison
- Automatic calibration support

### Evaluation & Calibration

- ROC-AUC, PR-AUC, Brier score, ECE
- Platt scaling (LogisticRegression on logits)
- Isotonic regression (non-parametric)
- Automatic best calibrator selection

### Explainability

- Permutation importance (OOF-based, no model refitting)
- Feature ranking with confidence intervals

---

## 📊 Workflow

```
data/sample.csv
    ↓
src/data/loader.py          (schema validation)
    ↓
lightweight_transformers.py (fold-safe preprocessing)
    ↓
pytorch_ann.py              (ANN training with CV)
xgb_lgb.py                  (GBDT baselines with CV)
    ↓
calibration.py              (probability calibration)
eval.py                     (metrics)
    ↓
compare_models.py           (side-by-side comparison)
    ↓
artifacts/cv/<run>/model_comparison.csv
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific suites
pytest tests/test_loader.py
pytest tests/test_lightweight_preprocessing.py
pytest tests/test_calibration.py
pytest tests/test_explainability.py
pytest tests/test_gbdt_baselines.py
```

---

## 📦 Dependencies

- `torch` — PyTorch ANN
- `xgboost`, `lightgbm` — GBDT baselines
- `scikit-learn`, `numpy`, `pandas` — Preprocessing & data
- `scipy`, `matplotlib`, `seaborn` — Metrics & plots
- `pyyaml` — Config loading

See `requirements.txt` for full list.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `docs/problem_spec.md` | Problem definition and business requirements |
| `docs/data_contract.md` | Data schema and temporal handling |
| `docs/eda_summary.md` | Exploratory data analysis findings |
| `docs/comparison.md` | Model comparison guide |
| `docs/PYTORCH_VS_TENSORFLOW.md` | Why PyTorch was chosen |
| `docs/LIGHTWEIGHT_OPTIONS.md` | Lightweight deployment options |

---

## 📝 License

MIT License
