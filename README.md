# Tabular ML Pipeline - Customer Churn Prediction

A production-ready Machine Learning pipeline for tabular data with comprehensive preprocessing, multiple model architectures, cross-validation, calibration, explainability, and REST API serving.

---

## 🎯 Project Overview

• **Problem**: Binary classification for customer churn prediction
• **Target**: Predict probability that customer will churn within 30 days
• **Architecture**: Flexible pipeline supporting ANN (TensorFlow), XGBoost, and LightGBM baselines
• **Status**: Production-ready with complete ML lifecycle support

---

## ✨ Core Features

### Data & Preprocessing
• **Fold-safe preprocessing pipeline** - Prevents data leakage in cross-validation
• **Lightweight preprocessing** - Fast sklearn-only pipeline (no TensorFlow dependency)
• **Feature-specific transformations** - Scale, one-hot, embedding strategies per feature
• **Data contract compliance** - Temporal splits, label latency handling
• **Schema validation** - YAML-based schema with type checking

### Models
• **Tabular ANN** - Neural network with embeddings for categoricals, configurable MLP trunk
• **XGBoost baseline** - Gradient boosting with same CV splits
• **LightGBM baseline** - Fast gradient boosting alternative
• **Calibration support** - Platt scaling and Isotonic regression for probability calibration
• **Model comparison** - Side-by-side metrics (ROC-AUC, PR-AUC, Brier, ECE)

### Training & Evaluation
• **K-Fold cross-validation** - Reproducible splits with per-fold artifacts
• **Nested CV with HPO** - Optional inner-loop hyperparameter optimization
• **OOF predictions** - Out-of-fold predictions for unbiased evaluation
• **Comprehensive metrics** - ROC-AUC, PR-AUC, Brier score, Expected Calibration Error (ECE)
• **Training callbacks** - Early stopping, learning rate reduction, checkpointing

### Explainability
• **Permutation importance** - OOF-based feature importance with confidence intervals
• **SHAP values** - Per-instance explanations with CSV export
• **Visualization notebooks** - Jupyter notebooks for feature analysis
• **Model-agnostic** - Works with any model type

### Model Serving
• **FastAPI REST API** - Production-ready inference server
• **Lazy TensorFlow loading** - Server starts in <1 second (TF loads on-demand)
• **Multi-model support** - TensorFlow SavedModel and sklearn (.pkl) models
• **Calibration in serving** - Automatic calibrated probability output
• **Batch predictions** - Multiple samples in single request

### Model Export
• **Complete artifact export** - SavedModel + calibrator + metadata
• **Version management** - Multiple model versions supported
• **Production-ready** - All artifacts needed for serving

---

## 📁 Project Structure

```
ml_pipeline/
├── src/
│   ├── data/                    # Data loading and schema validation
│   │   ├── loader.py            # YAML schema-based data loader
│   │   └── generate_sample.py   # Synthetic data generation
│   ├── preprocessing/           # Preprocessing pipelines
│   │   ├── pipeline.py          # Full TensorFlow preprocessing
│   │   └── lightweight_transformers.py  # Fast sklearn-only pipeline
│   ├── models/                  # Model architectures
│   │   ├── tabular_ann.py       # ANN with embeddings
│   │   ├── train_utils.py        # Training utilities & callbacks
│   │   └── export.py            # Model export utilities
│   ├── training/                # Training orchestration
│   │   └── cv.py                # KFold CV with nested HPO
│   ├── metrics/                 # Evaluation metrics
│   │   ├── calibration.py       # Platt & Isotonic calibration
│   │   └── eval.py              # ROC-AUC, PR-AUC, Brier, ECE
│   ├── explainability/         # Model interpretability
│   │   ├── permutation.py       # Permutation importance
│   │   └── shap.py              # SHAP value computation
│   ├── baselines/               # Baseline models
│   │   ├── xgb_lgb.py           # XGBoost & LightGBM training
│   │   └── compare_models.py    # Model comparison utilities
│   ├── serve/                   # Model serving
│   │   └── app.py               # FastAPI inference server
│   └── config/                   # Configuration
│       └── tf_config.py         # TensorFlow optimization
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_shap.ipynb            # SHAP explanations visualization
│   └── generate_plots.py        # Plot generation script
├── tests/                       # Comprehensive test suite
│   ├── test_loader.py           # Data loader tests
│   ├── test_preprocessing.py    # Preprocessing tests
│   ├── test_models.py           # Model architecture tests
│   ├── test_cv.py               # Cross-validation tests
│   ├── test_calibration.py     # Calibration tests
│   ├── test_explainability.py   # Explainability tests
│   ├── test_gbdt_baselines.py  # Baseline model tests
│   └── test_serve.py            # API serving tests
├── docs/                        # Documentation
│   ├── problem_spec.md          # Problem specification
│   ├── data_contract.md         # Data contract & requirements
│   ├── eda_summary.md           # EDA findings
│   ├── comparison.md            # Model comparison guide
│   ├── API_DOCUMENTATION.md     # Complete API reference
│   ├── API_QUICK_REFERENCE.md   # Quick API cheat sheet
│   ├── LIGHTWEIGHT_OPTIONS.md   # Lightweight deployment guide
│   └── PYTORCH_VS_TENSORFLOW.md # Framework comparison
├── configs/
│   ├── schema.yaml              # Data schema definition
│   └── feature_list.csv         # Feature preprocessing decisions
├── artifacts/                    # Model outputs
│   ├── cv/                      # Cross-validation runs
│   └── models/                  # Exported models
└── data/
    └── sample.csv                # Sample dataset
```

---

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Install lightweight TensorFlow (CPU-only, faster)
pip install tensorflow-cpu
```

### Run Preprocessing

```bash
# Lightweight pipeline (fast, no TF)
python -m src.preprocessing.lightweight_transformers

# Full pipeline (with TensorFlow)
python -m src.preprocessing.pipeline
```

### Train ANN Model

```bash
# Quick demo
python demo_ann_simple.py

# Full CV run
python -c "
from src.training.cv import run_kfold_cv
# ... load data and run CV
"
```

### Start API Server

```bash
# Start server (starts in <1 second)
./start_server.sh

# Or manually
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Load model and predict
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion"
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": 2.0}]}'
```

---

## 📋 Completed Tasks

### ✅ Task 1: Repository Skeleton + CI Basics
• Project structure with proper Python package layout
• Makefile for common tasks
• Requirements management
• Git workflow setup

### ✅ Task 2: Problem Specification & Data Contract
• `docs/problem_spec.md` - Complete problem definition
• `docs/data_contract.md` - Data requirements and temporal handling
• Schema definition in YAML format

### ✅ Task 3: Sample Dataset + Loader + Tests
• `src/data/loader.py` - Schema-based data loading
• `src/data/generate_sample.py` - Synthetic data generation
• `data/sample.csv` - 500-row sample dataset
• Comprehensive loader tests

### ✅ Task 4: EDA Notebook + Feature Decisions
• `notebooks/01_eda.ipynb` - Complete exploratory analysis
• `docs/eda_summary.md` - EDA findings summary
• `configs/feature_list.csv` - Feature preprocessing decisions
• Visualization plots (correlations, distributions, missing values)

### ✅ Task 5: Fold-Safe Preprocessing Pipeline
• `src/preprocessing/pipeline.py` - Full TensorFlow preprocessing
• `src/preprocessing/lightweight_transformers.py` - Fast sklearn pipeline
• Fold-safe design (fit on train, transform on val)
• Support for scale, one-hot, embedding strategies

### ✅ Task 6: ANN Baseline Model (Keras)
• `src/models/tabular_ann.py` - Tabular ANN with embeddings
• `src/models/train_utils.py` - Training utilities with callbacks
• Configurable architecture (layers, dropout, L2)
• Both probability and logit outputs

### ✅ Task 7: Cross-Validation & Nested CV
• `src/training/cv.py` - KFold CV with OOF predictions
• Optional inner-loop HPO (Grid/Random search)
• Per-fold artifacts (weights, metrics, normalizers)
• Reproducible with fixed seeds

### ✅ Task 8: Calibration & Evaluation Metrics
• `src/metrics/calibration.py` - Platt & Isotonic calibration
• `src/metrics/eval.py` - ROC-AUC, PR-AUC, Brier, ECE
• Automatic best calibrator selection
• Calibration curves and evaluation plots

### ✅ Task 9: Explainability (Permutation + SHAP)
• `src/explainability/permutation.py` - OOF-based permutation importance
• `src/explainability/shap.py` - SHAP value computation
• `notebooks/02_shap.ipynb` - Visualization notebook
• CSV export for external analysis

### ✅ Task 10: GBDT Baselines & Comparison
• `src/baselines/xgb_lgb.py` - XGBoost & LightGBM training
• Same CV splits as ANN for fair comparison
• Automatic calibration for GBDT models
• `docs/comparison.md` - Model comparison guide

### ✅ Task 11: Model Export & Serving
• `src/models/export.py` - Complete model export (SavedModel + calibrator)
• `src/serve/app.py` - FastAPI REST API server
• Lazy TensorFlow loading (server starts in <1s)
• Support for both TensorFlow and sklearn models

---

## 🔧 Key Components

### Preprocessing Pipelines

**Lightweight Pipeline** (Fast, sklearn-only):
• `LightweightNumericTransformer` - Imputation + scaling
• `LightweightCategoricalTransformer` - One-hot encoding
• No TensorFlow dependency - perfect for fast iteration

**Full Pipeline** (TensorFlow):
• `NumericTransformer` - Keras Normalization layers
• `CategoricalTransformer` - StringLookup + embeddings
• Production-ready with full feature support

### Models

**TabularANN**:
• Input layers for each feature type
• Embedding layers for high-cardinality categoricals
• MLP trunk with dropout and L2 regularization
• Dual outputs: probabilities + logits (for calibration)

**XGBoost / LightGBM**:
• Same CV splits as ANN
• Automatic hyperparameter defaults
• Calibration support
• Fast training and inference

### Training Infrastructure

**Cross-Validation**:
• KFold with reproducible splits
• Optional nested CV for hyperparameter search
• Per-fold model artifacts
• OOF predictions for unbiased evaluation

**Callbacks**:
• EarlyStopping - Prevent overfitting
• ReduceLROnPlateau - Adaptive learning rate
• ModelCheckpoint - Save best models
• TrainingLogger - Console metrics

### Evaluation & Calibration

**Metrics**:
• ROC-AUC, PR-AUC (ranking metrics)
• Brier score (probability accuracy)
• Expected Calibration Error (ECE)
• Accuracy, Precision, Recall, F1

**Calibration**:
• Platt scaling (LogisticRegression on logits)
• Isotonic regression (non-parametric)
• Automatic best method selection

### Explainability

**Permutation Importance**:
• OOF-based (no model refitting needed)
• Multiple repeats for confidence intervals
• Feature ranking by importance

**SHAP Values**:
• KernelExplainer for model-agnostic explanations
• Per-instance feature contributions
• CSV export for visualization

### Serving

**FastAPI Server**:
• RESTful API with OpenAPI docs
• Multi-model support (TensorFlow + sklearn)
• Batch predictions
• Health checks and monitoring

---

## 📊 Performance Characteristics

### Startup Times
• **Server startup**: ~0.5 seconds (no TensorFlow)
• **Sklearn model load**: ~1-2 seconds
• **TensorFlow model load**: ~5-10 seconds (first time)
• **After first load**: Subsequent requests <50ms

### Model Training
• **Lightweight preprocessing**: <1 second for 500 samples
• **ANN training**: ~30-60 seconds per fold
• **XGBoost training**: ~5-15 seconds per fold
• **LightGBM training**: ~3-10 seconds per fold

### Memory Usage
• **Server (no model)**: ~50-100 MB
• **With sklearn model**: ~100-200 MB
• **With TensorFlow model**: ~300-500 MB

---

## 🧪 Testing

### Run All Tests
```bash
pytest
```

### Run Specific Test Suites
```bash
# Data loading
pytest tests/test_loader.py

# Preprocessing
pytest tests/test_preprocessing.py
pytest tests/test_lightweight_preprocessing.py

# Models
pytest tests/test_models.py

# Cross-validation
pytest tests/test_cv.py

# Calibration
pytest tests/test_calibration.py

# Explainability
pytest tests/test_explainability.py

# GBDT baselines
pytest tests/test_gbdt_baselines.py

# API serving
pytest tests/test_serve.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| `docs/problem_spec.md` | Problem definition and business requirements |
| `docs/data_contract.md` | Data schema and temporal handling |
| `docs/eda_summary.md` | Exploratory data analysis findings |
| `docs/comparison.md` | Model comparison guide (ANN vs XGB vs LGB) |
| `docs/API_DOCUMENTATION.md` | Complete API reference |
| `docs/API_QUICK_REFERENCE.md` | Quick API cheat sheet |
| `docs/LIGHTWEIGHT_OPTIONS.md` | Lightweight deployment options |
| `docs/PYTORCH_VS_TENSORFLOW.md` | Framework comparison |

---

## 🎯 Usage Examples

### Complete Workflow

```python
# 1. Load data
from src.data.loader import DataLoader
loader = DataLoader("configs/schema.yaml")
df = loader.load_data("data/sample.csv")

# 2. Preprocess
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
pipeline = create_lightweight_pipeline()
pipeline.fit_on(df)
X, y = pipeline.transform(df)

# 3. Train with CV
from src.training.cv import run_kfold_cv
from src.models.tabular_ann import create_tabular_ann

feature_info = pipeline.get_feature_info()
summary = run_kfold_cv(
    X, y, feature_info,
    k_folds=5,
    seed=42,
    base_params={"epochs": 50, "batch_size": 64}
)

# 4. Export model
from src.models.export import export_from_cv_run
export_from_cv_run(
    cv_run_dir=summary["run_dir"],
    fold_idx=0,
    export_dir="artifacts/models/champion"
)

# 5. Serve model
# Start server: uvicorn src.serve.app:app --port 8000
# Load model via API and make predictions
```

### Model Comparison

```python
from src.baselines.xgb_lgb import run_gbdt_cv
from src.baselines.compare_models import compare_models

# Train GBDT baselines
gbdt_results = run_gbdt_cv(X, y, k_folds=5, seed=42)

# Compare all models
comparison = compare_models(
    ann_run_dir="artifacts/cv/ann_run",
    xgb_run_dir="artifacts/cv/gbdt_run"
)
print(comparison)
```

### Explainability

```python
from src.explainability.permutation import compute_permutation_importance_oof
from src.explainability.shap import compute_and_save_shap

# Load OOF predictions
oof_df = pd.read_csv("artifacts/cv/run_name/oof_predictions.csv")

# Permutation importance
perm_importance = compute_permutation_importance_oof(
    X, oof_df['y_true'], oof_df['oof_prob'],
    feature_names=feature_names
)

# SHAP values
shap_results = compute_and_save_shap(
    X[:100], model.predict_fn,
    output_path="artifacts/shap_values.csv"
)
```

---

## 🚀 Deployment

### Development
• Use lightweight preprocessing for fast iteration
• Sklearn models for quick baselines
• Jupyter notebooks for exploration

### Production
• Export models with complete artifacts
• Use FastAPI server with lazy loading
• Deploy with Docker (optional)
• Monitor with health checks

### Optimization Tips
• Use `tensorflow-cpu` for lighter installation
• Sklearn models for fastest serving
• Lazy TensorFlow loading (already implemented)
• Batch predictions for throughput

---

## 📦 Dependencies

### Core
• **numpy, pandas** - Data manipulation
• **scikit-learn** - Preprocessing and baselines
• **tensorflow-cpu** - Neural network models (recommended)
• **fastapi, uvicorn** - API server

### Optional
• **xgboost, lightgbm** - GBDT baselines
• **shap** - SHAP explanations
• **matplotlib, seaborn** - Visualizations
• **jupyter** - Notebooks

See `requirements.txt` for complete list.

---

## 🧪 Testing Status

✅ **All test suites passing:**
• Data loading: ✅
• Preprocessing: ✅
• Models: ✅
• Cross-validation: ✅
• Calibration: ✅
• Explainability: ✅
• GBDT baselines: ✅ (requires xgboost/lightgbm)
• API serving: ✅

---

## 🎓 Learning Resources

• **Problem Understanding**: `docs/problem_spec.md`
• **Data Contract**: `docs/data_contract.md`
• **EDA Analysis**: `notebooks/01_eda.ipynb`
• **SHAP Tutorial**: `notebooks/02_shap.ipynb`
• **API Guide**: `docs/API_DOCUMENTATION.md`

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

Built as a complete ML pipeline demonstrating best practices for:
• Fold-safe preprocessing
• Multiple model architectures
• Comprehensive evaluation
• Model explainability
• Production serving

---

## 🔗 Quick Links

• [API Documentation](docs/API_DOCUMENTATION.md)
• [API Quick Reference](docs/API_QUICK_REFERENCE.md)
• [Model Comparison](docs/comparison.md)
• [Lightweight Options](docs/LIGHTWEIGHT_OPTIONS.md)
