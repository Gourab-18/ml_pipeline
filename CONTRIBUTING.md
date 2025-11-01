# Contributing Guide

This guide explains how to contribute to the Tabular ML Pipeline project, including how to run full training, reproduce cross-validation results, and export models.

---

## 📋 Table of Contents

- [Development Setup](#development-setup)
- [Running Full Training](#running-full-training)
- [Reproducing Cross-Validation](#reproducing-cross-validation)
- [Exporting Models](#exporting-models)
- [Testing](#testing)
- [Code Standards](#code-standards)
- [Pull Request Process](#pull-request-process)

---

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ml_pipeline
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
make setup-venv
# or manually
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
make install
# or manually
pip install -r requirements.txt
pip install -e .
```

### 4. Verify Installation

```bash
make test
```

---

## Running Full Training

### Quick Training (Sample Dataset)

For quick iteration and testing:

```bash
make train-sample
```

This runs a quick training on the sample dataset (`data/sample.csv`) with minimal epochs.

### Full Training Pipeline

For production-ready training with full cross-validation:

```bash
make train-full
```

**What it does:**
1. Loads data from `data/sample.csv`
2. Applies lightweight preprocessing (sklearn-based, fast)
3. Runs 5-fold cross-validation with ANN model
4. Trains XGBoost and LightGBM baselines
5. Applies calibration (Platt/Isotonic)
6. Generates evaluation metrics and plots
7. Saves all artifacts to `artifacts/cv/<run_name>/`

**Configuration:**
- **K-folds**: 5 (default)
- **Random seed**: 42 (for reproducibility)
- **Epochs**: 50 (ANN)
- **Batch size**: 64 (ANN)

**Output:**
- Per-fold model weights and normalizers
- OOF (out-of-fold) predictions CSV
- Calibration plots and metrics
- Evaluation plots (ROC, PR curves, calibration curves)
- Summary JSON with all metrics

### Manual Training (Python)

```python
from src.data.loader import DataLoader
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
from src.training.cv import run_kfold_cv
from src.baselines.xgb_lgb import run_gbdt_cv

# 1. Load data
loader = DataLoader("configs/schema.yaml")
df = loader.load_data("data/sample.csv")

# 2. Preprocess
pipeline = create_lightweight_pipeline()
pipeline.fit_on(df)
X, y = pipeline.transform(df)
feature_info = pipeline.get_feature_info()

# 3. Train ANN with CV
summary = run_kfold_cv(
    X=X,
    y=y,
    feature_info=feature_info,
    k_folds=5,
    seed=42,
    base_params={
        "epochs": 50,
        "batch_size": 64,
        "embedding_dims": None,
        "hidden_layers": [128, 64, 32],
        "dropout_rate": 0.3,
        "l2_reg": 0.001,
        "learning_rate": 0.001
    },
    artifacts_dir="artifacts/cv",
    run_name="my_training_run"
)

print(f"Training complete! Artifacts saved to: {summary['run_dir']}")
print(f"OOF ROC-AUC: {summary['metrics']['roc_auc']:.4f}")

# 4. Train GBDT baselines
gbdt_results = run_gbdt_cv(
    X=X,
    y=y,
    k_folds=5,
    seed=42,
    run_name="my_gbdt_run"
)
```

---

## Reproducing Cross-Validation

### Reproducibility Guarantees

The pipeline uses fixed random seeds to ensure reproducibility:

- **Data splitting**: `random_state=42` in KFold
- **Model initialization**: `random_seed=42` in TabularANN
- **TensorFlow**: `tf.random.set_seed(42)` before training
- **NumPy**: `np.random.seed(42)` before data operations

### Reproducing Results

To reproduce exact CV results:

```bash
# 1. Ensure same seed
export SEED=42

# 2. Run training
make train-full

# 3. Expected OOF metrics (within tolerance):
# - ROC-AUC: ~0.85-0.90 (varies by dataset)
# - PR-AUC: ~0.70-0.80
# - Brier Score: ~0.15-0.20
# - ECE (before calibration): ~0.05-0.15
# - ECE (after calibration): <0.05
```

### Tolerance Thresholds

When comparing results, expect small variations due to:
- Floating-point precision differences
- Platform-specific implementations (macOS vs Linux)
- TensorFlow version differences

**Acceptable tolerances:**
- ROC-AUC: ±0.01
- PR-AUC: ±0.02
- Brier Score: ±0.01
- ECE: ±0.02

### Verifying Reproducibility

```bash
# Run twice with same seed
SEED=42 make train-full
# ... wait for completion ...

# Run again
SEED=42 make train-full

# Compare OOF metrics in:
# artifacts/cv/<run_name>/summary.json
```

The OOF predictions should match exactly (same splits, same seed).

---

## Exporting Models

### Using Makefile

```bash
# Export champion model from latest CV run
make export

# Or specify CV run and fold
make export RUN_NAME=20240101_120000 FOLD_IDX=0
```

### Manual Export (Python)

```python
from src.models.export import export_from_cv_run

# Export from CV run
export_from_cv_run(
    cv_run_dir="artifacts/cv/my_training_run",
    fold_idx=0,  # Export fold 0
    export_dir="artifacts/models/champion",
    version="1"
)
```

**What gets exported:**
- TensorFlow SavedModel (`saved_model/`)
- Calibrator pickle (`calibrator.pkl`)
- Metadata JSON (`metadata.json`) with:
  - Feature info
  - Calibration method
  - Model version
  - Training metrics

### Export Directory Structure

```
artifacts/models/champion/
└── 1/
    ├── saved_model/
    │   ├── assets/
    │   ├── variables/
    │   └── saved_model.pb
    ├── calibrator.pkl
    └── metadata.json
```

### Validating Export

```python
from src.models.export import load_model

# Load exported model
model_info = load_model("artifacts/models/champion", version="1")
print(model_info['feature_info'])
print(model_info['calibrator_method'])

# Test prediction
# ... load data and predict ...
```

---

## Testing

### Run All Tests

```bash
make test
```

### Run Specific Test Suites

```bash
# Data loading
pytest tests/test_loader.py -v

# Preprocessing
pytest tests/test_preprocessing.py -v

# Models
pytest tests/test_models.py -v

# Cross-validation
pytest tests/test_cv.py -v

# Calibration
pytest tests/test_calibration.py -v

# Explainability
pytest tests/test_explainability.py -v

# GBDT baselines
pytest tests/test_gbdt_baselines.py -v

# API serving
pytest tests/test_serve.py -v
```

### Test Coverage

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html  # macOS
# or: xdg-open htmlcov/index.html  # Linux
```

**Coverage Target**: Minimum 80% coverage for core modules.

### Running Tests in CI

Tests run automatically in GitHub Actions on:
- Pull requests
- Pushes to `main` branch
- Manual workflow triggers

CI runs:
- `make test` (all tests with coverage)
- `make lint` (code quality checks)

---

## Code Standards

### Code Formatting

```bash
# Check formatting
make lint

# Auto-format
make format
```

**Tools used:**
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Code Style

- Follow PEP 8
- Use type hints for function signatures
- Document all public functions with docstrings
- Keep functions focused and small (<100 lines when possible)
- Write descriptive variable names

### Example

```python
from typing import Dict, Any, Optional
import numpy as np

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train a model on the given data.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        params: Optional hyperparameters
    
    Returns:
        Dictionary with training metrics and model artifacts
    """
    # Implementation...
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/my-feature
# or
git checkout -b fix/my-bugfix
```

### 2. Make Changes

- Write code following style guidelines
- Add tests for new functionality
- Update documentation if needed

### 3. Test Locally

```bash
# Run tests
make test

# Check linting
make lint

# Fix any issues
make format  # Auto-fix formatting
```

### 4. Commit Changes

```bash
git add .
git commit -m "Add: feature description"
```

**Commit Message Format:**
- `Add: ` - New feature
- `Fix: ` - Bug fix
- `Update: ` - Update existing feature
- `Refactor: ` - Code refactoring
- `Docs: ` - Documentation changes

### 5. Push and Create PR

```bash
git push origin feature/my-feature
```

Create a pull request on GitHub with:
- Clear description of changes
- Link to related issues (if any)
- Screenshots/examples (if UI changes)
- Checklist of completed items

### 6. PR Review

- Address review comments
- Ensure all CI checks pass
- Update PR as needed

### 7. Merge

- Squash and merge (preferred)
- Delete feature branch after merge

---

## Development Tips

### Fast Iteration

- Use `make train-sample` for quick tests
- Use lightweight preprocessing (no TensorFlow) for faster development
- Test individual components before full pipeline

### Debugging

- Enable verbose logging: Set `TF_CPP_MIN_LOG_LEVEL=0`
- Use debugger: `python -m pdb script.py`
- Check artifacts: `artifacts/cv/<run_name>/` for intermediate results

### Performance Profiling

```bash
# Profile training time
python -m cProfile -o profile.stats demo_ann_simple.py
python -m pstats profile.stats
```

---

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Open an issue on GitHub
- **Questions**: Ask in PR comments or discussions

---

## Checklist for Contributors

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass (`make test`)
- [ ] Linting passes (`make lint`)
- [ ] Coverage maintained/improved
- [ ] PR description is clear
- [ ] Commit messages follow format

---


