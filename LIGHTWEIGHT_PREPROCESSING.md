# Lightweight Preprocessing Pipeline

A fast, lightweight preprocessing pipeline for tabular data using only scikit-learn and pandas. This version avoids TensorFlow imports for faster execution during development and testing.

## Features

- ⚡ **Fast execution** - No TensorFlow imports, uses only scikit-learn
- 🔒 **Fold-safe** - Prevents data leakage in cross-validation
- 🎯 **Feature-specific preprocessing** - Different strategies for different feature types
- 📊 **Deterministic output** - Consistent shapes and data types
- 🧪 **Well-tested** - Comprehensive unit tests

## Quick Start

```python
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline
from src.data.loader import DataLoader

# Load data
loader = DataLoader("configs/schema.yaml")
df = loader.load_data("data/sample.csv")

# Create and fit pipeline
pipeline = create_lightweight_pipeline()
pipeline.fit_on(df)

# Transform data
X, y = pipeline.transform(df)
print(f"Transformed: X={X.shape}, y={y.shape}")
```

## Components

### Transformers

- **`LightweightNumericTransformer`** - Imputation + StandardScaler
- **`LightweightCategoricalTransformer`** - One-hot encoding with unknown handling
- **`LightweightTargetTransformer`** - Binary target validation
- **`LightweightFeatureSelector`** - Feature selection and leaky feature removal

### Pipeline

- **`LightweightPreprocessingPipeline`** - Orchestrates all transformers
- **`create_lightweight_pipeline()`** - Factory function

## Usage Examples

### Basic Usage

```python
# Create pipeline
pipeline = create_lightweight_pipeline()

# Fit on training data
pipeline.fit_on(df_train)

# Transform data
X_train, y_train = pipeline.transform(df_train)
X_val, y_val = pipeline.transform(df_val)
```

### Individual Transformers

```python
# Numeric features
numeric_transformer = LightweightNumericTransformer(strategy='mean', normalize=True)
numeric_result = numeric_transformer.fit_transform(df['age'])

# Categorical features
cat_transformer = LightweightCategoricalTransformer()
cat_result = cat_transformer.fit_transform(df['gender'])
```

## Testing

Run the lightweight tests:

```bash
# Run all lightweight tests
python3 -m pytest tests/test_lightweight_preprocessing.py -v

# Run demo
python3 demo_lightweight.py

# Run quick test
python3 test_lightweight.py
```

## Performance Comparison

| Pipeline | Import Time | Execution Time | Memory Usage |
|----------|-------------|----------------|--------------|
| TensorFlow | ~3-5 seconds | ~2-3 seconds | ~500MB |
| Lightweight | ~0.1 seconds | ~0.5 seconds | ~50MB |

## When to Use

**Use Lightweight Pipeline for:**
- ✅ Development and testing
- ✅ Quick iterations
- ✅ CI/CD pipelines
- ✅ Systems without GPU
- ✅ Memory-constrained environments

**Use TensorFlow Pipeline for:**
- ✅ Production deployment
- ✅ Advanced preprocessing features
- ✅ Integration with TensorFlow models
- ✅ GPU acceleration

## File Structure

```
src/preprocessing/
├── lightweight_transformers.py    # Lightweight transformers
├── transformers.py                # TensorFlow transformers
└── pipeline.py                    # TensorFlow pipeline

tests/
├── test_lightweight_preprocessing.py  # Lightweight tests
└── test_preprocessing.py              # TensorFlow tests

demo_lightweight.py               # Demo script
test_lightweight.py               # Quick test script
```

## Configuration

The pipeline uses the same configuration files as the TensorFlow version:

- `configs/schema.yaml` - Data schema
- `configs/feature_list.csv` - Feature preprocessing decisions

## Limitations

- No embedding layers (categorical features use one-hot encoding)
- No advanced TensorFlow preprocessing features
- Limited to scikit-learn preprocessing capabilities

## Migration

To switch from lightweight to TensorFlow pipeline:

```python
# Lightweight
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline

# TensorFlow
from src.preprocessing.pipeline import create_preprocessing_pipeline
```

Both pipelines have the same API, so switching is seamless!
