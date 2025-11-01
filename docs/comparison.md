# Model Comparison: ANN vs XGBoost vs LightGBM

## Overview

This document compares the performance of three model architectures:
- **ANN** (Artificial Neural Network): Tabular ANN with embeddings
- **XGBoost**: Gradient boosting with XGBoost
- **LightGBM**: Gradient boosting with LightGBM

All models were trained using the same 5-fold cross-validation splits and evaluated on out-of-fold (OOF) predictions.

## Methodology

1. **Same CV Splits**: All models use identical KFold splits (same seed=42)
2. **Calibration**: All models are calibrated using Platt scaling or Isotonic regression
3. **Metrics**: Evaluation on calibrated OOF predictions
4. **Reproducibility**: Fixed random seeds for deterministic results

## Results Summary

| Model | ROC-AUC | PR-AUC | Brier Score | ECE | Accuracy | Training Time (s/fold) | Calibration Method |
|-------|---------|--------|-------------|-----|----------|----------------------|-------------------|
| **ANN** | TBD | TBD | TBD | TBD | TBD | ~30-60 | Platt/Isotonic |
| **XGBoost** | TBD | TBD | TBD | TBD | TBD | ~5-15 | Isotonic |
| **LightGBM** | TBD | TBD | TBD | TBD | TBD | ~3-10 | Isotonic |

*Note: TBD values are populated after running the full CV pipeline.*

## Per-Fold Metrics

### ANN Baseline

| Fold | ROC-AUC | PR-AUC | Brier | ECE | Accuracy |
|------|---------|--------|-------|-----|----------|
| 0 | TBD | TBD | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD |
| **Mean** | TBD | TBD | TBD | TBD | TBD |
| **Std** | TBD | TBD | TBD | TBD | TBD |

### XGBoost

| Fold | ROC-AUC | PR-AUC | Brier | ECE | Accuracy |
|------|---------|--------|-------|-----|----------|
| 0 | TBD | TBD | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD |
| **Mean** | TBD | TBD | TBD | TBD | TBD |
| **Std** | TBD | TBD | TBD | TBD | TBD |

### LightGBM

| Fold | ROC-AUC | PR-AUC | Brier | ECE | Accuracy |
|------|---------|--------|-------|-----|----------|
| 0 | TBD | TBD | TBD | TBD | TBD |
| 1 | TBD | TBD | TBD | TBD | TBD |
| 2 | TBD | TBD | TBD | TBD | TBD |
| 3 | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD |
| **Mean** | TBD | TBD | TBD | TBD | TBD |
| **Std** | TBD | TBD | TBD | TBD | TBD |

## Model Selection Guidelines

### When to Use ANN

- **Rich categorical features**: When you have high-cardinality categorical features that benefit from embeddings
- **Complex interactions**: When non-linear feature interactions are important
- **Large datasets**: When you have sufficient data for deep learning (typically >10K samples)
- **Feature engineering needed**: When you need to learn feature representations

**Pros:**
- Can learn complex patterns and feature interactions
- Embeddings capture semantic relationships in categorical data
- Flexible architecture for customization

**Cons:**
- Slower training and inference
- Requires more data to generalize
- More hyperparameters to tune
- GPU recommended for large models

### When to Use XGBoost

- **Tabular data with mixed types**: Works well with numeric and low-cardinality categorical features
- **Fast iteration**: When you need quick model development cycles
- **Interpretability**: Feature importance is readily available
- **Robust to outliers**: Less sensitive to outliers than neural networks

**Pros:**
- Fast training (seconds to minutes)
- Good performance out-of-the-box
- Built-in feature importance
- Handles missing values well

**Cons:**
- Limited ability to learn complex feature interactions
- May struggle with very high-cardinality categoricals
- Less flexible than neural networks

### When to Use LightGBM

- **Large datasets**: Efficient on datasets with millions of samples
- **Fast training**: When training speed is critical
- **Memory efficiency**: Lower memory footprint than XGBoost
- **Similar use cases to XGBoost**: When XGBoost is too slow

**Pros:**
- Fastest training time among the three
- Lower memory usage than XGBoost
- Good performance on large datasets
- Built-in categorical feature handling

**Cons:**
- Similar limitations to XGBoost
- May overfit on small datasets
- Less well-documented than XGBoost

## Latency Estimates

| Model | Training Time | Inference Time (single sample) | Batch Inference (1000 samples) |
|-------|--------------|-------------------------------|-------------------------------|
| **ANN** | 30-60s/fold | ~5-10ms | ~50-100ms |
| **XGBoost** | 5-15s/fold | ~1-2ms | ~10-20ms |
| **LightGBM** | 3-10s/fold | ~0.5-1ms | ~5-10ms |

*Note: Actual times depend on model size, hardware, and dataset characteristics.*

## Calibration Impact

All models benefit from calibration:

| Model | Uncalibrated Brier | Calibrated Brier | Improvement |
|-------|-------------------|------------------|-------------|
| **ANN** | TBD | TBD | TBD |
| **XGBoost** | TBD | TBD | TBD |
| **LightGBM** | TBD | TBD | TBD |

## Conclusion

The best model choice depends on:

1. **Dataset size**: Large datasets favor LightGBM or XGBoost; very large datasets favor ANN
2. **Feature types**: High-cardinality categoricals favor ANN with embeddings
3. **Latency requirements**: Real-time inference favors GBDT models
4. **Development time**: Fast iteration favors GBDT models
5. **Interpretability needs**: GBDT models provide clearer feature importance

**Recommended approach:**
- Start with LightGBM or XGBoost for quick baselines
- Try ANN if GBDT performance is insufficient and you have enough data
- Use calibration for all models to improve probability estimates
- Consider ensemble approaches combining multiple models

## Generating This Report

To regenerate this comparison table with actual results, run:

```bash
# Train ANN baseline
python -m src.training.cv <args>

# Train GBDT baselines
python -c "from src.baselines.xgb_lgb import run_gbdt_cv; run_gbdt_cv(X, y, ...)"

# Generate comparison
python -c "from src.baselines.compare_models import compare_models; df = compare_models(...); print(df)"
```

## References

- XGBoost: [Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)
- LightGBM: [Ke et al., 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree)
- Calibration: [Platt, 1999](https://www.cs.cornell.edu/~alexn/papers/calibration.icml99.camera.review.pdf)
