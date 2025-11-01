# Deployment Handover Guide

Complete guide for deploying the Tabular ML Pipeline to production.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Branch Strategy](#branch-strategy)
- [Model Versioning](#model-versioning)
- [Deployment Artifacts](#deployment-artifacts)
- [Deployment Steps](#deployment-steps)
- [Monitoring Checklist](#monitoring-checklist)
- [Rollback Procedure](#rollback-procedure)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers:
- Which branch to deploy from
- Model versioning strategy
- Artifacts to copy for deployment
- Monitoring metrics and alarms
- Rollback procedures

**Target Environment**: Production serving API

---

## Branch Strategy

### Deployment Branch

**Primary**: `main` branch (stable, tested code)

**Process:**
1. All features developed in feature branches
2. Merged to `main` after PR review and tests pass
3. `main` is always deployable
4. Tag releases: `v1.0.0`, `v1.1.0`, etc.

### Branch Verification

```bash
# Ensure you're on main
git checkout main
git pull origin main

# Verify latest commit
git log -1

# Check for uncommitted changes
git status
```

### Release Tags

Deploy specific versions using tags:

```bash
# List available tags
git tag -l

# Checkout specific release
git checkout v1.0.0
```

**Recommendation**: Always deploy from tagged releases for production.

---

## Model Versioning

### Version Format

**Format**: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes (different feature schema)
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes, recalibration

### Version Location

Models are versioned in the export directory:

```
artifacts/models/champion/
├── 1/           # Version 1
│   ├── saved_model/
│   ├── calibrator.pkl
│   └── metadata.json
├── 2/           # Version 2
│   └── ...
```

### Version in Metadata

Each model's `metadata.json` contains:

```json
{
  "version": "1",
  "model_type": "TabularANN",
  "calibrator_method": "isotonic",
  "training_date": "2024-01-15T10:30:00",
  "cv_metrics": {
    "roc_auc": 0.87,
    "pr_auc": 0.72,
    "brier": 0.18,
    "ece": 0.03
  }
}
```

### Version Selection

**In Production:**
- Load specific version via API: `/load?version=1`
- Support multiple versions simultaneously (A/B testing)
- Default to latest stable version

**Best Practice**: Keep at least 2 versions deployed for rollback capability.

---

## Deployment Artifacts

### Required Artifacts

Copy these to production:

```
production/
├── model/
│   └── champion/
│       └── 1/              # Model version
│           ├── saved_model/
│           │   ├── assets/
│           │   ├── variables/
│           │   └── saved_model.pb
│           ├── calibrator.pkl
│           └── metadata.json
├── configs/
│   ├── schema.yaml          # Data schema
│   └── feature_list.csv    # Feature preprocessing config
├── src/                     # Source code (for serving)
└── requirements.txt         # Dependencies
```

### Artifact Checklist

- [ ] Model SavedModel directory (`saved_model/`)
- [ ] Calibrator pickle (`calibrator.pkl`)
- [ ] Metadata JSON (`metadata.json`)
- [ ] Schema YAML (`configs/schema.yaml`)
- [ ] Feature list CSV (`configs/feature_list.csv`)
- [ ] Source code (`src/`)
- [ ] Requirements file (`requirements.txt`)

### Artifact Verification

```bash
# Verify model structure
ls -R artifacts/models/champion/1/

# Check metadata
cat artifacts/models/champion/1/metadata.json | python -m json.tool

# Verify calibrator
python -c "
import pickle
with open('artifacts/models/champion/1/calibrator.pkl', 'rb') as f:
    cal = pickle.load(f)
print('✅ Calibrator loaded:', type(cal).__name__)
"
```

---

## Deployment Steps

### Step 1: Prepare Artifacts

```bash
# On development machine
cd ml_pipeline

# Export champion model (if not already done)
make export

# Create deployment package
mkdir -p deployment/v1.0.0
cp -r artifacts/models/champion deployment/v1.0.0/model
cp -r configs deployment/v1.0.0/
cp requirements.txt deployment/v1.0.0/
cp -r src deployment/v1.0.0/

# Create ZIP
cd deployment
zip -r v1.0.0.zip v1.0.0/
```

### Step 2: Transfer to Production

```bash
# Transfer ZIP to production server
scp deployment/v1.0.0.zip user@production-server:/opt/ml_pipeline/

# On production server
cd /opt/ml_pipeline
unzip v1.0.0.zip
```

### Step 3: Set Up Environment

```bash
# On production server
cd /opt/ml_pipeline/v1.0.0

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Start Server

```bash
# Using uvicorn directly
uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Or using systemd service (recommended)
sudo systemctl start ml-pipeline-api

# Or using Docker (if containerized)
docker run -p 8000:8000 -v $(pwd)/model:/app/model ml-pipeline:latest
```

### Step 5: Load Model

```bash
# Load model via API
curl -X POST "http://localhost:8000/load?model_dir=/opt/ml_pipeline/v1.0.0/model/champion&version=1"

# Verify model loaded
curl http://localhost:8000/health
curl http://localhost:8000/model_info
```

### Step 6: Verify Predictions

```bash
# Test prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      {"feature_0": 1.0, "feature_1": 2.0, "feature_2": 0.5}
    ]
  }'

# Verify response format and values
```

### Step 7: Health Checks

```bash
# Set up health check monitoring
# Check every 30 seconds
watch -n 30 'curl -s http://localhost:8000/health | python -m json.tool'
```

---

## Monitoring Checklist

### Metrics to Watch

#### 1. Server Health

**Endpoint**: `GET /health`

**Monitor:**
- ✅ Status: `"healthy"`
- ✅ Model loaded: `true`

**Alarm if:**
- Status != `"healthy"`
- Model loaded = `false` (after deployment window)

**Frequency**: Every 30 seconds

#### 2. Prediction Latency

**Metric**: Response time from `/predict` endpoint

**Targets:**
- **P50 (median)**: < 50ms
- **P95**: < 100ms
- **P99**: < 200ms

**Alarm if:**
- P95 > 200ms for 5 minutes
- P99 > 500ms for 2 minutes

#### 3. Prediction Volume

**Metric**: Requests per second (RPS)

**Monitor:**
- Average RPS
- Peak RPS
- Request rate trends

**Alarm if:**
- Sudden drop > 50% (potential outage)
- Sudden spike > 200% (potential attack/bug)

#### 4. Expected Calibration Error (ECE)

**Metric**: ECE from calibration monitoring

**Target:**
- **ECE**: < 0.05 (well-calibrated)
- **Warning**: 0.05 ≤ ECE < 0.10
- **Alarm**: ECE ≥ 0.10

**Calculation:**
- Monitor predictions over time window (e.g., 1 hour)
- Bin predictions and compare to actual outcomes
- Compute ECE

**Alarm if:**
- ECE ≥ 0.10 for 2 consecutive hours
- ECE increasing trend over 24 hours

#### 5. Population Stability Index (PSI)

**Metric**: PSI comparing training vs production data

**Target:**
- **PSI < 0.1**: No significant shift
- **PSI 0.1-0.25**: Minor shift (monitor)
- **PSI > 0.25**: Significant shift (investigate)

**Calculation:**
```python
# Compare feature distributions
# Training data (baseline)
# Production data (monitoring window)

PSI = sum((prod_pct - train_pct) * log(prod_pct / train_pct))
```

**Alarm if:**
- PSI > 0.25 for any feature
- PSI > 0.15 for > 3 features simultaneously

**Frequency**: Daily calculation

#### 6. Prediction Distribution

**Monitor:**
- Mean prediction value
- Prediction variance
- Extreme predictions (prob > 0.95 or < 0.05)

**Alarm if:**
- Mean prediction shifts > 0.1 from baseline
- Variance increases > 50%
- Extreme predictions > 20% of total

#### 7. Error Rates

**Monitor:**
- HTTP 4xx errors (client errors)
- HTTP 5xx errors (server errors)
- Prediction failures (JSON parsing errors, missing features)

**Alarm if:**
- Error rate > 1% for 5 minutes
- 5xx errors > 0.1% for 10 minutes
- Continuous failures (potential model loading issue)

#### 8. Model Performance Metrics

**If labels available in production:**

**Monitor:**
- ROC-AUC (if can compute)
- Precision/Recall
- Actual vs predicted calibration

**Alarm if:**
- ROC-AUC drops > 0.05 from training
- Precision drops > 10 percentage points

**Note**: Requires labeled data (may have delay)

---

## Alarms Configuration

### Alarm Thresholds Summary

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| ECE | ≥ 0.05 | ≥ 0.10 | Recalibrate or retrain |
| PSI | ≥ 0.15 | ≥ 0.25 | Investigate data drift |
| P95 Latency | > 100ms | > 200ms | Scale or optimize |
| Error Rate | > 0.5% | > 1% | Investigate failures |
| Health Status | - | != healthy | Immediate response |

### Alert Channels

- **Critical**: PagerDuty / On-call
- **Warning**: Slack channel
- **Info**: Dashboard only

---

## Monitoring Dashboard

**Recommended Metrics Dashboard:**

```
┌─────────────────────────────────────────┐
│  ML Pipeline Monitoring Dashboard       │
├─────────────────────────────────────────┤
│ Health: [✓ Healthy] Model: [✓ Loaded]   │
│                                         │
│ Prediction Latency                      │
│ P50: 45ms | P95: 95ms | P99: 180ms     │
│                                         │
│ Requests                                │
│ Current: 120 req/min                    │
│                                         │
│ Model Metrics                           │
│ ECE: 0.03 ✓                            │
│ PSI: 0.08 ✓                            │
│ Mean Prediction: 0.35                   │
│                                         │
│ Errors                                  │
│ 4xx: 0.02% | 5xx: 0.01% ✓              │
└─────────────────────────────────────────┘
```

**Update Frequency**: Real-time (every 30 seconds)

---

## Rollback Procedure

### Quick Rollback (< 5 minutes)

```bash
# 1. Stop current server
sudo systemctl stop ml-pipeline-api
# or
pkill -f "uvicorn src.serve.app"

# 2. Load previous model version
curl -X POST "http://localhost:8000/load?model_dir=/opt/ml_pipeline/model/champion&version=1"

# 3. Verify
curl http://localhost:8000/health
curl http://localhost:8000/model_info

# 4. Restart server if needed
sudo systemctl start ml-pipeline-api
```

### Full Rollback (Previous Release)

```bash
# 1. Stop server
sudo systemctl stop ml-pipeline-api

# 2. Checkout previous release
cd /opt/ml_pipeline
git checkout v0.9.0  # Previous version

# 3. Reinstall dependencies (if changed)
source venv/bin/activate
pip install -r requirements.txt

# 4. Load previous model
curl -X POST "http://localhost:8000/load?model_dir=/opt/ml_pipeline/v0.9.0/model/champion&version=1"

# 5. Restart server
sudo systemctl start ml-pipeline-api

# 6. Verify
curl http://localhost:8000/health
```

---

## Troubleshooting

### Model Won't Load

**Symptoms:**
- `/load` returns 400 error
- Model info shows "no model loaded"

**Diagnosis:**
```bash
# Check model directory exists
ls -la /opt/ml_pipeline/model/champion/1/

# Verify files present
ls -la /opt/ml_pipeline/model/champion/1/saved_model/
ls -la /opt/ml_pipeline/model/champion/1/calibrator.pkl
ls -la /opt/ml_pipeline/model/champion/1/metadata.json

# Check permissions
chmod -R 755 /opt/ml_pipeline/model/
```

**Solutions:**
- Verify artifact paths
- Check file permissions
- Ensure TensorFlow is installed (for TensorFlow models)

### Predictions Failing

**Symptoms:**
- `/predict` returns 500 error
- JSON parsing errors

**Diagnosis:**
```bash
# Check server logs
journalctl -u ml-pipeline-api -n 50

# Test with minimal input
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0}]}'
```

**Solutions:**
- Verify feature names match schema
- Check feature types (must be numbers)
- Ensure model is loaded

### High Latency

**Symptoms:**
- P95 latency > 200ms
- Slow response times

**Solutions:**
- Scale horizontally (multiple instances)
- Use GPU (if available)
- Optimize model (quantization, pruning)
- Cache predictions for common inputs

### Model Drift (High PSI)

**Symptoms:**
- PSI > 0.25
- Performance degradation

**Solutions:**
- Retrain model on recent data
- Update feature preprocessing
- Investigate data source changes
- Check for data pipeline bugs

### Calibration Degradation (High ECE)

**Symptoms:**
- ECE > 0.10
- Predictions not calibrated

**Solutions:**
- Recalibrate on recent production data
- Retrain model
- Check for distribution shifts

---

## Post-Deployment Checklist

### Immediately After Deployment

- [ ] Server health check passes
- [ ] Model loads successfully
- [ ] Test predictions work
- [ ] Monitoring alerts configured
- [ ] Logs are being collected

### First Hour

- [ ] Monitor prediction latency
- [ ] Check error rates
- [ ] Verify prediction distribution
- [ ] Review server logs

### First Day

- [ ] Calculate PSI (if possible)
- [ ] Monitor ECE trend
- [ ] Review all metrics
- [ ] Confirm no degradation

### First Week

- [ ] Full metrics review
- [ ] Performance comparison with baseline
- [ ] Document any issues
- [ ] Update runbook if needed

---

## Contact Information

**On-Call Rotation**: See team calendar

**Escalation Path**:
1. Check documentation and troubleshooting
2. Contact on-call engineer
3. Escalate to ML team lead
4. Involve infrastructure team if server issues

---

## Additional Resources

- [API Documentation](API_DOCUMENTATION.md)
- [Release Guide](../RELEASE.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Model Comparison](comparison.md)

---

**Last Updated**: 2024-01-15
**Version**: 1.0.0

