# Release Guide

This document describes the release process for the Tabular ML Pipeline project.

---

## 📋 Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release tag created
- [ ] Release artifact prepared
- [ ] Release notes written

---

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

---

## Pre-Release Steps

### 1. Run Full Test Suite

```bash
# Run all tests with coverage
make test

# Ensure coverage is acceptable (target: >80%)
pytest tests/ --cov=src --cov-report=term --cov-report=html
```

### 2. Verify Reproducibility

```bash
# Run full training twice with same seed
SEED=42 make train-full

# Note the OOF metrics from first run
# Run again and verify metrics match (within tolerance)
SEED=42 make train-full

# Compare:
# artifacts/cv/<run1>/summary.json
# artifacts/cv/<run2>/summary.json
```

**Expected reproducibility:**
- Same OOF predictions (identical CSV)
- Metrics within tolerance (±0.01 ROC-AUC, ±0.02 PR-AUC)

### 3. Validate Model Export

```bash
# Export model from CV run
make export

# Verify export structure
ls -R artifacts/models/champion/1/

# Should contain:
# - saved_model/
# - calibrator.pkl
# - metadata.json
```

### 4. Test Model Serving

```bash
# Start server
make serve

# In another terminal, load and test model
curl -X POST "http://localhost:8000/load?model_dir=artifacts/models/champion&version=1"
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [{"feature_0": 1.0, "feature_1": 2.0}]}'

# Verify prediction response is valid
```

### 5. Update Documentation

- [ ] README.md is up-to-date
- [ ] API documentation is current
- [ ] Changelog updated (if maintained)
- [ ] Version numbers updated in docs

### 6. Code Quality Checks

```bash
# Run linting
make lint

# Fix any issues
make format

# Check type hints
mypy src/
```

---

## Release Process

### Step 1: Update Version

Update version in relevant files:

**Option A: Single location (if using setup.py)**
```python
# setup.py
version = "1.0.0"
```

**Option B: Version file**
```python
# src/__version__.py
__version__ = "1.0.0"
```

**Update README.md:**
```markdown
Current version: **v1.0.0**
```

### Step 2: Create Release Branch

```bash
git checkout -b release/v1.0.0
```

### Step 3: Run Full Training

```bash
# Ensure fresh training run
make clean
make train-full

# Verify metrics are acceptable
cat artifacts/cv/*/summary.json | grep roc_auc
```

### Step 4: Export Champion Model

```bash
# Export best fold as champion
make export

# Verify export
ls -la artifacts/models/champion/1/
```

### Step 5: Create Release Artifact

```bash
# Create release directory
mkdir -p releases/v1.0.0

# Copy essential artifacts
cp -r artifacts/models/champion releases/v1.0.0/model
cp -r artifacts/cv/* releases/v1.0.0/cv_runs  # Optional: include CV runs

# Copy configuration
cp configs/* releases/v1.0.0/configs/

# Copy documentation
cp README.md releases/v1.0.0/
cp -r docs/ releases/v1.0.0/docs/

# Create ZIP archive
cd releases
zip -r v1.0.0.zip v1.0.0/
cd ..
```

**Release artifact should contain:**
- Model artifacts (`champion/` directory)
- Configuration files (`configs/`)
- Documentation (`docs/`, `README.md`)
- Requirements file (`requirements.txt`)

### Step 6: Create Git Tag

```bash
# Tag the release
git tag -a v1.0.0 -m "Release v1.0.0: Initial production-ready version"

# Push tag to remote
git push origin v1.0.0
```

### Step 7: Create GitHub Release

1. Go to GitHub → Releases → Draft a new release
2. Choose tag: `v1.0.0`
3. Title: `Release v1.0.0`
4. Description:
   ```markdown
   ## Release v1.0.0
   
   First production-ready release of Tabular ML Pipeline.
   
   ### Features
   - Complete preprocessing pipeline (lightweight + full TensorFlow)
   - Tabular ANN with embeddings
   - XGBoost and LightGBM baselines
   - Cross-validation with calibration
   - Model explainability (permutation importance, SHAP)
   - FastAPI serving API
   - Comprehensive test suite
   
   ### Installation
   ```bash
   pip install -r requirements.txt
   ```
   
   ### Quick Start
   ```bash
   make train-sample
   make serve
   ```
   
   ### Documentation
   See `docs/` directory for complete documentation.
   
   ### Release Artifact
   Download: [v1.0.0.zip](link-to-artifact)
   ```

5. Attach release artifact ZIP file
6. Publish release

---

## Post-Release Steps

### 1. Update Documentation

- [ ] Update main branch with version number
- [ ] Update any "latest" references
- [ ] Archive old release notes (if applicable)

### 2. Notify Team

- Announce release in team channel
- Share release notes
- Highlight breaking changes (if any)

### 3. Monitor Deployment

- Watch for issues in production
- Monitor error rates
- Check model performance metrics

### 4. Prepare Next Release

- Create `release/v1.0.1` branch for patches
- Or start `release/v1.1.0` for next minor version

---

## Release Artifact Structure

```
v1.0.0/
├── model/
│   └── champion/
│       └── 1/
│           ├── saved_model/
│           ├── calibrator.pkl
│           └── metadata.json
├── configs/
│   ├── schema.yaml
│   └── feature_list.csv
├── docs/
│   ├── API_DOCUMENTATION.md
│   ├── problem_spec.md
│   └── ...
├── README.md
└── requirements.txt
```

---

## Hotfix Release Process

For urgent bug fixes:

```bash
# 1. Checkout release branch
git checkout release/v1.0.0

# 2. Create hotfix branch
git checkout -b hotfix/v1.0.1

# 3. Fix bug and test
# ... make changes ...
make test

# 4. Commit and tag
git commit -m "Fix: bug description"
git tag -a v1.0.1 -m "Hotfix v1.0.1: bug fix"
git push origin v1.0.1

# 5. Merge back to main
git checkout main
git merge hotfix/v1.0.1
git push origin main
```

---

## Release Notes Template

```markdown
## Release v1.0.0

**Release Date**: YYYY-MM-DD

### 🎉 New Features
- Feature 1
- Feature 2

### 🐛 Bug Fixes
- Fix 1
- Fix 2

### 🔄 Updates
- Update 1
- Update 2

### 📚 Documentation
- Docs update 1

### 🔧 Technical Changes
- Change 1
- Change 2

### 📦 Installation
```bash
pip install -r requirements.txt
```

### 🚀 Quick Start
```bash
make train-sample
make serve
```

### 📋 Full Changelog
See [CHANGELOG.md](CHANGELOG.md) for complete list of changes.
```

---

## Troubleshooting

### Tests Failing in CI

- Ensure local tests pass: `make test`
- Check Python version matches CI
- Verify all dependencies are in `requirements.txt`

### Model Export Failing

- Verify CV run completed successfully
- Check that fold artifacts exist
- Ensure export directory is writable

### Reproducibility Issues

- Verify seed is set correctly (42)
- Check TensorFlow version consistency
- Ensure no random operations outside seed control

---

## Version History

- **v1.0.0** (YYYY-MM-DD): Initial production release
  - Complete ML pipeline
  - ANN, XGBoost, LightGBM models
  - Calibration and explainability
  - FastAPI serving

---

For questions or issues with the release process, contact the maintainers or open an issue.

