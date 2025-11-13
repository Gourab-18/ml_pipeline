.PHONY: help install test test-fast test-full test-ci lint format clean train-sample train-full evaluate export serve setup-venv

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run all tests with coverage (includes TF, slow)"
	@echo "  make test-fast     - Run fast tests only (no TensorFlow, ~2s)"
	@echo "  make test-full     - Run all tests including TensorFlow (slow)"
	@echo "  make test-ci       - Run tests for CI (no strict coverage)"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean artifacts"
	@echo "  make train-sample  - Train sample model (quick)"
	@echo "  make train-full    - Full training with CV (all models)"
	@echo "  make evaluate      - Evaluate models from CV run"
	@echo "  make export        - Export champion model from CV"
	@echo "  make serve         - Start FastAPI serving server"
	@echo "  make compare       - Compare models from CV runs"
	@echo "  make setup-venv    - Setup virtual environment"

# Setup virtual environment
setup-venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Run all tests with coverage (includes TensorFlow, slow)
test:
	@if [ -d venv ]; then \
		venv/bin/pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-fail-under=80; \
	else \
		pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-fail-under=80; \
	fi

# Run fast tests only (no TensorFlow, ~2 seconds)
test-fast:
	@echo "🚀 Running fast tests (no TensorFlow)..."
	@if [ -d venv ]; then \
		venv/bin/pytest tests/test_loader.py tests/test_lightweight_preprocessing.py \
			tests/test_calibration.py tests/test_explainability.py \
			-v --tb=short; \
	else \
		pytest tests/test_loader.py tests/test_lightweight_preprocessing.py \
			tests/test_calibration.py tests/test_explainability.py \
			-v --tb=short; \
	fi

# Run all tests including TensorFlow (slow, 5-30s for TF import)
test-full:
	@echo "🐢 Running all tests (including TensorFlow - may be slow)..."
	@if [ -d venv ]; then \
		venv/bin/pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-fail-under=80; \
	else \
		pytest tests/ -v --cov=src --cov-report=html --cov-report=term --cov-fail-under=80; \
	fi

# Run tests (no coverage requirement, for CI)
test-ci:
	@if [ -d venv ]; then \
		venv/bin/pytest tests/ -v --cov=src --cov-report=term; \
	else \
		pytest tests/ -v --cov=src --cov-report=term; \
	fi

# Run linting
lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/

# Format code
format:
	black src/ tests/
	isort src/ tests/

# Clean artifacts
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf artifacts/

# Train sample model (quick - sklearn only)
train-sample:
	@echo "🚀 Training sample model (quick sklearn baseline)..."
	@python demo_cv_sklearn_quick.py

# Full training with CV (ANN + GBDT baselines)
train-full:
	@echo "🚀 Running full training pipeline..."
	@echo "This includes:"
	@echo "  - ANN with 5-fold CV"
	@echo "  - XGBoost baseline"
	@echo "  - LightGBM baseline"
	@echo "  - Calibration and evaluation"
	@echo ""
	@python3 -c " \
from src.data.loader import DataLoader; \
from src.preprocessing.lightweight_transformers import create_lightweight_pipeline; \
from src.training.cv import run_kfold_cv; \
from src.baselines.xgb_lgb import run_gbdt_cv; \
from src.baselines.compare_models import compare_models; \
import os; \
print('📊 Loading data...'); \
loader = DataLoader('configs/schema.yaml'); \
df = loader.load_data('data/sample.csv'); \
print('✅ Data loaded:', len(df), 'samples'); \
print('🔧 Preprocessing...'); \
pipeline = create_lightweight_pipeline(); \
pipeline.fit_on(df); \
X, y = pipeline.transform(df); \
feature_info = pipeline.get_feature_info(); \
print('✅ Preprocessing complete:', X.shape); \
print('🏋️  Training ANN with 5-fold CV...'); \
ann_summary = run_kfold_cv(X, y, feature_info, k_folds=5, seed=42, base_params={'epochs': 50, 'batch_size': 64}); \
print('✅ ANN training complete!'); \
cal_info = ann_summary.get('calibration', {}); \
cal_metrics = cal_info.get('calibrated_metrics', cal_info.get('uncalibrated_metrics', {})); \
print(f'📈 ANN OOF ROC-AUC: {cal_metrics.get(\"roc_auc\", 0.0):.4f}'); \
print('🏋️  Training GBDT baselines...'); \
gbdt_results = run_gbdt_cv(X, y, k_folds=5, seed=42, run_name=ann_summary['run_dir'].split('/')[-1]); \
print('✅ GBDT training complete!'); \
print(''); \
print('=' * 70); \
print('📊 MODEL COMPARISON'); \
print('=' * 70); \
comparison_df = compare_models(ann_run_dir=ann_summary['run_dir'], xgb_run_dir=gbdt_results['run_dir'], lgb_run_dir=gbdt_results['run_dir']); \
print(comparison_df.to_string(index=False)); \
print('=' * 70); \
comparison_path = os.path.join(ann_summary['run_dir'], 'model_comparison.csv'); \
comparison_df.to_csv(comparison_path, index=False); \
print(f'✅ Comparison saved to: {comparison_path}'); \
print('🎉 Full training pipeline completed!'); \
print('📁 Artifacts saved to:', ann_summary['run_dir']); \
"

# Evaluate models from CV run
evaluate:
	@echo "📊 Evaluating models..."
	@echo "Usage: make evaluate RUN_NAME=<cv_run_name>"
	@if [ -z "$(RUN_NAME)" ]; then \
		echo "❌ Please specify RUN_NAME, e.g.: make evaluate RUN_NAME=20240101_120000"; \
		echo "Available runs:"; \
		ls -d artifacts/cv/*/ 2>/dev/null | sed 's|artifacts/cv/||' | sed 's|/||' || echo "  No CV runs found"; \
	else \
		python3 -c " \
import json; \
import os; \
run_dir = f'artifacts/cv/$(RUN_NAME)'; \
summary_path = os.path.join(run_dir, 'summary.json'); \
if os.path.exists(summary_path): \
    with open(summary_path) as f: \
        summary = json.load(f); \
    print('📊 Evaluation Summary for $(RUN_NAME)'); \
    print('=' * 60); \
    print(f\"OOF ROC-AUC: {summary['metrics']['roc_auc']:.4f}\"); \
    print(f\"OOF PR-AUC:  {summary['metrics']['pr_auc']:.4f}\"); \
    print(f\"Brier Score: {summary['metrics']['brier']:.4f}\"); \
    print(f\"ECE (uncal): {summary['metrics']['ece']:.4f}\"); \
    if 'calibration' in summary and 'calibrated_metrics' in summary['calibration']: \
        print(f\"ECE (cal):   {summary['calibration']['calibrated_metrics']['ece']:.4f}\"); \
    print('=' * 60); \
else: \
    print(f'❌ Summary not found: {summary_path}'); \
"; \
	fi

# Export champion model from CV run
export:
	@echo "📦 Exporting champion model..."
	@if [ -z "$(RUN_NAME)" ]; then \
		echo "Finding latest CV run..."; \
		LATEST_RUN=$$(ls -td artifacts/cv/*/ 2>/dev/null | head -1 | xargs basename); \
		if [ -z "$$LATEST_RUN" ]; then \
			echo "❌ No CV runs found. Run 'make train-full' first."; \
			exit 1; \
		fi; \
		echo "Using latest run: $$LATEST_RUN"; \
		python3 -c " \
from src.models.export import export_from_cv_run; \
import os; \
run_dir = f'artifacts/cv/$$LATEST_RUN'; \
fold_idx = int(os.environ.get('FOLD_IDX', '0')); \
export_dir = 'artifacts/models/champion'; \
version = os.environ.get('VERSION', '1'); \
print(f'📦 Exporting from {run_dir}, fold {fold_idx}...'); \
export_from_cv_run(run_dir, fold_idx, export_dir, version=version); \
print(f'✅ Model exported to {export_dir}/{version}/'); \
"; \
	else \
		python3 -c " \
from src.models.export import export_from_cv_run; \
import os; \
run_dir = f'artifacts/cv/$(RUN_NAME)'; \
fold_idx = int(os.environ.get('FOLD_IDX', '0')); \
export_dir = 'artifacts/models/champion'; \
version = os.environ.get('VERSION', '1'); \
print(f'📦 Exporting from {run_dir}, fold {fold_idx}...'); \
export_from_cv_run(run_dir, fold_idx, export_dir, version=version); \
print(f'✅ Model exported to {export_dir}/{version}/'); \
"; \
	fi

# Start FastAPI serving server
serve:
	@echo "🚀 Starting FastAPI server..."
	@echo "Server will start on http://localhost:8000"
	@echo "Press Ctrl+C to stop"
	@uvicorn src.serve.app:app --host 0.0.0.0 --port 8000

# Compare models from CV runs
compare:
	@echo "📊 Comparing models..."
	@if [ -z "$(RUN_NAME)" ]; then \
		echo "Comparing latest runs..."; \
		python3 compare_models.py --latest; \
	else \
		echo "Comparing run: $(RUN_NAME)"; \
		python3 compare_models.py "artifacts/cv/$(RUN_NAME)" "artifacts/cv/$(RUN_NAME)"; \
	fi

# Development setup
dev-setup: setup-venv install
	@echo "Development environment setup complete!"
	@echo "Activate virtual environment: source venv/bin/activate"
   
