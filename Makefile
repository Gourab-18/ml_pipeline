.PHONY: help install test lint format clean train-sample setup-venv

# Default target
help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linting"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean artifacts"
	@echo "  make train-sample  - Train sample model"
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

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

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

# Train sample model (placeholder)
train-sample:
	@echo "Training sample model..."
	@echo "This is a placeholder - will be implemented in later tasks"
	python -c "print('Sample training completed!')"

# Development setup
dev-setup: setup-venv install
	@echo "Development environment setup complete!"
	@echo "Activate virtual environment: source venv/bin/activate"

