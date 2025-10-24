# Tabular ML System

A production-ready Machine Learning pipeline for tabular data, centered on an Artificial Neural Network (ANN), with all the components needed for a full ML system.

## Project Structure

```
tabular-ml-system/
├── src/                    # Source code
│   ├── data/              # Data ingestion and processing
│   ├── preprocessing/     # Data preprocessing pipelines
│   ├── models/            # Model architectures
│   ├── training/          # Training pipelines
│   ├── explainability/    # Model interpretability
│   ├── baselines/         # Baseline models
│   └── metrics/           # Evaluation metrics
├── notebooks/             # Jupyter notebooks for exploration
├── docs/                 # Documentation
├── tests/                # Test suite
├── configs/              # Configuration files
├── artifacts/            # Model artifacts and outputs
├── .github/              # GitHub Actions workflows
└── requirements.txt      # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Run tests:**
   ```bash
   make test
   ```

3. **Lint code:**
   ```bash
   make lint
   ```

4. **Train sample model:**
   ```bash
   make train-sample
   ```

## Development

This project uses:
- Python 3.9+
- Poetry for dependency management
- Pytest for testing
- Black and flake8 for code formatting
- GitHub Actions for CI/CD

## Contributing

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

## License

MIT License

