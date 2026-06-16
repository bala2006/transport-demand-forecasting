# Contributing

We welcome contributions of all sizes. Here's how to get started.

## Development Setup

```bash
git clone https://github.com/bala2006/transport-demand-forecasting
cd transport-demand-forecasting
pip install -r requirements.txt
pip install -e .  # editable install
```

## Code Standards

- Follow PEP 8 (100-120 char lines OK for readability)
- Type hints for all public functions
- Write tests for new features (pytest)
- Notebooks should be clean before PR (Restart & Run All)

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes
3. Run `pytest tests/` — all tests must pass
4. Run `flake8 src/ tests/` — no critical errors
5. Open a pull request with a clear description

## Adding a Model

1. Create `src/models/your_model.py`
2. Implement a class with `fit`/`predict` (sklearn) or `forward` (torch)
3. Add it to `src/models/train.py`
4. Register it in `src/pipeline.py`
5. Add tests in `tests/test_models.py`

## Adding a Dataset Adapter

1. Create `src/data/your_city.py`
2. Implement download and preprocessing functions
3. Return a DataFrame compatible with the feature pipeline
