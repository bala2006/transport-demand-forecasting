import numpy as np
import pytest
import torch

from src.models.baseline import build_linear_regression, build_random_forest, build_xgboost
from src.models.lstm_model import LSTMRegressor, prepare_sequences
from src.models.tcn_model import TCNRegressor
from src.models.train import evaluate_model


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10))
    y = X[:, 0] * 2 + X[:, 1] * 3 + rng.normal(0, 0.1, 200)
    return X, y


def test_linear_regression_trains(synthetic_data):
    X, y = synthetic_data
    model = build_linear_regression()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_random_forest_trains(synthetic_data):
    X, y = synthetic_data
    model = build_random_forest(n_estimators=10, max_depth=5, min_samples_leaf=2)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_xgboost_trains(synthetic_data):
    X, y = synthetic_data
    model = build_xgboost(n_estimators=10, max_depth=3)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_evaluate_model_returns_metrics(synthetic_data):
    X, y = synthetic_data
    model = build_linear_regression()
    model.fit(X, y)
    metrics = evaluate_model(model, X, y)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "MAPE" in metrics
    assert "R²" in metrics
    assert metrics["R²"] > 0.9


def test_lstm_forward_pass():
    model = LSTMRegressor(input_size=5, hidden_size=8, num_layers=1)
    x = torch.randn(4, 24, 5)
    out = model(x)
    assert out.shape == (4,)


def test_tcn_forward_pass():
    model = TCNRegressor(input_size=5, num_channels=[8, 16])
    x = torch.randn(4, 24, 5)
    out = model(x)
    assert out.shape == (4,)


def test_prepare_sequences():
    X = torch.randn(50, 3)
    y = torch.randn(50)
    X_seq, y_seq = prepare_sequences(X, y, seq_length=10)
    assert X_seq.shape == (40, 10, 3)
    assert y_seq.shape == (40,)
