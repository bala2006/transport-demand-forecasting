import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_sklearn_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> BaseEstimator:
    model.fit(X_train, y_train)
    return model


def train_torch_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor | None = None,
    y_val: torch.Tensor | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
) -> nn.Module:
    device = torch.device(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        epoch_loss /= len(train_dataset)

        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val.to(device))
                val_loss = loss_fn(val_preds, y_val.to(device)).item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        if epoch % 10 == 0:
            val_str = f", val_loss={val_loss:.4f}" if X_val is not None else ""
            print(f"Epoch {epoch}/{epochs} — train_loss={epoch_loss:.4f}{val_str}")

    return model


def evaluate_model(
    model,
    X_test: np.ndarray | torch.Tensor,
    y_test: np.ndarray | torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    if isinstance(model, nn.Module):
        model.eval()
        device = torch.device(device)
        with torch.no_grad():
            X_t = X_test.to(device) if isinstance(X_test, torch.Tensor) else torch.tensor(X_test, dtype=torch.float32).to(device)
            y_t = y_test.to(device) if isinstance(y_test, torch.Tensor) else torch.tensor(y_test, dtype=torch.float32).to(device)
            preds = model(X_t).cpu().numpy()
            y_true = y_t.cpu().numpy()
    else:
        preds = model.predict(X_test)
        y_true = y_test

    mae = mean_absolute_error(y_true, preds)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mape = np.mean(np.abs((y_true - preds) / np.maximum(y_true, 1))) * 100
    r2 = r2_score(y_true, preds)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "R²": float(r2),
    }
