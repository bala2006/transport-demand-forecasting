import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Path,
    sample: int = 500,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(y_true) > sample:
        idx = np.random.RandomState(42).choice(len(y_true), sample, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1)
    ax.set_xlabel("Actual Demand")
    ax.set_ylabel("Predicted Demand")
    ax.set_title(f"{model_name} — Actual vs Predicted")
    ax.set_aspect("equal")
    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{model_name}_actual_vs_predicted.png", dpi=150)
    plt.close(fig)


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Path,
):
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax1.axhline(0, color="r", linestyle="--")
    ax1.set_xlabel("Predicted Demand")
    ax1.set_ylabel("Residual")
    ax1.set_title(f"{model_name} — Residuals vs Predicted")

    ax2.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"{model_name} — Residual Distribution")

    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{model_name}_residuals.png", dpi=150)
    plt.close(fig)


def plot_time_series_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: Path,
    n_points: int = 200,
):
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(min(n_points, len(y_true)))
    ax.plot(x, y_true[:n_points], label="Actual", alpha=0.8)
    ax.plot(x, y_pred[:n_points], label="Predicted", alpha=0.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Demand")
    ax.set_title(f"{model_name} — Time Series Comparison")
    ax.legend()
    fig.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"{model_name}_time_series.png", dpi=150)
    plt.close(fig)
