from pathlib import Path

import numpy as np
import shap
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def explain_with_shap(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    model_name: str,
    feature_names: list[str],
    save_dir: Path,
    n_samples: int = 100,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(model, nn.Module):
        model.eval()
        device = next(model.parameters()).device
        X_bg_t = torch.tensor(X_background[:n_samples], dtype=torch.float32, device=device)
        X_ex_t = torch.tensor(X_explain[:n_samples], dtype=torch.float32, device=device)

        def model_fn(x):
            x_t = torch.tensor(x, dtype=torch.float32, device=device)
            with torch.no_grad():
                return model(x_t).cpu().numpy()

        explainer = shap.KernelExplainer(model_fn, X_bg_t.cpu().numpy()[:50])
        shap_values = explainer.shap_values(X_ex_t.cpu().numpy()[:50])
    else:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X_background[:50])
        shap_values = explainer.shap_values(X_explain[:n_samples])

    fig = plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        X_explain[:n_samples],
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"{model_name} — SHAP Summary")
    fig.tight_layout()
    fig.savefig(save_dir / f"{model_name}_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_explain[:n_samples],
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.title(f"{model_name} — SHAP Feature Importance")
    fig.tight_layout()
    fig.savefig(save_dir / f"{model_name}_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for i in range(min(3, len(X_explain))):
        base_val = explainer.expected_value if hasattr(explainer, "expected_value") else 0
        if hasattr(base_val, "__len__"):
            base_val = float(base_val[0]) if len(base_val) > 0 else 0.0
        else:
            base_val = float(base_val)
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[i],
                base_values=base_val,
                data=X_explain[i],
                feature_names=feature_names,
            ),
            show=False,
            max_display=15,
        )
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        fig.savefig(save_dir / f"{model_name}_force_plot_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return shap_values
