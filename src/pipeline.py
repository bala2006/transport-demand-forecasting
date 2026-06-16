import argparse
import json
from pathlib import Path

import polars as pl
import yaml
import torch

from src.data.download import download_tlc_data, download_zone_lookup
from src.data.preprocess import run_preprocessing
from src.data.features import build_feature_matrix
from src.models.baseline import build_linear_regression, build_random_forest, build_xgboost
from src.models.lstm_model import LSTMRegressor, prepare_sequences
from src.models.tcn_model import TCNRegressor
from src.models.train import train_sklearn_model, train_torch_model, evaluate_model
from src.evaluation.metrics import comparison_table
from src.evaluation.plots import plot_actual_vs_predicted
from src.explain.shap_analysis import explain_with_shap


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_pipeline(cfg: dict):
    base_dir = Path(cfg.get("base_dir", "."))
    data_cfg = cfg["data"]
    model_cfg = cfg["models"]
    output_cfg = cfg["output"]

    raw_dir = base_dir / "data" / "raw"
    proc_dir = base_dir / "data" / "processed"
    ext_dir = base_dir / "data" / "external"
    figures_dir = base_dir / output_cfg.get("figures_dir", "outputs/figures")
    models_dir = base_dir / output_cfg.get("models_dir", "outputs/models")
    results_dir = base_dir / output_cfg.get("results_dir", "outputs/results")

    years = data_cfg.get("years", [2024])
    months = data_cfg.get("months", [1, 2, 3])
    target_zones = data_cfg.get("target_zones", None)
    seq_length = model_cfg.get("seq_length", 24)
    weather_path = data_cfg.get("weather_path", None)

    print("=== Step 1: Download data ===")
    download_tlc_data(years, months, raw_dir)
    download_zone_lookup(ext_dir)
    zone_lookup_path = ext_dir / "taxi_zone_lookup.csv"

    print("\n=== Step 2: Preprocess ===")
    df = run_preprocessing(raw_dir, proc_dir, zone_lookup_path, years, months)

    print("\n=== Step 3: Feature engineering ===")
    df_feat = build_feature_matrix(df, weather_path)
    if target_zones:
        df_feat = df_feat.filter(pl.col("PULocationID").is_in(target_zones))

    feature_cols = [c for c in df_feat.columns if c not in (
        "pickup_hour", "PULocationID", "demand", "pickup_zone", "borough",
        "weather_date",
    )]
    df_feat = df_feat.drop_nulls(subset=feature_cols + ["demand"])
    print(f"Feature matrix: {df_feat.shape}")

    print("\n=== Step 4: Chronological split ===")
    df_feat = df_feat.sort("pickup_hour")
    n = df_feat.height
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_df = df_feat[:train_end]
    val_df = df_feat[train_end:val_end]
    test_df = df_feat[val_end:]

    X_train = train_df.select(feature_cols).to_numpy()
    y_train = train_df.select("demand").to_numpy().ravel()
    X_val = val_df.select(feature_cols).to_numpy()
    y_val = val_df.select("demand").to_numpy().ravel()
    X_test = test_df.select(feature_cols).to_numpy()
    y_test = test_df.select("demand").to_numpy().ravel()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = {}
    active_models = model_cfg.get("active", ["lr", "rf", "xgb", "lstm", "tcn"])

    print("\n=== Step 5: Train models ===")

    if "lr" in active_models:
        print("\n--- Linear Regression ---")
        lr = build_linear_regression()
        lr = train_sklearn_model(lr, X_train_s, y_train)
        results["LinearRegression"] = evaluate_model(lr, X_test_s, y_test)
        plot_actual_vs_predicted(y_test, lr.predict(X_test_s), "LinearRegression", figures_dir)
        torch.save(lr, models_dir / "linear_regression.pkl")

    if "rf" in active_models:
        print("\n--- Random Forest ---")
        rf = build_random_forest()
        rf = train_sklearn_model(rf, X_train_s, y_train)
        results["RandomForest"] = evaluate_model(rf, X_test_s, y_test)
        plot_actual_vs_predicted(y_test, rf.predict(X_test_s), "RandomForest", figures_dir)
        torch.save(rf, models_dir / "random_forest.pkl")

    if "xgb" in active_models:
        print("\n--- XGBoost ---")
        xgb = build_xgboost()
        xgb = train_sklearn_model(xgb, X_train_s, y_train)
        results["XGBoost"] = evaluate_model(xgb, X_test_s, y_test)
        plot_actual_vs_predicted(y_test, xgb.predict(X_test_s), "XGBoost", figures_dir)

    if "lstm" in active_models:
        print("\n--- LSTM ---")
        lstm = LSTMRegressor(input_size=X_train_s.shape[1])
        X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)
        X_train_seq, y_train_seq = prepare_sequences(X_train_t, y_train_t, seq_length)
        X_val_seq, y_val_seq = prepare_sequences(X_val_t, y_val_t, seq_length)
        X_test_seq, y_test_seq = prepare_sequences(X_test_t, y_test_t, seq_length)
        lstm = train_torch_model(lstm, X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        results["LSTM"] = evaluate_model(lstm, X_test_seq, y_test_seq)
        torch.save(lstm.state_dict(), models_dir / "lstm.pt")

    if "tcn" in active_models:
        print("\n--- TCN ---")
        tcn = TCNRegressor(input_size=X_train_s.shape[1])
        X_train_seq_tcn, y_train_seq_tcn = prepare_sequences(X_train_t, y_train_t, seq_length)
        X_val_seq_tcn, y_val_seq_tcn = prepare_sequences(X_val_t, y_val_t, seq_length)
        X_test_seq_tcn, y_test_seq_tcn = prepare_sequences(X_test_t, y_test_t, seq_length)
        tcn = train_torch_model(tcn, X_train_seq_tcn, y_train_seq_tcn, X_val_seq_tcn, y_val_seq_tcn)
        results["TCN"] = evaluate_model(tcn, X_test_seq_tcn, y_test_seq_tcn)
        torch.save(tcn.state_dict(), models_dir / "tcn.pt")

    print("\n=== Step 6: Results ===")
    comp_df = comparison_table(results)
    print(comp_df)
    results_dir.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(results_dir / "comparison.csv")
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Step 7: SHAP explanations ===")
    for name, model in [("LinearRegression", lr), ("RandomForest", rf), ("XGBoost", xgb)]:
        if name in results:
            explain_with_shap(
                model, X_train_s, X_test_s, name,
                feature_cols, figures_dir,
            )

    print("\nPipeline complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transportation Demand Forecasting Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_pipeline(cfg)
