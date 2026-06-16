from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_linear_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression()),
    ])


def build_random_forest(n_jobs: int = -1, **kwargs) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 15),
        min_samples_leaf=kwargs.get("min_samples_leaf", 5),
        n_jobs=n_jobs,
        random_state=42,
    )


def build_xgboost(**kwargs) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=kwargs.get("n_estimators", 300),
        max_depth=kwargs.get("max_depth", 8),
        learning_rate=kwargs.get("learning_rate", 0.1),
        subsample=kwargs.get("subsample", 0.8),
        colsample_bytree=kwargs.get("colsample_bytree", 0.8),
        random_state=42,
        verbosity=0,
    )
