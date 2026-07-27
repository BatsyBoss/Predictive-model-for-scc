"""
model_utils.py
---------------
Training, evaluating, saving, loading, and running inference with the
compressive-strength regression model.

XGBoost is the preferred regressor (per spec). If the xgboost package is not
available in the current environment, we transparently fall back to
scikit-learn's GradientBoostingRegressor so the app still runs end-to-end —
the UI clearly labels which model type is actually in use.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import FEATURE_COLUMNS, TARGET_COLUMN, MODEL_PATH

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when xgboost missing
    XGBOOST_AVAILABLE = False


@dataclass
class TrainingResult:
    model: object
    model_type: str
    metrics: dict
    X_test: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    feature_importance: pd.DataFrame
    trained_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    n_train: int = 0
    n_test: int = 0


def _build_model():
    """Instantiate the preferred regressor, with a graceful fallback."""
    if XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        )
        return model, "XGBoost Regressor"
    else:
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            random_state=42,
        )
        return model, "Gradient Boosting Regressor (sklearn fallback — xgboost not installed)"


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> TrainingResult:
    """Train the regression model and return a bundle of results/metrics."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model, model_type = _build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    # Cross-validated R2 gives a slightly more robust performance signal
    # than a single held-out split, shown alongside the primary metrics.
    try:
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        cv_r2_mean = float(np.mean(cv_scores))
    except Exception:
        cv_r2_mean = None

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        fi_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
    else:
        fi_df = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": np.nan})

    metrics = {
        "r2_score": r2,
        "rmse": rmse,
        "cv_r2_mean": cv_r2_mean,
        "n_samples": len(df),
    }

    return TrainingResult(
        model=model,
        model_type=model_type,
        metrics=metrics,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        feature_importance=fi_df,
        n_train=len(X_train),
        n_test=len(X_test),
    )


def save_model(result: TrainingResult, path=MODEL_PATH) -> str:
    """Persist the trained model + metadata bundle to disk with joblib."""
    bundle = {
        "model": result.model,
        "model_type": result.model_type,
        "metrics": result.metrics,
        "feature_importance": result.feature_importance,
        "trained_at": result.trained_at,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }
    joblib.dump(bundle, path)
    return str(path)


@st.cache_resource(show_spinner=False)
def _load_bundle_cached(path_str: str, mtime: float):
    """
    Internal cached loader. `mtime` is deliberately part of the cache key
    (NOT underscore-prefixed) so Streamlit busts the cache whenever the
    underlying file's modification time changes — e.g. after retraining and
    saving a new model — instead of serving a stale cached object.
    """
    return joblib.load(path_str)


def load_saved_model(path=MODEL_PATH) -> Optional[dict]:
    """Load a previously saved model bundle from disk, if one exists."""
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    return _load_bundle_cached(str(path), mtime)


def predict_single(model, input_dict: dict) -> float:
    """Run inference for a single mix design given as a dict of feature -> value."""
    X = pd.DataFrame([{col: input_dict[col] for col in FEATURE_COLUMNS}])
    pred = model.predict(X)[0]
    return float(pred)


def predict_batch(model, df_inputs: pd.DataFrame) -> np.ndarray:
    """Run inference for a batch of mix designs."""
    return model.predict(df_inputs[FEATURE_COLUMNS])
