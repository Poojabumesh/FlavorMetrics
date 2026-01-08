"""
Utility helpers for fitting a simple Linear Regression model, checking multicollinearity,
and making predictions.

Example:
    from pathlib import Path
    import pandas as pd
    from models import (
        train_linear_regression,
        predict_linear_regression,
        compute_vif,
        reduce_multicollinearity,
    )

    df = pd.read_parquet("data/marts/beer_kpi_date=2025-12-23.parquet")
    vif = compute_vif(df, feature_cols=["mean_value", "oos_rate"])
    kept, dropped = reduce_multicollinearity(df, feature_cols=["mean_value", "oos_rate"], vif_threshold=5.0)
    model, metrics = train_linear_regression(df, target_col="oos_rate", feature_cols=["mean_value"])
    preds = predict_linear_regression(model, df[["mean_value"]])
"""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _to_matrix(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    """Select columns and return as numpy matrix."""
    return df.loc[:, list(cols)].to_numpy()


def train_linear_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[LinearRegression, dict]:
    """
    Fit a Linear Regression model on the given DataFrame.

    Returns the trained model and a metrics dict with r2 and rmse on the holdout set.
    """
    X = _to_matrix(df, feature_cols)
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }
    return model, metrics


def compute_vif(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for each feature to diagnose multicollinearity.

    VIF = 1 / (1 - R^2) where R^2 is from regressing one feature on all the others.
    """
    features: List[str] = list(feature_cols)
    rows = []

    for i, col in enumerate(features):
        other = features[:i] + features[i + 1 :]
        if not other:
            rows.append({"feature": col, "vif": 1.0, "r2": 0.0})
            continue

        X = _to_matrix(df, other)
        y = df[col].to_numpy()
        reg = LinearRegression()
        reg.fit(X, y)
        r2 = reg.score(X, y)
        vif = 1.0 / max(1e-6, (1.0 - r2))
        rows.append({"feature": col, "vif": vif, "r2": r2})

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def reduce_multicollinearity(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    vif_threshold: float = 5.0,
) -> Tuple[List[str], List[str]]:
    """
    Iteratively drop the feature with the highest VIF until all remaining features are below the threshold.

    Returns (kept_features, dropped_features).
    """
    keep = list(feature_cols)
    dropped: List[str] = []

    while len(keep) > 1:
        vif_df = compute_vif(df, keep)
        max_vif = vif_df.iloc[0]
        if max_vif["vif"] <= vif_threshold:
            break
        dropped.append(max_vif["feature"])
        keep.remove(max_vif["feature"])

    return keep, dropped


def predict_linear_regression(model: LinearRegression, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    """
    Generate predictions from a trained Linear Regression model.

    Accepts either a pandas DataFrame or a numpy array containing the feature columns used for training.
    """
    X = features.to_numpy() if isinstance(features, pd.DataFrame) else np.asarray(features)
    return model.predict(X)
