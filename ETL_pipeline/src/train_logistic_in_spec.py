"""
Train a logistic regression classifier to predict in-spec readings from raw parquet data.

The script:
- Loads raw Parquet sensor readings (plant/line/batch/step/sensor/value/unit/in_spec).
- One-hot encodes categorical columns and scales numeric values.
- Fits a LogisticRegression model to classify in_spec.
- Prints standard classification metrics.

Usage:
    python src/train_logistic_in_spec.py --data-root data/raw --max-files 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_raw_data(data_root: Path, max_files: int | None) -> pd.DataFrame:
    files = sorted(data_root.glob("date=*/beer-*.parquet"))
    if max_files:
        files = files[-max_files:]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    frames: List[pd.DataFrame] = [pd.read_parquet(file, engine="fastparquet") for file in files]
    return pd.concat(frames, ignore_index=True)


def build_pipeline(categorical_cols: List[str], numeric_cols: List[str], class_weight: str | None) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight=class_weight,
        solver="liblinear",
        random_state=42,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a logistic regression for in_spec classification.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Path to the raw parquet data root (partitioned by date=...).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=400,
        help="Maximum parquet files to read (set higher for more training data).",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout fraction for evaluation.")
    parser.add_argument(
        "--class-weight",
        type=str,
        default=None,
        choices=["balanced"],
        help="Class weighting strategy for imbalanced data.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_root} (max_files={args.max_files}) ...")
    df = load_raw_data(args.data_root, args.max_files)

    df = df.dropna(subset=["in_spec", "value", "step", "sensor"])
    df["in_spec"] = df["in_spec"].astype(int)

    categorical_cols = ["plant_id", "line_id", "step", "sensor", "unit"]
    numeric_cols = ["value"]
    existing_cats = [c for c in categorical_cols if c in df.columns]
    existing_nums = [c for c in numeric_cols if c in df.columns]

    X = df[existing_cats + existing_nums]
    y = df["in_spec"]

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError("Need both in-spec and out-of-spec samples to train a classifier.")
    stratify = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=stratify
    )

    pipeline = build_pipeline(existing_cats, existing_nums, args.class_weight)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probas),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }

    print("=== Metrics ===")
    for key, val in metrics.items():
        if key.startswith("n_"):
            print(f"{key}: {val}")
        else:
            print(f"{key}: {val:.4f}")


if __name__ == "__main__":
    main()
