"""
Train a simple next-step parameter predictor from the existing Parquet data.

The script:
- Loads Parquet sensor readings (plant/line/batch/step/sensor/value).
- Aggregates per-step stats (mean, std, in-spec rate, count).
- Builds a training set where each row represents a completed step and the
  targets are the sensor means for the next step in the sequence.
- Trains a RandomForest-based multi-output regressor per transition
  (mashing→boiling, boiling→fermentation, fermentation→packaging).
- Prints MAE per target sensor plus a sample prediction vs. actual.

Usage:
    python src/predict_next_step.py --data-root data/raw --max-files 400
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# Manufacturing step order for sequencing.
STEP_ORDER: List[str] = ["mashing", "boiling", "fermentation", "packaging"]

# Aggregations applied to raw sensor readings.
AGGREGATIONS = {
    "mean_value": ("value", "mean"),
    "std_value": ("value", "std"),
    "in_spec_rate": ("in_spec", "mean"),
    "count": ("value", "size"),
}

# Columns that are not model features.
METADATA_COLS = {"plant_id", "line_id", "batch_id", "current_step", "next_step", "transition"}


def load_step_stats(data_root: Path, max_files: int | None) -> pd.DataFrame:
    """Read Parquet files and compute per-step, per-sensor aggregates."""
    files = sorted(data_root.glob("date=*/beer-*.parquet"))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found under {data_root}")

    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_parquet(file, engine="fastparquet")
        agg = (
            df.groupby(["plant_id", "line_id", "batch_id", "step", "sensor"])
            .agg(**AGGREGATIONS)
            .reset_index()
        )
        frames.append(agg)

    return pd.concat(frames, ignore_index=True)


def _ordered_steps(steps: Sequence[str]) -> List[str]:
    """Return steps sorted by predefined STEP_ORDER."""
    return sorted(steps, key=lambda s: STEP_ORDER.index(s) if s in STEP_ORDER else len(STEP_ORDER))


def build_transition_rows(step_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Construct rows where features describe all completed steps so far and targets
    are the sensor means for the next step.
    """
    rows: List[Dict[str, object]] = []

    for (plant_id, line_id, batch_id), batch_df in step_stats.groupby(
        ["plant_id", "line_id", "batch_id"]
    ):
        step_sequence = _ordered_steps(batch_df["step"].unique())
        if len(step_sequence) < 2:
            continue

        history_features: Dict[str, float] = {}
        for i, current_step in enumerate(step_sequence[:-1]):
            # Add aggregates from the current step into the feature history.
            current_stats = batch_df[batch_df["step"] == current_step]
            for _, row in current_stats.iterrows():
                sensor = row["sensor"]
                for metric in AGGREGATIONS.keys():
                    key = f"{current_step}__{sensor}__{metric}"
                    history_features[key] = row[metric]

            next_step = step_sequence[i + 1]
            next_stats = batch_df[batch_df["step"] == next_step]

            # Targets are the mean values of sensors in the next step.
            targets: Dict[str, float] = {}
            for _, row in next_stats.iterrows():
                targets[f"target_{row['sensor']}"] = row["mean_value"]

            row_record: Dict[str, object] = {
                "plant_id": plant_id,
                "line_id": line_id,
                "batch_id": batch_id,
                "current_step": current_step,
                "next_step": next_step,
                "transition": f"{current_step}->{next_step}",
            }
            row_record.update(history_features)
            row_record.update(targets)
            rows.append(row_record.copy())

    if not rows:
        raise ValueError("No transitions found. Ensure data has multiple steps per batch.")

    return pd.DataFrame(rows)


def train_transition_model(
    df: pd.DataFrame,
    transition: str,
    test_size: float,
    n_estimators: int,
) -> Dict[str, object]:
    """Train and evaluate a model for a specific step transition."""
    subset = df[df["transition"] == transition].copy()
    target_cols = [c for c in subset.columns if c.startswith("target_") and subset[c].notna().any()]
    feature_cols = [c for c in subset.columns if c not in target_cols and c not in METADATA_COLS]

    if not target_cols:
        raise ValueError(f"No targets found for transition {transition}")

    # Drop rows lacking targets; fill feature gaps with median.
    subset = subset.dropna(subset=target_cols)
    feature_medians = subset[feature_cols].median()
    X = subset[feature_cols].fillna(feature_medians)
    y = subset[target_cols]

    if len(X) < 3:
        # Not enough samples to create a holdout; train and evaluate on the same data.
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    base_regressor = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    model = MultiOutputRegressor(base_regressor)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds, multioutput="raw_values")
    mae_by_target = dict(zip(target_cols, mae))

    # Take a sample row for a quick qualitative view.
    sample_idx = X_test.index[0]
    sample_features = X.loc[sample_idx]
    sample_actual = y.loc[sample_idx]
    sample_pred = model.predict([sample_features.values])[0]
    sample_view = dict(actual=sample_actual.to_dict(), predicted=dict(zip(target_cols, sample_pred)))

    return {
        "transition": transition,
        "model": model,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "mae_by_target": mae_by_target,
        "sample": sample_view,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train next-step parameter predictors.")
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
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Holdout fraction for evaluation."
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in each RandomForest.",
    )
    args = parser.parse_args()

    print(f"Loading data from {args.data_root} (max_files={args.max_files}) ...")
    step_stats = load_step_stats(args.data_root, args.max_files)
    print(f"Loaded {len(step_stats)} aggregated step rows across {step_stats.batch_id.nunique()} batches.")

    print("Building transition rows ...")
    transitions_df = build_transition_rows(step_stats)
    print(f"Built {len(transitions_df)} transition examples across {transitions_df.transition.nunique()} transitions.")

    reports: List[Dict[str, object]] = []
    for transition in sorted(transitions_df.transition.unique()):
        report = train_transition_model(
            transitions_df, transition, test_size=args.test_size, n_estimators=args.n_estimators
        )
        reports.append(report)

    print("\n=== Evaluation ===")
    for report in reports:
        print(f"\nTransition: {report['transition']}")
        print(f"Rows: train={report['train_rows']} test={report['test_rows']}")
        for target, score in report["mae_by_target"].items():
            print(f"  MAE {target}: {score:.4f}")
        sample = report["sample"]
        print("  Sample prediction:")
        for sensor, actual_val in sample["actual"].items():
            pred_val = sample["predicted"][sensor]
            print(f"    {sensor}: actual={actual_val:.4f}, predicted={pred_val:.4f}")


if __name__ == "__main__":
    main()
