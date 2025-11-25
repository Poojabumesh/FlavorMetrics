"""
Generate synthetic manufacturing sensor data with variability and store it as Parquet.

Output schema matches existing raw data:
plant_id, line_id, batch_id, ts, step, sensor, value, unit, in_spec

Example:
    python src/generate_synthetic_data.py --batches 8 --points-per-step 60
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorSpec:
    unit: str
    mean: float
    std: float
    lower: float
    upper: float


# Specs per step-sensor; adjust ranges to change variability.
SPECS: Dict[Tuple[str, str], SensorSpec] = {
    ("mashing", "temp"): SensorSpec("C", mean=65.0, std=1.5, lower=62.0, upper=68.0),
    ("boiling", "temp"): SensorSpec("C", mean=99.5, std=0.8, lower=98.0, upper=101.0),
    ("fermentation", "temp"): SensorSpec("C", mean=20.0, std=0.8, lower=18.0, upper=23.0),
    ("fermentation", "gravity"): SensorSpec("SG", mean=1.014, std=0.004, lower=1.006, upper=1.020),
    ("packaging", "count"): SensorSpec("units", mean=105, std=6, lower=90, upper=120),
}

STEP_ORDER = ["mashing", "boiling", "fermentation", "packaging"]


def generate_reading(step: str, sensor: str, spec: SensorSpec, ts: datetime) -> Dict[str, object]:
    val = random.gauss(spec.mean, spec.std)
    in_spec = spec.lower <= val <= spec.upper
    return {
        "ts": ts.isoformat().replace("+00:00", "Z"),
        "step": step,
        "sensor": sensor,
        "value": float(round(val, 4)),
        "unit": spec.unit,
        "in_spec": bool(in_spec),
    }


def generate_batch(
    batch_id: str,
    plant_id: str,
    line_id: str,
    start_ts: datetime,
    points_per_step: int,
    step_gap_seconds: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    current_ts = start_ts

    for step in STEP_ORDER:
        specs = {k[1]: v for k, v in SPECS.items() if k[0] == step}
        for i in range(points_per_step):
            for sensor, spec in specs.items():
                ts = current_ts + timedelta(seconds=i, milliseconds=random.randint(0, 900))
                rows.append(
                    {
                        "plant_id": plant_id,
                        "line_id": line_id,
                        "batch_id": batch_id,
                        **generate_reading(step, sensor, spec, ts),
                    }
                )
        current_ts += timedelta(seconds=step_gap_seconds)

    return rows


def write_parquet(rows: List[Dict[str, object]], output_root: Path) -> Path:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No rows generated")

    date_str = datetime.now(timezone.utc).date().isoformat()
    output_dir = output_root / f"date={date_str}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts_suffix = int(time.time() * 1000)
    path = output_dir / f"beer-synth-{ts_suffix}.parquet"
    df.to_parquet(path, engine="fastparquet", index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic manufacturing data.")
    parser.add_argument("--batches", type=int, default=10, help="Number of batches to generate.")
    parser.add_argument("--points-per-step", type=int, default=120, help="Readings per step/sensor.")
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/raw"), help="Raw data root (partitioned by date=...)."
    )
    parser.add_argument("--plant-id", type=str, default="plantA")
    parser.add_argument("--line-id", type=str, default="line1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--step-gap-seconds",
        type=int,
        default=600,
        help="Gap between steps to advance timestamps.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    all_rows: List[Dict[str, object]] = []
    start_ts = datetime.now(timezone.utc) - timedelta(hours=1)

    for b in range(args.batches):
        batch_start = start_ts + timedelta(minutes=b * 15)
        batch_id = f"batch-synth-{int(batch_start.timestamp())}"
        batch_rows = generate_batch(
            batch_id=batch_id,
            plant_id=args.plant_id,
            line_id=args.line_id,
            start_ts=batch_start,
            points_per_step=args.points_per_step,
            step_gap_seconds=args.step_gap_seconds,
        )
        all_rows.extend(batch_rows)

    path = write_parquet(all_rows, args.output_root)
    print(f"Wrote {len(all_rows)} rows across {args.batches} batches to {path}")


if __name__ == "__main__":
    main()
