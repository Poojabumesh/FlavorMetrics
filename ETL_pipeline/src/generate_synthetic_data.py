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


# Specs per step-sensor; adjust ranges to change variability. These serve as
# defaults and can be overridden by summary stats from a brewery CSV.
SPECS: Dict[Tuple[str, str], SensorSpec] = {
    ("mashing", "temp"): SensorSpec("C", mean=65.0, std=1.5, lower=62.0, upper=68.0),
    ("boiling", "temp"): SensorSpec("C", mean=99.5, std=0.8, lower=98.0, upper=101.0),
    ("fermentation", "temp"): SensorSpec("C", mean=20.0, std=0.8, lower=18.0, upper=23.0),
    ("fermentation", "gravity"): SensorSpec("SG", mean=1.014, std=0.004, lower=1.006, upper=1.020),
    ("packaging", "count"): SensorSpec("units", mean=105, std=6, lower=90, upper=120),
}

STEP_ORDER = ["mashing", "boiling", "fermentation", "packaging"]
DEFAULT_PLANTS = ("plantA", "plantB")
DEFAULT_LINES = ("line1", "line2")

# Plant-level differences: plantA has intentionally higher variance on fermentation temperature.
PLANT_OVERRIDES: Dict[str, Dict[Tuple[str, str], SensorSpec]] = {
    "plantA": {
        ("fermentation", "temp"): SensorSpec(
            "C", mean=SPECS[("fermentation", "temp")].mean, std=1.25, lower=18.0, upper=23.5
        )
    },
    "plantB": {
        ("fermentation", "temp"): SensorSpec(
            "C", mean=SPECS[("fermentation", "temp")].mean, std=0.6, lower=18.5, upper=22.5
        )
    },
}

# Line-level differences: line1 represents bottling, line2 represents canning (higher throughput).
LINE_OVERRIDES: Dict[str, Dict[Tuple[str, str], SensorSpec]] = {
    "line1": {
        ("packaging", "count"): SensorSpec("units", mean=100, std=7, lower=85, upper=125),
    },
    "line2": {
        ("packaging", "count"): SensorSpec("units", mean=135, std=10, lower=110, upper=165),
    },
}


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


def load_brewery_informed_specs(csv_path: Path) -> Dict[Tuple[str, str], SensorSpec]:
    """
    Override default sensor specs using summary statistics from a brewery CSV.
    The CSV is expected to have batch-level columns such as Temperature,
    Gravity, and Volume_Produced. Missing columns fall back to defaults.
    """
    df = pd.read_csv(csv_path)
    specs = dict(SPECS)

    def update_from_column(
        step: str, sensor: str, col: str, unit: str, lower_quantile: float = 0.02, upper_quantile: float = 0.98
    ) -> None:
        if col not in df:
            return
        series = df[col].dropna()
        if series.empty:
            return

        mean = float(series.mean())
        std = float(series.std(ddof=0))
        if std == 0.0:
            std = max(abs(mean) * 0.01, 0.001)
        lower = float(series.quantile(lower_quantile))
        upper = float(series.quantile(upper_quantile))
        specs[(step, sensor)] = SensorSpec(unit, mean=mean, std=std, lower=lower, upper=upper)

    # Map batch-level columns to step-level sensors where sensible.
    update_from_column("fermentation", "temp", "Temperature", "C")
    update_from_column("fermentation", "gravity", "Gravity", "SG")
    update_from_column("packaging", "count", "Volume_Produced", "units")

    return specs


def build_specs_for_plant_line(
    plant_id: str, line_id: str, base_specs: Dict[Tuple[str, str], SensorSpec] | None = None
) -> Dict[Tuple[str, str], SensorSpec]:
    """
    Apply plant + line specific overrides (e.g., more variance for plantA fermentation temp,
    higher packaging throughput for canning line2).
    """
    specs = dict(base_specs or SPECS)

    for overrides in (PLANT_OVERRIDES.get(plant_id, {}), LINE_OVERRIDES.get(line_id, {})):
        for key, override in overrides.items():
            specs[key] = override

    return specs


def generate_batch(
    batch_id: str,
    plant_id: str,
    line_id: str,
    start_ts: datetime,
    points_per_step: int,
    step_gap_seconds: int,
    specs: Dict[Tuple[str, str], SensorSpec],
    mean_jitter_pct: float = 0.0,
    std_mult: float = 1.0,
) -> List[Dict[str, object]]:
    """Generate one batch worth of readings for a specific plant/line using resolved specs."""
    rows: List[Dict[str, object]] = []
    current_ts = start_ts
    # Introduce batch-level noise to help avoid overfitting on perfectly clean data.
    batch_specs: Dict[Tuple[str, str], SensorSpec] = {}
    for key, spec in specs.items():
        if mean_jitter_pct > 0:
            jitter = random.gauss(0, abs(spec.mean) * mean_jitter_pct)
        else:
            jitter = 0.0
        batch_specs[key] = SensorSpec(
            spec.unit,
            mean=spec.mean + jitter,
            std=spec.std * std_mult,
            lower=spec.lower,
            upper=spec.upper,
        )

    for step in STEP_ORDER:
        step_specs = {k[1]: v for k, v in batch_specs.items() if k[0] == step}
        for i in range(points_per_step):
            for sensor, spec in step_specs.items():
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
    parser.add_argument(
        "--plant-ids",
        nargs="+",
        default=list(DEFAULT_PLANTS),
        help="Plants to generate (e.g., plantA plantB).",
    )
    parser.add_argument(
        "--line-ids",
        nargs="+",
        default=list(DEFAULT_LINES),
        help="Lines to generate per plant (e.g., line1 line2).",
    )
    parser.add_argument("--plant-id", type=str, default=None, help="Deprecated: use --plant-ids.")
    parser.add_argument("--line-id", type=str, default=None, help="Deprecated: use --line-ids.")
    parser.add_argument(
        "--mean-jitter-pct",
        type=float,
        default=0.03,
        help="Per-batch Gaussian jitter applied to sensor means (as a fraction, e.g., 0.03 = 3%).",
    )
    parser.add_argument(
        "--std-mult",
        type=float,
        default=1.0,
        help="Multiplier applied to sensor std dev to add measurement noise.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--step-gap-seconds",
        type=int,
        default=600,
        help="Gap between steps to advance timestamps.",
    )
    parser.add_argument(
        "--brewery-csv",
        type=Path,
        default=None,
        help="Optional batch-level CSV (e.g., dataset/brewery_data.csv) to derive sensor means/stds.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    base_specs = load_brewery_informed_specs(args.brewery_csv) if args.brewery_csv else dict(SPECS)

    plant_ids = args.plant_ids or list(DEFAULT_PLANTS)
    line_ids = args.line_ids or list(DEFAULT_LINES)
    if args.plant_id:
        plant_ids = [args.plant_id]
    if args.line_id:
        line_ids = [args.line_id]

    all_rows: List[Dict[str, object]] = []
    start_ts = datetime.now(timezone.utc) - timedelta(hours=1)

    for plant_id in plant_ids:
        for line_id in line_ids:
            combo_specs = build_specs_for_plant_line(plant_id, line_id, base_specs)
            for b in range(args.batches):
                batch_start = start_ts + timedelta(minutes=b * 15)
                batch_id = f"batch-synth-{plant_id}-{line_id}-{int(batch_start.timestamp())}"
                batch_rows = generate_batch(
                    batch_id=batch_id,
                    plant_id=plant_id,
                    line_id=line_id,
                    start_ts=batch_start,
                    points_per_step=args.points_per_step,
                    step_gap_seconds=args.step_gap_seconds,
                    specs=combo_specs,
                    mean_jitter_pct=args.mean_jitter_pct,
                    std_mult=args.std_mult,
                )
                all_rows.extend(batch_rows)

    path = write_parquet(all_rows, args.output_root)
    total_batches = args.batches * len(plant_ids) * len(line_ids)
    print(f"Wrote {len(all_rows)} rows across {total_batches} batches to {path}")


if __name__ == "__main__":
    main()
