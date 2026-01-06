"""
Combine raw parquet parts for each day into a single parquet per date.

Usage:
    python src/combine_parquet_by_date.py --month 12 [--year 2025]

Outputs are written under data/combined/<month_name>/date=YYYY-MM-DD.parquet
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW_ROOT = Path("data/raw")
COMBINED_ROOT = Path("data/combined")


def combine_for_month(month: int, year: int | None = None) -> None:
    month_name = datetime(1900, month, 1).strftime("%B").lower()
    out_dir = COMBINED_ROOT / month_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for date_dir in sorted(RAW_ROOT.glob("date=*")):
        date_str = date_dir.name.split("=", 1)[-1]
        try:
            dt = datetime.fromisoformat(date_str)
        except ValueError:
            continue

        if dt.month != month or (year and dt.year != year):
            continue

        parts = sorted(date_dir.glob("*.parquet"))
        if not parts:
            continue

        frames = [pd.read_parquet(p) for p in parts]
        combined = pd.concat(frames, ignore_index=True)

        out_path = out_dir / f"date={dt.date().isoformat()}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"Wrote {out_path} ({len(combined)} rows from {len(parts)} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine raw parquet files by date for a given month.")
    parser.add_argument("--month", type=int, required=True, help="Month number (1-12)")
    parser.add_argument("--year", type=int, help="Optional year to filter")
    args = parser.parse_args()
    combine_for_month(args.month, args.year)


if __name__ == "__main__":
    main()
