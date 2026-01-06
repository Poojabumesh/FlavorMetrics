"""
Integration-style tests for the beer pipeline components.

These tests generate a small synthetic batch dataset, write it to a
date-partitioned Parquet location, and then exercise the next-step
prediction pipeline end-to-end (load → aggregate → build transitions → train).
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

import pandas as pd

# Ensure src/ is importable when tests are run from the repo root or ETL_pipeline/.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from generate_synthetic_data import build_specs_for_plant_line, generate_batch, SPECS  # type: ignore  # noqa: E402
from predict_next_step import (  # type: ignore  # noqa: E402
    build_transition_rows,
    load_step_stats,
    train_transition_model,
)


def _write_sample_parquet(data_root: Path, batches: int = 3, points_per_step: int = 6) -> Path:
    """Create a small synthetic dataset and write it under date=... with beer-*.parquet naming."""
    now = datetime.now(timezone.utc)
    rows = []
    plant_lines = [("plantA", "line1"), ("plantA", "line2"), ("plantB", "line1"), ("plantB", "line2")]

    for combo_idx, (plant_id, line_id) in enumerate(plant_lines):
        specs = build_specs_for_plant_line(plant_id, line_id, SPECS)
        for i in range(batches):
            batch_id = f"batch-test-{plant_id}-{line_id}-{i}"
            start_ts = now - timedelta(minutes=10 * (i + combo_idx))
            rows.extend(
                generate_batch(
                    batch_id=batch_id,
                    plant_id=plant_id,
                    line_id=line_id,
                    start_ts=start_ts,
                    points_per_step=points_per_step,
                    step_gap_seconds=60,
                    specs=specs,
                )
            )

    df = pd.DataFrame(rows)
    date_str = now.date().isoformat()
    out_dir = data_root / f"date={date_str}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "beer-000.parquet"
    df.to_parquet(out_path, engine="fastparquet", index=False)
    return out_path


class TestNextStepPipeline(unittest.TestCase):
    def test_end_to_end_training(self) -> None:
        """Verify we can load synthetic parquet, build transitions, and train per-step models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "raw"
            parquet_path = _write_sample_parquet(data_root)
            self.assertTrue(parquet_path.exists(), "Parquet file was not created")

            step_stats = load_step_stats(data_root, max_files=None)
            self.assertFalse(step_stats.empty, "Step stats should not be empty")

            transitions_df = build_transition_rows(step_stats)
            self.assertFalse(transitions_df.empty, "Transition rows should not be empty")
            self.assertGreaterEqual(
                transitions_df.transition.nunique(), 3, "Expected at least the three brewing transitions"
            )

            for transition in transitions_df.transition.unique():
                report = train_transition_model(
                    transitions_df, transition=transition, test_size=0.34, n_estimators=10
                )
                self.assertIn("mae_by_target", report)
                self.assertGreater(report["train_rows"], 0)
                self.assertGreater(report["test_rows"], 0)
                self.assertTrue(report["target_cols"], "Model should have target columns")


if __name__ == "__main__":
    unittest.main()
