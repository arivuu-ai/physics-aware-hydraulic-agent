"""
tests/test_train.py — Temporal split, metric computation
"""

import numpy as np
import pandas as pd
import pytest

from train import WaterOpsTrainer, evaluate_model, select_threshold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_table(tmp_path, n_rows=200):
    """Create a synthetic training_table.parquet."""
    np.random.seed(42)
    base_ts = 1700000060000  # after 60s warm-up
    rows = []
    for i in range(n_rows):
        row = {
            "incident_id": "incident_test",
            "tsms": base_ts + i * 1000,
            "feat_a": np.random.randn(),
            "feat_b": np.random.randn(),
            "feat_c": np.random.randn(),
            "risk_in_next_120s": 1 if i >= (n_rows - 40) else 0,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = tmp_path / "training"
    out_dir.mkdir()
    df.to_parquet(out_dir / "training_table.parquet", index=False)
    return out_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalSplit:
    def test_split_is_temporal_not_random(self, tmp_path):
        out_dir = _make_training_table(tmp_path)
        trainer = WaterOpsTrainer(
            model_dir=str(tmp_path / "artifacts"), log_level="WARNING"
        )
        df = trainer._load_training_table(str(out_dir))
        train_df, test_df = trainer._temporal_split(df, train_frac=0.8)

        # Every test timestamp must be >= every train timestamp per incident
        for inc_id in df["incident_id"].unique():
            train_max = train_df[train_df["incident_id"] == inc_id]["tsms"].max()
            test_min = test_df[test_df["incident_id"] == inc_id]["tsms"].min()
            assert test_min >= train_max, "Test set contains timestamps before train set"

    def test
