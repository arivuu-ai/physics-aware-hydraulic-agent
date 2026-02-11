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

    def test_split_ratio(self, tmp_path):
        out_dir = _make_training_table(tmp_path, n_rows=100)
        trainer = WaterOpsTrainer(
            model_dir=str(tmp_path / "artifacts"), log_level="WARNING"
        )
        df = trainer._load_training_table(str(out_dir))
        train_df, test_df = trainer._temporal_split(df, train_frac=0.8)
        assert len(train_df) == 80
        assert len(test_df) == 20


class TestSelectThreshold:
    def test_f1_strategy_returns_valid_threshold(self):
        np.random.seed(42)
        y_true = np.array([0]*80 + [1]*20)
        y_prob = np.random.rand(100)
        y_prob[80:] += 0.3
        y_prob = np.clip(y_prob, 0, 1)
        threshold, detail = select_threshold(y_true, y_prob, strategy="f1")
        assert 0.0 <= threshold <= 1.0
        assert "precision" in detail
        assert "recall" in detail
        assert "f1" in detail

    def test_high_recall_strategy(self):
        np.random.seed(42)
        y_true = np.array([0]*80 + [1]*20)
        y_prob = np.concatenate([np.random.rand(80) * 0.4, np.random.rand(20) * 0.5 + 0.5])
        threshold, detail = select_threshold(y_true, y_prob, strategy="high_recall")
        assert detail["recall"] >= 0.85  # should be near 0.90


class TestEvaluateModel:
    def test_all_metrics_present(self):
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        metrics = evaluate_model(y_true, y_prob, threshold=0.5)
        for key in ["threshold", "precision", "recall", "f1_score",
                     "mcc", "pr_auc", "roc_auc", "confusion_matrix", "support"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.9, 1.0])
        metrics = evaluate_model(y_true, y_prob, threshold=0.5)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_metrics_bounded(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        metrics = evaluate_model(y_true, y_prob, threshold=0.5)
        assert 0.0 <= metrics["pr_auc"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert -1.0 <= metrics["mcc"] <= 1.0


class TestEndToEnd:
    def test_train_and_evaluate_produces_artifacts(self, tmp_path):
        out_dir = _make_training_table(tmp_path, n_rows=200)
        artifact_dir = tmp_path / "artifacts"
        trainer = WaterOpsTrainer(
            model_dir=str(artifact_dir), log_level="WARNING"
        )
        results = trainer.train_and_evaluate(data_path=str(out_dir))
        assert "baseline_logistic_regression" in results
        assert "improved_xgboost" in results
        assert (artifact_dir / "model_latest.pkl").exists()
        assert (artifact_dir / "metadata_latest.json").exists()
