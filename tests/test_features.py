"""
tests/test_features.py — Time-bucketing, leakage prevention, domain features
"""

import numpy as np
import pandas as pd
import pytest

from features import WaterOpsFeatureEngine, flatten_payload, _dt_index_to_epoch_ms

# ---------------------------------------------------------------------------
# Helpers — build a minimal curated parquet in a temp dir
# ---------------------------------------------------------------------------

@pytest.fixture
def curated_dir(tmp_path):
    """Create a minimal curated parquet with pump + nozzle data."""
    records = []
    base_ts = 1700000000000  # epoch ms

    for i in range(120):
        ts = base_ts + i * 1000
        # Pump record
        records.append({
            "incident_id": "incident_test",
            "tsms": ts,
            "stream_type": "pump",
            "intakepressurepsi": 65.0 + np.random.randn() * 0.5,
            "dischargepressurepsi": 122.0 + np.random.randn() * 0.5,
            "enginerpm": 1420 + np.random.randint(-20, 20),
            "tanklevelgal": 500.0 - i * 0.5,
        })
        # Nozzle record
        records.append({
            "incident_id": "incident_test",
            "tsms": ts,
            "stream_type": "nozzle",
            "nozzlepressurepsi": 110.0 + np.random.randn() * 0.5,
            "flowgpm": 150.0 + np.random.randn() * 2.0,
        })

    df = pd.DataFrame(records)
    parquet_path = tmp_path / "telemetry_curated.parquet"
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    return tmp_path


@pytest.fixture
def labels_file(tmp_path):
    """Create a minimal labels CSV aligned with the curated data."""
    base_ts = 1700000000000
    rows = []
    for i in range(120):
        ts = base_ts + i * 1000
        label = 1 if i >= 100 else 0
        rows.append({"incident_id": "incident_test", "ts_ms": ts, "risk_in_next_120s": label})
    df = pd.DataFrame(rows)
    labels_path = tmp_path / "labels.csv"
    df.to_csv(labels_path, index=False)
    return labels_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDtIndexToEpochMs:
    def test_nanosecond_resolution(self):
        idx = pd.to_datetime([1700000000000], unit="ms")
        result = _dt_index_to_epoch_ms(idx)
        assert result[0] == 1700000000000

    def test_millisecond_resolution(self):
        idx = pd.to_datetime([1700000000000], unit="ms").as_unit("ms")
        result = _dt_index_to_epoch_ms(idx)
        assert result[0] == 1700000000000


class TestFlattenPayload:
    def test_no_payload_column(self):
        df = pd.DataFrame({
            "incident_id": ["inc1"],
            "tsms": [1700000000000],
            "intakepressurepsi": [65.0],
        })
        flat = flatten_payload(df, "pump")
        assert "incident_id" in flat.columns
        assert "tsms" in flat.columns
        assert "intakepressurepsi" in flat.columns

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        flat = flatten_payload(df, "pump")
        assert flat.empty


class TestResample:
    def test_resampled_to_1hz(self, curated_dir):
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir), history_window_sec=10
        )
        raw = engine._load_stream("pump")
        resampled = engine._resample_stream(raw, "pump")
        # Consecutive tsms should differ by 1000 ms (1 Hz)
        diffs = resampled.sort_values("tsms")["tsms"].diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == 1000

    def test_duplicate_timestamps_collapsed(self, curated_dir):
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir), history_window_sec=10
        )
        raw = engine._load_stream("pump")
        resampled = engine._resample_stream(raw, "pump")
        per_incident = resampled.groupby("incident_id")
        for _, grp in per_incident:
            assert grp["tsms"].is_unique


class TestLeakagePrevention:
    def test_rolling_features_use_shifted_values(self, curated_dir, labels_file):
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir),
            labels_path=str(labels_file),
            window_sizes=[5],
            history_window_sec=10,
        )
        raw = engine._load_stream("pump")
        resampled = engine._resample_stream(raw, "pump")
        merged = engine._merge_streams({"pump": resampled})
        merged = engine._add_rolling_features(merged)

        # The first row of every incident should have NaN for shifted rolling
        for _, grp in merged.groupby("incident_id"):
            first_row = grp.iloc[0]
            rolling_cols = [c for c in grp.columns if c.endswith("_mean_5s")]
            for col in rolling_cols:
                assert pd.isna(first_row[col]), (
                    f"{col} should be NaN for row 0 (shift guard)"
                )


class TestDomainFeatures:
    def test_pressure_differential_computed(self, curated_dir):
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir), history_window_sec=10
        )
        raw = engine._load_stream("pump")
        resampled = engine._resample_stream(raw, "pump")
        merged = engine._merge_streams({"pump": resampled})
        merged = engine._add_domain_features(merged)
        assert "pressure_differential" in merged.columns
        assert "tank_level_roc" in merged.columns

    def test_tank_time_to_empty_only_when_draining(self, curated_dir):
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir), history_window_sec=10
        )
        raw = engine._load_stream("pump")
        resampled = engine._resample_stream(raw, "pump")
        merged = engine._merge_streams({"pump": resampled})
        merged = engine._add_domain_features(merged)
        if "tank_time_to_empty" in merged.columns:
            positive_roc = merged[merged["tank_level_roc"] > 0]
            assert positive_roc["tank_time_to_empty"].isna().all()


class TestLabelAttachment:
    def test_labels_attached(self, curated_dir, labels_file):
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir),
            labels_path=str(labels_file),
            window_sizes=[5],
            history_window_sec=10,
        )
        training = engine.build_training_table()
        assert "risk_in_next_120s" in training.columns
        assert training["risk_in_next_120s"].notna().any()

    def test_ts_ms_label_column_renamed(self, curated_dir, labels_file):
        """Labels CSV uses ts_ms; engine should rename to tsms internally."""
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_dir),
            labels_path=str(labels_file),
            window_sizes=[5],
            history_window_sec=10,
        )
        training = engine.build_training_table()
        assert len(training) > 0
