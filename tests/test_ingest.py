"""
tests/test_ingest.py — Deduplication, schema validation, idempotency
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from ingest import build_curated_table, load_stream_jsonl, setup_logger, write_outputs

FIXTURE_DIR = Path("tests/_fixtures/incident_test")
OUTPUT_DIR = Path("tests/_output_ingest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Create minimal raw JSONL fixtures before each test, clean up after."""
    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    _write_jsonl(FIXTURE_DIR / "pump.jsonl", [
        {"tsms": 1700000000000, "pumpid": "pump1", "intakepressurepsi": 65.0,
         "dischargepressurepsi": 120.0, "enginerpm": 1400, "tanklevelgal": 500.0},
        # Duplicate timestamp — should both survive ingest (dedup happens in features)
        {"tsms": 1700000000000, "pumpid": "pump1", "intakepressurepsi": 65.5,
         "dischargepressurepsi": 121.0, "enginerpm": 1410, "tanklevelgal": 499.0},
        {"tsms": 1700000001000, "pumpid": "pump1", "intakepressurepsi": 66.0,
         "dischargepressurepsi": 122.0, "enginerpm": 1420, "tanklevelgal": 498.0},
        # Out-of-order timestamp
        {"tsms": 1700000000500, "pumpid": "pump1", "intakepressurepsi": 65.8,
         "dischargepressurepsi": 121.5, "enginerpm": 1415, "tanklevelgal": 499.5},
    ])

    _write_jsonl(FIXTURE_DIR / "nozzle.jsonl", [
        {"tsms": 1700000000000, "nozzlepressurepsi": 110.0, "flowgpm": 150.0},
        {"tsms": 1700000001000, "nozzlepressurepsi": 108.0, "flowgpm": 148.0},
    ])

    _write_jsonl(FIXTURE_DIR / "hydrant.jsonl", [
        {"tsms": 1700000000000, "residualpressurepsi": 70.0, "staticpressurepsi": 75.0},
    ])

    yield

    shutil.rmtree(FIXTURE_DIR, ignore_errors=True)
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadStreamJsonl:
    def test_loads_records(self):
        df = load_stream_jsonl(
            FIXTURE_DIR / "pump.jsonl", "pump", "incident_test"
        )
        assert not df.empty
        assert "stream_type" in df.columns
        assert "incident_id" in df.columns
        assert (df["stream_type"] == "pump").all()

    def test_missing_file_returns_empty(self):
        df = load_stream_jsonl(
            FIXTURE_DIR / "nonexistent.jsonl", "pump", "incident_test"
        )
        assert df.empty


class TestBuildCuratedTable:
    def test_schema_has_required_columns(self):
        logger = setup_logger("test")
        curated = build_curated_table(FIXTURE_DIR, logger)
        assert "incident_id" in curated.columns
        assert "tsms" in curated.columns
        assert "stream_type" in curated.columns

    def test_all_streams_present(self):
        logger = setup_logger("test")
        curated = build_curated_table(FIXTURE_DIR, logger)
        streams = set(curated["stream_type"].unique())
        assert "pump" in streams
        assert "nozzle" in streams
        assert "hydrant" in streams

    def test_sorted_by_tsms(self):
        logger = setup_logger("test")
        curated = build_curated_table(FIXTURE_DIR, logger)
        assert curated["tsms"].is_monotonic_increasing

    def test_out_of_order_events_sorted(self):
        """Out-of-order raw events should appear sorted in curated output."""
        logger = setup_logger("test")
        curated = build_curated_table(FIXTURE_DIR, logger)
        pump = curated[curated["stream_type"] == "pump"]
        tsms_list = pump["tsms"].tolist()
        assert tsms_list == sorted(tsms_list)

    def test_duplicates_preserved_in_ingest(self):
        """Ingest keeps duplicate timestamps; dedup is Component B's job."""
        logger = setup_logger("test")
        curated = build_curated_table(FIXTURE_DIR, logger)
        pump = curated[curated["stream_type"] == "pump"]
        dup_count = (pump["tsms"] == 1700000000000).sum()
        assert dup_count == 2


class TestIdempotency:
    def test_double_ingest_produces_identical_output(self):
        logger = setup_logger("test")
        curated1 = build_curated_table(FIXTURE_DIR, logger)
        write_outputs(curated1, OUTPUT_DIR, FIXTURE_DIR, logger)
        df1 = pd.read_parquet(OUTPUT_DIR / "telemetry_curated.parquet")

        curated2 = build_curated_table(FIXTURE_DIR, logger)
        write_outputs(curated2, OUTPUT_DIR, FIXTURE_DIR, logger)
        df2 = pd.read_parquet(OUTPUT_DIR / "telemetry_curated.parquet")

        pd.testing.assert_frame_equal(df1, df2)

    def test_row_count_stable_across_reruns(self):
        logger = setup_logger("test")
        curated1 = build_curated_table(FIXTURE_DIR, logger)
        curated2 = build_curated_table(FIXTURE_DIR, logger)
        assert len(curated1) == len(curated2)
