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
        {"tsms": 1700000001000, "nozzlepressurepsi
