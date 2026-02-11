import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

STREAM_FILES: Dict[str, str] = {
    "pump": "pump.jsonl",
    "nozzle": "nozzle.jsonl",
    "hydrant": "hydrant.jsonl",
}


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def load_stream_jsonl(path: Path, stream_type: str, incident_id: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    records: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["stream_type"] = stream_type
            rec["incident_id"] = incident_id
            records.append(rec)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def build_curated_table(input_root: Path, logger: logging.Logger) -> pd.DataFrame:
    """
    Component A: read raw JSONL streams for one incident and build a single
    curated telemetry table.

    Assumptions:
    - input_root contains pump.jsonl, nozzle.jsonl, hydrant.jsonl
    - Each JSON line has at least a tsms field (epoch milliseconds)
    """
    incident_id = input_root.name
    parts: List[pd.DataFrame] = []

    for stream_type, fname in STREAM_FILES.items():
        df = load_stream_jsonl(input_root / fname, stream_type, incident_id)
        if df.empty:
            continue

        if "tsms" in df.columns:
            ts_col = "tsms"
        elif "ts_ms" in df.columns:
            ts_col = "ts_ms"
        else:
            raise ValueError(f"{fname} missing tsms/ts_ms timestamp column")

        df = df.rename(columns={ts_col: "tsms"})
        parts.append(df)

    if not parts:
        logger.warning(f"No telemetry files found under {input_root}")
        return pd.DataFrame()

    curated = pd.concat(parts, ignore_index=True)
    curated = curated.sort_values(["incident_id", "tsms"]).reset_index(drop=True)

    logger.info(
        "Built curated telemetry table with %d rows for incident %s",
        len(curated),
        incident_id,
    )
    return curated


def write_outputs(
    curated: pd.DataFrame, output_dir: Path, input_root: Path, logger: logging.Logger
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "telemetry_curated.parquet"
    curated.to_parquet(parquet_path, index=False, engine="pyarrow")
    logger.info("Wrote curated telemetry to %s", parquet_path)

    manifest = {
        "n_rows": int(len(curated)),
        "incidents": (
            sorted(curated["incident_id"].unique().tolist())
            if not curated.empty
            else []
        ),
        "streams": (
            curated["stream_type"].value_counts().to_dict()
            if "stream_type" in curated.columns and not curated.empty
            else {}
        ),
        "input_root": str(input_root),
    }

    manifest_path = output_dir / "ingest_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Wrote ingest manifest to %s", manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "WaterOps Component A – Ingest raw JSONL telemetry and build curated table "
            "(single incident or project root)."
        )
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help=(
            "Path to a single incident directory (with pump/nozzle/hydrant.jsonl), "
            "or a project root containing incident_* subdirectories."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output directory (single-incident) or output root (for multi-incident mode)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logger = setup_logger("waterops.ingest", args.log_level)

    input_root = Path(args.input_root)
    output_root = Path(args.output)

    # Case 1: single-incident (current behavior) — input_root directly has JSONL files
    if any((input_root / fname).exists() for fname in STREAM_FILES.values()):
        logger.info("Detected single-incident layout at %s", input_root)
        curated = build_curated_table(input_root, logger)
        if curated.empty:
            logger.error("No curated telemetry built; nothing to write")
            sys.exit(1)
        write_outputs(curated, output_root, input_root, logger)
        return

    # Case 2: project root — scan incident_* subdirectories
    logger.info(
        "Detected project layout under %s; scanning incident_* subdirectories",
        input_root,
    )
    incident_dirs = sorted(
        d for d in input_root.iterdir()
        if d.is_dir() and d.name.startswith("incident_")
    )

    if not incident_dirs:
        logger.error(f"No incident_* directories found under {input_root}")
        sys.exit(1)

    any_success = False
    for incident_dir in incident_dirs:
        logger.info("Processing incident directory %s", incident_dir)
        curated = build_curated_table(incident_dir, logger)
        if curated.empty:
            logger.warning("Skipping %s: no telemetry files found", incident_dir)
            continue

        out_dir = output_root / incident_dir.name
        write_outputs(curated, out_dir, incident_dir, logger)
        any_success = True

    if not any_success:
        logger.error("No curated telemetry built for any incident; nothing to write")
        sys.exit(1)


if __name__ == "__main__":
    main()
