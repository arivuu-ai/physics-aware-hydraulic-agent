"""
waterops/features.py — Component B: Build Training Examples (Windowing + Joins)

- Reads curated telemetry (Component A) with incident_id, tsms, stream_type, sensor fields.
- Resamples pump/nozzle/supply to 1 Hz per incident with forward-fill, collapsing duplicate timestamps.
- Builds domain + rolling features with leakage guards.
- Attaches labels from labels.csv (ts_ms → tsms) using per-incident merge_asof.
- Supports single-incident and project layouts.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------

STREAM_TYPES = ["pump", "nozzle", "supply"]
DEFAULT_WINDOW_SIZES = [10, 30, 60]
RESAMPLE_FREQ_MS = 1000  # 1 Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def flatten_payload(df: pd.DataFrame, stream_prefix: str) -> pd.DataFrame:
    """
    For current ingest, there is no payload column; just keep incident_id, tsms,
    and numeric fields. If payload exists, flatten it.
    """
    if df.empty:
        return df

    if "payload" not in df.columns:
        keep_cols = ["incident_id", "tsms"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in keep_cols]
        return df[keep_cols + numeric_cols].copy()

    payload_df = pd.json_normalize(df["payload"].tolist())
    payload_df.columns = [f"{stream_prefix}_{c}" for c in payload_df.columns]
    result = pd.concat(
        [df[["incident_id", "tsms"]].reset_index(drop=True), payload_df],
        axis=1,
    )
    return result

EPOCH = pd.Timestamp("1970-01-01")
ONE_MS = pd.Timedelta("1ms")

def _dt_index_to_epoch_ms(idx: pd.DatetimeIndex) -> pd.Index:
    return (idx - EPOCH) // ONE_MS
# ---------------------------------------------------------------------------
# Feature Engine
# ---------------------------------------------------------------------------

class WaterOpsFeatureEngine:
    def __init__(
        self,
        curated_dir: str,
        labels_path: Optional[str] = None,
        window_sizes: Optional[List[int]] = None,
        history_window_sec: int = 60,
        log_level: str = "INFO",
    ):
        self.curated_dir = Path(curated_dir)
        self.labels_path = Path(labels_path) if labels_path else None
        self.window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
        self.history_sec = history_window_sec
        self.logger = _setup_logger("waterops.features", log_level)

    # ------------------------------------------------------------------ #
    # 1. Load curated streams
    # ------------------------------------------------------------------ #

    def _load_stream(self, stream_type: str) -> pd.DataFrame:
        candidates = [
            self.curated_dir / stream_type / "data.parquet",
            self.curated_dir / f"{stream_type}.parquet",
            self.curated_dir / "telemetry_curated.parquet",
        ]

        for path in candidates:
            if path.exists():
                df = pd.read_parquet(path)
                if "stream_type" in df.columns:
                    df = df[df["stream_type"] == stream_type]
                self.logger.info(
                    "Loaded %d records for stream '%s' from %s",
                    len(df),
                    stream_type,
                    path,
                )
                return df

        self.logger.warning("No curated file found for stream '%s'", stream_type)
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # 2. Flatten & resample each stream to 1 Hz
    # ------------------------------------------------------------------ #

    def _resample_stream(self, df: pd.DataFrame, stream_type: str) -> pd.DataFrame:
        flat = flatten_payload(df, stream_type)
        if flat.empty:
            return flat

        if "tsms" not in flat.columns:
            raise ValueError(f"Expected 'tsms' column for {stream_type}")

        resampled_parts: List[pd.DataFrame] = []

        for incident_id, grp in flat.groupby("incident_id"):
            grp = grp.sort_values("tsms").copy()
            grp["ts_dt"] = pd.to_datetime(grp["tsms"], unit="ms")
            grp = grp.set_index("ts_dt")

            # Drop duplicate timestamps; keep the latest sample at that instant
            grp = grp[~grp.index.duplicated(keep="last")]

            numeric_cols = grp.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != "tsms"]

            resampled = grp[numeric_cols].resample("1s").ffill()
            resampled["incident_id"] = incident_id
            resampled["tsms"] = _dt_index_to_epoch_ms(resampled.index)

            resampled_parts.append(resampled.reset_index(drop=True))

        result = pd.concat(resampled_parts, ignore_index=True)
        self.logger.info(
            "Resampled %s to 1 Hz: %d rows across %d incidents",
            stream_type,
            len(result),
            result["incident_id"].nunique(),
        )
        return result

    # ------------------------------------------------------------------ #
    # 3. Merge streams
    # ------------------------------------------------------------------ #

    def _merge_streams(self, streams: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        anchor = None

        for stype in ["pump", "nozzle", "supply"]:
            df = streams.get(stype)
            if df is None or df.empty:
                continue

            if anchor is None:
                anchor = df
            else:
                anchor = anchor.merge(
                    df,
                    on=["incident_id", "tsms"],
                    how="outer",
                    suffixes=("", f"_dup_{stype}"),
                )

        if anchor is None:
            return pd.DataFrame()

        anchor = anchor.sort_values(["incident_id", "tsms"]).reset_index(drop=True)
        self.logger.info(
            "Merged timeline: %d rows, %d columns", len(anchor), anchor.shape[1]
        )
        return anchor

    # ------------------------------------------------------------------ #
    # 4. Rolling features
    # ------------------------------------------------------------------ #

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_cols = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c != "tsms"
        ]
        base_cols = len(df.columns)

        for col in numeric_cols:
            shifted = df.groupby("incident_id")[col].shift(1)

            for w in self.window_sizes:
                rolled = shifted.groupby(df["incident_id"]).rolling(
                    window=w, min_periods=1
                )
                df[f"{col}_mean_{w}s"] = rolled.mean().reset_index(level=0, drop=True)
                df[f"{col}_std_{w}s"] = rolled.std().reset_index(level=0, drop=True)
                df[f"{col}_min_{w}s"] = rolled.min().reset_index(level=0, drop=True)
                df[f"{col}_max_{w}s"] = rolled.max().reset_index(level=0, drop=True)

        self.logger.info(
            "Added rolling features: %d → %d columns",
            base_cols,
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------ #
    # 5. Domain features
    # ------------------------------------------------------------------ #

    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Intake / discharge pressure differential
        if "intakepressurepsi" in df.columns and "dischargepressurepsi" in df.columns:
            df["pressure_differential"] = (
                df["dischargepressurepsi"] - df["intakepressurepsi"]
            )
            df["pressure_diff_roc"] = df.groupby("incident_id")[
                "pressure_differential"
            ].diff()

        # Tank drain rate and time to empty
        if "tanklevelgal" in df.columns:
            df["tank_level_roc"] = df.groupby("incident_id")["tanklevelgal"].diff()
            df["tank_time_to_empty"] = np.where(
                df["tank_level_roc"] < 0,
                -df["tanklevelgal"] / df["tank_level_roc"].replace(0, np.nan),
                np.nan,
            )

        # Nozzle-pump divergence
        if "nozzlepressurepsi" in df.columns and "dischargepressurepsi" in df.columns:
            df["nozzle_pump_ratio"] = (
                df["nozzlepressurepsi"]
                / df["dischargepressurepsi"].replace(0, np.nan)
            )
            df["nozzle_pump_divergence"] = df.groupby("incident_id")[
                "nozzle_pump_ratio"
            ].diff()

        # Supply residual trend
        if "residualpressurepsi" in df.columns:
            shifted = df.groupby("incident_id")["residualpressurepsi"].shift(1)
            for w in [10, 30]:
                df[f"supply_residual_trend_{w}s"] = (
                    shifted.groupby(df["incident_id"])
                    .rolling(window=w, min_periods=1)
                    .apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0]
                        if len(x) > 1
                        else 0.0,
                        raw=True,
                    )
                    .reset_index(level=0, drop=True)
                )

        self.logger.info("Added domain-specific features")
        return df

    # ------------------------------------------------------------------ #
    # 6. Labels
    # ------------------------------------------------------------------ #

    def _load_labels(self) -> pd.DataFrame:
        if self.labels_path is None:
            for candidate in [
                self.curated_dir / "labels.csv",
                self.curated_dir / "labels.parquet",
                self.curated_dir.parent / "data_raw" / "labels.csv",
            ]:
                if candidate.exists():
                    self.labels_path = candidate
                    break

        if self.labels_path is None or not self.labels_path.exists():
            self.logger.warning("No labels file found — returning features only")
            return pd.DataFrame()

        ext = self.labels_path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(self.labels_path)
        return pd.read_parquet(self.labels_path)

    def _attach_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        labels_df = self._load_labels()
        if labels_df.empty:
            df["risk_in_next_120s"] = np.nan
            return df

        # Normalize ts_ms → tsms
        if "ts_ms" in labels_df.columns and "tsms" not in labels_df.columns:
            labels_df = labels_df.rename(columns={"ts_ms": "tsms"})

        if "tsms" not in labels_df.columns:
            self.logger.error("Labels file missing 'tsms' column")
            df["risk_in_next_120s"] = np.nan
            return df

        labels_df["tsms"] = labels_df["tsms"].astype(np.int64)
        df["tsms"] = df["tsms"].astype(np.int64)

        parts: List[pd.DataFrame] = []
        for incident_id, grp in df.groupby("incident_id"):
            lgrp = labels_df[labels_df["incident_id"] == incident_id].copy()
            if lgrp.empty:
                grp = grp.copy()
                grp["risk_in_next_120s"] = np.nan
                parts.append(grp)
                continue

            grp = grp.sort_values("tsms").reset_index(drop=True)
            lgrp = lgrp.sort_values("tsms").reset_index(drop=True)

            joined = pd.merge_asof(
                grp,
                lgrp[["tsms", "risk_in_next_120s"]],
                on="tsms",
                direction="backward",
                tolerance=500,  # ms
            )
            parts.append(joined)

        merged = pd.concat(parts, ignore_index=True)
        n_labelled = merged["risk_in_next_120s"].notna().sum()
        self.logger.info(
            "Labels attached: %d/%d rows have labels", n_labelled, len(merged)
        )
        return merged

    # ------------------------------------------------------------------ #
    # 7. Filter valid rows
    # ------------------------------------------------------------------ #

    def _filter_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        valid_parts: List[pd.DataFrame] = []

        for incident_id, grp in df.groupby("incident_id"):
            if grp.empty:
                continue
            min_ts = grp["tsms"].min()
            cutoff = min_ts + self.history_sec * 1000
            valid = grp[grp["tsms"] >= cutoff]
            valid_parts.append(valid)

        if not valid_parts:
            return pd.DataFrame()

        result = pd.concat(valid_parts, ignore_index=True)

        if "risk_in_next_120s" in result.columns:
            before = len(result)
            result = result.dropna(subset=["risk_in_next_120s"])
            self.logger.info(
                "Filtered: %d → %d rows (dropped %d warm-up/unlabelled)",
                before,
                len(result),
                before - len(result),
            )

        return result

    # ------------------------------------------------------------------ #
    # 8. Single-incident pipeline
    # ------------------------------------------------------------------ #

    def build_training_table(self, output_path: Optional[str] = None) -> pd.DataFrame:
        streams: Dict[str, pd.DataFrame] = {}

        for stype in STREAM_TYPES:
            raw = self._load_stream(stype)
            if not raw.empty:
                streams[stype] = self._resample_stream(raw, stype)

        if not streams:
            self.logger.error("No stream data loaded — aborting")
            return pd.DataFrame()

        merged = self._merge_streams(streams)
        if merged.empty:
            return merged

        merged = self._add_domain_features(merged)
        merged = self._add_rolling_features(merged)
        merged = self._attach_labels(merged)
        training = self._filter_valid_rows(merged)

        if output_path:
            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            out_file = out / "training_table.parquet"
            training.to_parquet(out_file, index=False, engine="pyarrow")
            self.logger.info(
                "Training table written → %s (%d rows)", out_file, len(training)
            )

            manifest = {
                "history_window_sec": self.history_sec,
                "window_sizes": self.window_sizes,
                "n_features": len(
                    [
                        c
                        for c in training.columns
                        if c not in ["incident_id", "tsms", "risk_in_next_120s"]
                    ]
                ),
                "n_rows": len(training),
                "incidents": sorted(training["incident_id"].unique().tolist())
                if not training.empty
                else [],
                "label_distribution": (
                    training["risk_in_next_120s"].value_counts().to_dict()
                    if "risk_in_next_120s" in training.columns
                    else {}
                ),
            }
            (out / "feature_manifest.json").write_text(
                json.dumps(manifest, indent=2, default=str)
            )

        return training


# ---------------------------------------------------------------------------
# Project-mode CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WaterOps Component B — Build Training Examples"
    )
    parser.add_argument(
        "--curated",
        required=True,
        help="Path to curated data directory (single incident) or project root (incident_* dirs)",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to labels file (CSV or Parquet). Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory (single incident) or project root for training tables.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=60,
        help="History window in seconds (default: 60)",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[10, 30, 60],
        help="Rolling window sizes in seconds (default: 10 30 60)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    curated_root = Path(args.curated)
    output_root = Path(args.output)

    # Single-incident
    if (curated_root / "telemetry_curated.parquet").exists():
        logger = _setup_logger("waterops.features.cli", args.log_level)
        logger.info("Detected single-incident layout at %s", curated_root)
        engine = WaterOpsFeatureEngine(
            curated_dir=str(curated_root),
            labels_path=args.labels,
            window_sizes=args.window_sizes,
            history_window_sec=args.history_window,
            log_level=args.log_level,
        )
        engine.build_training_table(output_path=str(output_root))
        return

    # Project mode
    logger = _setup_logger("waterops.features.cli", args.log_level)
    logger.info(
        "Detected project layout under %s; scanning incident_* subdirectories",
        curated_root,
    )
    incident_dirs = sorted(
        d for d in curated_root.iterdir() if d.is_dir() and d.name.startswith("incident_")
    )

    if not incident_dirs:
        logger.error("No incident_* directories found under %s", curated_root)
        sys.exit(1)

    all_training: List[pd.DataFrame] = []
    for incident_dir in incident_dirs:
        logger.info("Processing incident directory %s", incident_dir)
        out_dir = output_root / incident_dir.name
        engine = WaterOpsFeatureEngine(
            curated_dir=str(incident_dir),
            labels_path=args.labels,
            window_sizes=args.window_sizes,
            history_window_sec=args.history_window,
            log_level=args.log_level,
        )
        training = engine.build_training_table(output_path=str(out_dir))
        if not training.empty:
            all_training.append(training)

    if all_training:
        combined = pd.concat(all_training, ignore_index=True)
        combined_out = output_root / "combined_training"
        combined_out.mkdir(parents=True, exist_ok=True)
        combined_file = combined_out / "training_table.parquet"
        combined.to_parquet(combined_file, index=False, engine="pyarrow")
        logger.info(
            "Combined training table written → %s (%d rows)",
            combined_file,
            len(combined),
        )


if __name__ == "__main__":
    main()
