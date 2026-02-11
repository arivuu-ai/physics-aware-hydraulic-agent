"""
waterops/serve.py — Component D: Inference Interface (JSONL-aligned version)
"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import json
import logging
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_dir: str = os.getenv("WATEROPS_MODEL_DIR", "./artifacts")
    log_level: str = os.getenv("WATEROPS_LOG_LEVEL", "INFO")
    enable_shap: bool = os.getenv("WATEROPS_ENABLE_SHAP", "true").lower() == "true"

    class Config:
        env_prefix = "WATEROPS_"


settings = Settings()


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


logger = _setup_logger("waterops.serve", settings.log_level)


class ModelState:
    def __init__(self):
        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.feature_columns: List[str] = []
        self.threshold: float = 0.5
        self.explainer = None
        self.ready: bool = False

    def load(self, model_dir: str):
        model_path = Path(model_dir)
        model_file = model_path / "model_latest.pkl"
        meta_file = model_path / "metadata_latest.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        self.pipeline = joblib.load(model_file)
        logger.info(f"Model loaded from {model_file}")
        self.metadata = json.loads(meta_file.read_text())
        self.feature_columns = self.metadata.get("feature_columns", [])
        self.threshold = self.metadata.get("threshold", 0.5)
        logger.info(
            f"Metadata loaded: version={self.metadata.get('version')}, "
            f"threshold={self.threshold}, n_features={len(self.feature_columns)}"
        )
        if settings.enable_shap:
            try:
                model_step = self.pipeline.named_steps.get("model")
                if model_step is not None:
                    self.explainer = shap.TreeExplainer(model_step)
                    logger.info("SHAP TreeExplainer initialized")
                else:
                    logger.warning("Could not find 'model' step in pipeline — SHAP disabled")
            except Exception as e:
                logger.warning(f"SHAP init failed (non-fatal): {e}")
        self.ready = True


model_state = ModelState()


FEATURE_INSIGHT_MAP = {
    "tank_level": (
        "Urgent: Tank level dropping rapidly. "
        "Verify supply line or reduce flow demand."
    ),
    "intake_pressure": (
        "Warning: Intake pressure collapse detected. Possible cavitation — "
        "consider throttling down pump RPM."
    ),
    "pressure_differential": (
        "Warning: Intake-discharge pressure differential is abnormal. "
        "Check for supply obstruction or air in the line."
    ),
    "nozzle_pump": (
        "Anomaly: Nozzle pressure dropping while pump discharge is steady. "
        "Check for hose kink, burst line, or nozzle blockage."
    ),
    "supply_residual": (
        "Alert: Hydrant/supply residual pressure is critically low. "
        "Prepare to switch to tank supply or request additional hydrant."
    ),
    "tank_time_to_empty": (
        "Urgent: Tank projected to empty soon at current flow rate. "
        "Reduce flow or establish additional water supply immediately."
    ),
    "discharge_pressure": (
        "Warning: Discharge pressure trending abnormally. "
        "Verify pump throttle setting and relief valve."
    ),
}

DEFAULT_INSIGHT = (
    "Alert: Elevated risk detected. Monitor all pressure gauges and flow rates closely."
)
STABLE_INSIGHT = "System operating within normal parameters. Continue current operations."


def _match_insight(feature_name: str) -> str:
    feature_lower = feature_name.lower()
    for key, recommendation in FEATURE_INSIGHT_MAP.items():
        if key in feature_lower:
            return recommendation
    return DEFAULT_INSIGHT


def build_reasoning(risk_score: float, threshold: float, top_features: List[Dict[str, Any]]):
    is_unstable = risk_score >= threshold
    if not is_unstable:
        return {"top_drivers": top_features, "insight": STABLE_INSIGHT}
    primary = top_features[0]["feature"] if top_features else "unknown"
    insight = _match_insight(primary)
    return {"top_drivers": top_features, "insight": insight}


def build_agent_protocol(risk_score: float, threshold: float) -> Dict[str, str]:
    if risk_score < threshold:
        return {
            "suggested_action": "MAINTAIN",
            "priority": "NORMAL",
            "escalation_required": "false",
        }
    if risk_score >= 0.8:
        return {
            "suggested_action": "EMERGENCY_RESPONSE",
            "priority": "CRITICAL",
            "escalation_required": "true",
        }
    elif risk_score >= 0.6:
        return {
            "suggested_action": "REDUCE_THROTTLE",
            "priority": "HIGH",
            "escalation_required": "true",
        }
    else:
        return {
            "suggested_action": "INCREASE_MONITORING",
            "priority": "ELEVATED",
            "escalation_required": "false",
        }


class TelemetryReading(BaseModel):
    ts_ms: int = Field(..., gt=0, description="Event timestamp in epoch ms")
    stream_type: str = Field(..., description="pump, nozzle, or supply")
    payload: Dict[str, Any] = Field(default_factory=dict)


class PredictionRequest(BaseModel):
    incident_id: str = Field(..., min_length=1)
    telemetry: List[TelemetryReading] = Field(..., min_length=1)

    @field_validator("telemetry")
    @classmethod
    def validate_telemetry_not_empty(cls, v):
        if not v:
            raise ValueError("At least one telemetry reading is required")
        return v


class InferenceResult(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    is_unstable: bool
    threshold_used: float


class ReasoningResult(BaseModel):
    top_drivers: List[Dict[str, Any]]
    insight: str


class AgentProtocol(BaseModel):
    suggested_action: str
    priority: str
    escalation_required: str


class PredictionResponse(BaseModel):
    incident_id: str
    inference: InferenceResult
    reasoning: ReasoningResult
    agent_protocol: AgentProtocol
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None


def construct_features(telemetry: List[TelemetryReading], feature_columns: List[str]) -> np.ndarray:
    rows = []
    for reading in telemetry:
        row: Dict[str, Any] = {"ts_ms": reading.ts_ms, "stream_type": reading.stream_type}
        if reading.stream_type == "pump":
            key_map = {
                "engine_rpm": "rpm",
                "throttle_pct": "throttle",
                "intake_pressure_psi": "intake_pressure",
                "discharge_pressure_psi": "discharge_pressure",
                "tank_level_gal": "tank_level",
            }
        elif reading.stream_type == "nozzle":
            key_map = {
                "nozzle_pressure_psi": "nozzle_pressure",
                "flow_gpm": "flow_estimate",
                "nozzle_setting_gpm": "nozzle_setting",
            }
        elif reading.stream_type == "supply":
            key_map = {
                "residual_pressure_psi": "residual_pressure",
                "static_pressure_psi": "static_pressure",
            }
        else:
            key_map = {}

        for k, v in reading.payload.items():
            renamed = key_map.get(k, k)
            col_name = f"{reading.stream_type}_{renamed}"
            row[col_name] = v

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("ts_ms")
    feature_dict: Dict[str, Any] = {}

    numeric_cols = [
        c
        for c in df.columns
        if c not in ("ts_ms", "stream_type") and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            feature_dict[col] = np.nan
            continue
        feature_dict[col] = series.iloc[-1]
        for suffix, func in [("mean", np.nanmean), ("std", np.nanstd), ("min", np.nanmin), ("max", np.nanmax)]:
            for w_label in ["10s", "30s", "60s"]:
                key = f"{col}_{suffix}_{w_label}"
                if key in feature_columns:
                    feature_dict[key] = func(series.values)

    ip = feature_dict.get("pump_intake_pressure", np.nan)
    dp = feature_dict.get("pump_discharge_pressure", np.nan)
    if not (np.isnan(ip) or np.isnan(dp)):
        feature_dict["pressure_differential"] = dp - ip
        feature_dict["pressure_diff_roc"] = 0.0

    np_ = feature_dict.get("nozzle_nozzle_pressure", np.nan)
    if not np.isnan(np_) and not np.isnan(dp) and dp != 0:
        feature_dict["nozzle_pump_ratio"] = np_ / dp
        feature_dict["nozzle_pump_divergence"] = 0.0

    tl = feature_dict.get("pump_tank_level", np.nan)
    if not np.isnan(tl):
        tl_series = df.get("pump_tank_level", pd.Series(dtype=float)).dropna()
        if len(tl_series) > 1:
            feature_dict["tank_level_roc"] = float(tl_series.diff().iloc[-1])
            roc = feature_dict["tank_level_roc"]
            if roc < 0:
                feature_dict["tank_time_to_empty"] = -tl / roc
            else:
                feature_dict["tank_time_to_empty"] = np.nan

    vector = [feature_dict.get(col, np.nan) for col in feature_columns]
    return np.array(vector, dtype=np.float64).reshape(1, -1)


def explain_prediction(
    feature_vector: np.ndarray,
    feature_columns: List[str],
    explainer,
    pipeline,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    if explainer is None:
        model = pipeline.named_steps.get("model")
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:top_k]
            return [
                {
                    "feature": feature_columns[i],
                    "importance": round(float(importances[i]), 4),
                    "type": "global_importance",
                }
                for i in top_idx
            ]
        return []
    try:
        imputer = pipeline.named_steps.get("imputer")
        if imputer is not None:
            X_imp = imputer.transform(feature_vector)
        else:
            X_imp = feature_vector
        shap_values = explainer.shap_values(X_imp)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_k]
        return [
            {
                "feature": feature_columns[i],
                "shap_value": round(float(sv[i]), 4),
                "feature_value": round(float(feature_vector[0, i]), 4)
                if not np.isnan(feature_vector[0, i])
                else None,
                "type": "shap",
            }
            for i in top_idx
        ]
    except Exception as e:
        logger.warning(f"SHAP explanation failed (non-fatal): {e}")
        return []


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting WaterOps inference service...")
    try:
        model_state.load(settings.model_dir)
        logger.info("Model loaded successfully — service is ready")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    yield
    logger.info("Shutting down WaterOps inference service")


app = FastAPI(
    title="WaterOps Risk Predictor",
    description="Predicts water supply instability risk from telemetry data",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("WATEROPS_CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Inference-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.get("/livez", response_model=HealthResponse, tags=["health"])
async def liveness():
    return HealthResponse(
        status="alive",
        model_loaded=model_state.ready,
        model_version=model_state.metadata.get("version"),
    )


@app.get("/readyz", response_model=HealthResponse, tags=["health"])
async def readiness():
    if not model_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ready",
        model_loaded=True,
        model_version=model_state.metadata.get("version"),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest):
    if not model_state.ready:
        raise HTTPException(status_code=503, detail="Model not loaded — service not ready")

    start = time.perf_counter()
    try:
        features = construct_features(request.telemetry, model_state.feature_columns)
    except Exception as e:
        logger.error(f"Feature construction failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to construct features: {e}")

    try:
        risk_score = float(model_state.pipeline.predict_proba(features)[:, 1][0])
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model prediction failed")

    top_features = explain_prediction(
        features,
        model_state.feature_columns,
        model_state.explainer,
        model_state.pipeline,
        top_k=5,
    )

    reasoning = build_reasoning(risk_score, model_state.threshold, top_features)
    agent_protocol = build_agent_protocol(risk_score, model_state.threshold)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"Prediction: incident={request.incident_id} "
        f"risk={risk_score:.4f} unstable={risk_score >= model_state.threshold} "
        f"action={agent_protocol['suggested_action']} "
        f"priority={agent_protocol['priority']} "
        f"latency={elapsed_ms:.1f}ms"
    )

    return PredictionResponse(
        incident_id=request.incident_id,
        inference=InferenceResult(
            risk_score=round(risk_score, 4),
            is_unstable=bool(risk_score >= model_state.threshold),
            threshold_used=model_state.threshold,
        ),
        reasoning=ReasoningResult(
            top_drivers=reasoning["top_drivers"],
            insight=reasoning["insight"],
        ),
        agent_protocol=AgentProtocol(
            suggested_action=agent_protocol["suggested_action"],
            priority=agent_protocol["priority"],
            escalation_required=agent_protocol["escalation_required"],
        ),
        model_version=model_state.metadata.get("version", "unknown"),
        inference_time_ms=round(elapsed_ms, 1),
    )


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="WaterOps Component D — Inference Service (JSONL-aligned)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--model-dir", default="./artifacts", help="Path to model artifacts")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn workers (use 1 for model serving)",
    )
    args = parser.parse_args()

    os.environ["WATEROPS_MODEL_DIR"] = args.model_dir
    os.environ["WATEROPS_LOG_LEVEL"] = args.log_level

    uvicorn.run(
        "waterops.serve:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()
