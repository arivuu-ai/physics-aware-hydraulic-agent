# WaterOps Predictor: Agentic AI for Hydraulic Failure Prediction
Architecture: Physics-Informed XGBoost + Agentic Decision Layer Domain: High-Pressure Firefighting Hydraulics

# 1. Executive Summary
This repository contains a production-ready prototype for detecting hydraulic failure modes (specifically cavitation and mechanical degradation).
Unlike standard "black box" ML approaches, this solution utilizes First-Principles Physics (e.g., Efficiency Ratios, Cavitation Signatures) to ensure the model is robust, interpretable, and actionable. It wraps the prediction in an Agentic Protocol that translates risk scores into structured operational commands (MAINTAIN, THROTTLE, EMERGENCY).
# Domain: 
Firefighting water operations — pump, nozzle, and water supply telemetry.
# Objective: 
Predict a *Water Supply Risk Score* — the probability that water delivery will become unstable within the next 2 minutes, based on the last 60 seconds of telemetry.

---

# 2. Architecture Overview 
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WaterOps Predictor Pipeline                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  data_raw/                Component A              data_curated/        │
│  ├── pump_telemetry.csv   ─────────────────────►   ├── pump/data.parquet│
│  ├── nozzle_telemetry.csv  waterops/ingest.py      ├── nozzle/          │
│  ├── supply_telemetry.csv                          ├── supply/          │
│  └── labels.csv                                    └── ingest_manifest  │
│                                                         │               │
│                           Component B                   │               │
│                           ─────────────────────►  data_training/        │
│                            waterops/features.py   ├── training_table    │
│                                                   └── feature_manifest  │
│                                                         │               │
│                           Component C                   │               │
│                           ─────────────────────►  artifacts/            │
│                            waterops/train.py      ├── model_latest.pkl  │
│                                                   └── metadata_latest   │
│                                                         │               │
│                           Component D                   │               │
│                           ─────────────────────►  :8000                 │
│                            waterops/serve.py      ├── POST /predict     │
│                                                   ├── GET  /livez       │
│                                                   └── GET  /readyz      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# 3. Quick Start

```bash
# 1. Set up environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Ingest raw telemetry → curated Parquet
python -m waterops.ingest --input ./data_raw --output ./data_curated

# 3. Build training features
python -m waterops.features --curated ./data_curated --labels ./data_raw/labels.csv --output ./data_training

# 4. Train & evaluate models
python -m waterops.train --data ./data_training --model-out ./artifacts

# 5. Serve predictions
python -m waterops.serve --model-dir ./artifacts --port 8000

# 6. Test a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @examples/sample_request.json
```

---

# 4. Key Design Decisions & Tradeoffs

### A. Ingest (Component A)

| Decision | Rationale |
|---|---|
| Pydantic schema with `StreamType` enum | Catches invalid stream types at ingest time, not downstream |
| Composite dedup key (incident_id + ts_ms + stream_type + payload_hash) | Handles records that differ only in float formatting |
| Idempotent delete-write pattern | Re-running on same input produces identical output without stale partitions |
| NaN → None in payload; drop rows missing incident_id or ts_ms | Sensor gaps are preserved for downstream imputation; identity/time are non-negotiable |
| Per-stream Parquet output | Preserves stream identity for Component B joins |

### B. Feature Engineering (Component B)

| Decision | Rationale |
|---|---|
| 1 Hz resampling with forward-fill only | Aligns multi-rate streams (pump 2Hz, nozzle 1Hz, supply 0.2Hz) without future leakage |
| `.shift(1)` before `.rolling()` | Ensures rolling window at time t covers [t-W, t-1], never including current row |
| Domain features targeting 4 instability modes | pressure_differential → cavitation; tank_drain_rate → depletion; nozzle_pump_divergence → line failure; supply_residual_trend → hydrant crash |
| Drop first N seconds per incident | Rolling features are incomplete during warm-up; including them would add noise |

### C. Training (Component C)

| Decision | Rationale |
|---|---|
| Per-incident temporal split (80/20) | Prevents future leakage that random splitting would cause for time-series data |
| PR-AUC as primary metric | Robust to class imbalance; ROC-AUC can be misleadingly high when negatives dominate |
| Threshold selected via PR curve (configurable: f1 / high_recall / high_prec) | Data-driven, not arbitrary 0.5. For firefighting, `high_recall` is recommended |
| sklearn Pipeline (imputer + model) | Single serializable artifact ensures identical preprocessing at train and serve time |
| XGBoost early stopping on aucpr | Prevents overfitting without manually tuning n_estimators |

### D. Inference (Component D)

| Decision | Rationale |
|---|---|
| Model loaded once at startup (lifespan handler) | Avoids per-request disk I/O; model stays in memory |
| SHAP TreeExplainer for per-instance explanations | Real feature attribution, not hardcoded lists |
| Severity-categorized insights (Urgent/Warning/Anomaly/Alert) | Maps SHAP output to actionable firefighting recommendations |
| Separate `agent_protocol` block | Machine-readable action/priority for downstream automation, decoupled from human reasoning |
| /livez and /readyz health probes | Standard K8s contract for container orchestration |

### Threshold Tradeoff (Firefighting Context)

In firefighting, a **missed instability event (false negative)** can result in loss of water supply during active operations — a life-safety risk. A **false alarm (false positive)** causes temporary throttle-down or increased monitoring — an operational inconvenience.

Therefore, we recommend `--threshold-strategy high_recall` for production deployment, which targets ≥90% recall at the cost of lower precision. The exact threshold value is persisted in `artifacts/metadata_latest.json` and automatically loaded by the inference service.

---

# 5. Sample Output (Agent Protocol)
The system outputs a decision block ready for the control loop:

```json
{
  "incident_id": "incident_000",
  "inference": {
    "risk_score": 0.1021,
    "is_unstable": false,
    "threshold_used": 0.9844
  },
  "reasoning": {
    "top_drivers": [
      {
        "feature": "flow_gpm_dup_nozzle_max_30s",
        "shap_value": 0.8048,
        "feature_value": null,
        "type": "shap"
      },
      {
        "feature": "intake_pressure_psi_dup_nozzle_max_60s",
        "shap_value": -0.7386,
        "feature_value": null,
        "type": "shap"
      },
      {
        "feature": "intake_pressure_psi_dup_nozzle_mean_60s",
        "shap_value": -0.7011,
        "feature_value": null,
        "type": "shap"
      },
      {
        "feature": "discharge_pressure_psi_dup_nozzle_min_30s",
        "shap_value": -0.6505,
        "feature_value": null,
        "type": "shap"
      },
      {
        "feature": "intake_pressure_psi_dup_nozzle_min_10s",
        "shap_value": -0.4575,
        "feature_value": null,
        "type": "shap"
      }
    ],
    "insight": "System operating within normal parameters. Continue current operations."
  },
  "agent_protocol": {
    "suggested_action": "MAINTAIN",
    "priority": "NORMAL",
    "escalation_required": "false"
  },
  "model_version": "20260210_132521",
  "inference_time_ms": 103.1
}

```

# 6. Project Structure

```
waterops/
├── __init__.py
├── ingest.py          # Component A — Ingest & canonicalize telemetry
├── features.py        # Component B — Windowing, joins, feature engineering
├── train.py           # Component C — Train, evaluate, and save models
└── serve.py           # Component D — FastAPI inference service

tests/
├── __init__.py
├── test_ingest.py     # Deduplication, schema validation, idempotency
├── test_features.py   # Time-bucketing, leakage prevention, domain features
├── test_train.py      # Temporal split, metric computation
└── test_serve.py      # Model load, /predict endpoint, response schema

examples/
└── sample_request.json

data_raw/              # Raw CSV files (provided, not committed)
data_curated/          # Output of Component A
data_training/         # Output of Component B
artifacts/             # Output of Component C (model + metadata)

requirements.txt
pyproject.toml
README.md
```

---

# 7. Results

# Model Comparison

| Model | PR-AUC | ROC-AUC | F1 | MCC | Threshold |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | — | — | — | — | — |
| XGBoost (improved) | — | — | — | — | — |

*Fill in after running `python -m waterops.train`.*

# Error Analysis

Key observations to document after training:
- **False negatives**: What patterns does the model miss? (e.g., slow-onset supply degradation)
- **False positives**: What triggers spurious alerts? (e.g., brief sensor noise in nozzle readings)
- **Feature importance**: Which features dominate? Is the model relying on expected physical signals?
- **Incident-level variance**: Does performance vary significantly across incidents?

---

# 8. Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test module
python -m pytest tests/test_ingest.py -v

# Run with coverage
python -m pytest tests/ --cov=waterops --cov-report=term-missing
```

---

# 9. Environment

- Python 3.10+
- All dependencies pinned in `requirements.txt`
- No proprietary datasets included — use only the provided data
