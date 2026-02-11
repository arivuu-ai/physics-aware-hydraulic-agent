"""
tests/test_serve.py — Model load, /predict endpoint, response schema
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

# We import specific helpers from serve to test in isolation
# before testing the full endpoint
import serve as serve_module
from serve import (
    ModelState,
    build_agent_protocol,
    build_reasoning,
    construct_features,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

class TestBuildAgentProtocol:
    def test_below_threshold_maintain(self):
        result = build_agent_protocol(risk_score=0.3, threshold=0.5)
        assert result["suggested_action"] == "MAINTAIN"
        assert result["priority"] == "NORMAL"

    def test_high_risk_emergency(self):
        result = build_agent_protocol(risk_score=0.85, threshold=0.5)
        assert result["suggested_action"] == "EMERGENCY_RESPONSE"
        assert result["priority"] == "CRITICAL"
        assert result["escalation_required"] == "true"

    def test_medium_risk_reduce_throttle(self):
        result = build_agent_protocol(risk_score=0.65, threshold=0.5)
        assert result["suggested_action"] == "REDUCE_THROTTLE"
        assert result["priority"] == "HIGH"


class TestBuildReasoning:
    def test_stable_insight_below_threshold(self):
        result = build_reasoning(
            risk_score=0.2,
            threshold=0.5,
            top_features=[{"feature": "intake_pressure_psi", "shap_value": 0.01}],
        )
        assert "normal parameters" in result["insight"].lower()

    def test_unstable_returns_domain_insight(self):
        result = build_reasoning(
            risk_score=0.9,
            threshold=0.5,
            top_features=[{"feature": "tank_level_roc", "shap_value": 0.5}],
        )
        assert "tank" in result["insight"].lower() or "Urgent" in result["insight"]


class TestConstructFeatures:
    def test_returns_correct_shape(self):
        from serve import TelemetryReading
        readings = [
            TelemetryReading(
                ts_ms=1700000000000,
                stream_type="pump",
                payload={"intake_pressure_psi": 65.0, "discharge_pressure_psi": 120.0},
            ),
        ]
        feature_columns = [
            "pump_intake_pressure_psi",
            "pump_discharge_pressure_psi",
            "pump_intake_pressure_psi_mean_10s",
        ]
        result = construct_features(readings, feature_columns)
        assert result.shape == (1, 3)

    def test_missing_features_filled_with_nan(self):
        from serve import TelemetryReading
        readings = [
            TelemetryReading(
                ts_ms=1700000000000,
                stream_type="pump",
                payload={"intake_pressure_psi": 65.0},
            ),
        ]
        feature_columns = ["pump_intake_pressure_psi", "nonexistent_feature"]
        result = construct_features(readings, feature_columns)
        assert result.shape == (1, 2)
        assert np.isnan(result[0, 1])


class TestModelState:
    def test_initial_state_not_ready(self):
        state = ModelState()
        assert state.ready is False
        assert state.pipeline is None
        assert state.threshold == 0.5

    def test_load_missing_files_raises(self, tmp_path):
        state = ModelState()
        with pytest.raises(FileNotFoundError):
            state.load(str(tmp_path))


# ---------------------------------------------------------------------------
# Integration test for /predict and health endpoints
# ---------------------------------------------------------------------------

class TestEndpoints:
    @pytest.fixture
    def client(self):
        """Create a test client with a mocked model state."""
        # Mock the model so we don't need real artifacts
        serve_module.model_state.ready = True
        serve_module.model_state.threshold = 0.5
        serve_module.model_state.metadata = {"version": "test_v1"}
        serve_module.model_state.feature_columns = [
            "pump_intake_pressure_psi",
            "pump_discharge_pressure_psi",
        ]

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.predict_proba.return_value = np.array([[0.7, 0.3]])
        mock_pipeline.named_steps = {"model": MagicMock(), "imputer": None}
        serve_module.model_state.pipeline = mock_pipeline
        serve_module.model_state.explainer = None

        from serve import app
        return TestClient(app)

    def test_liveness(self, client):
        resp = client.get("/livez")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"
        assert data["model_loaded"] is True

    def test_readiness(self, client):
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_predict_returns_valid_response(self, client):
        payload = {
            "incident_id": "incident_test",
            "telemetry": [
                {
                    "ts_ms": 1700000000000,
                    "stream_type": "pump",
                    "payload": {
                        "intake_pressure_psi": 65.0,
                        "discharge_pressure_psi": 120.0,
                    },
                }
            ],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "inference" in data
        assert "reasoning" in data
        assert "agent_protocol" in data
        assert 0.0 <= data["inference"]["risk_score"] <= 1.0
        assert isinstance(data["inference"]["is_unstable"], bool)
        assert data["model_version"] == "test_v1"

    def test_predict_empty_telemetry_rejected(self, client):
        payload = {
            "incident_id": "incident_test",
            "telemetry": [],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_missing_incident_id_rejected(self, client):
        payload = {
            "telemetry": [
                {
                    "ts_ms": 1700000000000,
                    "stream_type": "pump",
                    "payload": {"intake_pressure_psi": 65.0},
                }
            ],
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_response_schema_fields(self, client):
        payload = {
            "incident_id": "incident_test",
            "telemetry": [
                {
                    "ts_ms": 1700000000000,
                    "stream_type": "pump",
                    "payload": {"intake_pressure_psi": 65.0},
                }
            ],
        }
        resp = client.post("/predict", json=payload)
        data = resp.json()
        # Check nested schema
        assert "risk_score" in data["inference"]
        assert "is_unstable" in data["inference"]
        assert "threshold_used" in data["inference"]
        assert "top_drivers" in data["reasoning"]
        assert "insight" in data["reasoning"]
        assert "suggested_action" in data["agent_protocol"]
        assert "priority" in data["agent_protocol"]
        assert "escalation_required" in data["agent_protocol"]
        assert "inference_time_ms" in data
            risk_score=0.9,
            threshold=0.5,
            top_features=[{"feature": "tank_level_roc", "shap_value": 0.5}],
        )
        assert "tank" in result["insigh
