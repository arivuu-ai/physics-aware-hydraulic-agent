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
        assert "tank" in result["insigh
