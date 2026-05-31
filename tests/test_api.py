"""
tests/test_api.py

Unit tests for the FastAPI prediction API.
Runs without GCP credentials (Vertex AI is mocked).
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Patch Vertex AI before importing the app
with patch("google.cloud.aiplatform.init"):
    with patch("google.cloud.aiplatform.Endpoint"):
        from api.main import app


@pytest.fixture
def client():
    """Test client with no live GCP connection (demo mode)."""
    with TestClient(app) as c:
        yield c


# ── Health check ─────────────────────────────────────────────────────────────
class TestHealthCheck:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_status(self, client):
        data = response = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "model_version" in data


# ── Input validation ──────────────────────────────────────────────────────────
class TestInputValidation:
    BASE_PAYLOAD = {
        "Age": 32,
        "MonthlyIncome": 4500,
        "YearsAtCompany": 4,
        "OverTime": "No",
        "Department": "Sales",
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
    }

    def test_valid_payload_returns_200(self, client):
        response = client.post("/predict", json=self.BASE_PAYLOAD)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client):
        data = client.post("/predict", json=self.BASE_PAYLOAD).json()
        assert "attrition_probability" in data
        assert "prediction" in data
        assert "threshold_used" in data
        assert "model_version" in data
        assert "latency_ms" in data

    def test_probability_in_range(self, client):
        data = client.post("/predict", json=self.BASE_PAYLOAD).json()
        assert 0.0 <= data["attrition_probability"] <= 1.0

    def test_prediction_is_valid_label(self, client):
        data = client.post("/predict", json=self.BASE_PAYLOAD).json()
        assert data["prediction"] in ("High Risk", "Low Risk")

    def test_age_below_minimum_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "Age": 15}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_age_above_maximum_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "Age": 80}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_overtime_value_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "OverTime": "Maybe"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_invalid_department_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "Department": "Accounting"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_job_satisfaction_out_of_range_rejected(self, client):
        payload = {**self.BASE_PAYLOAD, "JobSatisfaction": 5}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_missing_required_fields_rejected(self, client):
        response = client.post("/predict", json={"Age": 30})
        assert response.status_code == 422


# ── Business logic ────────────────────────────────────────────────────────────
class TestPredictionLogic:
    """Demo mode predictions (no Vertex AI) — validates risk classification logic."""

    HIGH_RISK = {
        "Age": 24,
        "MonthlyIncome": 1500,
        "YearsAtCompany": 1,
        "OverTime": "Yes",
        "Department": "Sales",
        "JobSatisfaction": 1,
        "WorkLifeBalance": 1,
    }

    LOW_RISK = {
        "Age": 45,
        "MonthlyIncome": 12000,
        "YearsAtCompany": 15,
        "OverTime": "No",
        "Department": "Research & Development",
        "JobSatisfaction": 4,
        "WorkLifeBalance": 4,
    }

    def test_high_risk_signals_produce_high_risk(self, client):
        data = client.post("/predict", json=self.HIGH_RISK).json()
        assert data["prediction"] == "High Risk"

    def test_low_risk_signals_produce_low_risk(self, client):
        data = client.post("/predict", json=self.LOW_RISK).json()
        assert data["prediction"] == "Low Risk"

    def test_high_risk_probability_above_threshold(self, client):
        data = client.post("/predict", json=self.HIGH_RISK).json()
        assert data["attrition_probability"] >= data["threshold_used"]

    def test_low_risk_probability_below_threshold(self, client):
        data = client.post("/predict", json=self.LOW_RISK).json()
        assert data["attrition_probability"] < data["threshold_used"]

    def test_latency_is_positive(self, client):
        data = client.post("/predict", json=self.HIGH_RISK).json()
        assert data["latency_ms"] > 0


# ── Batch endpoint ────────────────────────────────────────────────────────────
class TestBatchEndpoint:
    EMPLOYEE = {
        "Age": 32,
        "MonthlyIncome": 4500,
        "YearsAtCompany": 4,
        "OverTime": "No",
        "Department": "Sales",
        "JobSatisfaction": 3,
        "WorkLifeBalance": 3,
    }

    def test_batch_over_limit_rejected(self, client):
        batch = [self.EMPLOYEE] * 51
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 422

    def test_batch_without_endpoint_returns_503(self, client):
        # Without a configured endpoint, batch prediction should fail gracefully
        batch = [self.EMPLOYEE] * 3
        response = client.post("/predict/batch", json=batch)
        # In demo mode (no endpoint), should return 503
        assert response.status_code == 503
