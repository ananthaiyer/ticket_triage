"""Tests for triage.api endpoints — no real models or API calls."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from triage.llm import LLMUnavailableError


def _llm_result():
    return {
        "category": "hardware",
        "urgency": "low",
        "reason": "Screen is a hardware component.",
        "suggested_reply": "We will arrange a repair shortly.",
    }


@pytest.fixture
def client():
    """TestClient with mocked model loading and Anthropic client."""
    with (
        patch("triage.baseline.load_or_train", return_value=(MagicMock(), MagicMock())),
        patch("anthropic.Anthropic", return_value=MagicMock()),
    ):
        from triage.api import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_returns_ok(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_no_auth_required(self, client, monkeypatch):
        monkeypatch.setenv("SERVICE_API_KEY", "secret")
        assert client.get("/health").status_code == 200


class TestAuth:
    def test_passes_without_service_key_set(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        with patch("triage.llm.classify", return_value=_llm_result()):
            resp = client.post("/triage", json={"subject": "x", "body": "y"})
        assert resp.status_code == 200

    def test_401_with_wrong_key(self, client, monkeypatch):
        monkeypatch.setenv("SERVICE_API_KEY", "correct")
        resp = client.post(
            "/triage",
            json={"subject": "x", "body": "y"},
            headers={"X-Api-Key": "wrong"},
        )
        assert resp.status_code == 401

    def test_401_with_missing_key(self, client, monkeypatch):
        monkeypatch.setenv("SERVICE_API_KEY", "correct")
        resp = client.post("/triage", json={"subject": "x", "body": "y"})
        assert resp.status_code == 401

    def test_200_with_correct_key(self, client, monkeypatch):
        monkeypatch.setenv("SERVICE_API_KEY", "correct")
        with patch("triage.llm.classify", return_value=_llm_result()):
            resp = client.post(
                "/triage",
                json={"subject": "broken screen", "body": "cracked"},
                headers={"X-Api-Key": "correct"},
            )
        assert resp.status_code == 200


class TestTriage:
    def test_required_fields_present(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        with patch("triage.llm.classify", return_value=_llm_result()):
            data = client.post("/triage", json={"subject": "x", "body": "y"}).json()
        for field in ("ticket_id", "category", "urgency", "reason", "suggested_reply", "source"):
            assert field in data

    def test_source_is_llm_on_success(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        with patch("triage.llm.classify", return_value=_llm_result()):
            data = client.post("/triage", json={"subject": "x", "body": "y"}).json()
        assert data["source"] == "llm"

    def test_ticket_id_echoed(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        with patch("triage.llm.classify", return_value=_llm_result()):
            data = client.post(
                "/triage",
                json={"ticket_id": "tid-42", "subject": "x", "body": "y"},
            ).json()
        assert data["ticket_id"] == "tid-42"

    def test_llm_category_in_response(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        result = {**_llm_result(), "category": "network"}
        with patch("triage.llm.classify", return_value=result):
            data = client.post("/triage", json={"subject": "x", "body": "y"}).json()
        assert data["category"] == "network"
        assert data["source"] == "llm"

    def test_baseline_fallback_on_llm_failure(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        with (
            patch("triage.llm.classify", side_effect=LLMUnavailableError("api down")),
            patch("triage.baseline.predict", return_value=[
                {"ticket_id": "", "category": "network", "urgency": "high"}
            ]),
        ):
            resp = client.post("/triage", json={"subject": "x", "body": "y"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "baseline"
        assert data["category"] == "network"
        assert data["urgency"] == "high"

    def test_503_when_both_classifiers_fail(self, client, monkeypatch):
        monkeypatch.delenv("SERVICE_API_KEY", raising=False)
        with (
            patch("triage.llm.classify", side_effect=LLMUnavailableError("api down")),
            patch("triage.baseline.predict", side_effect=Exception("models corrupt")),
        ):
            resp = client.post("/triage", json={"subject": "x", "body": "y"})
        assert resp.status_code == 503
