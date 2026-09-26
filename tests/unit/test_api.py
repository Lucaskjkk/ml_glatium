"""API smoke tests (no external DB required for not_ready paths)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ml_pdv.api.app import create_app
from ml_pdv.config.settings import get_settings
from ml_pdv.inference.service import reset_prediction_service
from ml_pdv.models.local_registry import reset_local_registry


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Force not_ready in unit tests (ignore real artifacts on disk).
    missing = tmp_path / "missing.joblib"
    monkeypatch.setenv("MODEL_ARTIFACT_PATH", str(missing))
    get_settings.cache_clear()
    reset_local_registry()
    reset_prediction_service()
    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
    reset_local_registry()
    reset_prediction_service()


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_ready"
    assert body["status"] in {"ready", "degraded", "not_ready"}


def test_list_models(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert body["model_status"] == "not_ready"


def test_production_model_not_ready(client: TestClient) -> None:
    response = client.get("/api/v1/models/production")
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_ready"
    assert body["model"] is None


def test_predict_demand_not_ready(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predictions/demand",
        json={"tenant_id": "t1", "product_ids": ["p1"]},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["model_status"] == "not_ready"
    assert detail["code"] == "MODEL_NOT_READY"


def test_product_prediction_not_ready(client: TestClient) -> None:
    response = client.get("/api/v1/predictions/products/p1", params={"tenant_id": "t1"})
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_ready"
    assert body["product_id"] == "p1"
