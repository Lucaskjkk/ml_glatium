"""API smoke tests (no external DB required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ml_pdv.api.app import create_app


client = TestClient(create_app())


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_ready() -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_ready"
    assert body["status"] in {"ready", "degraded", "not_ready"}


def test_list_models() -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert body["model_status"] == "not_ready"


def test_production_model_not_ready() -> None:
    response = client.get("/api/v1/models/production")
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_ready"
    assert body["model"] is None


def test_predict_demand_not_ready() -> None:
    response = client.post(
        "/api/v1/predictions/demand",
        json={"tenant_id": "t1", "product_ids": ["p1"]},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["model_status"] == "not_ready"
    assert detail["code"] == "MODEL_NOT_READY"


def test_product_prediction_not_ready() -> None:
    response = client.get("/api/v1/predictions/products/p1", params={"tenant_id": "t1"})
    assert response.status_code == 200
    body = response.json()
    assert body["model_status"] == "not_ready"
    assert body["product_id"] == "p1"
