"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ml_pdv import __version__
from ml_pdv.api.schemas import HealthResponse, ReadyResponse
from ml_pdv.database.connection import check_erp_connectivity, check_ml_connectivity
from ml_pdv.inference import get_prediction_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness for orchestration. ERP is optional for serving predictions."""
    erp_ok: bool | None
    ml_ok: bool | None
    try:
        erp_ok = check_erp_connectivity()
    except Exception:
        erp_ok = False
    try:
        ml_ok = check_ml_connectivity()
    except Exception:
        ml_ok = False

    model_status = get_prediction_service().model_status()
    if model_status == "ready":
        status = "ready"
        detail = None
    else:
        status = "degraded"
        detail = "API is up but production model is not_ready."

    return ReadyResponse(
        status=status,
        erp_reachable=erp_ok,
        ml_db_reachable=ml_ok,
        model_status=model_status,  # type: ignore[arg-type]
        detail=detail,
    )
