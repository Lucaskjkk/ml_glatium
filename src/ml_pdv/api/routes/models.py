"""Model registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ml_pdv.api.deps import verify_api_key
from ml_pdv.api.schemas import ModelsListResponse, ProductionModelResponse
from ml_pdv.inference import PredictionService, get_prediction_service

router = APIRouter(
    prefix="/api/v1/models",
    tags=["models"],
    dependencies=[Depends(verify_api_key)],
)


@router.get("", response_model=ModelsListResponse)
def list_models(
    service: PredictionService = Depends(get_prediction_service),
) -> ModelsListResponse:
    return service.list_models()


@router.get("/production", response_model=ProductionModelResponse)
def production_model(
    service: PredictionService = Depends(get_prediction_service),
) -> ProductionModelResponse:
    return service.get_production_model()
