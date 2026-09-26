"""Prediction endpoints — no ML logic here."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ml_pdv.api.deps import verify_api_key
from ml_pdv.api.schemas import (
    DemandPredictionRequest,
    DemandPredictionResponse,
    ErrorResponse,
    ProductPredictionResponse,
)
from ml_pdv.inference import ModelNotReadyError, PredictionService, get_prediction_service
from ml_pdv.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["predictions"],
    dependencies=[Depends(verify_api_key)],
)


@router.post(
    "/demand",
    response_model=DemandPredictionResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
)
def predict_demand(
    body: DemandPredictionRequest,
    service: PredictionService = Depends(get_prediction_service),
) -> DemandPredictionResponse:
    try:
        result = service.predict_demand(body)
        logger.info(
            "prediction_demand",
            tenant_id=body.tenant_id,
            products=len(body.product_ids),
            model_status=result.model_status,
        )
        return result
    except ModelNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "detail": exc.message,
                "model_status": "not_ready",
                "code": "MODEL_NOT_READY",
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"detail": str(exc), "code": "BAD_REQUEST"},
        ) from exc


@router.get(
    "/products/{product_id}",
    response_model=ProductPredictionResponse,
)
def product_prediction(
    product_id: str,
    tenant_id: str = Query(..., min_length=1, description="Tenant / empresa id"),
    service: PredictionService = Depends(get_prediction_service),
) -> ProductPredictionResponse:
    return service.get_product_prediction(tenant_id=tenant_id, product_id=product_id)
