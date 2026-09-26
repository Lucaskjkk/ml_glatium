"""Inference services — keep ML logic out of FastAPI routes."""

from __future__ import annotations

from datetime import date, timedelta

from ml_pdv.api.schemas import (
    DemandPredictionItem,
    DemandPredictionRequest,
    DemandPredictionResponse,
    ModelInfo,
    ModelsListResponse,
    ProductionModelResponse,
    ProductPredictionResponse,
)
from ml_pdv.config import Settings, get_settings
from ml_pdv.models.registry import ModelRegistry, get_model_registry
from ml_pdv.utils.logging import get_logger

logger = get_logger(__name__)


class ModelNotReadyError(Exception):
    """Raised when no production model is available for inference."""

    def __init__(self, message: str = "Production model is not ready") -> None:
        super().__init__(message)
        self.message = message


class PredictionService:
    """Loads production model once (when available) and serves predictions."""

    def __init__(
        self,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_model_registry()
        self._model: object | None = None
        self._model_version: str | None = None

    def model_status(self) -> str:
        return "ready" if self.registry.is_ready(self.settings.model_name) else "not_ready"

    def ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        prod = self.registry.get_production(self.settings.model_name)
        if prod is None:
            raise ModelNotReadyError(
                "No production model registered yet. Train and promote a model first."
            )
        # Future: load artifact from MLflow / disk and keep in memory.
        self._model = object()
        self._model_version = prod.version
        logger.info(
            "production_model_loaded",
            model=self.settings.model_name,
            version=self._model_version,
        )

    def list_models(self) -> ModelsListResponse:
        models = [
            ModelInfo(
                name=m.name,
                version=m.version,
                stage=m.stage,  # type: ignore[arg-type]
                framework=m.framework,
                metrics=m.metrics,
                registered_at=m.registered_at,
            )
            for m in self.registry.list_models()
        ]
        return ModelsListResponse(models=models, model_status=self.model_status())  # type: ignore[arg-type]

    def get_production_model(self) -> ProductionModelResponse:
        prod = self.registry.get_production(self.settings.model_name)
        if prod is None:
            return ProductionModelResponse(
                model=None,
                model_status="not_ready",
                detail="No production model registered.",
            )
        return ProductionModelResponse(
            model=ModelInfo(
                name=prod.name,
                version=prod.version,
                stage=prod.stage,  # type: ignore[arg-type]
                framework=prod.framework,
                metrics=prod.metrics,
                registered_at=prod.registered_at,
            ),
            model_status="ready",
        )

    def predict_demand(self, request: DemandPredictionRequest) -> DemandPredictionResponse:
        if self.model_status() == "not_ready":
            raise ModelNotReadyError(
                "Demand model not ready (model_status=not_ready). "
                "Endpoints and schemas are stable for ERP integration."
            )

        self.ensure_model_loaded()
        as_of = request.as_of_date or date.today()
        horizon = request.horizon_days or self.settings.prediction_horizon_days
        horizon_start = as_of + timedelta(days=1)
        horizon_end = as_of + timedelta(days=horizon)

        # Placeholder path once a model exists — real feature lookup + predict later.
        predictions = [
            DemandPredictionItem(
                product_id=pid,
                predicted_quantity=0.0,
                horizon_start=horizon_start,
                horizon_end=horizon_end,
            )
            for pid in request.product_ids
        ]

        return DemandPredictionResponse(
            tenant_id=request.tenant_id,
            as_of_date=as_of,
            horizon_days=horizon,
            model_name=self.settings.model_name,
            model_version=self._model_version,
            model_status="ready",
            predictions=predictions,
        )

    def get_product_prediction(
        self,
        tenant_id: str,
        product_id: str,
    ) -> ProductPredictionResponse:
        if self.model_status() == "not_ready":
            return ProductPredictionResponse(
                tenant_id=tenant_id,
                product_id=product_id,
                model_status="not_ready",
                detail="No production model / stored predictions yet.",
            )
        return ProductPredictionResponse(
            tenant_id=tenant_id,
            product_id=product_id,
            model_status="ready",
            latest=None,
            history=[],
            detail="Prediction history storage not implemented yet.",
        )


_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service
