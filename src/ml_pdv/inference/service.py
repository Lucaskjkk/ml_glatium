"""Inference services — keep ML logic out of FastAPI routes."""

from __future__ import annotations

import re
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
from ml_pdv.inference.features import build_inference_features
from ml_pdv.models.local_registry import LocalArtifactRegistry, get_local_registry
from ml_pdv.utils.logging import get_logger

logger = get_logger(__name__)

_TENANT_RE = re.compile(r"^tenant_[a-z0-9_]+$")


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
        registry: LocalArtifactRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or get_local_registry()
        self._model: object | None = None
        self._model_version: str | None = None
        self._model_name: str | None = None

    def model_status(self) -> str:
        return "ready" if self.registry.is_ready() else "not_ready"

    def ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.registry.is_ready():
            raise ModelNotReadyError(
                f"No model artifact at {self.registry.artifact_path}. "
                "Run: uv run train-demand --tenant tenant_mrcoutinho"
            )
        self._model = self.registry.load_model()
        prod = self.registry.get_production(self.settings.model_name)
        self._model_version = prod.version if prod else "local"
        self._model_name = prod.name if prod else getattr(self._model, "name", "unknown")
        logger.info(
            "production_model_loaded",
            model=self._model_name,
            version=self._model_version,
            path=str(self.registry.artifact_path),
        )

    @staticmethod
    def _normalize_tenant(tenant_id: str) -> str:
        value = tenant_id.strip()
        if not value.startswith("tenant_"):
            value = f"tenant_{value}"
        if not _TENANT_RE.match(value):
            raise ValueError(f"Invalid tenant_id: {tenant_id}")
        return value

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
        return ModelsListResponse(
            models=models,
            model_status=self.model_status(),  # type: ignore[arg-type]
        )

    def get_production_model(self) -> ProductionModelResponse:
        prod = self.registry.get_production(self.settings.model_name)
        if prod is None:
            return ProductionModelResponse(
                model=None,
                model_status="not_ready",
                detail=f"No artifact at {self.registry.artifact_path}",
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
        self.ensure_model_loaded()
        assert self._model is not None

        tenant = self._normalize_tenant(request.tenant_id)
        as_of = request.as_of_date or date.today()
        horizon = request.horizon_days or self.settings.prediction_horizon_days
        horizon_start = as_of + timedelta(days=1)
        horizon_end = as_of + timedelta(days=horizon)

        try:
            product_ids = [int(p) for p in request.product_ids]
        except ValueError as exc:
            raise ValueError("product_ids must be integers (produto.id)") from exc

        X, meta = build_inference_features(tenant, product_ids, as_of)
        preds = self._model.predict(X)  # type: ignore[attr-defined]

        predictions = [
            DemandPredictionItem(
                product_id=str(int(meta.iloc[i]["product_id"])),
                predicted_quantity=float(preds[i]),
                horizon_start=horizon_start,
                horizon_end=horizon_end,
            )
            for i in range(len(meta))
        ]

        logger.info(
            "prediction_demand_done",
            tenant=tenant,
            as_of=str(as_of),
            n=len(predictions),
            model=self._model_name,
        )

        return DemandPredictionResponse(
            tenant_id=tenant,
            as_of_date=as_of,
            horizon_days=horizon,
            model_name=self._model_name or self.settings.model_name,
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
                detail=f"No artifact at {self.registry.artifact_path}",
            )
        req = DemandPredictionRequest(tenant_id=tenant_id, product_ids=[product_id])
        result = self.predict_demand(req)
        latest = result.predictions[0] if result.predictions else None
        return ProductPredictionResponse(
            tenant_id=result.tenant_id,
            product_id=product_id,
            model_status="ready",
            latest=latest,
            history=[],
            detail=None,
        )


_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def reset_prediction_service() -> None:
    global _prediction_service
    _prediction_service = None
