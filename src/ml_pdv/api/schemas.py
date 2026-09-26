"""API request/response schemas."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ReadyResponse(BaseModel):
    status: Literal["ready", "degraded", "not_ready"]
    erp_reachable: bool | None = None
    ml_db_reachable: bool | None = None
    model_status: Literal["ready", "not_ready"] = "not_ready"
    detail: str | None = None


class ModelInfo(BaseModel):
    name: str
    version: str | None = None
    stage: Literal["production", "staging", "candidate", "none"] = "none"
    framework: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    registered_at: datetime | None = None


class ModelsListResponse(BaseModel):
    models: list[ModelInfo]
    model_status: Literal["ready", "not_ready"] = "not_ready"


class ProductionModelResponse(BaseModel):
    model: ModelInfo | None
    model_status: Literal["ready", "not_ready"] = "not_ready"
    detail: str | None = None


class DemandPredictionRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1, description="Tenant / empresa identifier")
    product_ids: list[str] = Field(..., min_length=1)
    as_of_date: date | None = Field(
        default=None,
        description="Feature cutoff date (D). Defaults to today (UTC).",
    )
    horizon_days: int | None = Field(
        default=None,
        ge=1,
        le=90,
        description="Prediction window length. Defaults to settings.",
    )


class DemandPredictionItem(BaseModel):
    product_id: str
    predicted_quantity: float
    horizon_start: date
    horizon_end: date
    unit: str = "units"


class DemandPredictionResponse(BaseModel):
    tenant_id: str
    as_of_date: date
    horizon_days: int
    model_name: str
    model_version: str | None
    model_status: Literal["ready", "not_ready"]
    predictions: list[DemandPredictionItem] = Field(default_factory=list)
    detail: str | None = None


class ProductPredictionResponse(BaseModel):
    tenant_id: str
    product_id: str
    model_status: Literal["ready", "not_ready"]
    latest: DemandPredictionItem | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)
    detail: str | None = None


class ErrorResponse(BaseModel):
    detail: str
    model_status: Literal["ready", "not_ready"] | None = None
    code: str | None = None
