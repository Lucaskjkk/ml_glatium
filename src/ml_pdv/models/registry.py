"""Model registry facade (stub until MLflow Model Registry is wired)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RegisteredModel:
    name: str
    version: str | None = None
    stage: str = "none"
    framework: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    registered_at: datetime | None = None


class ModelRegistry:
    """In-memory / future MLflow-backed registry."""

    def list_models(self) -> list[RegisteredModel]:
        return []

    def get_production(self, model_name: str) -> RegisteredModel | None:
        return None

    def is_ready(self, model_name: str) -> bool:
        return self.get_production(model_name) is not None


_registry: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
