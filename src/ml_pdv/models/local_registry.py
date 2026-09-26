"""Local artifact model registry (file-based until MLflow)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from ml_pdv.config import get_settings
from ml_pdv.models.registry import RegisteredModel


class LocalArtifactRegistry:
    """Reads production candidate from disk (`MODEL_ARTIFACT_PATH`)."""

    def __init__(self, artifact_path: Path | None = None) -> None:
        settings = get_settings()
        self.artifact_path = Path(
            artifact_path or settings.model_artifact_path
        ).expanduser().resolve()
        self._model: object | None = None
        self._meta: RegisteredModel | None = None

    def _meta_path(self) -> Path:
        return self.artifact_path.with_suffix(self.artifact_path.suffix + ".meta.json")

    def is_ready(self, model_name: str | None = None) -> bool:
        return self.artifact_path.is_file()

    def list_models(self) -> list[RegisteredModel]:
        prod = self.get_production(get_settings().model_name)
        return [prod] if prod else []

    def get_production(self, model_name: str) -> RegisteredModel | None:
        if not self.is_ready():
            return None
        if self._meta is not None:
            return self._meta

        meta_file = self._meta_path()
        payload: dict = {}
        if meta_file.is_file():
            payload = json.loads(meta_file.read_text(encoding="utf-8"))

        registered_at = None
        if "registered_at" in payload:
            try:
                registered_at = datetime.fromisoformat(payload["registered_at"])
            except ValueError:
                registered_at = None
        if registered_at is None:
            registered_at = datetime.fromtimestamp(
                self.artifact_path.stat().st_mtime, tz=timezone.utc
            )

        self._meta = RegisteredModel(
            name=str(payload.get("model_name") or payload.get("best_model") or model_name),
            version=str(payload.get("version") or payload.get("artifacts_dir") or "local"),
            stage="production",
            framework=str(payload.get("framework") or "sklearn/xgboost"),
            metrics={
                k: float(v)
                for k, v in (payload.get("metrics") or {}).items()
                if isinstance(v, (int, float))
            },
            registered_at=registered_at,
        )
        return self._meta

    def load_model(self) -> object:
        if not self.is_ready():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")
        if self._model is None:
            self._model = joblib.load(self.artifact_path)
        return self._model


_local_registry: LocalArtifactRegistry | None = None


def get_local_registry() -> LocalArtifactRegistry:
    global _local_registry
    if _local_registry is None:
        _local_registry = LocalArtifactRegistry()
    return _local_registry


def reset_local_registry() -> None:
    global _local_registry
    _local_registry = None
