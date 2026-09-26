"""Treino de modelos — fit + evaluate, sem I/O de API."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml_pdv.models.base import BaseForecastModel
from ml_pdv.models.demand_forecasting import build_model_zoo
from ml_pdv.training.evaluate import regression_metrics
from ml_pdv.training.split import TemporalSplit


@dataclass
class ModelResult:
    name: str
    model: BaseForecastModel
    metrics_valid: dict[str, float]
    metrics_test: dict[str, float]


def train_one(
    model: BaseForecastModel,
    split: TemporalSplit,
) -> ModelResult:
    model.fit(split.X_train, split.y_train)
    pred_valid = model.predict(split.X_valid)
    pred_test = model.predict(split.X_test)
    return ModelResult(
        name=model.name,
        model=model,
        metrics_valid=regression_metrics(split.y_valid, pred_valid),
        metrics_test=regression_metrics(split.y_test, pred_test),
    )


def train_model_zoo(
    split: TemporalSplit,
    *,
    random_state: int = 42,
    model_names: list[str] | None = None,
) -> list[ModelResult]:
    zoo = build_model_zoo(random_state=random_state)
    if model_names:
        missing = set(model_names) - set(zoo)
        if missing:
            raise KeyError(f"Unknown models: {sorted(missing)}")
        zoo = {k: zoo[k] for k in model_names}
    return [train_one(model, split) for model in zoo.values()]


def pick_best_by_mae(results: list[ModelResult], *, on: str = "valid") -> ModelResult:
    if not results:
        raise ValueError("No model results to compare")
    key = "metrics_valid" if on == "valid" else "metrics_test"

    def mae_of(r: ModelResult) -> float:
        return float(getattr(r, key)["mae"])

    return min(results, key=mae_of)


def metrics_table(results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "model": r.name,
                "valid_mae": r.metrics_valid["mae"],
                "valid_rmse": r.metrics_valid["rmse"],
                "valid_wape": r.metrics_valid["wape"],
                "test_mae": r.metrics_test["mae"],
                "test_rmse": r.metrics_test["rmse"],
                "test_wape": r.metrics_test["wape"],
            }
        )
    return pd.DataFrame(rows).sort_values("valid_mae").reset_index(drop=True)
