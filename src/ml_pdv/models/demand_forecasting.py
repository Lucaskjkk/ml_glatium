"""Modelos de previsão de demanda: baselines + ML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml_pdv.models.base import BaseForecastModel
from ml_pdv.training.evaluate import regression_metrics


class NaiveLastValueModel(BaseForecastModel):
    """Baseline: prevê a quantidade do dia atual (qty) como proxy do horizonte."""

    name = "naive_last_value"

    def __init__(self) -> None:
        self._fallback: float = 0.0

    def fit(self, X: Any, y: Any) -> NaiveLastValueModel:
        y_arr = np.asarray(y, dtype=float)
        self._fallback = float(np.nanmean(y_arr)) if len(y_arr) else 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X)
        if "qty" in frame.columns:
            pred = frame["qty"].astype(float).to_numpy()
        else:
            pred = np.full(len(frame), self._fallback, dtype=float)
        return pred

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        return regression_metrics(y, self.predict(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> NaiveLastValueModel:
        return joblib.load(path)


class MovingAverageBaseline(BaseForecastModel):
    """Baseline: usa rolling_mean_7 (já na feature matrix) como previsão."""

    name = "moving_average_7"

    def __init__(self) -> None:
        self._fallback: float = 0.0

    def fit(self, X: Any, y: Any) -> MovingAverageBaseline:
        y_arr = np.asarray(y, dtype=float)
        self._fallback = float(np.nanmean(y_arr)) if len(y_arr) else 0.0
        return self

    def predict(self, X: Any) -> np.ndarray:
        frame = pd.DataFrame(X)
        if "rolling_mean_7" in frame.columns:
            pred = frame["rolling_mean_7"].astype(float).to_numpy() * 7.0
        elif "sales_7d" in frame.columns:
            pred = frame["sales_7d"].astype(float).to_numpy()
        else:
            pred = np.full(len(frame), self._fallback, dtype=float)
        return np.clip(pred, a_min=0.0, a_max=None)

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        return regression_metrics(y, self.predict(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> MovingAverageBaseline:
        return joblib.load(path)


class RandomForestDemandModel(BaseForecastModel):
    name = "random_forest"

    def __init__(
        self,
        *,
        n_estimators: int = 200,
        max_depth: int | None = 12,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: Any, y: Any) -> RandomForestDemandModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(self.model.predict(X), a_min=0.0, a_max=None)

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        return regression_metrics(y, self.predict(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> RandomForestDemandModel:
        return joblib.load(path)


class XGBoostDemandModel(BaseForecastModel):
    name = "xgboost"

    def __init__(
        self,
        *,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        try:
            from xgboost import XGBRegressor
        except Exception as exc:  # noqa: BLE001 — missing libomp etc.
            raise RuntimeError(
                "XGBoost unavailable. On macOS run: brew install libomp"
            ) from exc
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: Any, y: Any) -> XGBoostDemandModel:
        self.model.fit(X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.clip(self.model.predict(X), a_min=0.0, a_max=None)

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        return regression_metrics(y, self.predict(X))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> XGBoostDemandModel:
        return joblib.load(path)


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def build_model_zoo(random_state: int = 42) -> dict[str, BaseForecastModel]:
    """Modelos padrão para comparar no pipeline de treino."""
    zoo: dict[str, BaseForecastModel] = {
        NaiveLastValueModel.name: NaiveLastValueModel(),
        MovingAverageBaseline.name: MovingAverageBaseline(),
        RandomForestDemandModel.name: RandomForestDemandModel(random_state=random_state),
    }
    if _xgboost_available():
        try:
            zoo[XGBoostDemandModel.name] = XGBoostDemandModel(random_state=random_state)
        except RuntimeError:
            pass
    return zoo
