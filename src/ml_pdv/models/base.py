"""ML model package — base interfaces and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseForecastModel(ABC):
    """Interface for demand / sales forecasting models."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "BaseForecastModel":
        ...

    @abstractmethod
    def predict(self, X: Any) -> Any:
        ...

    @abstractmethod
    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseForecastModel":
        ...
