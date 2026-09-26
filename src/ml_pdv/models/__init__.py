"""Models package."""

from ml_pdv.models.base import BaseForecastModel
from ml_pdv.models.registry import ModelRegistry, get_model_registry

__all__ = [
    "BaseForecastModel",
    "ModelRegistry",
    "get_model_registry",
    "build_model_zoo",
]


def __getattr__(name: str):
    if name in {
        "MovingAverageBaseline",
        "NaiveLastValueModel",
        "RandomForestDemandModel",
        "XGBoostDemandModel",
        "build_model_zoo",
    }:
        from ml_pdv.models import demand_forecasting as df

        return getattr(df, name)
    raise AttributeError(name)
