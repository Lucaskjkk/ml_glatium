"""Feature engineering package."""

from ml_pdv.features.sales import FEATURE_COLUMNS, build_feature_matrix

__all__ = ["FEATURE_COLUMNS", "build_demand_dataset", "build_feature_matrix"]


def __getattr__(name: str):
    if name == "build_demand_dataset":
        from ml_pdv.features.pipeline import build_demand_dataset

        return build_demand_dataset
    raise AttributeError(name)
