"""Training pipelines package."""

from ml_pdv.training.dataset import HORIZON_DAYS, TARGET_COL, load_daily_sales

__all__ = [
    "HORIZON_DAYS",
    "TARGET_COL",
    "load_daily_sales",
    "run_demand_training",
]


def __getattr__(name: str):
    if name == "run_demand_training":
        from ml_pdv.training.pipeline import run_demand_training

        return run_demand_training
    raise AttributeError(name)
