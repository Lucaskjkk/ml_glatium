"""Orquestra feature engineering para o problema de demanda."""

from __future__ import annotations

import pandas as pd

from ml_pdv.features.sales import build_feature_matrix
from ml_pdv.training.dataset import HORIZON_DAYS, build_multi_tenant_frame, build_supervised_frame


def build_demand_dataset(
    tenant_schema: str | None = None,
    tenant_schemas: list[str] | None = None,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Monta X, y, meta para um tenant ou vários.

    Use tenant_schema para um cliente; tenant_schemas / None para multi-tenant.
    """
    if tenant_schema is not None:
        supervised = build_supervised_frame(tenant_schema, horizon_days=horizon_days)
    else:
        supervised = build_multi_tenant_frame(tenant_schemas, horizon_days=horizon_days)
    return build_feature_matrix(supervised)
