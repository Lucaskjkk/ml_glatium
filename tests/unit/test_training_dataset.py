"""Unit tests for demand dataset / target / metrics (no DB required for pure logic)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from ml_pdv.features.sales import build_feature_matrix
from ml_pdv.training.dataset import TARGET_COL, add_target, densify_daily_sales
from ml_pdv.training.evaluate import regression_metrics
from ml_pdv.training.split import temporal_train_valid_test_split


def _toy_daily() -> pd.DataFrame:
    rows = []
    start = datetime(2026, 1, 1)
    for product_id in (1, 2):
        for day in range(40):
            rows.append(
                {
                    "tenant_schema": "tenant_demo",
                    "product_id": product_id,
                    "sale_date": start + pd.Timedelta(days=day),
                    "qty": 1 if day % 2 == 0 else 0,
                }
            )
    return pd.DataFrame(rows)


def test_densify_and_target_no_leakage_window() -> None:
    dense = densify_daily_sales(_toy_daily())
    supervised = add_target(dense, horizon_days=7)
    assert TARGET_COL in supervised.columns
    # Últimos 7 dias de calendário por produto não podem ter target completo
    assert supervised["feature_as_of"].max() < dense["sale_date"].max()


def test_feature_matrix_shapes() -> None:
    dense = densify_daily_sales(_toy_daily())
    supervised = add_target(dense, horizon_days=7)
    X, y, meta = build_feature_matrix(supervised)
    assert len(X) == len(y) == len(meta)
    assert y.name == TARGET_COL
    assert "sales_7d" in X.columns


def test_temporal_split_is_ordered() -> None:
    dense = densify_daily_sales(_toy_daily())
    supervised = add_target(dense, horizon_days=7)
    X, y, meta = build_feature_matrix(supervised)
    split = temporal_train_valid_test_split(X, y, meta)
    assert split.meta_train["feature_as_of"].max() <= split.meta_valid["feature_as_of"].min()
    assert split.meta_valid["feature_as_of"].max() <= split.meta_test["feature_as_of"].min()


def test_metrics() -> None:
    m = regression_metrics([1, 2, 3], [1, 2, 2])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] > 0
