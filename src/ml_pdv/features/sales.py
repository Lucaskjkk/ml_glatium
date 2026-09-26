"""Feature engineering de demanda — sem data leakage.

Regra: na data feature_as_of = D, só usar informação com event_time <= D.
Rolling usa a série de qty já conhecida até D (calendário densificado).
"""

from __future__ import annotations

import pandas as pd

from ml_pdv.training.dataset import DATE_COL, TARGET_COL

FEATURE_COLUMNS: list[str] = [
    "qty",
    "sales_7d",
    "sales_14d",
    "sales_30d",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_30",
    "day_of_week",
    "week_of_year",
    "month",
    "price",
    "category_id",
    "stock",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[DATE_COL])
    out["day_of_week"] = ts.dt.dayofweek.astype("int64")
    out["week_of_year"] = ts.dt.isocalendar().week.astype("int64")
    out["month"] = ts.dt.month.astype("int64")
    return out


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Janelas rolling por (tenant, product), ordenadas por data.

    Como o calendário está densificado, rolling(7) = últimos 7 dias de calendário.
    """
    out = df.sort_values(["tenant_schema", "product_id", DATE_COL]).copy()
    group = out.groupby(["tenant_schema", "product_id"], sort=False)["qty"]

    out["sales_7d"] = group.transform(lambda s: s.rolling(7, min_periods=1).sum())
    out["sales_14d"] = group.transform(lambda s: s.rolling(14, min_periods=1).sum())
    out["sales_30d"] = group.transform(lambda s: s.rolling(30, min_periods=1).sum())
    out["rolling_mean_7"] = group.transform(lambda s: s.rolling(7, min_periods=1).mean())
    out["rolling_mean_30"] = group.transform(lambda s: s.rolling(30, min_periods=1).mean())
    out["rolling_std_30"] = group.transform(
        lambda s: s.rolling(30, min_periods=2).std().fillna(0.0)
    )
    return out


def build_feature_matrix(supervised: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Retorna (X, y, meta).

    meta guarda ids para auditoria (tenant/product/date) sem entrar no modelo.
    """
    if supervised.empty:
        empty_x = pd.DataFrame(columns=FEATURE_COLUMNS)
        empty_y = pd.Series(dtype="float64", name=TARGET_COL)
        empty_meta = pd.DataFrame(columns=["tenant_schema", "product_id", DATE_COL])
        return empty_x, empty_y, empty_meta

    feats = add_lag_rolling_features(supervised)
    feats = add_calendar_features(feats)

    for col in ("price", "category_id", "stock"):
        if col not in feats.columns:
            feats[col] = 0.0
        feats[col] = feats[col].fillna(0.0)

    meta = feats[["tenant_schema", "product_id", DATE_COL]].copy()
    y = feats[TARGET_COL].astype("float64")
    X = feats[FEATURE_COLUMNS].astype("float64")
    return X, y, meta
