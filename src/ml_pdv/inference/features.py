"""Build feature rows for online demand inference (no target / no leakage)."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from ml_pdv.features.sales import FEATURE_COLUMNS, add_calendar_features, add_lag_rolling_features
from ml_pdv.training.dataset import (
    DATE_COL,
    densify_daily_sales,
    load_daily_sales,
    load_product_attrs,
)


def _normalize_as_of(as_of: date | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(as_of).normalize()


def densify_through_as_of(daily: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Densify each product calendar from first sale through as_of (inclusive)."""
    if daily.empty:
        return daily.copy()

    as_of_ts = _normalize_as_of(as_of)
    frames: list[pd.DataFrame] = []
    for (tenant, product_id), group in daily.groupby(["tenant_schema", "product_id"], sort=False):
        g = group.sort_values("sale_date").copy()
        g["sale_date"] = pd.to_datetime(g["sale_date"]).dt.normalize()
        start = g["sale_date"].min()
        end = max(g["sale_date"].max(), as_of_ts)
        idx = pd.date_range(start, end, freq="D")
        dens = (
            g.set_index("sale_date")
            .reindex(idx)
            .assign(qty=lambda d: d["qty"].fillna(0).astype("int64"))
        )
        dens["tenant_schema"] = tenant
        dens["product_id"] = int(product_id)
        dens = dens.reset_index(names="sale_date")
        frames.append(dens[["tenant_schema", "product_id", "sale_date", "qty"]])

    out = pd.concat(frames, ignore_index=True)
    out["sale_date"] = pd.to_datetime(out["sale_date"]).dt.normalize()
    return out


def build_inference_features(
    tenant_schema: str,
    product_ids: list[int],
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (X, meta) for the requested products at feature_as_of = as_of.

    Products with no sales history get a zero row (cold start).
    """
    daily = load_daily_sales(tenant_schema)
    as_of_ts = _normalize_as_of(as_of)
    wanted = sorted({int(p) for p in product_ids})

    if daily.empty:
        dense = pd.DataFrame(
            {
                "tenant_schema": [tenant_schema] * len(wanted),
                "product_id": wanted,
                "sale_date": [as_of_ts] * len(wanted),
                "qty": [0] * len(wanted),
            }
        )
    else:
        # keep only requested products when possible; still densify history
        hist = daily[daily["product_id"].isin(wanted)].copy()
        missing = [p for p in wanted if p not in set(hist["product_id"].tolist())]
        if missing:
            extra = pd.DataFrame(
                {
                    "tenant_schema": tenant_schema,
                    "product_id": missing,
                    "sale_date": as_of_ts,
                    "qty": 0,
                }
            )
            hist = pd.concat([hist, extra], ignore_index=True)
        dense = densify_through_as_of(hist, as_of)

    frame = dense.rename(columns={"sale_date": DATE_COL})
    products = load_product_attrs(tenant_schema)
    if not products.empty:
        frame = frame.merge(products, on="product_id", how="left")

    feats = add_lag_rolling_features(frame)
    feats = add_calendar_features(feats)
    for col in ("price", "category_id", "stock"):
        if col not in feats.columns:
            feats[col] = 0.0
        feats[col] = feats[col].fillna(0.0)

    at_as_of = feats[pd.to_datetime(feats[DATE_COL]).dt.normalize() == as_of_ts].copy()
    # one row per product
    at_as_of = at_as_of.drop_duplicates(subset=["product_id"], keep="last")
    at_as_of = at_as_of.set_index("product_id").reindex(wanted).reset_index()
    at_as_of["tenant_schema"] = tenant_schema
    at_as_of[DATE_COL] = as_of_ts
    for col in FEATURE_COLUMNS:
        if col not in at_as_of.columns:
            at_as_of[col] = 0.0
        at_as_of[col] = at_as_of[col].fillna(0.0)

    meta = at_as_of[["tenant_schema", "product_id", DATE_COL]].copy()
    X = at_as_of[FEATURE_COLUMNS].astype("float64")
    return X, meta
