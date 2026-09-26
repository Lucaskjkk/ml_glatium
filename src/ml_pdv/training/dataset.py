"""Dataset de demanda — grain, target e frame de treino.

Contrato
--------
grain:   (tenant_schema, product_id, feature_as_of)
X:       features com informação <= feature_as_of (sem leakage)
y:       soma de quantidade vendida em [feature_as_of+1 .. feature_as_of+7]

Isso se repete em outros modelos: sempre defina grain + target antes do fit.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

from ml_pdv.database.connection import get_erp_engine

HORIZON_DAYS = 7
TARGET_COL = "qty_next_7d"
DATE_COL = "feature_as_of"
ID_COLS = ("tenant_schema", "product_id", DATE_COL)

_TENANT_RE = re.compile(r"^tenant_[a-z0-9_]+$")


def _validate_tenant_schema(tenant_schema: str) -> str:
    """Evita SQL injection no nome do schema (identificador, não bind param)."""
    if not _TENANT_RE.match(tenant_schema):
        raise ValueError(f"Invalid tenant schema: {tenant_schema}")
    return tenant_schema


def list_tenant_schemas() -> list[str]:
    """Lista schemas de tenant ativos a partir de public.tenants."""
    sql = text(
        """
        SELECT schema_name
        FROM public.tenants
        WHERE is_active = true
          AND schema_name LIKE 'tenant_%'
        ORDER BY schema_name
        """
    )
    engine = get_erp_engine()
    with engine.connect() as conn:
        rows = conn.execute(sql).scalars().all()
    return [_validate_tenant_schema(str(r)) for r in rows]


def load_daily_sales(tenant_schema: str) -> pd.DataFrame:
    """
    Agrega vendas finalizadas por produto e dia.

    Colunas: tenant_schema, product_id, sale_date, qty
    """
    schema = _validate_tenant_schema(tenant_schema)
    sql = text(
        f"""
        SELECT
            :tenant AS tenant_schema,
            i.produto_id AS product_id,
            (v.data_venda AT TIME ZONE 'America/Sao_Paulo')::date AS sale_date,
            SUM(i.quantidade)::bigint AS qty
        FROM {schema}.vendas v
        JOIN {schema}.itens_venda i
          ON i.venda_id = v.id
        WHERE v.status = 'finalizada'
        GROUP BY 1, 2, 3
        ORDER BY 3, 2
        """
    )
    engine = get_erp_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"tenant": schema})

    if df.empty:
        return pd.DataFrame(columns=["tenant_schema", "product_id", "sale_date", "qty"])

    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.normalize()
    df["product_id"] = df["product_id"].astype("int64")
    df["qty"] = df["qty"].astype("int64")
    return df


def load_product_attrs(tenant_schema: str) -> pd.DataFrame:
    """Atributos de produto (preço/categoria) para enriquecer features."""
    schema = _validate_tenant_schema(tenant_schema)
    sql = text(
        f"""
        SELECT
            id AS product_id,
            preco_venda AS price,
            categoria_id AS category_id,
            quantidade_estoque AS stock
        FROM {schema}.produtos
        WHERE ativo = true
        """
    )
    engine = get_erp_engine()
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    if df.empty:
        return pd.DataFrame(columns=["product_id", "price", "category_id", "stock"])
    df["product_id"] = df["product_id"].astype("int64")
    return df


def densify_daily_sales(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche dias sem venda com qty=0 por produto.

    Necessário para shift/rolling por calendário (não por 'próxima venda').
    """
    if daily.empty:
        return daily.copy()

    frames: list[pd.DataFrame] = []
    for (tenant, product_id), group in daily.groupby(["tenant_schema", "product_id"], sort=False):
        g = group.sort_values("sale_date")
        start = g["sale_date"].min()
        end = g["sale_date"].max()
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


def add_target(daily_dense: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    """
    Cria y = soma de qty nos próximos `horizon_days` dias (D+1 .. D+H).

    Linhas sem horizonte completo à frente são removidas (evita target parcial).
    """
    if daily_dense.empty:
        return daily_dense.copy()

    df = daily_dense.sort_values(["tenant_schema", "product_id", "sale_date"]).copy()
    parts: list[pd.Series] = []
    for i in range(1, horizon_days + 1):
        parts.append(
            df.groupby(["tenant_schema", "product_id"], sort=False)["qty"].shift(-i)
        )
    df[TARGET_COL] = sum(parts)
    # Se qualquer dia futuro faltar no calendário densificado, shift devolve NaN.
    df = df.dropna(subset=[TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype("float64")
    df = df.rename(columns={"sale_date": DATE_COL})
    return df


def build_supervised_frame(
    tenant_schema: str,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Pipeline local do dataset (ainda sem features de engenharia):

    load → densify → target
    """
    daily = load_daily_sales(tenant_schema)
    dense = densify_daily_sales(daily)
    supervised = add_target(dense, horizon_days=horizon_days)
    products = load_product_attrs(tenant_schema)
    if not products.empty and not supervised.empty:
        supervised = supervised.merge(products, on="product_id", how="left")
    return supervised.reset_index(drop=True)


def build_multi_tenant_frame(
    tenant_schemas: list[str] | None = None,
    *,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Concatena frames de vários tenants (mesmo grain)."""
    schemas = tenant_schemas or list_tenant_schemas()
    frames = [build_supervised_frame(s, horizon_days=horizon_days) for s in schemas]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
