"""ERP schema introspection (read-only)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ml_pdv.config import get_settings
from ml_pdv.database.connection import get_erp_engine
from ml_pdv.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Heuristic keywords to classify domain tables (matched against table names).
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sales": (
        "venda",
        "vendas",
        "pedido",
        "pedidos",
        "cupom",
        "nota",
        "nfe",
        "nfce",
        "sale",
        "order",
        "fatura",
    ),
    "sale_items": (
        "item",
        "itens",
        "produto_venda",
        "pedido_item",
        "venda_item",
        "itens_venda",
        "order_item",
        "sale_item",
    ),
    "products": (
        "produto",
        "produtos",
        "product",
        "sku",
        "mercadoria",
        "item_estoque",
    ),
    "customers": (
        "cliente",
        "clientes",
        "customer",
        "pessoa",
        "pessoas",
    ),
    "stock": (
        "estoque",
        "stock",
        "inventory",
        "saldo",
        "movimento_estoque",
        "movimentacao",
    ),
    "tenant": (
        "empresa",
        "empresas",
        "tenant",
        "filial",
        "filiais",
        "loja",
        "lojas",
        "organization",
    ),
}

WATERMARK_CANDIDATES = (
    "updated_at",
    "updatedat",
    "dt_atualizacao",
    "data_atualizacao",
    "dh_atualizacao",
    "alterado_em",
    "modified_at",
    "modification_date",
    "created_at",
    "createdat",
    "dt_criacao",
    "data_criacao",
    "dh_criacao",
    "data_venda",
    "data_pedido",
    "data_emissao",
)

TENANT_CANDIDATES = (
    "tenant_id",
    "empresa_id",
    "id_empresa",
    "empresa",
    "filial_id",
    "id_filial",
    "loja_id",
    "id_loja",
    "organization_id",
    "org_id",
    "client_id",  # SaaS client — not shopper
)


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    is_nullable: bool
    ordinal_position: int


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[dict[str, Any]] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    watermark_columns: list[str] = field(default_factory=list)
    tenant_columns: list[str] = field(default_factory=list)
    estimated_rows: int | None = None


def _classify_domains(table_name: str) -> list[str]:
    lower = table_name.lower()
    found: list[str] = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in lower for k in keywords):
            found.append(domain)
    return found


def _match_columns(columns: list[ColumnInfo], candidates: tuple[str, ...]) -> list[str]:
    names = {c.name.lower(): c.name for c in columns}
    matched: list[str] = []
    for cand in candidates:
        if cand.lower() in names:
            matched.append(names[cand.lower()])
    return matched


def introspect_erp() -> list[TableInfo]:
    engine = get_erp_engine()
    tables: list[TableInfo] = []

    with engine.connect() as conn:
        table_rows = conn.execute(
            text(
                """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                  AND table_schema NOT IN ('pg_catalog', 'information_schema')
                ORDER BY table_schema, table_name
                """
            )
        ).mappings().all()

        for row in table_rows:
            schema = row["table_schema"]
            name = row["table_name"]

            cols = conn.execute(
                text(
                    """
                    SELECT column_name, data_type, is_nullable, ordinal_position
                    FROM information_schema.columns
                    WHERE table_schema = :schema AND table_name = :name
                    ORDER BY ordinal_position
                    """
                ),
                {"schema": schema, "name": name},
            ).mappings().all()

            columns = [
                ColumnInfo(
                    name=c["column_name"],
                    data_type=c["data_type"],
                    is_nullable=c["is_nullable"] == "YES",
                    ordinal_position=c["ordinal_position"],
                )
                for c in cols
            ]

            pks = conn.execute(
                text(
                    """
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                      AND tc.table_schema = :schema
                      AND tc.table_name = :name
                    ORDER BY kcu.ordinal_position
                    """
                ),
                {"schema": schema, "name": name},
            ).scalars().all()

            fks = conn.execute(
                text(
                    """
                    SELECT
                        kcu.column_name AS column_name,
                        ccu.table_schema AS foreign_table_schema,
                        ccu.table_name AS foreign_table_name,
                        ccu.column_name AS foreign_column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                     AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage ccu
                      ON ccu.constraint_name = tc.constraint_name
                     AND ccu.table_schema = tc.table_schema
                    WHERE tc.constraint_type = 'FOREIGN KEY'
                      AND tc.table_schema = :schema
                      AND tc.table_name = :name
                    """
                ),
                {"schema": schema, "name": name},
            ).mappings().all()

            est = conn.execute(
                text(
                    """
                    SELECT COALESCE(c.reltuples, 0)::bigint AS estimated_rows
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = :schema AND c.relname = :name
                    """
                ),
                {"schema": schema, "name": name},
            ).scalar()

            info = TableInfo(
                schema=schema,
                name=name,
                columns=columns,
                primary_keys=list(pks),
                foreign_keys=[dict(fk) for fk in fks],
                domains=_classify_domains(name),
                watermark_columns=_match_columns(columns, WATERMARK_CANDIDATES),
                tenant_columns=_match_columns(columns, TENANT_CANDIDATES),
                estimated_rows=int(est) if est is not None else None,
            )
            tables.append(info)

    return tables


def tables_to_markdown(tables: list[TableInfo]) -> str:
    lines: list[str] = [
        "# ERP Schema Map",
        "",
        "Generated by `introspect-erp` against the configured ERP source "
        "(`ERP_SOURCE_MODE=dump` → local DB restored from dump; "
        "`live` → production READ ONLY).",
        "Do not invent tables — this document reflects the live catalog.",
        "",
        f"**Tables discovered:** {len(tables)}",
        "",
    ]

    by_domain: dict[str, list[TableInfo]] = {}
    unclassified: list[TableInfo] = []
    for t in tables:
        if not t.domains:
            unclassified.append(t)
            continue
        for d in t.domains:
            by_domain.setdefault(d, []).append(t)

    priority = ["tenant", "sales", "sale_items", "products", "customers", "stock"]
    lines.append("## Domain candidates (heuristic by table name)")
    lines.append("")
    for domain in priority:
        items = by_domain.get(domain, [])
        lines.append(f"### {domain}")
        lines.append("")
        if not items:
            lines.append("_No tables matched this domain by name heuristics._")
            lines.append("")
            continue
        for t in items:
            fq = f"{t.schema}.{t.name}"
            lines.append(f"- `{fq}` — ~{t.estimated_rows} rows")
            lines.append(f"  - PK: `{', '.join(t.primary_keys) or 'n/a'}`")
            lines.append(
                f"  - Watermark candidates: "
                f"`{', '.join(t.watermark_columns) or 'none'}`"
            )
            lines.append(
                f"  - Tenant candidates: `{', '.join(t.tenant_columns) or 'none'}`"
            )
        lines.append("")

    lines.append("## All tables")
    lines.append("")
    for t in tables:
        fq = f"{t.schema}.{t.name}"
        lines.append(f"### `{fq}`")
        lines.append("")
        lines.append(f"- Estimated rows: `{t.estimated_rows}`")
        lines.append(f"- Domains (heuristic): `{', '.join(t.domains) or 'unclassified'}`")
        lines.append(f"- Primary keys: `{', '.join(t.primary_keys) or 'n/a'}`")
        lines.append(
            f"- Watermark candidates: `{', '.join(t.watermark_columns) or 'none'}`"
        )
        lines.append(f"- Tenant candidates: `{', '.join(t.tenant_columns) or 'none'}`")
        if t.foreign_keys:
            lines.append("- Foreign keys:")
            for fk in t.foreign_keys:
                lines.append(
                    f"  - `{fk['column_name']}` → "
                    f"`{fk['foreign_table_schema']}.{fk['foreign_table_name']}."
                    f"{fk['foreign_column_name']}`"
                )
        lines.append("")
        lines.append("| Column | Type | Nullable |")
        lines.append("| --- | --- | --- |")
        for c in t.columns:
            lines.append(
                f"| `{c.name}` | `{c.data_type}` | `{c.is_nullable}` |"
            )
        lines.append("")

    lines.append("## Gaps / follow-ups")
    lines.append("")
    lines.append(
        "- Confirm which heuristic matches are the real sales/products/stock sources."
    )
    lines.append(
        "- Prefer a dedicated READ ONLY DB user (not superuser/root) for ML ingestion."
    )
    lines.append(
        "- Soft-deletes: look for `deleted_at`, `ativo`, `status` columns per table."
    )
    lines.append(
        "- If a critical table has no watermark, document PK-range sync + reconciliation."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info(
        "introspect_erp_start",
        source_mode=settings.erp_source_mode,
        erp_url_host=_safe_host(settings.erp_database_url),
        dump_path=str(settings.erp_dump_path),
    )
    try:
        tables = introspect_erp()
    except Exception as exc:
        logger.error("introspect_erp_failed", error=str(exc))
        docs = Path("docs")
        docs.mkdir(parents=True, exist_ok=True)
        hint = (
            "Dump mode: place the file at `data/dumps/pdv_prod.dump`, then run:\n"
            "  docker compose up -d postgres-erp\n"
            "  uv run restore-erp-dump\n"
            "  uv run introspect-erp\n"
            if settings.erp_source_mode == "dump"
            else "Live mode: verify ERP_DATABASE_URL and network reachability.\n"
        )
        (docs / "erp-schema-map.md").write_text(
            "# ERP Schema Map\n\n"
            "## Connection failure\n\n"
            f"Could not connect or introspect ERP source: `{exc}`\n\n"
            f"Mode: `{settings.erp_source_mode}`\n\n"
            f"{hint}",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc

    docs = Path("docs")
    docs.mkdir(parents=True, exist_ok=True)
    md_path = docs / "erp-schema-map.md"
    md_path.write_text(tables_to_markdown(tables), encoding="utf-8")

    json_path = docs / "erp-schema-map.json"
    json_path.write_text(
        json.dumps([asdict(t) for t in tables], indent=2, default=str),
        encoding="utf-8",
    )
    logger.info(
        "introspect_erp_done",
        tables=len(tables),
        markdown=str(md_path),
        json=str(json_path),
    )
    print(f"Wrote {md_path} ({len(tables)} tables)")


def _safe_host(url: str) -> str:
    try:
        from urllib.parse import urlparse

        return urlparse(url).hostname or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
