"""CLI entrypoints."""

from __future__ import annotations


def main() -> None:
    print("ml-pdv — use: uvicorn ml_pdv.api.app:app --reload")
    print("1) Place dump at data/dumps/pdv_prod.dump")
    print("2) docker compose up -d postgres-erp")
    print("3) uv run restore-erp-dump")
    print("4) uv run introspect-erp")
