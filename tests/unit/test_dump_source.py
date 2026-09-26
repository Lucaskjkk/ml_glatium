"""Settings / dump-source unit tests."""

from __future__ import annotations

from pathlib import Path

from ml_pdv.config.settings import Settings
from ml_pdv.scripts.restore_erp_dump import _detect_dump_format


def test_default_source_mode_is_dump() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        erp_source_mode="dump",
        erp_dump_path="data/dumps/pdv_prod.dump",
        erp_database_url="postgresql://erp:erp@127.0.0.1:5434/pdv_prod",
    )
    assert settings.erp_source_mode == "dump"
    assert settings.erp_dump_path == Path("data/dumps/pdv_prod.dump")


def test_detect_dump_format_by_extension(tmp_path: Path) -> None:
    custom = tmp_path / "x.dump"
    custom.write_bytes(b"PGDMP\x00")
    plain = tmp_path / "x.sql"
    plain.write_text("-- dump", encoding="utf-8")
    assert _detect_dump_format(custom) == "custom"
    assert _detect_dump_format(plain) == "plain"


def test_sanitize_plain_sql_strips_pg18_bits() -> None:
    from ml_pdv.scripts.restore_erp_dump import sanitize_plain_sql

    raw = (
        "--\n"
        "\\restrict ABC\n"
        "SET transaction_timeout = 0;\n"
        "CREATE SCHEMA public;\n"
        "CREATE SCHEMA tenant_x;\n"
        "ALTER TABLE x OWNER TO postgres;\n"
        "\\unrestrict ABC\n"
    )
    out = sanitize_plain_sql(raw, owner="erp")
    assert "\\restrict" not in out
    assert "\\unrestrict" not in out
    assert "transaction_timeout" not in out
    assert "CREATE SCHEMA public;" not in out
    assert "CREATE SCHEMA IF NOT EXISTS tenant_x;" in out
    assert "OWNER TO erp" in out

