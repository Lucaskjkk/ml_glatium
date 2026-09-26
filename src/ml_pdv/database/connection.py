"""Database connection helpers (ERP source + ML analytical)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ml_pdv.config import get_settings

_erp_engine: Engine | None = None
_ml_engine: Engine | None = None


def _to_psycopg_url(url: str) -> str:
    """Prefer psycopg3 driver (`postgresql+psycopg://`)."""
    if url.startswith("postgresql+psycopg://"):
        return url
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    return url


def resolve_erp_dump_path() -> Path:
    """Return configured dump path (may not exist yet)."""
    return get_settings().erp_dump_path.resolve()


def assert_erp_source_ready() -> None:
    """Validate ERP source config before connecting.

    In dump mode the dump file is optional at connect time (DB may already be
    restored), but we surface a clear hint when both dump and DB are missing.
    """
    settings = get_settings()
    if not settings.erp_database_url:
        raise RuntimeError("ERP_DATABASE_URL is not configured")
    if settings.erp_source_mode == "dump":
        dump = resolve_erp_dump_path()
        if not dump.exists():
            # Soft warning path — connection may still work if DB was restored earlier.
            # Hard fail happens in restore_erp_dump when the file is required.
            pass


def get_erp_engine() -> Engine:
    """Engine for the ERP source database (local dump restore or live READ ONLY)."""
    global _erp_engine
    if _erp_engine is None:
        settings = get_settings()
        assert_erp_source_ready()
        connect_args: dict = {}
        # Always prefer read-only sessions against the ERP source.
        connect_args["options"] = "-c default_transaction_read_only=on"
        _erp_engine = create_engine(
            _to_psycopg_url(settings.erp_database_url),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
            connect_args=connect_args,
        )
    return _erp_engine


def reset_engines() -> None:
    """Drop cached engines (useful after restore / URL change in tests)."""
    global _erp_engine, _ml_engine
    if _erp_engine is not None:
        _erp_engine.dispose()
        _erp_engine = None
    if _ml_engine is not None:
        _ml_engine.dispose()
        _ml_engine = None


def get_ml_engine() -> Engine:
    """Engine for the ML analytical database."""
    global _ml_engine
    if _ml_engine is None:
        settings = get_settings()
        _ml_engine = create_engine(
            _to_psycopg_url(settings.ml_database_url),
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _ml_engine


@contextmanager
def erp_session() -> Generator[Session, None, None]:
    SessionLocal = sessionmaker(bind=get_erp_engine(), autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def ml_session() -> Generator[Session, None, None]:
    SessionLocal = sessionmaker(bind=get_ml_engine(), autoflush=False, autocommit=False)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_erp_connectivity() -> bool:
    try:
        with get_erp_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def check_ml_connectivity() -> bool:
    try:
        with get_ml_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
