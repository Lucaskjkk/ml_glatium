"""Application settings loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # dump = local Postgres restored from file (default for local ML)
    # live = direct read-only connection to production ERP (avoid when possible)
    erp_source_mode: Literal["dump", "live"] = Field(
        default="dump",
        description="Data source for ERP reads: local dump restore or live DB.",
    )
    erp_dump_path: Path = Field(
        default=Path("data/dumps/pdv_prod.dump"),
        description="Path to pg_dump custom (.dump) or plain SQL (.sql) file.",
    )
    erp_database_url: str = Field(
        default="postgresql://erp:erp@127.0.0.1:5432/pdv_prod",
        description="PostgreSQL URL used as ERP source (local dump DB or live).",
    )
    ml_database_url: str = Field(
        default="postgresql://ml:ml@127.0.0.1:5433/ml_pdv",
        description="Analytical ML PostgreSQL URL (separate from ERP).",
    )

    redis_url: str = "redis://localhost:6379/0"
    mlflow_tracking_uri: str = "http://localhost:5000"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-in-production"

    model_name: str = "demand_forecasting"
    prediction_horizon_days: int = 7

    sync_interval_minutes: int = 5
    training_schedule_cron: str = "0 3 * * 0"

    log_level: str = "INFO"

    @field_validator("erp_dump_path", mode="before")
    @classmethod
    def _expand_dump_path(cls, value: object) -> Path:
        return Path(str(value)).expanduser()


@lru_cache
def get_settings() -> Settings:
    return Settings()
