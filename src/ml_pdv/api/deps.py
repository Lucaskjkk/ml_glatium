"""API dependencies (auth, settings)."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from ml_pdv.config import Settings, get_settings


def get_api_settings() -> Settings:
    return get_settings()


async def verify_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Settings = Depends(get_api_settings),
) -> None:
    """Simple API-key auth. Replace with OAuth/JWT later without changing routes."""
    expected = settings.api_key
    if not expected or expected == "change-me-in-production":
        # Allow local bootstrap when key is still default; warn via OpenAPI docs.
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
