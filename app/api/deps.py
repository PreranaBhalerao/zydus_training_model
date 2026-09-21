"""FastAPI dependency helpers."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.core.model_loader import ModelBundle


def get_model_bundle(request: Request) -> ModelBundle:
    bundle = getattr(request.app.state, "model_bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded yet. Try again shortly.",
        )
    return bundle


def is_ready(request: Request) -> bool:
    return getattr(request.app.state, "ready", False) is True
