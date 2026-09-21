"""Health check route."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.config import APP_VERSION
from app.schemas.responses import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    ready = getattr(request.app.state, "ready", False) is True
    adapters = getattr(request.app.state, "loaded_adapters", None)
    startup_error = getattr(request.app.state, "startup_error", None)

    if ready:
        body = HealthResponse(
            status="ok",
            ready=True,
            version=APP_VERSION,
            loaded_adapters=adapters,
        )
        return body

    body = HealthResponse(
        status="unavailable",
        ready=False,
        version=APP_VERSION,
        loaded_adapters=adapters,
        detail=startup_error or "Models not ready",
    )
    return JSONResponse(
        status_code=503,
        content=body.model_dump(),
    )
