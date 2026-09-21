"""
FastAPI entrypoint for the invoice extraction API.

Run from repository root:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import extract, health
from app.core.config import APP_NAME, APP_VERSION
from app.core.model_loader import load_model_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ready = False
    app.state.model_bundle = None
    app.state.loaded_adapters = None
    app.state.startup_error = None

    try:
        logger.info("Starting model load...")
        bundle = load_model_bundle()
        app.state.model_bundle = bundle
        app.state.loaded_adapters = bundle.loaded_adapters
        app.state.ready = True
        logger.info("API ready. Adapters: %s", bundle.loaded_adapters)
    except Exception as exc:
        app.state.startup_error = str(exc)
        app.state.ready = False
        logger.exception("Failed to load models at startup: %s", exc)
        # Keep process up so /health can report the failure reason.
        # Extraction endpoint will return 503 via get_model_bundle.

    yield

    app.state.ready = False
    app.state.model_bundle = None


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(extract.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Request validation failed.",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error.",
        },
    )
