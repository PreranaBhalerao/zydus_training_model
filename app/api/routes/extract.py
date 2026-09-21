"""Invoice extraction endpoint."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_model_bundle
from app.core.config import MAX_UPLOAD_BYTES, MAX_UPLOAD_MB
from app.core.model_loader import ModelBundle
from app.schemas.responses import ErrorResponse, ExtractResponse, OcrMetadata
from app.services.inference import run_inference, run_ocr

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/invoice", tags=["invoice"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}


def _is_pdf_upload(filename: str | None, content_type: str | None) -> bool:
    name_ok = bool(filename) and filename.lower().endswith(".pdf")
    type_ok = (
        content_type is None
        or content_type.lower().split(";")[0].strip()
        in ALLOWED_CONTENT_TYPES
    )
    return name_ok and type_ok


@router.post(
    "/extract",
    response_model=ExtractResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def extract_invoice(
    file: UploadFile = File(..., description="Invoice PDF"),
    bundle: ModelBundle = Depends(get_model_bundle),
):
    request_id = str(uuid.uuid4())
    logger.info(
        "request_id=%s filename=%s content_type=%s",
        request_id,
        file.filename,
        file.content_type,
    )

    if not _is_pdf_upload(file.filename, file.content_type):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "request_id": request_id,
                "error": "Only PDF uploads are accepted.",
                "detail": (
                    f"filename={file.filename!r} "
                    f"content_type={file.content_type!r}"
                ),
            },
        )

    tmp_path: str | None = None

    try:
        raw = await file.read()

        if not raw:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "request_id": request_id,
                    "error": "Uploaded file is empty.",
                },
            )

        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "request_id": request_id,
                    "error": (
                        f"File exceeds max upload size of "
                        f"{MAX_UPLOAD_MB} MB."
                    ),
                },
            )

        # PDF magic bytes check
        if not raw.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "request_id": request_id,
                    "error": "File does not look like a valid PDF.",
                },
            )

        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="invoice_")
        os.close(fd)
        Path(tmp_path).write_bytes(raw)

        try:
            ocr_text, ocr_meta = run_ocr(tmp_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "request_id": request_id,
                    "error": str(exc),
                },
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "success": False,
                    "request_id": request_id,
                    "error": str(exc),
                },
            ) from exc

        try:
            result, fallback_used = run_inference(bundle, ocr_text)
        except Exception as exc:
            logger.exception(
                "request_id=%s inference failed", request_id
            )
            raise HTTPException(
                status_code=500,
                detail={
                    "success": False,
                    "request_id": request_id,
                    "error": "Invoice extraction failed.",
                    "detail": str(exc),
                },
            ) from exc

        logger.info(
            "request_id=%s ocr_chars=%s fallback_used=%s",
            request_id,
            ocr_meta.get("total_chars"),
            fallback_used,
        )

        return ExtractResponse(
            success=True,
            request_id=request_id,
            result=result,
            fallback_used=fallback_used,
            ocr=OcrMetadata(
                page_count=int(ocr_meta.get("page_count") or 0),
                pdfplumber_pages=int(
                    ocr_meta.get("pdfplumber_pages") or 0
                ),
                pymupdf_pages=int(ocr_meta.get("pymupdf_pages") or 0),
                failed_pages=int(ocr_meta.get("failed_pages") or 0),
                total_chars=int(ocr_meta.get("total_chars") or 0),
                extraction_time=float(
                    ocr_meta.get("extraction_time") or 0.0
                ),
            ),
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(
                    "request_id=%s failed to delete temp file %s",
                    request_id,
                    tmp_path,
                )
        await file.close()
