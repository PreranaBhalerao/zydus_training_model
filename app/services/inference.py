"""OCR + locked GPU inference orchestration."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure repository root is importable so `ocr.py` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ocr import extract_invoice_text  # noqa: E402

from app.core.model_loader import ModelBundle
from app.services.pipeline import run_pipeline

logger = logging.getLogger(__name__)

# Single-flight lock: one generate/run_pipeline at a time on the GPU.
_INFERENCE_LOCK = threading.Lock()


def run_ocr(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
    """Extract OCR text from a PDF. Raises ValueError on failure."""
    text, metadata = extract_invoice_text(pdf_path)

    if not text or not metadata.get("success"):
        raise ValueError(
            "No usable text could be extracted from the PDF. "
            "Ensure the file is a text-based invoice PDF."
        )

    return text, metadata


def run_inference(
    bundle: ModelBundle,
    ocr_text: str,
) -> Tuple[Dict[str, Any], list]:
    """
    Run V3→V1 fallback pipeline under a process-wide GPU lock.

    Returns (result_obj, fallback_used).
    """
    with _INFERENCE_LOCK:
        pred, used, _raw = run_pipeline(
            bundle.model,
            bundle.tokenizer,
            ocr_text,
        )

    if not isinstance(pred, dict):
        pred = {
            "invoice_summary": {},
            "line_items": None,
        }

    # Strip internal raw dump from client response if present.
    result = {
        "invoice_summary": pred.get("invoice_summary") or {},
        "line_items": pred.get("line_items"),
    }

    logger.info("fallback_used=%s", used)
    return result, list(used)
