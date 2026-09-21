"""Application configuration loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# app/core/config.py -> parents[2] == repository root
REPO_ROOT = Path(__file__).resolve().parents[2]

V3_PATH = REPO_ROOT / "invoice_lora_v3"
V1_PATH = REPO_ROOT / "invoice_lora"

BASE_MODEL = os.environ.get(
    "BASE_MODEL",
    "unsloth/Qwen2.5-1.5B-Instruct",
)

MAX_SEQ = int(os.environ.get("MAX_SEQ", "8192"))

MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

APP_NAME = "Invoice Extraction API"
APP_VERSION = "1.0.0"
