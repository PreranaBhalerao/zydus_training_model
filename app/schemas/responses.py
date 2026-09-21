"""API response schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class OcrMetadata(BaseModel):
    page_count: int = 0
    pdfplumber_pages: int = 0
    pymupdf_pages: int = 0
    failed_pages: int = 0
    total_chars: int = 0
    extraction_time: float = 0.0


class ExtractResponse(BaseModel):
    success: bool = True
    request_id: str
    result: Dict[str, Any] = Field(default_factory=dict)
    fallback_used: List[str] = Field(default_factory=list)
    ocr: OcrMetadata


class HealthResponse(BaseModel):
    status: str
    ready: bool
    version: str
    loaded_adapters: Optional[List[str]] = None
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    request_id: Optional[str] = None
    error: str
    detail: Optional[str] = None
