"""
OCR / PDF TEXT EXTRACTION MODULE
--------------------------------
Purpose:
    Extract usable text from invoice PDFs.

Flow:
    1. Try PDFPlumber first for digitally generated / typed PDFs.
    2. If PDFPlumber fails or returns insufficient text,
       fall back to PyMuPDF text extraction.
    3. Return the extracted text and extraction metadata.

This file does NOT:
    - Load Qwen
    - Run LoRA adapters
    - Run regex invoice extraction
    - Run V1/V3 fallback
    - Call FastAPI

Those responsibilities will be handled separately.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import pdfplumber
import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


# Minimum amount of extracted text considered usable
MIN_TEXT_CHARS = 100


def extract_text_with_pdfplumber(
    pdf_path: str,
    page_num: int
) -> Tuple[Optional[str], float]:
    """
    Extract text from one PDF page using PDFPlumber.

    Best suited for:
        - Digitally generated PDFs
        - Typed invoices
        - PDFs containing selectable text
        - PDFs containing tables

    Returns:
        (text, confidence_score)

    Confidence:
        95.0 when sufficient typed text is extracted.
        0.0 when extraction fails or text is insufficient.
    """

    try:
        start_time = time.time()

        with pdfplumber.open(pdf_path) as pdf:

            if page_num >= len(pdf.pages):
                return None, 0.0

            page = pdf.pages[page_num]

            # Extract normal page text
            text = page.extract_text()

            if not text:
                return None, 0.0

            # Extract tables if present
            tables = page.extract_tables()

            if tables:
                for table in tables:
                    for row in table:

                        if row:
                            row_text = " | ".join(
                                str(cell) if cell else ""
                                for cell in row
                            )

                            text += "\n" + row_text

            extraction_time = time.time() - start_time

            text = text.strip()
            char_count = len(text)

            # Quality check
            if char_count > MIN_TEXT_CHARS:

                logger.info(
                    f"PDFPlumber: {char_count} chars "
                    f"in {extraction_time:.2f}s"
                )

                return text, 95.0

            logger.warning(
                f"PDFPlumber extracted only {char_count} chars"
            )

            return None, 0.0

    except Exception as e:

        logger.warning(
            f"PDFPlumber failed on page {page_num}: {e}"
        )

        return None, 0.0


def extract_text_with_pymupdf(
    pdf_path: str,
    page_num: int
) -> Tuple[Optional[str], float]:
    """
    Fallback text extraction using PyMuPDF.

    This handles PDFs where PDFPlumber does not successfully
    extract enough text.

    Returns:
        (text, confidence_score)
    """

    try:

        start_time = time.time()

        document = fitz.open(pdf_path)

        try:

            if page_num >= len(document):
                return None, 0.0

            page = document[page_num]

            text = page.get_text("text")

            if not text:
                return None, 0.0

            text = text.strip()

            char_count = len(text)

            extraction_time = time.time() - start_time

            if char_count > MIN_TEXT_CHARS:

                logger.info(
                    f"PyMuPDF: {char_count} chars "
                    f"in {extraction_time:.2f}s"
                )

                return text, 85.0

            return None, 0.0

        finally:

            document.close()

    except Exception as e:

        logger.warning(
            f"PyMuPDF failed on page {page_num}: {e}"
        )

        return None, 0.0


def extract_invoice_text(
    pdf_path: str
) -> Tuple[Optional[str], Dict[str, Any]]:
    """
    Extract text from the complete invoice PDF.

    Extraction strategy:

        PDF
         ↓
        PDFPlumber
         ↓
        if insufficient
         ↓
        PyMuPDF
         ↓
        combined text

    Returns:

        text:
            Complete extracted text or None.

        metadata:
            Information about the extraction process.
    """

    pdf_path = str(pdf_path)

    if not Path(pdf_path).exists():

        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    try:

        document = fitz.open(pdf_path)
        page_count = len(document)
        document.close()

    except Exception as e:

        raise ValueError(
            f"Unable to open PDF: {e}"
        )

    extracted_pages = []

    pdfplumber_pages = 0
    pymupdf_pages = 0
    failed_pages = 0

    start_time = time.time()

    for page_num in range(page_count):

        # -----------------------------------------
        # TIER 1: PDFPlumber
        # -----------------------------------------

        logger.info(
            f"Page {page_num + 1}/{page_count}: "
            f"Trying PDFPlumber..."
        )

        text, confidence = extract_text_with_pdfplumber(
            pdf_path,
            page_num
        )

        if text:

            extracted_pages.append(
                f"\n--- PAGE {page_num + 1} ---\n{text}"
            )

            pdfplumber_pages += 1

            continue

        # -----------------------------------------
        # TIER 2: PyMuPDF
        # -----------------------------------------

        logger.info(
            f"Page {page_num + 1}/{page_count}: "
            f"PDFPlumber insufficient. Trying PyMuPDF..."
        )

        text, confidence = extract_text_with_pymupdf(
            pdf_path,
            page_num
        )

        if text:

            extracted_pages.append(
                f"\n--- PAGE {page_num + 1} ---\n{text}"
            )

            pymupdf_pages += 1

            continue

        # -----------------------------------------
        # FAILED PAGE
        # -----------------------------------------

        logger.warning(
            f"Page {page_num + 1}: "
            f"No usable text extracted."
        )

        failed_pages += 1

    total_time = time.time() - start_time

    if not extracted_pages:

        logger.error(
            "No usable text could be extracted from PDF."
        )

        return None, {
            "success": False,
            "page_count": page_count,
            "pdfplumber_pages": 0,
            "pymupdf_pages": 0,
            "failed_pages": failed_pages,
            "total_chars": 0,
            "extraction_time": total_time,
        }

    full_text = "\n".join(extracted_pages).strip()

    metadata = {
        "success": True,
        "page_count": page_count,
        "pdfplumber_pages": pdfplumber_pages,
        "pymupdf_pages": pymupdf_pages,
        "failed_pages": failed_pages,
        "total_chars": len(full_text),
        "extraction_time": round(total_time, 3),
    }

    logger.info(
        f"Extraction complete: "
        f"{len(full_text)} chars, "
        f"{page_count} pages, "
        f"{total_time:.2f}s"
    )

    return full_text, metadata


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    print(
        "OCR module loaded successfully."
    )
