"""
Ingestion Submodule
Responsible for loading medical reports in various formats:
- Digital PDF documents (Stage 2)
- Scanned PDF documents with OCR fallback (Stage 5)
- Medical report images: PNG, JPG, JPEG (Stage 5)
- Plain text reports
"""

from src.ingestion.pdf_loader import PDFReportLoader, extract_text_from_pdf
from src.ingestion.ocr_loader import (
    MultimodalReportLoader,
    load_medical_document,
    extract_text_from_image,
    is_tesseract_available,
    resolve_tesseract_binary,
)

__all__ = [
    "PDFReportLoader",
    "extract_text_from_pdf",
    "MultimodalReportLoader",
    "load_medical_document",
    "extract_text_from_image",
    "is_tesseract_available",
    "resolve_tesseract_binary",
]
