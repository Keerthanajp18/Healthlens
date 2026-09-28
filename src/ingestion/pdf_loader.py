"""
HealthLens - Stage 2: PDF Document Ingestion and Text Extraction
Module: src/ingestion/pdf_loader.py

This module provides reusable utilities to ingest digital/text-based medical
PDF reports and extract their text content page-by-page using pypdf.
"""

from pathlib import Path
import sys
import pypdf
from pypdf.errors import PdfReadError

# Import project configurations
from src.config import PROCESSED_DATA_DIR


class PDFReportLoader:
    """
    Reusable loader for digital medical PDF reports.
    
    Handles:
    - File existence & extension validation
    - Page-by-page text extraction using pypdf
    - Preservation of page boundaries
    - Error detection (corrupted files, empty pages, missing files)
    - Optional export of extracted text to data/processed/
    """

    def __init__(self, pdf_path: str | Path):
        """
        Initialize loader with a target PDF path and validate path attributes.
        
        Args:
            pdf_path: Path or string pointing to the PDF report.
            
        Raises:
            ValueError: If path is empty or does not have a .pdf extension.
            FileNotFoundError: If the file does not exist on disk.
        """
        if not pdf_path:
            raise ValueError("PDF file path cannot be empty or None.")

        self.pdf_path = Path(pdf_path).resolve()
        self._validate_file()

    def _validate_file(self) -> None:
        """Verify that the target file exists and is a PDF."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"Medical report PDF not found at: '{self.pdf_path}'"
            )

        if not self.pdf_path.is_file():
            raise ValueError(
                f"Provided path is a directory, not a file: '{self.pdf_path}'"
            )

        if self.pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Invalid file type: '{self.pdf_path.name}'. "
                f"Expected a '.pdf' document, but received '{self.pdf_path.suffix}'."
            )

    def extract_text(
        self,
        save_to_processed: bool = False,
        output_filename: str | None = None
    ) -> dict:
        """
        Extract text from each page of the digital PDF.
        
        Args:
            save_to_processed: If True, saves the extracted combined text to data/processed/.
            output_filename: Optional custom filename for the saved text file.
            
        Returns:
            dict containing:
                - file_name: Name of the PDF
                - file_path: Absolute path string
                - total_pages: Number of pages
                - pages: List of dicts per page (page_number, text, char_count, has_text)
                - combined_text: Full report text with page boundaries
                - warnings: List of warning messages (e.g. empty or scanned pages)
                - saved_path: Path to saved text file (if save_to_processed is True)
                
        Raises:
            RuntimeError: If pypdf fails to open or read the PDF (e.g., encrypted/corrupted).
        """
        try:
            reader = pypdf.PdfReader(str(self.pdf_path))
        except PdfReadError as e:
            raise RuntimeError(
                f"Cannot read PDF '{self.pdf_path.name}'. The file may be corrupted, "
                f"encrypted, or have an invalid PDF header: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Unexpected error while opening PDF '{self.pdf_path.name}': {e}"
            ) from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            raise RuntimeError(f"The PDF file '{self.pdf_path.name}' contains 0 pages.")

        pages_data = []
        page_text_blocks = []
        warnings = []

        for index, page in enumerate(reader.pages):
            page_number = index + 1
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                # If extraction fails for a specific page, report it explicitly
                raw_text = ""
                warnings.append(
                    f"Warning: Failed to extract text from Page {page_number} ({e})."
                )

            cleaned_text = raw_text.strip()
            has_text = len(cleaned_text) > 0

            if not has_text:
                warning_msg = (
                    f"Warning: Page {page_number} contains no extractable digital text. "
                    "It may be a scanned image or an empty page (OCR will be required in Stage 3)."
                )
                warnings.append(warning_msg)
                print(f"[!] {warning_msg}", file=sys.stderr)

            pages_data.append({
                "page_number": page_number,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "has_text": has_text,
            })

            # Format text block with clear boundary marker
            page_banner = f"--- Page {page_number} of {total_pages} ---"
            page_text_blocks.append(f"{page_banner}\n{cleaned_text}")

        combined_text = "\n\n".join(page_text_blocks).strip()

        result = {
            "file_name": self.pdf_path.name,
            "file_path": str(self.pdf_path),
            "total_pages": total_pages,
            "pages": pages_data,
            "combined_text": combined_text,
            "warnings": warnings,
            "is_fully_digital": all(p["has_text"] for p in pages_data),
            "saved_path": None,
        }

        # Optionally persist extracted text to data/processed/
        if save_to_processed:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            if output_filename:
                out_name = output_filename
            else:
                out_name = f"{self.pdf_path.stem}_extracted_text.txt"

            destination = PROCESSED_DATA_DIR / out_name
            destination.write_text(combined_text, encoding="utf-8")
            result["saved_path"] = str(destination)

        return result


def extract_text_from_pdf(
    pdf_path: str | Path,
    save_to_processed: bool = False,
    output_filename: str | None = None
) -> dict:
    """
    Convenience functional wrapper around PDFReportLoader.
    
    Args:
        pdf_path: Path to the target medical PDF.
        save_to_processed: Whether to write extracted text to data/processed/.
        output_filename: Optional custom output text filename.
        
    Returns:
        dict containing extraction results, metadata, and combined text.
    """
    loader = PDFReportLoader(pdf_path)
    return loader.extract_text(
        save_to_processed=save_to_processed,
        output_filename=output_filename
    )
