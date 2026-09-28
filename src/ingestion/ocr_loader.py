"""
HealthLens - Stage 5: Image and Scanned Medical Report Processing
Module: src/ingestion/ocr_loader.py

Extends Person 1's ingestion pipeline to handle:
1. Native digital PDFs (via pypdf)
2. Scanned PDFs (via pypdfium2 rasterization + pytesseract OCR)
3. Image medical reports (.png, .jpg, .jpeg via Pillow + pytesseract OCR)

Key Architecture:
- Hybrid Ingestion Pipeline: Attempts native digital text extraction first.
- If pages lack extractable text (scanned PDF) or an image is provided, OCR is invoked.
- Tracks source_type ('digital_pdf', 'scanned_pdf', 'image') and ocr_used (True/False).
- Formats text with preserved page boundaries for Stage 3 extraction.
"""

from pathlib import Path
import shutil
import sys
from PIL import Image, ImageEnhance, ImageFilter
import pypdf
import pytesseract

from src.config import PROCESSED_DATA_DIR, SUPPORTED_DOCUMENT_EXTENSIONS, SUPPORTED_IMAGE_EXTENSIONS
from src.ingestion.pdf_loader import PDFReportLoader


# ==============================================================================
# TESSERACT ENVIRONMENT RESOLUTION & PRE-FLIGHT CHECKS
# ==============================================================================

COMMON_WINDOWS_TESSERACT_PATHS = [
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Users\kiran\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\tools\tesseract\tesseract.exe"),
]


def resolve_tesseract_binary() -> str | None:
    """
    Locates the tesseract executable on Windows or POSIX systems.
    Auto-configures pytesseract.pytesseract.tesseract_cmd if located.
    
    Returns:
        Path string to tesseract binary, or None if not found.
    """
    # 1. Check if already configured and valid
    configured = pytesseract.pytesseract.tesseract_cmd
    if configured and configured != "tesseract" and Path(configured).is_file():
        return configured

    # 2. Check system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return which_path

    # 3. Check known standard Windows installation directories
    for candidate in COMMON_WINDOWS_TESSERACT_PATHS:
        if candidate.exists() and candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return str(candidate)

    return None


def is_tesseract_available() -> bool:
    """Check if the native Tesseract OCR engine is installed and reachable."""
    return resolve_tesseract_binary() is not None


def require_tesseract() -> str:
    """
    Ensure Tesseract is installed, or raise a clear explanatory error.
    
    Raises:
        RuntimeError: Detailing download links and instructions for Windows/Linux/Mac.
    """
    binary_path = resolve_tesseract_binary()
    if not binary_path:
        raise RuntimeError(
            "Tesseract OCR Engine Not Found!\n"
            "----------------------------------------------------------------------\n"
            "The Python wrapper 'pytesseract' is installed, but the underlying native\n"
            "Tesseract OCR executable (tesseract.exe) was not found on your system.\n\n"
            "To enable OCR for scanned PDFs and image files (.png, .jpg):\n"
            "1. Download and install Tesseract for Windows:\n"
            "   https://github.com/UB-Mannheim/tesseract/wiki\n"
            "   (or via winget: 'winget install UB-Mannheim.TesseractOCR')\n"
            "2. Ensure tesseract.exe is in your PATH or installed at:\n"
            "   'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n"
            "----------------------------------------------------------------------"
        )
    return binary_path


# ==============================================================================
# IMAGE PREPROCESSING & OCR UTILITIES
# ==============================================================================

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Preprocess image to maximize OCR character recognition accuracy.
    Applies grayscale conversion and contrast enhancement.
    """
    # 1. Convert to grayscale (removes color noise)
    gray = image.convert("L")

    # 2. Increase contrast so text stands out against background
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.8)

    return enhanced


def extract_text_from_image(image_path: str | Path) -> str:
    """
    Run OCR on a single image file (.png, .jpg, .jpeg) using Pillow and Tesseract.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Extracted raw text string.
        
    Raises:
        FileNotFoundError: If image file does not exist.
        ValueError: If file format is not a supported image.
        RuntimeError: If Tesseract OCR binary is missing.
    """
    path = Path(image_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Medical report image not found: '{path}'")

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    if path.suffix.lower() not in valid_exts:
        raise ValueError(
            f"Unsupported image extension '{path.suffix}'. "
            f"Expected one of: {sorted(list(valid_exts))}"
        )

    # Verify Tesseract is available
    require_tesseract()

    with Image.open(path) as img:
        processed_img = preprocess_image_for_ocr(img)
        # Run OCR with page segmentation mode 6 (Assume a single uniform block of text / table)
        text = pytesseract.image_to_string(processed_img, config="--psm 6")
        return text.strip()


# ==============================================================================
# UNIFIED MULTIMODAL DOCUMENT LOADER
# ==============================================================================

class MultimodalReportLoader:
    """
    Unified Ingestion Loader for HealthLens.
    
    Automatically routes documents:
    - Digital PDFs  -> Extracted via pypdf (fast, lossless)
    - Scanned PDFs  -> Rendered via pypdfium2 + OCR via pytesseract
    - Images (.png, .jpg) -> Preprocessed via Pillow + OCR via pytesseract
    
    Always preserves page boundaries and formats output identically for Stage 3.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

    def __init__(self, file_path: str | Path):
        if not file_path:
            raise ValueError("Report file path cannot be empty or None.")

        self.path = Path(file_path).resolve()
        self._validate_file()

    def _validate_file(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Report file not found at: '{self.path}'")

        if not self.path.is_file():
            raise ValueError(f"Path is not a regular file: '{self.path}'")

        ext = self.path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document format '{ext}'. "
                f"Supported formats: {sorted(list(self.SUPPORTED_EXTENSIONS))}"
            )

    def load(
        self,
        save_to_processed: bool = False,
        output_filename: str | None = None
    ) -> dict:
        """
        Ingest the document and extract text using the appropriate digital or OCR pipeline.
        
        Returns:
            dict containing:
                - file_name: str
                - file_path: str
                - source_type: 'digital_pdf' | 'scanned_pdf' | 'image'
                - ocr_used: bool
                - total_pages: int
                - pages: list of dicts (page_number, text, char_count, has_text, ocr_applied)
                - combined_text: str (with page boundaries)
                - saved_path: str | None
                - warnings: list[str]
        """
        ext = self.path.suffix.lower()

        if ext in {".png", ".jpg", ".jpeg"}:
            return self._load_image(save_to_processed, output_filename)
        elif ext == ".pdf":
            return self._load_pdf(save_to_processed, output_filename)
        else:
            raise ValueError(f"Unsupported extension: {ext}")

    def _load_image(self, save_to_processed: bool, output_filename: str | None) -> dict:
        """Process standalone image medical report."""
        warnings = []
        require_tesseract()

        with Image.open(self.path) as img:
            processed_img = preprocess_image_for_ocr(img)
            text = pytesseract.image_to_string(processed_img, config="--psm 6").strip()

        has_text = len(text) > 0
        if not has_text:
            warnings.append("OCR completed, but zero characters were detected on the image.")

        page_block = f"--- Page 1 of 1 ---\n{text}"

        result = {
            "file_name": self.path.name,
            "file_path": str(self.path),
            "source_type": "image",
            "ocr_used": True,
            "total_pages": 1,
            "pages": [{
                "page_number": 1,
                "text": text,
                "char_count": len(text),
                "has_text": has_text,
                "ocr_applied": True,
            }],
            "combined_text": page_block,
            "saved_path": None,
            "warnings": warnings,
        }

        if save_to_processed:
            result["saved_path"] = self._persist_extracted_text(page_block, output_filename)

        return result

    def _load_pdf(self, save_to_processed: bool, output_filename: str | None) -> dict:
        """Process PDF document, using pypdf for digital text and falling back to OCR when needed."""
        # 1. First run digital extraction using Stage 2 loader
        digital_loader = PDFReportLoader(self.path)
        digital_result = digital_loader.extract_text(save_to_processed=False)

        # 2. Check if all pages already contain clean digital text
        if digital_result["is_fully_digital"]:
            digital_result["source_type"] = "digital_pdf"
            digital_result["ocr_used"] = False
            for p in digital_result["pages"]:
                p["ocr_applied"] = False

            if save_to_processed:
                digital_result["saved_path"] = self._persist_extracted_text(
                    digital_result["combined_text"], output_filename
                )
            return digital_result

        # 3. If one or more pages had no digital text, OCR is required for those pages!
        # Scanned PDF detected!
        require_tesseract()

        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise RuntimeError(
                "Package 'pypdfium2' is required to rasterize scanned PDF pages for OCR. "
                "Install it via: pip install pypdfium2"
            )

        pdf = pdfium.PdfDocument(str(self.path))
        total_pages = len(pdf)
        updated_pages = []
        page_text_blocks = []
        warnings = list(digital_result["warnings"])

        for i in range(total_pages):
            page_num = i + 1
            existing_page = digital_result["pages"][i]

            # If digital text exists on this page, keep it
            if existing_page["has_text"]:
                cleaned_text = existing_page["text"]
                ocr_applied = False
            else:
                # Render page to image using pdfium and run OCR
                page = pdf[i]
                pil_image = page.render(scale=2.0).to_pil()
                preprocessed = preprocess_image_for_ocr(pil_image)
                cleaned_text = pytesseract.image_to_string(preprocessed, config="--psm 6").strip()
                ocr_applied = True

            has_text = len(cleaned_text) > 0
            updated_pages.append({
                "page_number": page_num,
                "text": cleaned_text,
                "char_count": len(cleaned_text),
                "has_text": has_text,
                "ocr_applied": ocr_applied,
            })

            page_banner = f"--- Page {page_num} of {total_pages} ---"
            page_text_blocks.append(f"{page_banner}\n{cleaned_text}")

        combined_text = "\n\n".join(page_text_blocks).strip()

        result = {
            "file_name": self.path.name,
            "file_path": str(self.path),
            "source_type": "scanned_pdf",
            "ocr_used": True,
            "total_pages": total_pages,
            "pages": updated_pages,
            "combined_text": combined_text,
            "saved_path": None,
            "warnings": warnings,
        }

        if save_to_processed:
            result["saved_path"] = self._persist_extracted_text(combined_text, output_filename)

        return result

    def _persist_extracted_text(self, text: str, custom_name: str | None) -> str:
        """Write extracted combined text to data/processed/."""
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        fname = custom_name or f"{self.path.stem}_extracted_text.txt"
        dest = PROCESSED_DATA_DIR / fname
        dest.write_text(text, encoding="utf-8")
        return str(dest)


def load_medical_document(
    file_path: str | Path,
    save_to_processed: bool = False,
    output_filename: str | None = None
) -> dict:
    """
    Unified function to ingest any supported medical report (Digital PDF, Scanned PDF, or Image).
    """
    loader = MultimodalReportLoader(file_path)
    return loader.load(save_to_processed=save_to_processed, output_filename=output_filename)
