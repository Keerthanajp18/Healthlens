"""
Unit and integration test suite for Image & Scanned Medical Report Processing (Stage 5).
Tests:
1. Image loading and preprocessing with Pillow (grayscale & contrast).
2. Supported extensions validation (.pdf, .png, .jpg, .jpeg) and rejection of invalid types.
3. Robust error handling (missing files, empty paths, non-image files).
4. Digital PDF routing (ensures digital PDFs do NOT needlessly invoke OCR).
5. OCR pipeline execution:
   - If Tesseract OCR binary is installed: verifies live OCR extraction on PNG, JPG, and scanned PDF.
   - If Tesseract OCR binary is NOT installed: explicitly verifies that the loader detects
     the missing dependency and raises a clear explanatory RuntimeError instead of silently faking results.
"""

import sys
from pathlib import Path
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_REPORTS_DIR, PROCESSED_DATA_DIR
from src.ingestion.ocr_loader import (
    MultimodalReportLoader,
    load_medical_document,
    preprocess_image_for_ocr,
    is_tesseract_available,
    resolve_tesseract_binary,
    extract_text_from_image,
)


def test_image_loading_and_preprocessing():
    """Test 1: Verify Pillow loads images and preprocessor converts to enhanced grayscale."""
    img_path = SAMPLE_REPORTS_DIR / "sample_medical_report_image.png"
    assert img_path.exists(), f"Sample image missing at: {img_path}"

    with Image.open(img_path) as img:
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

        preprocessed = preprocess_image_for_ocr(img)
        assert preprocessed.mode == "L", f"Expected mode 'L' (grayscale), got {preprocessed.mode}"

    print(f"  [PASS] 1. Image loading & preprocessing verified (PNG, {img.size[0]}x{img.size[1]} -> Grayscale L)")


def test_supported_extensions():
    """Test 2: Verify loader accepts .png, .jpg, .jpeg, .pdf and rejects unsupported types."""
    png_path = SAMPLE_REPORTS_DIR / "sample_medical_report_image.png"
    jpg_path = SAMPLE_REPORTS_DIR / "sample_medical_report_image.jpg"
    pdf_path = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"

    # Supported formats should instantiate without error
    loader_png = MultimodalReportLoader(png_path)
    loader_jpg = MultimodalReportLoader(jpg_path)
    loader_pdf = MultimodalReportLoader(pdf_path)
    assert loader_png.path == png_path
    assert loader_jpg.path == jpg_path
    assert loader_pdf.path == pdf_path

    # Unsupported formats (.txt or .docx) must raise ValueError
    txt_path = SAMPLE_REPORTS_DIR / "sample_cbc_report.txt"
    try:
        MultimodalReportLoader(txt_path)
        assert False, "Should have raised ValueError for .txt extension in MultimodalReportLoader"
    except ValueError as e:
        assert "Unsupported document format" in str(e)

    print("  [PASS] 2. Extension validation: .png, .jpg, .pdf accepted; unsupported types rejected")


def test_invalid_file_handling():
    """Test 3: Verify missing files, empty paths, and invalid paths raise expected exceptions."""
    # 1. Non-existent image file
    try:
        MultimodalReportLoader("non_existent_report.png")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    # 2. Empty string path
    try:
        MultimodalReportLoader("")
        assert False, "Should have raised ValueError for empty path"
    except ValueError:
        pass

    print("  [PASS] 3. Invalid file handling: FileNotFoundError and ValueError correctly raised")


def test_digital_pdf_routing():
    """Test 4: Verify digital PDFs bypass OCR entirely and are marked 'digital_pdf' with ocr_used=False."""
    pdf_path = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"
    assert pdf_path.exists()

    result = load_medical_document(pdf_path, save_to_processed=False)
    assert result["source_type"] == "digital_pdf"
    assert result["ocr_used"] is False
    assert result["total_pages"] == 2
    assert "Hemoglobin" in result["combined_text"]
    assert "Total Cholesterol" in result["combined_text"]

    print("  [PASS] 4. Digital PDF routing verified (source_type='digital_pdf', ocr_used=False)")


def test_ocr_pipeline_and_tesseract_availability():
    """
    Test 5: Verify OCR execution or explicit dependency reporting.
    
    If Tesseract is installed: verifies live OCR on PNG image and scanned PDF.
    If Tesseract is missing: verifies that an explicit RuntimeError is raised rather
    than silently faking output.
    """
    tesseract_available = is_tesseract_available()
    tesseract_bin = resolve_tesseract_binary()

    png_path = SAMPLE_REPORTS_DIR / "sample_medical_report_image.png"
    scanned_pdf_path = SAMPLE_REPORTS_DIR / "sample_scanned_report.pdf"

    if tesseract_available:
        print(f"  [INFO] Tesseract binary detected at: {tesseract_bin}")

        # Test A: Image OCR
        img_result = load_medical_document(png_path)
        assert img_result["source_type"] == "image"
        assert img_result["ocr_used"] is True
        assert len(img_result["combined_text"]) > 50
        print(f"  -> Extracted {len(img_result['combined_text'])} characters via image OCR")

        # Test B: Scanned PDF OCR
        pdf_result = load_medical_document(scanned_pdf_path)
        assert pdf_result["source_type"] == "scanned_pdf"
        assert pdf_result["ocr_used"] is True
        print(f"  -> Extracted {len(pdf_result['combined_text'])} characters via scanned PDF OCR")
        print("  [PASS] 5. Live OCR execution succeeded on both image and scanned PDF")

    else:
        print("  [NOTICE] Native Tesseract OCR binary (tesseract.exe) is not installed on this system.")
        print("           Verifying that the loader raises a clear, informative RuntimeError...")

        # Test that calling OCR on image raises RuntimeError with clear instructions
        try:
            extract_text_from_image(png_path)
            assert False, "Should have raised RuntimeError for missing Tesseract binary"
        except RuntimeError as e:
            assert "Tesseract OCR Engine Not Found" in str(e)
            assert "https://github.com/UB-Mannheim/tesseract/wiki" in str(e)

        # Test that calling OCR on scanned PDF raises RuntimeError with instructions
        try:
            load_medical_document(scanned_pdf_path)
            assert False, "Should have raised RuntimeError for missing Tesseract binary"
        except RuntimeError as e:
            assert "Tesseract OCR Engine Not Found" in str(e)

        print("  [PASS] 5. Explicit error reporting verified: Non-silent failure with clear install instructions")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens Stage 5: Image & OCR Loader Verification Suite")
    print("=" * 70)
    test_image_loading_and_preprocessing()
    test_supported_extensions()
    test_invalid_file_handling()
    test_digital_pdf_routing()
    test_ocr_pipeline_and_tesseract_availability()
    print("=" * 70)
    print("  ALL STAGE 5 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
