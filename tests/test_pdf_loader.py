"""
Unit and integration test script for PDF Document Ingestion (Stage 2).
Tests:
1. Successful extraction from the sample digital PDF medical report.
2. Page boundary preservation and page counts.
3. Output persistence to data/processed/.
4. Error handling for missing files, invalid extensions, and empty paths.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.pdf_loader import PDFReportLoader, extract_text_from_pdf
from src.config import SAMPLE_REPORTS_DIR, PROCESSED_DATA_DIR


def test_valid_pdf_extraction():
    """Test standard extraction from sample_medical_report.pdf."""
    pdf_path = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"
    print(f"\n[TEST 1] Testing extraction from valid PDF: {pdf_path.name}")
    
    assert pdf_path.exists(), f"Sample PDF not found at {pdf_path}. Run scripts/create_sample_pdf.py first."

    result = extract_text_from_pdf(pdf_path, save_to_processed=True)

    # Assertions
    assert result["total_pages"] == 2, f"Expected 2 pages, got {result['total_pages']}"
    assert len(result["pages"]) == 2, "Pages list should contain 2 page records"
    assert result["is_fully_digital"] is True, "Digital test report should be fully digital"
    assert "Hemoglobin" in result["combined_text"], "Combined text should contain 'Hemoglobin'"
    assert "Lipid" in result["combined_text"] or "Cholesterol" in result["combined_text"], "Combined text should contain Lipid/Cholesterol"
    assert "--- Page 1 of 2 ---" in result["combined_text"], "Page 1 boundary marker missing"
    assert "--- Page 2 of 2 ---" in result["combined_text"], "Page 2 boundary marker missing"
    assert result["saved_path"] is not None, "Saved path should be returned when save_to_processed=True"
    assert Path(result["saved_path"]).exists(), "Extracted text file must exist in data/processed/"

    print("  -> Page Count :", result["total_pages"])
    print("  -> Page 1 Chars:", result["pages"][0]["char_count"])
    print("  -> Page 2 Chars:", result["pages"][1]["char_count"])
    print("  -> Output Saved:", result["saved_path"])
    print("  [PASS] Valid PDF extraction succeeded.")


def test_error_handling():
    """Test that invalid inputs and missing files raise clear, expected errors."""
    print("\n[TEST 2] Testing error handling (non-silent failures):")

    # 1. Non-existent file
    try:
        extract_text_from_pdf("non_existent_report.pdf")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"  [PASS] Correctly caught missing file: {type(e).__name__}")

    # 2. Invalid file extension (.txt instead of .pdf)
    txt_file = SAMPLE_REPORTS_DIR / "sample_cbc_report.txt"
    try:
        extract_text_from_pdf(txt_file)
        assert False, "Should have raised ValueError for non-PDF file"
    except ValueError as e:
        print(f"  [PASS] Correctly caught invalid file extension: {type(e).__name__}")

    # 3. Empty string / None
    try:
        extract_text_from_pdf("")
        assert False, "Should have raised ValueError for empty path"
    except ValueError as e:
        print(f"  [PASS] Correctly caught empty path: {type(e).__name__}")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens Stage 2: PDF Ingestion Verification Suite")
    print("=" * 70)
    test_valid_pdf_extraction()
    test_error_handling()
    print("\n" + "=" * 70)
    print("  ALL STAGE 2 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
