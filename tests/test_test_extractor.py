"""
Unit and integration test suite for Medical Test Information Extraction (Stage 3).
Tests:
1. Normal decimal value extraction
2. Integer value with comma extraction
3. Reference range extraction (standard ranges)
4. '<' reference value extraction (e.g. '< 200')
5. '>' reference value extraction (e.g. '> 50')
6. Measurement unit extraction
7. Multiple test extraction (full real synthetic report)
8. Filtering out patient/report metadata numbers (age, IDs, dates, page numbers)
9. JSON output generation and validation (integration contract for Person 2)
10. pandas DataFrame output format
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_REPORTS_DIR, PROCESSED_DATA_DIR
from src.ingestion.pdf_loader import extract_text_from_pdf
from src.extraction.test_extractor import MedicalTestExtractor, extract_medical_tests


def test_decimal_value_extraction():
    """Requirement 1: Test normal decimal value extraction."""
    snippet = """
    Hemoglobin (Hb)
    11.5
    g/dL
    12.0 - 15.5
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)
    assert len(records) == 1, f"Expected 1 record, got {len(records)}"
    rec = records[0]
    assert rec["test_name"] == "Hemoglobin (Hb)"
    assert rec["value"] == 11.5
    assert isinstance(rec["value"], float)
    print("  [PASS] 1. Decimal value extracted correctly (11.5 float)")


def test_integer_with_comma():
    """Requirement 2: Test integer value with comma formatting."""
    snippet = """
    Total Leukocyte Count (WBC)
    11,200
    cells/mcL
    4,000 - 11,000
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)
    assert len(records) == 1
    rec = records[0]
    assert rec["test_name"] == "Total Leukocyte Count (WBC)"
    assert rec["value"] == 11200
    assert isinstance(rec["value"], (int, float))
    print("  [PASS] 2. Comma-separated integer extracted correctly (11,200 -> 11200)")


def test_reference_range():
    """Requirement 3: Test standard biological reference ranges."""
    snippet = """
    RBC Count
    3.95
    million/mcL
    3.80 - 5.10
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)
    assert len(records) == 1
    assert records[0]["reference_range"] == "3.80 - 5.10"
    print("  [PASS] 3. Standard reference range extracted correctly ('3.80 - 5.10')")


def test_less_than_reference_range():
    """Requirement 4: Test '<' reference range."""
    snippet = """
    Total Cholesterol
    215
    mg/dL
    < 200 Desirable
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)
    assert len(records) == 1
    assert records[0]["test_name"] == "Total Cholesterol"
    assert "< 200" in records[0]["reference_range"]
    print("  [PASS] 4. '<' reference range extracted correctly ('< 200 Desirable')")


def test_greater_than_reference_range():
    """Requirement 5: Test '>' reference range."""
    snippet = """
    HDL Cholesterol
    48
    mg/dL
    > 50 Optimal
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)
    assert len(records) == 1
    assert records[0]["test_name"] == "HDL Cholesterol"
    assert "> 50" in records[0]["reference_range"]
    print("  [PASS] 5. '>' reference range extracted correctly ('> 50 Optimal')")


def test_unit_extraction():
    """Requirement 6: Test multiple clinical measurement units."""
    snippet = """
    Mean Corpuscular Volume (MCV)
    89.1
    fL
    80.0 - 100.0
    MCH
    29.1
    pg
    27.0 - 33.0
    Packed Cell Volume (PCV)
    35.2
    %
    36.0 - 46.0
    Cholesterol / HDL Ratio
    4.48
    Ratio
    3.3 - 4.4
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)
    assert len(records) == 4
    units = [r["unit"] for r in records]
    assert units == ["fL", "pg", "%", "Ratio"]
    print("  [PASS] 6. Units extracted correctly: fL, pg, %, Ratio")


def test_metadata_exclusion():
    """Requirement 8: Ensure non-test metadata numbers are excluded from test records."""
    snippet = """
    HEALTHLENS DIAGNOSTIC LABORATORY
    Patient Name: Sarah Jenkins
    Age / Gender: 35 Y / Female
    Patient ID: HL-2024-5501
    Sample ID: SMP-77102
    Collection Date: 10-Feb-2024 08:15 AM
    Reporting Date: 10-Feb-2024 01:30 PM
    Page 1 of 2
    
    DEPARTMENT OF HEMATOLOGY: COMPLETE BLOOD COUNT (CBC)
    Investigation / Test
    Observed Value
    Unit
    Biological Reference Range
    Hemoglobin (Hb)
    11.5
    g/dL
    12.0 - 15.5
    
    CLINICAL IMPRESSIONS & LAB NOTES
    - Borderline elevation in Total Cholesterol and LDL.
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(snippet)

    # Only Hemoglobin should be extracted
    assert len(records) == 1, f"Expected 1 test record, but got {len(records)}"
    assert records[0]["test_name"] == "Hemoglobin (Hb)"

    # Verify no PII or demographic numbers are extracted
    extracted_names = [r["test_name"] for r in records]
    assert "Age" not in extracted_names
    assert "Sarah Jenkins" not in str(records)
    assert "HL-2024-5501" not in str(records)
    assert "SMP-77102" not in str(records)
    print("  [PASS] 8. Patient metadata & non-test numbers correctly excluded (no PII leaked)")


def test_full_report_integration():
    """Requirement 7: Test full multi-test extraction from sample PDF."""
    pdf_path = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"
    assert pdf_path.exists(), "Sample PDF not found"

    # Step 1: Ingest PDF (Stage 2)
    ingestion_result = extract_text_from_pdf(pdf_path)
    raw_text = ingestion_result["combined_text"]

    # Step 2: Extract structured tests (Stage 3)
    records, df, json_path = extract_medical_tests(
        raw_text=raw_text,
        save_to_json=True,
        output_filename="sample_medical_report_structured.json",
        source_file=pdf_path.name
    )

    # Verifications
    assert len(records) >= 15, f"Expected at least 15 tests, got {len(records)}"
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == len(records)
    assert "test_name" in df.columns
    assert "value" in df.columns
    assert "unit" in df.columns
    assert "reference_range" in df.columns
    assert "category" in df.columns

    # Verify JSON file
    assert json_path is not None and json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "metadata" in data
    assert "tests" in data
    assert data["metadata"]["total_tests_extracted"] == len(records)
    assert data["metadata"]["source_file"] == "sample_medical_report.pdf"

    print(f"  [PASS] 7. Full report integration: Extracted {len(records)} tests across categories: {data['metadata']['categories']}")
    print(f"  [PASS] 9. JSON integration file verified at: {json_path}")
    print(f"  [PASS] 10. pandas DataFrame verified with shape {df.shape}")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens Stage 3: Medical Test Extractor Verification Suite")
    print("=" * 70)
    test_decimal_value_extraction()
    test_integer_with_comma()
    test_reference_range()
    test_less_than_reference_range()
    test_greater_than_reference_range()
    test_unit_extraction()
    test_metadata_exclusion()
    test_full_report_integration()
    print("=" * 70)
    print("  ALL STAGE 3 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
