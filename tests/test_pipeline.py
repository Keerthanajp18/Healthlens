"""
HealthLens - Person 1 Unified Pipeline Integration Test Suite
Module: tests/test_pipeline.py

Verifies the complete Person 1 pipeline integration:
Input Medical Document (PDF)
  ↓
Multimodal Ingestion (File type detection & digital text extraction)
  ↓
Deterministic Parsing (Medical test extraction without PII)
  ↓
Validation (Deterministic laboratory reference interval evaluation)
  ↓
Standardized Ground-Truth JSON Output (Contract for Person 2 RAG)
  ↓
Optional Longitudinal Comparison & Trend Analysis
"""

import sys
import tempfile
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_REPORTS_DIR, PROCESSED_DATA_DIR, CHARTS_DIR
from src.pipeline import (
    MedicalDataPipeline,
    process_medical_report,
    compare_reports,
    analyze_trends,
)


def test_pdf_end_to_end_pipeline():
    """
    Test full end-to-end pipeline:
    PDF -> extraction -> structured records -> validation -> final JSON
    """
    sample_pdf = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"
    assert sample_pdf.exists(), f"Sample PDF missing at: {sample_pdf}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_output = Path(tmp_dir) / "test_pipeline_output.json"

        # Execute unified pipeline function
        result = process_medical_report(
            file_path=sample_pdf,
            save_to_json=True,
            output_filename=tmp_output,
        )

        # 1. Verify Top-Level Structure
        assert "metadata" in result, "Missing 'metadata' key in pipeline result"
        assert "tests" in result, "Missing 'tests' key in pipeline result"
        assert "processing" in result, "Missing 'processing' key in pipeline result"

        # 2. Verify Ingestion & Processing Details
        proc = result["processing"]
        assert proc["source_type"] == "digital_pdf", f"Expected 'digital_pdf', got {proc['source_type']}"
        assert proc["ocr_used"] is False, "Digital PDF should not invoke OCR"
        assert proc["total_pages"] == 2
        assert proc["characters_extracted"] > 1000
        assert proc["saved_json_path"] == str(tmp_output)

        # 3. Verify Document Metadata
        meta = result["metadata"]
        assert meta["source_file"] == "sample_medical_report.pdf"
        assert meta["report_date"] == "10-Feb-2024"
        assert meta["total_tests"] == 19
        assert "Complete Blood Count (CBC)" in meta["categories"]
        assert "Lipid Profile" in meta["categories"]

        # 4. Verify Validation Summary Counts
        summary = meta["status_summary"]
        assert summary["NORMAL"] == 9
        assert summary["LOW"] == 3
        assert summary["HIGH"] == 7
        assert summary["UNKNOWN"] == 0

        # 5. Verify Test Records Schema Contract
        tests = result["tests"]
        assert len(tests) == 19

        test_map = {t["test_name"].lower(): t for t in tests}

        # Check Hemoglobin (Hb): 11.5 g/dL in [12.0 - 15.5] -> LOW
        hb = test_map.get("hemoglobin (hb)")
        assert hb is not None
        assert hb["value"] == 11.5
        assert hb["unit"] == "g/dL"
        assert hb["reference_range"] == "12.0 - 15.5"
        assert hb["category"] == "Complete Blood Count (CBC)"
        assert hb["report_date"] == "10-Feb-2024"
        assert hb["status"] == "LOW"

        # Check Total Leukocyte Count (WBC): 11,200 cells/mcL in [4000 - 11000] -> HIGH
        wbc = test_map.get("total leukocyte count (wbc)")
        assert wbc is not None
        assert wbc["value"] == 11200.0
        assert wbc["unit"] == "cells/mcL"
        assert wbc["status"] == "HIGH"

        # Check Total Cholesterol: 215 mg/dL (< 200) -> HIGH
        chol = test_map.get("total cholesterol")
        assert chol is not None
        assert chol["value"] == 215.0
        assert chol["status"] == "HIGH"

        # Check HDL Cholesterol: 48 mg/dL (> 50 Optimal) -> LOW
        hdl = test_map.get("hdl cholesterol")
        assert hdl is not None
        assert hdl["value"] == 48
        assert hdl["status"] == "LOW"

        # 6. Verify Persisted Final JSON
        assert tmp_output.exists(), "Final JSON was not saved to disk"
        with open(tmp_output, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        assert disk_data["metadata"]["total_tests"] == 19
        assert len(disk_data["tests"]) == 19

    print("  [PASS] 1. End-to-end PDF -> Extraction -> Validation -> JSON verified")


def test_pipeline_zero_pii_contract():
    """Verify that patient PII is completely excluded from the standardized JSON."""
    sample_pdf = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"
    result = process_medical_report(sample_pdf, save_to_json=False)

    disallowed_keys = [
        "patient_name", "patient_id", "patient_age", "patient_gender",
        "physician_name", "doctor_name", "phone", "email", "address",
        "ssn", "mrn", "dob"
    ]

    # Check top-level metadata
    for key in disallowed_keys:
        assert key not in result["metadata"], f"Leaked PII key '{key}' in metadata"

    # Check individual test records
    for t in result["tests"]:
        for key in disallowed_keys:
            assert key not in t, f"Leaked PII key '{key}' in test record {t}"

        # Ensure required keys exist
        assert "test_name" in t
        assert "value" in t
        assert "unit" in t
        assert "reference_range" in t
        assert "category" in t
        assert "report_date" in t
        assert "status" in t
        assert t["status"] in ["NORMAL", "LOW", "HIGH", "UNKNOWN"]

    print("  [PASS] 2. Zero PII contract and required test schema verified")


def test_pipeline_comparison_and_trends():
    """Verify comparison and trend analysis orchestration through pipeline functions."""
    rep_jan = PROCESSED_DATA_DIR / "sample_patient_report_2024_01_validated.json"
    rep_apr = PROCESSED_DATA_DIR / "sample_patient_report_2024_04_validated.json"
    rep_jul = PROCESSED_DATA_DIR / "sample_patient_report_2024_07_validated.json"

    assert rep_jan.exists() and rep_apr.exists() and rep_jul.exists()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_charts = Path(tmp_dir) / "charts"
        tmp_comp_json = Path(tmp_dir) / "comp.json"
        tmp_trend_json = Path(tmp_dir) / "trends.json"

        # 1. Compare two reports through pipeline orchestrator
        comp = compare_reports(
            report_a=rep_jan,
            report_b=rep_apr,
            save_to_json=True,
            output_filename=tmp_comp_json,
        )
        assert "comparisons" in comp
        assert tmp_comp_json.exists()

        # 2. Analyze trends across 3 reports with chart generation
        trends = analyze_trends(
            reports=[rep_jan, rep_apr, rep_jul],
            generate_charts=True,
            save_to_json=True,
            output_filename=tmp_trend_json,
            charts_dir=tmp_charts,
        )
        assert "test_trends" in trends
        assert "charts_generated" in trends
        assert len(trends["charts_generated"]) >= 7
        assert tmp_trend_json.exists()

        for c_file in trends["charts_generated"]:
            assert Path(c_file).exists()
            assert Path(c_file).stat().st_size > 1000

    print("  [PASS] 3. Pipeline compare_reports and analyze_trends orchestration verified")


def test_pipeline_error_handling():
    """Verify non-silent, clear error handling for invalid or missing inputs."""
    pipeline = MedicalDataPipeline()

    # Missing file
    try:
        pipeline.process_medical_report("non_existent_report.pdf")
        assert False, "Failed to raise FileNotFoundError for missing report"
    except FileNotFoundError:
        pass

    # Unsupported file extension
    with tempfile.NamedTemporaryFile(suffix=".docx") as tmp:
        try:
            pipeline.process_medical_report(tmp.name)
            assert False, "Failed to raise ValueError for unsupported format"
        except ValueError:
            pass

    print("  [PASS] 4. Safe error handling for missing and unsupported files verified")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens: Person 1 Unified Pipeline Integration Test Suite")
    print("=" * 70)
    test_pdf_end_to_end_pipeline()
    test_pipeline_zero_pii_contract()
    test_pipeline_comparison_and_trends()
    test_pipeline_error_handling()
    print("=" * 70)
    print("  ALL PIPELINE INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
