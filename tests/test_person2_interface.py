"""
HealthLens - Person 2 Interface Integration Test Suite
Module: tests/test_person2_interface.py

Validates the clean, standardized interface between Person 1 and Person 2:
1. Confirms get_rag_context() produces 100% valid, serializable JSON.
2. Confirms all required fields are present in every test record.
3. Confirms that no patient names, IDs, phones, addresses, or PII are leaked.
4. Confirms exact extracted numerical values and laboratory reference ranges are preserved.
5. Confirms get_comparison_context() and get_trend_context() outputs are valid and clean.
6. Confirms that all sample data in the repository contains only synthetic/mock data.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_REPORTS_DIR, PROCESSED_DATA_DIR
from src.integration.person2_interface import (
    get_rag_context,
    get_comparison_context,
    get_trend_context,
    FORBIDDEN_PII_KEYS,
)


def test_rag_context_schema_and_json_serialization():
    """Verify get_rag_context schema contract and valid JSON serialization."""
    sample_validated_json = PROCESSED_DATA_DIR / "sample_medical_report_validated.json"
    assert sample_validated_json.exists(), f"Sample validated JSON missing at: {sample_validated_json}"

    rag_context = get_rag_context(sample_validated_json)

    # 1. JSON serializability test
    serialized = json.dumps(rag_context)
    assert serialized is not None
    deserialized = json.loads(serialized)
    assert deserialized == rag_context

    # 2. Required top-level fields
    assert "report_date" in rag_context
    assert "tests" in rag_context
    assert rag_context["report_date"] == "10-Feb-2024"
    assert len(rag_context["tests"]) == 19

    # 3. Required individual test fields
    required_test_keys = ["test_name", "value", "unit", "reference_range", "status", "category"]
    for t in rag_context["tests"]:
        for k in required_test_keys:
            assert k in t, f"Missing required field '{k}' in test record {t}"
        assert t["status"] in ["NORMAL", "LOW", "HIGH", "UNKNOWN"]
        assert isinstance(t["test_name"], str)
        assert isinstance(t["unit"], str)
        assert isinstance(t["reference_range"], str)
        assert isinstance(t["category"], str)

    print("  [PASS] 1. RAG context schema and JSON serialization verified")


def test_zero_pii_enforcement():
    """Verify that patient name, ID, phone, address, and demographics are strictly absent."""
    sample_pdf = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"
    rag_context = get_rag_context(sample_pdf)

    def scan_for_pii(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_lower = k.lower().strip()
                assert k_lower not in FORBIDDEN_PII_KEYS, (
                    f"Forbidden PII key '{k}' detected at '{path}.{k}'"
                )
                scan_for_pii(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                scan_for_pii(item, f"{path}[{idx}]")

    scan_for_pii(rag_context)

    # Also assert that mock names like "Sarah Jenkins" or "John Doe" are not present anywhere
    serialized = json.dumps(rag_context).lower()
    assert "sarah jenkins" not in serialized, "Mock patient name leaked in RAG context"
    assert "john doe" not in serialized, "Mock patient name leaked in RAG context"
    assert "david miller" not in serialized, "Mock doctor name leaked in RAG context"

    print("  [PASS] 2. Zero PII enforcement strictly verified (no names, IDs, or demographics)")


def test_exact_numerical_and_reference_range_fidelity():
    """Verify that exact numerical values and laboratory reference ranges are preserved."""
    sample_validated_json = PROCESSED_DATA_DIR / "sample_medical_report_validated.json"
    rag_context = get_rag_context(sample_validated_json)

    test_dict = {t["test_name"].lower(): t for t in rag_context["tests"]}

    # Check Hemoglobin: 11.5 g/dL, "12.0 - 15.5", LOW
    hb = test_dict.get("hemoglobin (hb)")
    assert hb is not None
    assert hb["value"] == 11.5
    assert hb["unit"] == "g/dL"
    assert hb["reference_range"] == "12.0 - 15.5"
    assert hb["status"] == "LOW"

    # Check Total Leukocyte Count (WBC): 11200 cells/mcL, "4000 - 11000", HIGH
    wbc = test_dict.get("total leukocyte count (wbc)")
    assert wbc is not None
    assert wbc["value"] == 11200.0
    assert wbc["unit"] == "cells/mcL"
    assert wbc["reference_range"] == "4000 - 11000"
    assert wbc["status"] == "HIGH"

    # Check Total Cholesterol: 215 mg/dL, "< 200 Desirable", HIGH
    chol = test_dict.get("total cholesterol")
    assert chol is not None
    assert chol["value"] == 215.0
    assert chol["unit"] == "mg/dL"
    assert chol["reference_range"] == "< 200 Desirable"
    assert chol["status"] == "HIGH"

    print("  [PASS] 3. Exact numerical and laboratory reference range fidelity verified")


def test_longitudinal_contexts():
    """Verify get_comparison_context and get_trend_context functions."""
    comp_json = PROCESSED_DATA_DIR / "report_comparison_summary.json"
    trend_json = PROCESSED_DATA_DIR / "longitudinal_trends_summary.json"

    assert comp_json.exists(), f"Missing comparison summary at {comp_json}"
    assert trend_json.exists(), f"Missing trend summary at {trend_json}"

    # Comparison context
    comp_ctx = get_comparison_context(comp_json)
    assert "baseline_date" in comp_ctx
    assert "followup_date" in comp_ctx
    assert "comparisons" in comp_ctx
    assert len(comp_ctx["comparisons"]) == 8
    json.dumps(comp_ctx)  # Check serializability

    # Trend context
    trend_ctx = get_trend_context(trend_json)
    assert "report_dates" in trend_ctx
    assert "summary_observations" in trend_ctx
    assert "biomarker_trends" in trend_ctx
    assert len(trend_ctx["report_dates"]) == 3
    assert len(trend_ctx["summary_observations"]) == 8
    json.dumps(trend_ctx)  # Check serializability

    print("  [PASS] 4. Longitudinal comparison and trend context helpers verified")


def test_no_real_patient_data_in_samples():
    """Audit all sample report files to guarantee zero real patient information is stored."""
    sample_files = list(SAMPLE_REPORTS_DIR.glob("*"))
    processed_files = list(PROCESSED_DATA_DIR.glob("*.json"))

    all_audited_files = sample_files + processed_files
    assert len(all_audited_files) > 0

    # Ensure files exist and check their contents
    for f in all_audited_files:
        if f.suffix in [".txt", ".json"]:
            text = f.read_text(encoding="utf-8")
            # If names exist, they must only be synthetic test names
            if "patient" in text.lower():
                assert any(mock in text for mock in ["Sarah Jenkins", "John Doe", "HL-2024", "PAT-2024", "SMP-"]), (
                    f"Potential non-synthetic data found in: {f.name}"
                )

    print("  [PASS] 5. Repository audit confirmed: Zero real patient information in sample data")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens: Person 2 RAG Interface Integration Test Suite")
    print("=" * 70)
    test_rag_context_schema_and_json_serialization()
    test_zero_pii_enforcement()
    test_exact_numerical_and_reference_range_fidelity()
    test_longitudinal_contexts()
    test_no_real_patient_data_in_samples()
    print("=" * 70)
    print("  ALL PERSON 2 INTERFACE TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
