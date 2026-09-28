"""
Unit and integration test suite for Medical Test Data Validation (Stage 4).
Tests:
1. Normal range classification (value between lower and upper bound -> NORMAL)
2. Low value classification (value < lower_bound -> LOW)
3. High value classification (value > upper_bound -> HIGH)
4. '<' threshold classification (value < threshold -> NORMAL, value >= threshold -> HIGH)
5. '>' threshold classification (value > threshold -> NORMAL, value <= threshold -> LOW)
6. Invalid / unparseable reference range -> UNKNOWN status and None bounds
7. Full integration validation of sample_medical_report_structured.json
8. Verification of sample_medical_report_validated.json output
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.validator import MedicalTestValidator, validate_medical_tests
from src.config import PROCESSED_DATA_DIR


def test_normal_range():
    """Test value within standard biological range -> NORMAL."""
    validator = MedicalTestValidator()
    rec = {
        "test_name": "Hemoglobin (Hb)",
        "value": 14.2,
        "unit": "g/dL",
        "reference_range": "12.0 - 15.5",
    }
    result = validator.validate_record(rec)
    assert result["status"] == "NORMAL"
    assert result["lower_bound"] == 12.0
    assert result["upper_bound"] == 15.5
    assert result["reference_type"] == "RANGE"
    print("  [PASS] 1. Normal range: 14.2 within [12.0 - 15.5] -> NORMAL")


def test_low_value():
    """Test value below lower bound -> LOW."""
    validator = MedicalTestValidator()
    rec = {
        "test_name": "Hemoglobin (Hb)",
        "value": 11.5,
        "unit": "g/dL",
        "reference_range": "12.0 - 15.5",
    }
    result = validator.validate_record(rec)
    assert result["status"] == "LOW"
    assert result["lower_bound"] == 12.0
    assert result["upper_bound"] == 15.5
    print("  [PASS] 2. Low value: 11.5 below 12.0 -> LOW")


def test_high_value():
    """Test value above upper bound -> HIGH."""
    validator = MedicalTestValidator()
    rec = {
        "test_name": "Total Leukocyte Count (WBC)",
        "value": 11200,
        "unit": "cells/mcL",
        "reference_range": "4000 - 11000",
    }
    result = validator.validate_record(rec)
    assert result["status"] == "HIGH"
    assert result["lower_bound"] == 4000.0
    assert result["upper_bound"] == 11000.0
    print("  [PASS] 3. High value: 11,200 above 11,000 -> HIGH")


def test_less_than_range():
    """Test '<' threshold range for both NORMAL (< threshold) and HIGH (>= threshold)."""
    validator = MedicalTestValidator()

    # Case A: value < threshold -> NORMAL
    rec_normal = {
        "test_name": "Total Cholesterol",
        "value": 180,
        "unit": "mg/dL",
        "reference_range": "< 200 Desirable",
    }
    res_normal = validator.validate_record(rec_normal)
    assert res_normal["status"] == "NORMAL"
    assert res_normal["upper_bound"] == 200.0
    assert res_normal["reference_type"] == "LESS_THAN"

    # Case B: value >= threshold -> HIGH
    rec_high = {
        "test_name": "Total Cholesterol",
        "value": 215,
        "unit": "mg/dL",
        "reference_range": "< 200 Desirable",
    }
    res_high = validator.validate_record(rec_high)
    assert res_high["status"] == "HIGH"
    assert res_high["upper_bound"] == 200.0
    print("  [PASS] 4. '<' threshold: 180 < 200 -> NORMAL, 215 >= 200 -> HIGH")


def test_greater_than_range():
    """Test '>' threshold range for both NORMAL (> threshold) and LOW (<= threshold)."""
    validator = MedicalTestValidator()

    # Case A: value > threshold -> NORMAL
    rec_normal = {
        "test_name": "HDL Cholesterol",
        "value": 58,
        "unit": "mg/dL",
        "reference_range": "> 50 Optimal",
    }
    res_normal = validator.validate_record(rec_normal)
    assert res_normal["status"] == "NORMAL"
    assert res_normal["lower_bound"] == 50.0
    assert res_normal["reference_type"] == "GREATER_THAN"

    # Case B: value <= threshold -> LOW
    rec_low = {
        "test_name": "HDL Cholesterol",
        "value": 48,
        "unit": "mg/dL",
        "reference_range": "> 50 Optimal",
    }
    res_low = validator.validate_record(rec_low)
    assert res_low["status"] == "LOW"
    assert res_low["lower_bound"] == 50.0
    print("  [PASS] 5. '>' threshold: 58 > 50 -> NORMAL, 48 <= 50 -> LOW")


def test_invalid_or_unknown_range():
    """Test unparseable range strings return UNKNOWN without guessing."""
    validator = MedicalTestValidator()
    rec = {
        "test_name": "Blood Culture",
        "value": 0,
        "unit": "N/A",
        "reference_range": "Negative / No growth",
    }
    res = validator.validate_record(rec)
    assert res["status"] == "UNKNOWN"
    assert res["lower_bound"] is None
    assert res["upper_bound"] is None
    assert res["reference_type"] == "UNKNOWN"
    print("  [PASS] 6. Invalid/Qualitative range ('Negative / No growth') -> UNKNOWN")


def test_multiple_tests_full_report():
    """Test full integration on Stage 3 structured JSON file."""
    json_path = PROCESSED_DATA_DIR / "sample_medical_report_structured.json"
    assert json_path.exists(), f"Stage 3 output missing at: {json_path}"

    validated_records, df, output_path = validate_medical_tests(
        input_data=json_path,
        save_to_json=True,
        output_filename="sample_medical_report_validated.json"
    )

    assert len(validated_records) == 19
    assert output_path.exists()

    # Load and verify JSON file structure
    with open(output_path, "r", encoding="utf-8") as f:
        validated_json = json.load(f)

    meta = validated_json["metadata"]
    counts = meta["status_summary"]

    assert counts["NORMAL"] > 0, "Expected some NORMAL findings"
    assert counts["LOW"] > 0, "Expected some LOW findings (e.g. Hemoglobin)"
    assert counts["HIGH"] > 0, "Expected some HIGH findings (e.g. Total Cholesterol, WBC)"
    assert counts["UNKNOWN"] == 0, "All 19 tests in synthetic report should be recognized"

    # Specific tests verification
    test_lookup = {r["test_name"]: r for r in validated_records}
    assert test_lookup["Hemoglobin (Hb)"]["status"] == "LOW"
    assert test_lookup["Packed Cell Volume (PCV)"]["status"] == "LOW"
    assert test_lookup["Total Leukocyte Count (WBC)"]["status"] == "HIGH"
    assert test_lookup["Neutrophils"]["status"] == "HIGH"
    assert test_lookup["Total Cholesterol"]["status"] == "HIGH"
    assert test_lookup["Triglycerides"]["status"] == "HIGH"
    assert test_lookup["HDL Cholesterol"]["status"] == "LOW"
    assert test_lookup["Platelet Count"]["status"] == "NORMAL"

    # Verify original fields preserved
    for r in validated_records:
        assert "test_name" in r
        assert "value" in r
        assert "unit" in r
        assert "reference_range" in r
        assert "category" in r
        assert "report_date" in r
        assert "status" in r
        assert "lower_bound" in r
        assert "upper_bound" in r
        assert "reference_type" in r

    print(f"  [PASS] 7. Full report integration: 19 tests validated ({counts['NORMAL']} NORMAL, {counts['LOW']} LOW, {counts['HIGH']} HIGH)")
    print(f"  [PASS] 8. Validated JSON successfully saved to: {output_path}")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens Stage 4: Medical Test Validator Verification Suite")
    print("=" * 70)
    test_normal_range()
    test_low_value()
    test_high_value()
    test_less_than_range()
    test_greater_than_range()
    test_invalid_or_unknown_range()
    test_multiple_tests_full_report()
    print("=" * 70)
    print("  ALL STAGE 4 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
