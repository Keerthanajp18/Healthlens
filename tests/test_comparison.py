"""
Unit and integration test suite for Medical Report Comparison (Stage 6).
Tests:
1. Increase calculation (current > previous -> positive absolute and percentage change)
2. Decrease calculation (current < previous -> negative absolute and percentage change)
3. Unchanged value calculation (current == previous -> delta is zero)
4. Missing test handling (tests present only in first or only in second report)
5. Zero previous value handling (prevents division by zero, returns None for percentage change)
6. Unit mismatch handling (prevents arithmetic on incompatible units)
7. Chronological ordering (sorts reports chronologically regardless of argument order)
8. Multi-test comparison on validated sample reports with JSON export
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR
from src.comparison.report_comparator import MedicalReportComparator, compare_medical_reports


def test_increase_calculation():
    """Test 1: Increase in observed value."""
    comparator = MedicalReportComparator()
    rep_a = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [{"test_name": "Hemoglobin (Hb)", "value": 10.2, "unit": "g/dL", "status": "LOW"}]
    }
    rep_b = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [{"test_name": "Hemoglobin (Hb)", "value": 11.5, "unit": "g/dL", "status": "LOW"}]
    }
    res = comparator.compare_two_reports(rep_a, rep_b)
    item = res["comparisons"][0]
    assert item["test_name"] == "Hemoglobin (Hb)"
    assert item["previous_value"] == 10.2
    assert item["current_value"] == 11.5
    assert item["absolute_change"] == 1.3
    assert item["percentage_change"] == 12.75
    assert item["change_direction"] == "INCREASE"
    assert item["status_previous"] == "LOW"
    assert item["status_current"] == "LOW"
    print("  [PASS] 1. Value increase: 10.2 -> 11.5 (abs: +1.3, pct: +12.75%, dir: INCREASE)")


def test_decrease_calculation():
    """Test 2: Decrease in observed value."""
    comparator = MedicalReportComparator()
    rep_a = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [{"test_name": "Total Leukocyte Count (WBC)", "value": 11800, "unit": "cells/mcL", "status": "HIGH"}]
    }
    rep_b = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [{"test_name": "Total Leukocyte Count (WBC)", "value": 9400, "unit": "cells/mcL", "status": "NORMAL"}]
    }
    res = comparator.compare_two_reports(rep_a, rep_b)
    item = res["comparisons"][0]
    assert item["absolute_change"] == -2400.0
    assert item["percentage_change"] == -20.34
    assert item["change_direction"] == "DECREASE"
    assert item["status_previous"] == "HIGH"
    assert item["status_current"] == "NORMAL"
    print("  [PASS] 2. Value decrease: 11,800 -> 9,400 (abs: -2400.0, pct: -20.34%, dir: DECREASE)")


def test_unchanged_value():
    """Test 3: Unchanged observed value."""
    comparator = MedicalReportComparator()
    rep_a = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [{"test_name": "Platelet Count", "value": 250000, "unit": "cells/mcL", "status": "NORMAL"}]
    }
    rep_b = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [{"test_name": "Platelet Count", "value": 250000, "unit": "cells/mcL", "status": "NORMAL"}]
    }
    res = comparator.compare_two_reports(rep_a, rep_b)
    item = res["comparisons"][0]
    assert item["absolute_change"] == 0.0
    assert item["percentage_change"] == 0.0
    assert item["change_direction"] == "UNCHANGED"
    print("  [PASS] 3. Unchanged value: 250,000 -> 250,000 (abs: 0.0, pct: 0.0%, dir: UNCHANGED)")


def test_missing_test_handling():
    """Test 4: Tests present only in first or second report."""
    comparator = MedicalReportComparator()
    rep_a = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [
            {"test_name": "Fasting Blood Sugar", "value": 92, "unit": "mg/dL", "status": "NORMAL"},
            {"test_name": "Hemoglobin", "value": 12.0, "unit": "g/dL", "status": "NORMAL"}
        ]
    }
    rep_b = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [
            {"test_name": "Hemoglobin", "value": 12.5, "unit": "g/dL", "status": "NORMAL"},
            {"test_name": "HbA1c", "value": 5.4, "unit": "%", "status": "NORMAL"}
        ]
    }
    res = comparator.compare_two_reports(rep_a, rep_b)
    items = {c["test_name"]: c for c in res["comparisons"]}

    assert items["Fasting Blood Sugar"]["change_direction"] == "ONLY_IN_PREVIOUS"
    assert items["Fasting Blood Sugar"]["current_value"] is None
    assert items["Fasting Blood Sugar"]["absolute_change"] is None

    assert items["HbA1c"]["change_direction"] == "ONLY_IN_CURRENT"
    assert items["HbA1c"]["previous_value"] is None
    assert items["HbA1c"]["absolute_change"] is None

    print("  [PASS] 4. Missing tests handled: ONLY_IN_PREVIOUS and ONLY_IN_CURRENT detected")


def test_zero_previous_value():
    """Test 5: Handle previous value of zero without ZeroDivisionError."""
    comparator = MedicalReportComparator()
    rep_a = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [{"test_name": "Basophils", "value": 0, "unit": "%", "status": "NORMAL"}]
    }
    rep_b = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [{"test_name": "Basophils", "value": 1, "unit": "%", "status": "NORMAL"}]
    }
    res = comparator.compare_two_reports(rep_a, rep_b)
    item = res["comparisons"][0]
    assert item["absolute_change"] == 1.0
    assert item["percentage_change"] is None, "Percentage change must be None when previous value is 0"
    print("  [PASS] 5. Zero previous value: 0 -> 1 (abs: +1.0, pct: None without division-by-zero)")


def test_unit_mismatch():
    """Test 6: Incompatible units prevent invalid delta calculations."""
    comparator = MedicalReportComparator()
    rep_a = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [{"test_name": "Glucose", "value": 100, "unit": "mg/dL", "status": "NORMAL"}]
    }
    rep_b = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [{"test_name": "Glucose", "value": 5.5, "unit": "mmol/L", "status": "NORMAL"}]
    }
    res = comparator.compare_two_reports(rep_a, rep_b)
    item = res["comparisons"][0]
    assert item["change_direction"] == "UNIT_MISMATCH"
    assert item["absolute_change"] is None
    assert item["percentage_change"] is None
    print("  [PASS] 6. Unit mismatch detected: 'mg/dL' vs 'mmol/L' (deltas suppressed)")


def test_chronological_ordering():
    """Test 7: Reports provided in reverse chronological order are auto-sorted."""
    comparator = MedicalReportComparator()
    rep_april = {
        "metadata": {"report_date": "20-Apr-2024"},
        "tests": [{"test_name": "Hemoglobin", "value": 11.5, "unit": "g/dL", "status": "LOW"}]
    }
    rep_jan = {
        "metadata": {"report_date": "15-Jan-2024"},
        "tests": [{"test_name": "Hemoglobin", "value": 10.2, "unit": "g/dL", "status": "LOW"}]
    }
    # Pass April first, Jan second
    res = comparator.compare_two_reports(rep_april, rep_jan)
    item = res["comparisons"][0]
    assert item["previous_date"] == "15-Jan-2024"
    assert item["current_date"] == "20-Apr-2024"
    assert item["previous_value"] == 10.2
    assert item["current_value"] == 11.5
    assert item["absolute_change"] == 1.3
    print("  [PASS] 7. Chronological auto-sorting: Jan 15 correctly assigned previous, Apr 20 current")


def test_full_sample_reports_comparison():
    """Test 8: Full integration test comparing two longitudinal validated JSON files."""
    path_1 = PROCESSED_DATA_DIR / "sample_patient_report_2024_01_validated.json"
    path_2 = PROCESSED_DATA_DIR / "sample_patient_report_2024_04_validated.json"
    assert path_1.exists() and path_2.exists(), "Sample comparison files missing"

    result, df, output_path = compare_medical_reports(
        report_a=path_1,
        report_b=path_2,
        save_to_json=True,
        output_filename="report_comparison_summary.json"
    )

    meta = result["metadata"]
    assert meta["total_comparable_tests"] == 6
    assert meta["total_increased"] == 2  # Hemoglobin, RBC Count
    assert meta["total_decreased"] == 3  # WBC, Total Cholesterol, Triglycerides
    assert meta["total_unchanged"] == 1  # Platelet Count
    assert meta["only_in_previous"] == 1 # Fasting Blood Sugar
    assert meta["only_in_current"] == 1  # HbA1c
    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        saved_json = json.load(f)
    assert len(saved_json["comparisons"]) == 8

    print(f"  [PASS] 8. Full report comparison: 8 tests analyzed (2 increased, 3 decreased, 1 unchanged, 2 unshared)")
    print(f"  [PASS] Saved comparison summary to: {output_path}")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens Stage 6: Medical Report Comparator Verification Suite")
    print("=" * 70)
    test_increase_calculation()
    test_decrease_calculation()
    test_unchanged_value()
    test_missing_test_handling()
    test_zero_previous_value()
    test_unit_mismatch()
    test_chronological_ordering()
    test_full_sample_reports_comparison()
    print("=" * 70)
    print("  ALL STAGE 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
