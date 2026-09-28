"""
HealthLens - Stage 7 Test Suite: Trend Analysis & Visualization
Module: tests/test_visualizer.py

Comprehensive test suite verifying:
1. Chronological ordering (reports provided in scrambled/reverse order are sorted correctly)
2. Multi-report trend analysis and trend DataFrame schema
3. Missing values, unshared tests, and missing dates safely handled
4. Chart generation (single-panel Matplotlib line charts saved to disk with non-zero size)
5. Incompatible units protection (strictly prevents comparing or connecting points across differing units)
6. Non-clinical numerical summaries (strictly zero medical diagnosis or subjective claims)
7. Full pipeline integration with sample patient reports
"""

import sys
import tempfile
from pathlib import Path
import json

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, CHARTS_DIR
from src.comparison.trend_analyzer import (
    MedicalTrendAnalyzer,
    analyze_medical_trends,
)
from src.comparison.visualizer import (
    MedicalTrendVisualizer,
    visualize_medical_trends,
    sanitize_filename,
)


def get_sample_reports():
    """Generates 3 synthetic longitudinal reports for testing."""
    r1 = {
        "metadata": {
            "source_file": "report_jan.pdf",
            "report_date": "15-Jan-2024",
        },
        "tests": [
            {
                "test_name": "Hemoglobin (Hb)",
                "value": 10.2,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "report_date": "15-Jan-2024",
                "status": "LOW",
                "lower_bound": 12.0,
                "upper_bound": 15.5,
            },
            {
                "test_name": "Total Cholesterol",
                "value": 230,
                "unit": "mg/dL",
                "reference_range": "< 200",
                "report_date": "15-Jan-2024",
                "status": "HIGH",
                "upper_bound": 200.0,
            }
        ]
    }

    r2 = {
        "metadata": {
            "source_file": "report_apr.pdf",
            "report_date": "20-Apr-2024",
        },
        "tests": [
            {
                "test_name": "Hemoglobin (Hb)",
                "value": 11.5,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "report_date": "20-Apr-2024",
                "status": "LOW",
                "lower_bound": 12.0,
                "upper_bound": 15.5,
            },
            {
                "test_name": "Total Cholesterol",
                "value": 210,
                "unit": "mg/dL",
                "reference_range": "< 200",
                "report_date": "20-Apr-2024",
                "status": "HIGH",
                "upper_bound": 200.0,
            },
            {
                "test_name": "HbA1c",
                "value": 5.9,
                "unit": "%",
                "reference_range": "< 5.7",
                "report_date": "20-Apr-2024",
                "status": "HIGH",
                "upper_bound": 5.7,
            }
        ]
    }

    r3 = {
        "metadata": {
            "source_file": "report_jul.pdf",
            "report_date": "15-Jul-2024",
        },
        "tests": [
            {
                "test_name": "Hemoglobin (Hb)",
                "value": 12.8,
                "unit": "g/dL",
                "reference_range": "12.0 - 15.5",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "lower_bound": 12.0,
                "upper_bound": 15.5,
            },
            {
                "test_name": "Total Cholesterol",
                "value": 195,
                "unit": "mg/dL",
                "reference_range": "< 200",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "upper_bound": 200.0,
            },
            {
                "test_name": "HbA1c",
                "value": 5.4,
                "unit": "%",
                "reference_range": "< 5.7",
                "report_date": "15-Jul-2024",
                "status": "NORMAL",
                "upper_bound": 5.7,
            }
        ]
    }

    return [r1, r2, r3]


def test_chronological_ordering():
    """Requirement: Sort values chronologically, even when input order is scrambled."""
    analyzer = MedicalTrendAnalyzer()
    r1, r2, r3 = get_sample_reports()

    # Provide reports out-of-order: July -> January -> April
    scrambled = [r3, r1, r2]
    aggregated = analyzer.load_and_aggregate_reports(scrambled)

    dates = [meta.get("report_date") for meta, _, _ in aggregated]
    assert dates == ["15-Jan-2024", "20-Apr-2024", "15-Jul-2024"], (
        f"Expected ['15-Jan-2024', '20-Apr-2024', '15-Jul-2024'], got {dates}"
    )

    df = analyzer.build_trend_dataframe(scrambled)
    hb_dates = df[df["test_name"] == "Hemoglobin (Hb)"]["date"].tolist()
    assert hb_dates == ["15-Jan-2024", "20-Apr-2024", "15-Jul-2024"], (
        f"Trend DataFrame not sorted chronologically: {hb_dates}"
    )
    print("  [PASS] 1. Chronological ordering verified across scrambled inputs")


def test_trend_dataframe_schema():
    """Requirement: Trend DataFrame containing: date, test_name, value, unit, status."""
    analyzer = MedicalTrendAnalyzer()
    reports = get_sample_reports()
    df = analyzer.build_trend_dataframe(reports)

    required_cols = ["date", "test_name", "value", "unit", "status"]
    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' missing from DataFrame"

    assert len(df) == 8  # 2 in r1 + 3 in r2 + 3 in r3
    assert df["value"].dtype in ["float64", "int64"]
    print("  [PASS] 2. Trend DataFrame schema verified (date, test_name, value, unit, status)")


def test_multiple_reports():
    """Requirement: Group records by test_name and analyze across multiple reports."""
    analyzer = MedicalTrendAnalyzer()
    reports = get_sample_reports()
    result = analyzer.analyze_trends(reports)

    meta = result["metadata"]
    assert meta["total_reports"] == 3
    assert meta["total_unique_tests"] == 3  # Hemoglobin, Total Cholesterol, HbA1c

    trends = result["test_trends"]
    assert "hemoglobin (hb)" in trends
    assert "total cholesterol" in trends
    assert "hba1c" in trends

    hb = trends["hemoglobin (hb)"]
    assert hb["first_value"] == 10.2
    assert hb["last_value"] == 12.8
    assert hb["min_value"] == 10.2
    assert hb["max_value"] == 12.8
    assert hb["absolute_change"] == round(12.8 - 10.2, 4)
    assert hb["direction"] == "INCREASE"
    assert hb["summary"] == "Hemoglobin (Hb) changed from 10.2 to 12.8."

    # HbA1c present only in report 2 and 3
    hba1c = trends["hba1c"]
    assert hba1c["total_observations"] == 2
    assert hba1c["first_value"] == 5.9
    assert hba1c["last_value"] == 5.4
    assert hba1c["direction"] == "DECREASE"
    assert hba1c["summary"] == "HbA1c changed from 5.9 to 5.4."
    print("  [PASS] 3. Multiple reports grouping & longitudinal metrics verified")


def test_missing_values_and_dates():
    """Requirement: Handle missing dates and missing values safely."""
    analyzer = MedicalTrendAnalyzer()

    # Report without metadata date
    r_undated = {
        "metadata": {},
        "tests": [
            {"test_name": "TestA", "value": None, "unit": "mg/dL", "status": "UNKNOWN"},
            {"test_name": "TestB", "value": "15.0", "unit": "g/dL", "status": "NORMAL"},
        ]
    }
    r_dated = {
        "metadata": {"report_date": "10-Oct-2024"},
        "tests": [
            {"test_name": "TestA", "value": 45.0, "unit": "mg/dL", "status": "NORMAL"},
            {"test_name": "TestB", "value": None, "unit": "g/dL", "status": "UNKNOWN"},
        ]
    }

    # Should not raise exception
    df = analyzer.build_trend_dataframe([r_undated, r_dated])
    assert not df.empty
    assert len(df) == 4

    result = analyzer.analyze_trends([r_undated, r_dated])
    test_a = result["test_trends"]["testa"]
    assert test_a["numeric_observations"] == 1
    assert test_a["direction"] == "SINGLE_OBSERVATION"

    test_b = result["test_trends"]["testb"]
    assert test_b["numeric_observations"] == 1
    assert test_b["first_value"] == 15.0

    print("  [PASS] 4. Missing dates & None values handled safely without exceptions")


def test_incompatible_units():
    """Requirement: Incompatible units - do not calculate deltas or connect."""
    analyzer = MedicalTrendAnalyzer()

    r1 = {
        "metadata": {"report_date": "01-Jan-2024"},
        "tests": [{"test_name": "Glucose", "value": 90.0, "unit": "mg/dL", "status": "NORMAL"}]
    }
    r2 = {
        "metadata": {"report_date": "01-Apr-2024"},
        "tests": [{"test_name": "Glucose", "value": 5.0, "unit": "mmol/L", "status": "NORMAL"}]
    }

    result = analyzer.analyze_trends([r1, r2])
    glucose = result["test_trends"]["glucose"]

    assert glucose["units_consistent"] is False
    assert glucose["direction"] == "UNIT_INCOMPATIBLE"
    assert glucose["absolute_change"] is None
    assert glucose["percentage_change"] is None
    assert "incompatible units" in glucose["summary"].lower()
    print("  [PASS] 5. Incompatible units properly flagged and numerical delta blocked")


def test_chart_generation():
    """Requirement: Generate line charts using Matplotlib and save to data/processed/charts/."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        visualizer = MedicalTrendVisualizer(output_dir=tmp_path)

        analyzer = MedicalTrendAnalyzer()
        reports = get_sample_reports()
        trend_results = analyzer.analyze_trends(reports)

        # 1. Single test chart
        hb_records = trend_results["test_trends"]["hemoglobin (hb)"]["records"]
        chart_path = visualizer.plot_test_trend("Hemoglobin (Hb)", hb_records)

        assert chart_path is not None
        assert chart_path.exists(), f"Chart file not created at {chart_path}"
        assert chart_path.stat().st_size > 1000, "Chart file is empty or suspiciously small"
        assert chart_path.suffix == ".png"

        # 2. Generate all charts (one per unique test)
        all_charts = visualizer.generate_all_charts(trend_results)
        assert len(all_charts) == 3, f"Expected 3 charts, generated {len(all_charts)}"
        for p in all_charts:
            assert p.exists()
            assert p.stat().st_size > 1000

        # 3. Chart with incompatible units (should plot without connecting across units)
        mixed_records = [
            {"date": "01-Jan-2024", "value": 10.2, "unit": "g/dL", "status": "NORMAL"},
            {"date": "01-Apr-2024", "value": 110.0, "unit": "g/L", "status": "NORMAL"},
        ]
        incompat_chart = visualizer.plot_test_trend("Mixed Units Test", mixed_records)
        assert incompat_chart is not None
        assert incompat_chart.exists()
        assert incompat_chart.stat().st_size > 1000

    print("  [PASS] 6. Matplotlib chart generation verified (single-panel, headless, non-zero file sizes)")


def test_numerical_summary_non_clinical():
    """Requirement: Objective numerical observations such as 'Hemoglobin changed from 10.2 to 11.1.' without diagnosis."""
    analyzer = MedicalTrendAnalyzer()
    reports = get_sample_reports()
    result = analyzer.analyze_trends(reports)

    # Check the exact requested pattern format
    summaries = result["summary_observations"]
    assert any("Hemoglobin (Hb) changed from 10.2 to 12.8." in s for s in summaries)
    assert any("Total Cholesterol changed from 230.0 to 195.0." in s for s in summaries)

    # Prohibited clinical diagnosis terms
    forbidden_terms = [
        "improve", "improved", "improving",
        "worse", "worsened", "worsening",
        "cure", "healthy", "unhealthy",
        "disease", "diagnos", "anemia", "anemic",
        "good", "bad", "danger", "risk"
    ]

    for s in summaries:
        s_lower = s.lower()
        for forbidden in forbidden_terms:
            assert forbidden not in s_lower, (
                f"Forbidden clinical bias '{forbidden}' detected in summary: '{s}'"
            )

    print("  [PASS] 7. Objective numerical observations confirmed (zero clinical diagnosis or bias)")


def test_full_pipeline_with_disk_reports():
    """Integration test with the 3 Sarah Jenkins sample patient reports stored in data/processed."""
    rep_1 = PROCESSED_DATA_DIR / "sample_patient_report_2024_01_validated.json"
    rep_2 = PROCESSED_DATA_DIR / "sample_patient_report_2024_04_validated.json"
    rep_3 = PROCESSED_DATA_DIR / "sample_patient_report_2024_07_validated.json"

    assert rep_1.exists(), f"Sample report 1 missing: {rep_1}"
    assert rep_2.exists(), f"Sample report 2 missing: {rep_2}"
    assert rep_3.exists(), f"Sample report 3 missing: {rep_3}"

    # Analyze in scrambled order to verify automatic chronological sorting
    trend_results, df, json_path = analyze_medical_trends(
        [rep_3, rep_1, rep_2],
        save_to_json=True,
        output_filename="longitudinal_trends_summary.json"
    )

    assert json_path is not None and json_path.exists()
    assert trend_results["metadata"]["total_reports"] == 3
    assert not df.empty
    assert len(trend_results["test_trends"]) >= 7

    # Generate charts to data/processed/charts/
    visualizer = MedicalTrendVisualizer(output_dir=CHARTS_DIR)
    generated_charts = visualizer.generate_all_charts(trend_results)
    assert len(generated_charts) >= 7

    for chart in generated_charts:
        assert chart.exists()
        assert chart.stat().st_size > 1000

    print(f"  [PASS] 8. Full longitudinal pipeline: {len(trend_results['test_trends'])} tests analyzed across 3 reports")
    print(f"  [PASS] Generated {len(generated_charts)} charts in: {CHARTS_DIR}")


def run_all_tests():
    print("=" * 70)
    print("  HealthLens Stage 7: Medical Trend Analysis & Visualization Test Suite")
    print("=" * 70)
    test_chronological_ordering()
    test_trend_dataframe_schema()
    test_multiple_reports()
    test_missing_values_and_dates()
    test_incompatible_units()
    test_chart_generation()
    test_numerical_summary_non_clinical()
    test_full_pipeline_with_disk_reports()
    print("=" * 70)
    print("  ALL STAGE 7 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
