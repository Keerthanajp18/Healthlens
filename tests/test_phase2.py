import json
from pathlib import Path
import pytest

from src.report_loader import ReportLoader, ReportValidationError, PatientLabReport, PatientTestResult
from src.report_query import build_report_query, identify_relevant_tests
from src.rag_pipeline import RAGPipeline


@pytest.fixture(scope="module")
def sample_report_path():
    p = Path(__file__).resolve().parent.parent / "data" / "sample_report.json"
    assert p.exists(), f"Sample report file not found at {p}"
    return p


@pytest.fixture(scope="module")
def sample_report(sample_report_path):
    return ReportLoader.load_from_file(sample_report_path)


@pytest.fixture(scope="module")
def pipeline():
    # Use existing FAISS index without rebuilding knowledge base
    return RAGPipeline()


# ============================================================================
# Phase 2 Test Cases
# ============================================================================

def test_1_valid_report_loading(sample_report_path):
    """Test 1: Verifies that a valid structured medical report JSON loads correctly."""
    report = ReportLoader.load_from_file(sample_report_path)

    assert isinstance(report, PatientLabReport), "Loaded object must be an instance of PatientLabReport"
    assert len(report.tests) == 3, f"Expected 3 tests in sample report, got {len(report.tests)}"
    assert report.report_date == "2026-09-28"

    # Validate individual test extraction
    hgb = report.get_test("Hemoglobin")
    assert hgb is not None, "Hemoglobin test not found in loaded report"
    assert hgb.name == "Hemoglobin"
    assert hgb.value == 10.2
    assert hgb.unit == "g/dL"
    assert hgb.reference_range == "12.0-16.0 g/dL"
    assert hgb.status == "low"

    wbc = report.get_test("WBC")
    assert wbc is not None, "WBC test not found in loaded report"
    assert wbc.value == 8200
    assert wbc.status == "normal"


def test_2_invalid_report_json():
    """Test 2: Verifies that malformed or non-compliant JSON structures raise ReportValidationError."""
    # Malformed JSON syntax
    with pytest.raises(ReportValidationError, match="Invalid JSON format"):
        ReportLoader.load_from_json_string("{invalid_json: true")

    # Non-dictionary root
    with pytest.raises(ReportValidationError, match="Report root must be a JSON object"):
        ReportLoader.validate_and_parse(["not", "a", "dict"])

    # Missing 'tests' field
    with pytest.raises(ReportValidationError, match="missing mandatory 'tests' field"):
        ReportLoader.validate_and_parse({"report_date": "2026-09-28"})

    # Empty 'tests' list
    with pytest.raises(ReportValidationError, match="must be a non-empty list"):
        ReportLoader.validate_and_parse({"tests": []})


def test_3_missing_test_name():
    """Test 3: Verifies that a test missing the 'name' field is rejected."""
    bad_data = {
        "tests": [
            {
                # Missing "name"
                "value": 14.0,
                "unit": "g/dL",
                "reference_range": "12.0-16.0 g/dL"
            }
        ]
    }
    with pytest.raises(ReportValidationError, match="missing required field 'name'"):
        ReportLoader.validate_and_parse(bad_data)


def test_4_missing_reference_range():
    """Test 4: Verifies that a test missing the 'reference_range' field is rejected."""
    bad_data = {
        "tests": [
            {
                "name": "Hemoglobin",
                "value": 10.2,
                "unit": "g/dL",
                # Missing "reference_range"
            }
        ]
    }
    with pytest.raises(ReportValidationError, match="missing required field 'reference_range'"):
        ReportLoader.validate_and_parse(bad_data)


def test_5_report_query_construction(sample_report):
    """Test 5: Verifies report query builder matches the correct test and constructs a grounded query."""
    user_question = "Explain my hemoglobin result."
    ctx = build_report_query(sample_report, user_question)

    assert len(ctx.matched_tests) == 1, f"Expected 1 matched test, got {len(ctx.matched_tests)}"
    assert ctx.matched_tests[0].name == "Hemoglobin"

    # Query must contain key patient lab parameters without inventing diagnostic claims
    q = ctx.retrieval_query
    assert "Hemoglobin" in q
    assert "10.2" in q
    assert "12.0-16.0" in q
    assert "what Hemoglobin measures" in q or "what hemoglobin measures" in q.lower()
    assert "Explain my hemoglobin result." in q


def test_6_retrieval_using_report_aware_query(pipeline, sample_report_path):
    """Test 6: Verifies end-to-end report-aware retrieval against the existing FAISS knowledge base."""
    user_question = "Explain my hemoglobin result."
    res = pipeline.query_with_report(sample_report_path, user_question, top_k=3)

    assert "retrieval_result" in res
    retrieval_res = res["retrieval_result"]
    assert retrieval_res.has_results, "Report-aware retrieval returned no results"

    top_chunk = retrieval_res.scored_chunks[0]
    assert top_chunk.score > 0.4, f"Retrieval score unexpectedly low: {top_chunk.score}"
    assert "Hemoglobin" in top_chunk.chunk.metadata.get("title", ""), (
        f"Top retrieved chunk '{top_chunk.chunk.metadata.get('title')}' is not Hemoglobin"
    )


def test_7_lab_reference_range_preserved_exactly(pipeline, sample_report_path, sample_report):
    """Test 7: Critical Grounding Rule - Verifies laboratory-provided reference range is preserved exactly

    and never replaced with a generic knowledge base range.
    """
    # Patient lab report range from sample_report.json
    expected_lab_range = "12.0-16.0 g/dL"
    hgb_test = sample_report.get_test("Hemoglobin")
    assert hgb_test.reference_range == expected_lab_range, (
        f"ReportLoader altered lab range: {hgb_test.reference_range} != {expected_lab_range}"
    )

    # Execute report-aware retrieval
    res = pipeline.query_with_report(sample_report_path, "Explain my hemoglobin result.")
    patient_info = res["patient_report_info"]

    # Verify patient information block contains exact laboratory reference range
    assert "Hemoglobin = 10.2 g/dL" in patient_info
    assert f"Laboratory reference range = {expected_lab_range}" in patient_info
    assert "REPORT INFORMATION:" in patient_info

    # Verify that generic KB ranges (e.g. 13.8 - 17.2 g/dL) did NOT overwrite the lab's range
    assert hgb_test.reference_range != "13.8 - 17.2 g/dL"
