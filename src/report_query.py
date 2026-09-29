import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.report_loader import PatientLabReport, PatientTestResult, ReportLoader

logger = logging.getLogger(__name__)


# Common medical lab synonyms and aliases for robust test detection
TEST_SYNONYMS = {
    "hemoglobin": ["hemoglobin", "hb", "hgb", "haemoglobin"],
    "wbc": ["wbc", "white blood cell", "white blood cells", "leukocyte", "leukocytes"],
    "platelets": ["platelets", "platelet", "plt", "thrombocyte", "thrombocytes"],
    "rbc": ["rbc", "red blood cell", "red blood cells", "erythrocyte", "erythrocytes"],
    "glucose": ["glucose", "blood sugar", "fasting blood glucose", "fbg"],
    "creatinine": ["creatinine", "creat", "serum creatinine"],
    "bun": ["bun", "blood urea nitrogen", "urea"],
    "alt": ["alt", "alanine aminotransferase", "sgpt"],
    "ast": ["ast", "aspartate aminotransferase", "sgot"],
    "cholesterol": ["cholesterol", "lipid", "total cholesterol"],
    "crp": ["crp", "c-reactive protein", "hs-crp"],
    "esr": ["esr", "erythrocyte sedimentation rate", "sed rate"],
}


@dataclass
class ReportQueryContext:
    """Encapsulates a report-aware retrieval query, matched patient tests, and grounding metadata."""
    retrieval_query: str
    user_question: str
    report: PatientLabReport
    matched_tests: List[PatientTestResult] = field(default_factory=list)

    @property
    def has_matched_tests(self) -> bool:
        return len(self.matched_tests) > 0

    def format_patient_report_info(self) -> str:
        """Returns the isolated, explicit patient-specific report section.

        This guarantees that patient-specific lab ranges are never mixed or replaced
        with generic knowledge base ranges.
        """
        lines = ["REPORT INFORMATION:"]
        tests_to_show = self.matched_tests if self.matched_tests else self.report.tests
        for t in tests_to_show:
            lines.append(f"{t.name} = {t.value} {t.unit}")
            lines.append(f"Laboratory reference range = {t.reference_range}")
            if t.status:
                lines.append(f"Flag / Status = {t.status.upper()}")
        return "\n".join(lines)


def identify_relevant_tests(
    report: PatientLabReport,
    user_question: str,
) -> List[PatientTestResult]:
    """Identifies tests in the patient report that are relevant to the user's question."""
    q_lower = user_question.lower()
    matched: List[PatientTestResult] = []

    for test in report.tests:
        t_name_lower = test.name.lower()

        # 1. Direct name match
        if t_name_lower in q_lower:
            matched.append(test)
            continue

        # 2. Known alias / synonym match
        for canonical, aliases in TEST_SYNONYMS.items():
            if t_name_lower == canonical or any(a in t_name_lower for a in aliases):
                if any(alias in q_lower for alias in aliases):
                    if test not in matched:
                        matched.append(test)
                    break

    # If user asks a general question covering the whole report (e.g. "Explain my results")
    if not matched:
        # Match all tests in report
        matched = list(report.tests)

    return matched


def build_report_query(
    report_data: Union[PatientLabReport, Dict[str, Any], Path, str],
    user_question: str,
) -> ReportQueryContext:
    """Builds a grounded, report-aware semantic retrieval query from a structured lab report

    and user question.

    Flow:
        Structured report JSON -> Identify relevant test(s) -> Build report-aware query

    Grounding Rules:
    - Never replaces laboratory-provided reference range with generic knowledge base range.
    - Does not generate diagnostic rules or medical diagnoses.
    - Isolates patient report facts from general educational medical facts.
    """
    # 1. Parse / normalize report input
    if isinstance(report_data, PatientLabReport):
        report = report_data
    elif isinstance(report_data, (str, Path)) and (str(report_data).endswith(".json") or Path(str(report_data)).exists()):
        report = ReportLoader.load_from_file(report_data)
    elif isinstance(report_data, str):
        report = ReportLoader.load_from_json_string(report_data)
    elif isinstance(report_data, dict):
        report = ReportLoader.validate_and_parse(report_data)
    else:
        raise TypeError(f"Unsupported report_data type: {type(report_data).__name__}")

    # 2. Identify relevant test(s)
    matched_tests = identify_relevant_tests(report, user_question)

    # 3. Construct report-aware semantic query
    query_parts: List[str] = []

    for test in matched_tests:
        query_parts.append(test.name)
        query_parts.append(f"{test.value} {test.unit}")
        query_parts.append(f"{test.reference_range}")
        query_parts.append(f"what {test.name} measures general educational interpretation")

    query_parts.append(user_question.strip())

    retrieval_query = " ".join(query_parts)

    return ReportQueryContext(
        retrieval_query=retrieval_query,
        user_question=user_question,
        report=report,
        matched_tests=matched_tests,
    )
