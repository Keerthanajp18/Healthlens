import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ReportValidationError(ValueError):
    """Raised when a patient medical report JSON does not conform to required schema."""
    pass


@dataclass
class PatientTestResult:
    """Represents a validated, single laboratory test entry from a patient's report."""
    name: str
    value: Union[float, int, str]
    unit: str
    reference_range: str
    status: Optional[str] = None

    def __post_init__(self):
        # Ensure values are cleanly formatted and trimmed
        if isinstance(self.name, str):
            self.name = self.name.strip()
        if isinstance(self.unit, str):
            self.unit = self.unit.strip()
        if isinstance(self.reference_range, str):
            self.reference_range = self.reference_range.strip()
        if isinstance(self.status, str):
            self.status = self.status.strip().lower()

    @property
    def display_string(self) -> str:
        """Formatted string of patient's lab result without exposing PII."""
        status_part = f" ({self.status.upper()})" if self.status else ""
        return f"{self.name} = {self.value} {self.unit}{status_part} [Lab Reference: {self.reference_range}]"


@dataclass
class PatientLabReport:
    """Represents a validated patient laboratory report containing one or more test results.

    Adheres to privacy-by-design: Strips and never stores or exposes patient-identifying info (PII).
    """
    tests: List[PatientTestResult]
    report_date: Optional[str] = None

    def get_test(self, test_name: str) -> Optional[PatientTestResult]:
        """Case-insensitive search for a test by name or common alias."""
        clean_target = test_name.strip().lower()
        # Direct exact match
        for test in self.tests:
            if test.name.lower() == clean_target:
                return test

        # Common synonym / substring match
        alias_map = {
            "hb": "hemoglobin",
            "hgb": "hemoglobin",
            "haemoglobin": "hemoglobin",
            "wbc": "wbc",
            "white blood cell": "wbc",
            "white blood cells": "wbc",
            "leukocyte": "wbc",
            "leukocytes": "wbc",
            "plt": "platelets",
            "platelet": "platelets",
            "thrombocyte": "platelets",
            "rbc": "rbc",
            "red blood cell": "rbc",
            "red blood cells": "rbc",
        }
        canonical_target = alias_map.get(clean_target, clean_target)

        for test in self.tests:
            t_canonical = alias_map.get(test.name.lower(), test.name.lower())
            if canonical_target == t_canonical:
                return test
            if canonical_target in test.name.lower() or test.name.lower() in canonical_target:
                return test

        return None

    @property
    def summary(self) -> str:
        lines = [f"Report Date: {self.report_date or 'N/A'}"]
        for t in self.tests:
            lines.append(f"  - {t.display_string}")
        return "\n".join(lines)


class ReportLoader:
    """Modular loader and validator for Person 1 structured medical report JSON."""

    REQUIRED_TEST_FIELDS = ("name", "value", "unit", "reference_range")

    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> PatientLabReport:
        """Loads and validates a patient lab report from a JSON file path."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Medical report file not found: {p}")

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ReportValidationError(f"Invalid JSON format in report file: {e}") from e

        return cls.validate_and_parse(data)

    @classmethod
    def load_from_json_string(cls, json_str: str) -> PatientLabReport:
        """Loads and validates a patient lab report from a JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ReportValidationError(f"Invalid JSON format: {e}") from e

        return cls.validate_and_parse(data)

    @classmethod
    def validate_and_parse(cls, data: Any) -> PatientLabReport:
        """Validates dictionary structure against contract and returns a PatientLabReport."""
        if not isinstance(data, dict):
            raise ReportValidationError(
                f"Report root must be a JSON object (dict), received: {type(data).__name__}"
            )

        if "tests" not in data:
            raise ReportValidationError("Report JSON is missing mandatory 'tests' field.")

        raw_tests = data.get("tests")
        if not isinstance(raw_tests, list) or len(raw_tests) == 0:
            raise ReportValidationError("'tests' must be a non-empty list of test items.")

        parsed_tests: List[PatientTestResult] = []

        for idx, item in enumerate(raw_tests):
            if not isinstance(item, dict):
                raise ReportValidationError(
                    f"Test item at index {idx} must be an object (dict), received: {type(item).__name__}"
                )

            # Check all required fields
            for req_field in cls.REQUIRED_TEST_FIELDS:
                if req_field not in item:
                    raise ReportValidationError(
                        f"Test item at index {idx} is missing required field '{req_field}'."
                    )
                val = item[req_field]
                if val is None or (isinstance(val, str) and not val.strip()):
                    raise ReportValidationError(
                        f"Required field '{req_field}' at index {idx} cannot be empty or null."
                    )

            # Extract and validate fields
            name = str(item["name"]).strip()
            value = item["value"]
            # Validate value is numeric or valid non-empty string representation
            if not isinstance(value, (int, float, str)):
                raise ReportValidationError(
                    f"Test '{name}' has invalid value type: {type(value).__name__}"
                )

            unit = str(item["unit"]).strip()
            # Preserve laboratory-provided reference range exactly as specified
            ref_range = str(item["reference_range"]).strip()
            status = str(item.get("status", "")).strip().lower() if item.get("status") else None

            parsed_tests.append(
                PatientTestResult(
                    name=name,
                    value=value,
                    unit=unit,
                    reference_range=ref_range,
                    status=status,
                )
            )

        # Non-PII metadata only
        report_date = data.get("report_date")
        if report_date and isinstance(report_date, str):
            report_date = report_date.strip()

        return PatientLabReport(
            tests=parsed_tests,
            report_date=report_date,
        )
