"""
HealthLens - Stage 6: Medical Report Comparison
Module: src/comparison/report_comparator.py

Compares two or more validated medical reports across different dates.
Calculates numerical changes:
- absolute_change = current_value - previous_value
- percentage_change = ((current_value - previous_value) / previous_value) * 100

STRICT RULES:
1. Do NOT diagnose diseases or interpret clinical significance.
2. Do NOT label changes as 'improved' or 'worsened' clinically.
3. Only describe objective numerical changes (e.g., INCREASE, DECREASE, UNCHANGED).
4. Do NOT calculate mathematical deltas if units are incompatible (flag unit mismatch).
5. If previous_value is 0, percentage_change must be None/null (no division by zero).
6. Explicitly handle tests missing from either report.
7. Sort reports chronologically by report date automatically.
"""

from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from dateutil import parser as date_parser

from src.config import PROCESSED_DATA_DIR


def parse_date(date_str: str | None) -> datetime | None:
    """Safely parse a date string into a datetime object for chronological sorting."""
    if not date_str or not isinstance(date_str, str):
        return None
    try:
        return date_parser.parse(date_str.strip())
    except Exception:
        return None


def normalize_report_input(
    report_input: dict | list | str | Path
) -> tuple[dict, list[dict]]:
    """
    Normalizes different input formats (JSON file path, full dictionary, or test list)
    into a standard (metadata_dict, tests_list) tuple.
    """
    metadata = {}
    tests = []

    # 1. Path to JSON file
    if isinstance(report_input, (str, Path)):
        p = Path(report_input)
        if not p.is_absolute():
            p = PROCESSED_DATA_DIR / p
        if not p.exists():
            raise FileNotFoundError(f"Report file for comparison not found: '{p}'")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            metadata = data.get("metadata", {})
            tests = data.get("tests", [])
        elif isinstance(data, list):
            tests = data
        else:
            raise ValueError(f"Unrecognized JSON structure in '{p}'")

    # 2. Dictionary input
    elif isinstance(report_input, dict):
        metadata = report_input.get("metadata", {})
        tests = report_input.get("tests", [])

    # 3. List of test records
    elif isinstance(report_input, list):
        tests = report_input

    else:
        raise TypeError(f"Unsupported report input type: {type(report_input)}")

    # Ensure report_date is resolved in metadata if available in records
    if not metadata.get("report_date") and tests:
        metadata["report_date"] = tests[0].get("report_date")

    return metadata, tests


class MedicalReportComparator:
    """
    Deterministic comparator for longitudinal medical laboratory test reports.
    """

    def compare_two_reports(
        self,
        report_a: dict | list | str | Path,
        report_b: dict | list | str | Path,
    ) -> dict:
        """
        Compare two validated reports, automatically ordering them chronologically.
        
        Returns:
            dict containing:
                - metadata (chronological previous and current report metadata)
                - comparisons (list of test comparison dictionaries)
        """
        meta_a, tests_a = normalize_report_input(report_a)
        meta_b, tests_b = normalize_report_input(report_b)

        date_a = parse_date(meta_a.get("report_date"))
        date_b = parse_date(meta_b.get("report_date"))

        # Determine chronological order: earlier is previous, later is current
        if date_a and date_b and date_a > date_b:
            meta_prev, tests_prev = meta_b, tests_b
            meta_curr, tests_curr = meta_a, tests_a
        else:
            meta_prev, tests_prev = meta_a, tests_a
            meta_curr, tests_curr = meta_b, tests_b

        prev_date_str = meta_prev.get("report_date")
        curr_date_str = meta_curr.get("report_date")

        # Map tests by test_name (case-insensitive key, preserving original display name)
        prev_map = {t["test_name"].strip().lower(): t for t in tests_prev if "test_name" in t}
        curr_map = {t["test_name"].strip().lower(): t for t in tests_curr if "test_name" in t}

        all_keys = []
        key_display_names = {}

        # Preserve ordering from previous report first, then any new tests in current
        for t in tests_prev:
            k = t["test_name"].strip().lower()
            if k not in key_display_names:
                all_keys.append(k)
                key_display_names[k] = t["test_name"]

        for t in tests_curr:
            k = t["test_name"].strip().lower()
            if k not in key_display_names:
                all_keys.append(k)
                key_display_names[k] = t["test_name"]

        comparisons = []

        for k in all_keys:
            display_name = key_display_names[k]
            rec_prev = prev_map.get(k)
            rec_curr = curr_map.get(k)

            # Case 1: Present in both reports
            if rec_prev is not None and rec_curr is not None:
                val_prev = rec_prev.get("value")
                val_curr = rec_curr.get("value")
                unit_prev = rec_prev.get("unit", "")
                unit_curr = rec_curr.get("unit", "")

                status_prev = rec_prev.get("status")
                status_curr = rec_curr.get("status")

                # Check unit compatibility
                if unit_prev.strip().lower() != unit_curr.strip().lower():
                    comparisons.append({
                        "test_name": display_name,
                        "previous_date": prev_date_str,
                        "previous_value": val_prev,
                        "current_date": curr_date_str,
                        "current_value": val_curr,
                        "absolute_change": None,
                        "percentage_change": None,
                        "change_direction": "UNIT_MISMATCH",
                        "unit": f"{unit_prev} vs {unit_curr}",
                        "status_previous": status_prev,
                        "status_current": status_curr,
                    })
                else:
                    # Valid unit match -> calculate changes
                    abs_change = None
                    pct_change = None
                    direction = "UNKNOWN"

                    if val_prev is not None and val_curr is not None:
                        abs_change = round(float(val_curr) - float(val_prev), 4)
                        if val_prev == 0:
                            pct_change = None  # Prevent division by zero
                        else:
                            pct_change = round(((float(val_curr) - float(val_prev)) / abs(float(val_prev))) * 100, 2)

                        if abs_change > 0:
                            direction = "INCREASE"
                        elif abs_change < 0:
                            direction = "DECREASE"
                        else:
                            direction = "UNCHANGED"

                    comparisons.append({
                        "test_name": display_name,
                        "previous_date": prev_date_str,
                        "previous_value": val_prev,
                        "current_date": curr_date_str,
                        "current_value": val_curr,
                        "absolute_change": abs_change,
                        "percentage_change": pct_change,
                        "change_direction": direction,
                        "unit": unit_curr or unit_prev,
                        "status_previous": status_prev,
                        "status_current": status_curr,
                    })

            # Case 2: Present only in previous report
            elif rec_prev is not None and rec_curr is None:
                comparisons.append({
                    "test_name": display_name,
                    "previous_date": prev_date_str,
                    "previous_value": rec_prev.get("value"),
                    "current_date": curr_date_str,
                    "current_value": None,
                    "absolute_change": None,
                    "percentage_change": None,
                    "change_direction": "ONLY_IN_PREVIOUS",
                    "unit": rec_prev.get("unit"),
                    "status_previous": rec_prev.get("status"),
                    "status_current": None,
                })

            # Case 3: Present only in current report
            elif rec_prev is None and rec_curr is not None:
                comparisons.append({
                    "test_name": display_name,
                    "previous_date": prev_date_str,
                    "previous_value": None,
                    "current_date": curr_date_str,
                    "current_value": rec_curr.get("value"),
                    "absolute_change": None,
                    "percentage_change": None,
                    "change_direction": "ONLY_IN_CURRENT",
                    "unit": rec_curr.get("unit"),
                    "status_previous": None,
                    "status_current": rec_curr.get("status"),
                })

        summary = {
            "metadata": {
                "previous_report": {
                    "source_file": meta_prev.get("source_file"),
                    "report_date": prev_date_str,
                    "total_tests": len(tests_prev),
                },
                "current_report": {
                    "source_file": meta_curr.get("source_file"),
                    "report_date": curr_date_str,
                    "total_tests": len(tests_curr),
                },
                "total_comparable_tests": sum(1 for c in comparisons if c["change_direction"] in ["INCREASE", "DECREASE", "UNCHANGED"]),
                "total_increased": sum(1 for c in comparisons if c["change_direction"] == "INCREASE"),
                "total_decreased": sum(1 for c in comparisons if c["change_direction"] == "DECREASE"),
                "total_unchanged": sum(1 for c in comparisons if c["change_direction"] == "UNCHANGED"),
                "only_in_previous": sum(1 for c in comparisons if c["change_direction"] == "ONLY_IN_PREVIOUS"),
                "only_in_current": sum(1 for c in comparisons if c["change_direction"] == "ONLY_IN_CURRENT"),
                "unit_mismatches": sum(1 for c in comparisons if c["change_direction"] == "UNIT_MISMATCH"),
            },
            "comparisons": comparisons,
        }

        return summary

    def compare_to_dataframe(
        self,
        report_a: dict | list | str | Path,
        report_b: dict | list | str | Path,
    ) -> pd.DataFrame:
        """
        Compare two reports and return the results as a clean pandas DataFrame.
        """
        result = self.compare_two_reports(report_a, report_b)
        columns = [
            "test_name",
            "previous_date",
            "previous_value",
            "current_date",
            "current_value",
            "absolute_change",
            "percentage_change",
            "change_direction",
            "unit",
            "status_previous",
            "status_current",
        ]
        return pd.DataFrame(result["comparisons"])[columns]

    def save_comparison_to_json(
        self,
        comparison_result: dict,
        output_filename: str | Path = "report_comparison_summary.json"
    ) -> Path:
        """
        Save comparison results to JSON in data/processed/.
        """
        dest = Path(output_filename)
        if not dest.is_absolute():
            dest = PROCESSED_DATA_DIR / dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False)

        return dest


def compare_medical_reports(
    report_a: dict | list | str | Path,
    report_b: dict | list | str | Path,
    save_to_json: bool = True,
    output_filename: str = "report_comparison_summary.json"
) -> tuple[dict, pd.DataFrame, Path | None]:
    """
    Convenience function to compare two medical reports, returning dict, DataFrame, and saved JSON path.
    """
    comparator = MedicalReportComparator()
    result = comparator.compare_two_reports(report_a, report_b)
    df = pd.DataFrame(result["comparisons"])
    saved_path = None

    if save_to_json:
        saved_path = comparator.save_comparison_to_json(result, output_filename)

    return result, df, saved_path
