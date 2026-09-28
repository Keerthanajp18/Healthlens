"""
HealthLens - Stage 7: Medical Report Trend Analysis and Visualization
Module: src/comparison/trend_analyzer.py

Analyzes longitudinal medical laboratory data across multiple dated reports.
Groups records by test_name, verifies chronological ordering, checks unit consistency,
and produces objective numerical summaries and trend DataFrames.

STRICT CONSTRAINTS:
1. Visualization and numerical trend analysis only.
2. Strictly NO medical diagnosis or clinical interpretation.
3. Do NOT claim that a trend is clinically good or bad (no 'improved' or 'worsened').
4. Do NOT use an LLM.
5. Deterministic, explainable, student capstone-level code.
6. Handle missing dates, missing values, and unshared tests safely.
7. Do not connect or mathematically compare values with incompatible units.
"""

from pathlib import Path
from datetime import datetime
import json
import re
from typing import Any
import pandas as pd
from dateutil import parser as date_parser

from src.config import PROCESSED_DATA_DIR
from src.comparison.report_comparator import parse_date, normalize_report_input


def safe_float(val: Any) -> float | None:
    """Safely convert a value to float, returning None if invalid."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        cleaned = re.sub(r"[^\d.-]", "", val.strip())
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class MedicalTrendAnalyzer:
    """
    Longitudinal trend analyzer for multiple validated medical laboratory reports.
    Groups records by test name, sorts chronologically, checks unit compatibility,
    and produces structured trend tables and objective numerical observations.
    """

    def __init__(self):
        pass

    def load_and_aggregate_reports(
        self,
        report_sources: list[dict | list | str | Path]
    ) -> list[tuple[dict, list[dict], datetime | None]]:
        """
        Loads multiple reports, extracts their metadata, test records, and dates,
        and sorts them chronologically from earliest to latest.

        Returns:
            List of tuples: (metadata, tests, parsed_date)
        """
        if not report_sources:
            return []

        loaded_reports = []
        for idx, src in enumerate(report_sources):
            meta, tests = normalize_report_input(src)
            date_str = meta.get("report_date")
            if not date_str and tests:
                date_str = tests[0].get("report_date")
            
            p_date = parse_date(date_str)
            loaded_reports.append((meta, tests, p_date, date_str or f"Report_{idx + 1}"))

        # Sort chronologically. Reports without valid dates placed at the end while preserving order.
        loaded_reports.sort(
            key=lambda x: (x[2] is None, x[2] if x[2] is not None else datetime.max)
        )

        return [(item[0], item[1], item[2]) for item in loaded_reports]

    def build_trend_dataframe(
        self,
        report_sources: list[dict | list | str | Path]
    ) -> pd.DataFrame:
        """
        Extracts all test records from the provided reports, sorts them chronologically,
        and returns a pandas DataFrame with columns:
        ['date', 'test_name', 'value', 'unit', 'status'] (plus auxiliary bound/reference columns).
        """
        aggregated = self.load_and_aggregate_reports(report_sources)
        records = []

        for meta, tests, p_date in aggregated:
            report_date = meta.get("report_date")
            for t in tests:
                t_date = t.get("report_date") or report_date or "Undated"
                records.append({
                    "date": t_date,
                    "test_name": t.get("test_name", "Unknown Test"),
                    "value": safe_float(t.get("value")),
                    "unit": t.get("unit", "").strip(),
                    "status": t.get("status", "UNKNOWN"),
                    "reference_range": t.get("reference_range"),
                    "lower_bound": safe_float(t.get("lower_bound")),
                    "upper_bound": safe_float(t.get("upper_bound")),
                    "reference_type": t.get("reference_type"),
                    "category": t.get("category"),
                    "_parsed_date": p_date or parse_date(t_date),
                })

        if not records:
            return pd.DataFrame(columns=["date", "test_name", "value", "unit", "status"])

        df = pd.DataFrame(records)
        
        # Chronological sort by parsed date, then test_name
        df["_sort_date"] = df["_parsed_date"].apply(
            lambda d: d if d is not None else datetime.max
        )
        df = df.sort_values(by=["_sort_date", "test_name"]).reset_index(drop=True)
        df = df.drop(columns=["_parsed_date", "_sort_date"])

        # Reorder so required columns come first
        required_cols = ["date", "test_name", "value", "unit", "status"]
        other_cols = [c for c in df.columns if c not in required_cols]
        return df[required_cols + other_cols]

    def analyze_trends(
        self,
        report_sources: list[dict | list | str | Path]
    ) -> dict:
        """
        Groups test records by test_name across multiple reports, evaluates unit consistency,
        computes numerical deltas, and creates objective, non-clinical summary observations.

        Returns:
            Dictionary containing:
                - metadata (report count, dates, total unique tests)
                - test_trends (dict keyed by normalized test name)
                - summary_observations (list of plain-text numerical observations)
        """
        aggregated = self.load_and_aggregate_reports(report_sources)
        if not aggregated:
            return {
                "metadata": {
                    "total_reports": 0,
                    "report_dates": [],
                    "total_unique_tests": 0,
                },
                "test_trends": {},
                "summary_observations": [],
            }

        report_dates = [
            meta.get("report_date") or "Undated" for meta, _, _ in aggregated
        ]

        # Group records by test_name (case-insensitive, preserving display name)
        groups: dict[str, dict] = {}

        for meta, tests, p_date in aggregated:
            report_date = meta.get("report_date")
            for t in tests:
                name = t.get("test_name", "Unknown Test").strip()
                k = name.lower()

                if k not in groups:
                    groups[k] = {
                        "display_name": name,
                        "category": t.get("category"),
                        "records": [],
                    }

                t_date = t.get("report_date") or report_date or "Undated"
                point = {
                    "date": t_date,
                    "parsed_date": p_date.isoformat() if p_date else None,
                    "value": safe_float(t.get("value")),
                    "unit": t.get("unit", "").strip(),
                    "status": t.get("status", "UNKNOWN"),
                    "reference_range": t.get("reference_range"),
                    "lower_bound": safe_float(t.get("lower_bound")),
                    "upper_bound": safe_float(t.get("upper_bound")),
                    "reference_type": t.get("reference_type"),
                }
                groups[k]["records"].append(point)

        test_trends = {}
        summary_observations = []

        for k, grp in groups.items():
            display_name = grp["display_name"]
            records = grp["records"]

            # Filter valid numerical observations
            numeric_records = [r for r in records if r["value"] is not None]
            units = list({r["unit"] for r in records if r["unit"]})
            units_consistent = len(units) <= 1
            primary_unit = units[0] if units else ""

            first_val = numeric_records[0]["value"] if numeric_records else None
            last_val = numeric_records[-1]["value"] if numeric_records else None
            first_date = numeric_records[0]["date"] if numeric_records else None
            last_date = numeric_records[-1]["date"] if numeric_records else None

            # Calculate numerical changes if units are compatible and at least 2 points exist
            absolute_change = None
            percentage_change = None
            direction = "UNKNOWN"
            summary_text = ""

            if not units_consistent:
                # Incompatible units: Do NOT calculate deltas or connect
                direction = "UNIT_INCOMPATIBLE"
                summary_text = (
                    f"{display_name} has incompatible units across reports "
                    f"({', '.join(units)}). Numerical change not calculated."
                )
            elif len(numeric_records) >= 2:
                if first_val is not None and last_val is not None:
                    absolute_change = round(last_val - first_val, 4)
                    if first_val != 0:
                        percentage_change = round(
                            ((last_val - first_val) / abs(first_val)) * 100, 2
                        )
                    
                    if absolute_change > 0:
                        direction = "INCREASE"
                    elif absolute_change < 0:
                        direction = "DECREASE"
                    else:
                        direction = "UNCHANGED"

                    # Objective numerical observation (no clinical interpretation)
                    if absolute_change == 0:
                        summary_text = f"{display_name} remained unchanged at {first_val}."
                    else:
                        summary_text = f"{display_name} changed from {first_val} to {last_val}."
            elif len(numeric_records) == 1:
                direction = "SINGLE_OBSERVATION"
                summary_text = f"{display_name} recorded once at {first_val}."
            else:
                direction = "NO_NUMERICAL_VALUES"
                summary_text = f"{display_name} has no numerical values recorded."

            summary_observations.append(summary_text)

            all_values = [r["value"] for r in numeric_records]
            test_trends[k] = {
                "test_name": display_name,
                "category": grp.get("category"),
                "total_observations": len(records),
                "numeric_observations": len(numeric_records),
                "units_consistent": units_consistent,
                "units": units,
                "unit": primary_unit,
                "first_date": first_date,
                "first_value": first_val,
                "last_date": last_date,
                "last_value": last_val,
                "min_value": min(all_values) if all_values else None,
                "max_value": max(all_values) if all_values else None,
                "absolute_change": absolute_change,
                "percentage_change": percentage_change,
                "direction": direction,
                "summary": summary_text,
                "records": records,
            }

        result = {
            "metadata": {
                "total_reports": len(aggregated),
                "report_dates": report_dates,
                "total_unique_tests": len(test_trends),
                "units_consistent_tests": sum(
                    1 for t in test_trends.values() if t["units_consistent"]
                ),
                "unit_incompatible_tests": sum(
                    1 for t in test_trends.values() if not t["units_consistent"]
                ),
            },
            "test_trends": test_trends,
            "summary_observations": summary_observations,
        }

        return result

    def save_trends_to_json(
        self,
        trend_results: dict,
        output_filename: str | Path = "longitudinal_trends_summary.json"
    ) -> Path:
        """
        Saves trend analysis results to JSON in data/processed/.
        """
        dest = Path(output_filename)
        if not dest.is_absolute():
            dest = PROCESSED_DATA_DIR / dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(trend_results, f, indent=2, ensure_ascii=False)

        return dest


def analyze_medical_trends(
    report_sources: list[dict | list | str | Path],
    save_to_json: bool = True,
    output_filename: str = "longitudinal_trends_summary.json"
) -> tuple[dict, pd.DataFrame, Path | None]:
    """
    Convenience function to run longitudinal trend analysis on multiple reports.

    Returns:
        tuple: (trend_results_dict, trend_dataframe, saved_json_path_or_none)
    """
    analyzer = MedicalTrendAnalyzer()
    df = analyzer.build_trend_dataframe(report_sources)
    trend_results = analyzer.analyze_trends(report_sources)
    saved_path = None

    if save_to_json:
        saved_path = analyzer.save_trends_to_json(trend_results, output_filename)

    return trend_results, df, saved_path
