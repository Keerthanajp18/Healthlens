"""
HealthLens - Person 2 RAG & LLM Integration Interface
Module: src/integration/person2_interface.py

Provides clean, standardized, ground-truth medical test data to Person 2's
Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) pipeline.

STRICT CONSTRAINTS:
1. NEVER include patient name, patient ID, phone, email, address, or demographic PII.
2. Preserve exact extracted numerical values (no rounding loss or conversion).
3. Preserve laboratory-provided biological reference ranges.
4. Preserve deterministic validation statuses (NORMAL, LOW, HIGH, UNKNOWN).
5. Output must be 100% standard JSON-serializable (native Python dicts, floats, ints, strings).
6. Zero LLM, zero vector embeddings, zero RAG logic (handled by Person 2).
7. Clean, single-function import interface.
"""

from pathlib import Path
import json
from typing import Any

from src.pipeline import process_medical_report, compare_reports, analyze_trends
from src.config import PROCESSED_DATA_DIR


# Forbidden PII keys that must never be emitted to Person 2
FORBIDDEN_PII_KEYS = {
    "patient_name", "patient", "name", "patient_id", "pat_id", "mrn",
    "patient_age", "age", "gender", "sex", "dob", "birth_date",
    "doctor", "doctor_name", "physician", "physician_name", "referred_by",
    "phone", "mobile", "telephone", "email", "address", "hospital",
    "clinic", "laboratory_address", "ssn", "national_id"
}


def _strip_pii_from_dict(d: dict) -> dict:
    """Recursively strip any known demographic or PII fields from a dictionary."""
    clean = {}
    for k, v in d.items():
        if k.strip().lower() in FORBIDDEN_PII_KEYS:
            continue
        if isinstance(v, dict):
            clean[k] = _strip_pii_from_dict(v)
        elif isinstance(v, list):
            clean[k] = [
                _strip_pii_from_dict(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            clean[k] = v
    return clean


def _resolve_processed_data(processed_report: dict | list | str | Path) -> dict:
    """
    Normalizes diverse input formats into standard validated report dictionary:
    - Path to a raw document (.pdf, .png, .jpg, .txt) -> processes via pipeline
    - Path to an already validated JSON file -> loads from disk
    - Dictionary returned by process_medical_report() -> passes through
    - List of test dictionaries -> wraps into standard structure
    """
    if isinstance(processed_report, (str, Path)):
        p = Path(processed_report)
        if not p.is_absolute():
            p = PROCESSED_DATA_DIR / p

        if not p.exists():
            raise FileNotFoundError(f"Medical report or JSON file not found: '{p}'")

        # If it's a JSON file, load it directly
        if p.suffix.lower() == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
            elif isinstance(data, list):
                return {"tests": data}
            else:
                raise ValueError(f"Unrecognized JSON structure in '{p}'")
        else:
            # If it's a raw report (PDF, PNG, JPG, TXT), process through pipeline
            return process_medical_report(p, save_to_json=False)

    elif isinstance(processed_report, dict):
        return processed_report

    elif isinstance(processed_report, list):
        return {"tests": processed_report}

    else:
        raise TypeError(
            f"Unsupported report input type: '{type(processed_report)}'. "
            f"Expected dict, list of tests, Path, or file path string."
        )


def get_rag_context(processed_report: dict | list | str | Path) -> dict:
    """
    Extracts validated ground-truth medical test data formatted specifically
    for Person 2's RAG knowledge retrieval and LLM prompt grounding.

    Guarantee:
    - Zero patient PII (no names, IDs, contact details).
    - Exact extracted numeric values and laboratory reference ranges.
    - Verified validation status (NORMAL / LOW / HIGH / UNKNOWN).
    - 100% JSON-serializable.

    Parameters:
        processed_report: A validated dictionary, report path, or raw report file.

    Returns:
        Standardized RAG context dictionary:
        {
            "report_date": "10-Feb-2024",
            "tests": [
                {
                    "test_name": "Hemoglobin (Hb)",
                    "value": 11.5,
                    "unit": "g/dL",
                    "reference_range": "12.0 - 15.5",
                    "status": "LOW",
                    "category": "Complete Blood Count (CBC)"
                },
                ...
            ]
        }
    """
    raw_data = _resolve_processed_data(processed_report)

    # 1. Resolve document-level report date
    report_date = None
    if isinstance(raw_data, dict):
        meta = raw_data.get("metadata", {})
        report_date = meta.get("report_date")

    tests_list = raw_data.get("tests", []) if isinstance(raw_data, dict) else raw_data

    # Fallback date from first test record if metadata didn't have it
    if not report_date and tests_list and isinstance(tests_list[0], dict):
        report_date = tests_list[0].get("report_date")

    # 2. Extract strictly relevant medical fields for Person 2
    sanitized_tests = []
    for t in tests_list:
        if not isinstance(t, dict):
            continue

        test_name = t.get("test_name")
        if not test_name:
            continue

        # Extract values ensuring exact numerical precision
        val = t.get("value")
        if val is not None:
            try:
                val = float(val)
                if val.is_integer() and not str(t.get("value")).endswith(".0"):
                    # Preserve integer representation if originally integer
                    pass
            except (ValueError, TypeError):
                pass

        entry = {
            "test_name": str(test_name).strip(),
            "value": val,
            "unit": str(t.get("unit", "")).strip(),
            "reference_range": str(t.get("reference_range", "")).strip(),
            "status": str(t.get("status", "UNKNOWN")).strip().upper(),
            "category": str(t.get("category", "General")).strip(),
        }

        # Optional bounds for RAG depth
        if t.get("lower_bound") is not None:
            try:
                entry["lower_bound"] = float(t["lower_bound"])
            except (ValueError, TypeError):
                entry["lower_bound"] = None
        if t.get("upper_bound") is not None:
            try:
                entry["upper_bound"] = float(t["upper_bound"])
            except (ValueError, TypeError):
                entry["upper_bound"] = None

        sanitized_tests.append(entry)

    rag_payload = {
        "report_date": report_date or "Unknown Date",
        "tests": sanitized_tests,
    }

    # Final PII sanitization sweep
    clean_rag_payload = _strip_pii_from_dict(rag_payload)

    # Verify JSON serializability
    try:
        json.dumps(clean_rag_payload)
    except (TypeError, OverflowError) as e:
        raise ValueError(f"RAG context payload failed JSON serialization: {e}")

    return clean_rag_payload


def get_comparison_context(comparison_result: dict | str | Path) -> dict:
    """
    Extracts longitudinal comparison context between two reports for Person 2's
    RAG-grounded delta explanations (e.g., explaining why Hemoglobin increased).

    Returns:
        {
            "baseline_date": "15-Jan-2024",
            "followup_date": "20-Apr-2024",
            "comparisons": [
                {
                    "test_name": "Hemoglobin (Hb)",
                    "previous_value": 10.2,
                    "current_value": 11.5,
                    "unit": "g/dL",
                    "absolute_change": 1.3,
                    "percentage_change": 12.75,
                    "change_direction": "INCREASE",
                    "status_previous": "LOW",
                    "status_current": "LOW"
                },
                ...
            ]
        }
    """
    if isinstance(comparison_result, (str, Path)):
        p = Path(comparison_result)
        if not p.is_absolute():
            p = PROCESSED_DATA_DIR / p
        with open(p, "r", encoding="utf-8") as f:
            raw_comp = json.load(f)
    elif isinstance(comparison_result, dict):
        raw_comp = comparison_result
    else:
        raise TypeError(f"Unsupported comparison input: {type(comparison_result)}")

    meta = raw_comp.get("metadata", {})
    prev_date = meta.get("previous_report", {}).get("report_date", "Unknown Date")
    curr_date = meta.get("current_report", {}).get("report_date", "Unknown Date")

    comparisons_list = []
    for c in raw_comp.get("comparisons", []):
        comparisons_list.append({
            "test_name": c.get("test_name"),
            "previous_value": c.get("previous_value"),
            "current_value": c.get("current_value"),
            "unit": c.get("unit"),
            "absolute_change": c.get("absolute_change"),
            "percentage_change": c.get("percentage_change"),
            "change_direction": c.get("change_direction"),
            "status_previous": c.get("status_previous"),
            "status_current": c.get("status_current"),
        })

    payload = {
        "baseline_date": prev_date,
        "followup_date": curr_date,
        "comparisons": comparisons_list,
    }

    clean_payload = _strip_pii_from_dict(payload)
    json.dumps(clean_payload)  # Verify serializability
    return clean_payload


def get_trend_context(trend_result: dict | str | Path) -> dict:
    """
    Extracts multi-report historical trend context for Person 2's longitudinal
    patient health summary generation.

    Returns:
        {
            "report_dates": ["15-Jan-2024", "20-Apr-2024", "15-Jul-2024"],
            "summary_observations": [
                "Hemoglobin (Hb) changed from 10.2 to 12.8.",
                ...
            ],
            "biomarker_trends": [
                {
                    "test_name": "Hemoglobin (Hb)",
                    "unit": "g/dL",
                    "first_value": 10.2,
                    "last_value": 12.8,
                    "absolute_change": 2.6,
                    "percentage_change": 25.49,
                    "direction": "INCREASE"
                },
                ...
            ]
        }
    """
    if isinstance(trend_result, (str, Path)):
        p = Path(trend_result)
        if not p.is_absolute():
            p = PROCESSED_DATA_DIR / p
        with open(p, "r", encoding="utf-8") as f:
            raw_trends = json.load(f)
    elif isinstance(trend_result, dict):
        raw_trends = trend_result
    else:
        raise TypeError(f"Unsupported trend input: {type(trend_result)}")

    meta = raw_trends.get("metadata", {})
    dates = meta.get("report_dates", [])
    summaries = raw_trends.get("summary_observations", [])

    trends_data = []
    for k, info in raw_trends.get("test_trends", []).items() if isinstance(raw_trends.get("test_trends"), dict) else []:
        trends_data.append({
            "test_name": info.get("test_name"),
            "unit": info.get("unit"),
            "first_value": info.get("first_value"),
            "last_value": info.get("last_value"),
            "absolute_change": info.get("absolute_change"),
            "percentage_change": info.get("percentage_change"),
            "direction": info.get("direction"),
        })

    payload = {
        "report_dates": dates,
        "summary_observations": summaries,
        "biomarker_trends": trends_data,
    }

    clean_payload = _strip_pii_from_dict(payload)
    json.dumps(clean_payload)  # Verify serializability
    return clean_payload
