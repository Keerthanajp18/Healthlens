"""
HealthLens - Stage 4: Medical Test Data Validation
Module: src/validation/validator.py

Validates extracted laboratory test results against the reference ranges
provided directly by the diagnostic laboratory report.

Classifies each test value strictly as:
- NORMAL
- LOW
- HIGH
- UNKNOWN

STRICT RULES:
1. No external or internet reference ranges are guessed; only the report's own
   reported reference range is used.
2. No disease diagnosis or clinical condition inferences are made.
3. If a reference range is unparseable or absent, status is set to "UNKNOWN".
4. All original fields from Stage 3 are preserved.
"""

from pathlib import Path
import re
import json
import pandas as pd

from src.config import PROCESSED_DATA_DIR


class MedicalTestValidator:
    """
    Deterministic validator for medical laboratory test records.
    
    Parses biological reference intervals (ranges, '< threshold', '> threshold')
    and evaluates observed numeric values without clinical speculation or disease diagnosis.
    """

    def __init__(self):
        # Regex for '<' or '<=' upper threshold (e.g., "< 200", "< 200 Desirable", "<= 150")
        self._less_than_pattern = re.compile(
            r'^\s*<(=)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            re.IGNORECASE
        )

        # Regex for '>' or '>=' lower threshold (e.g., "> 50", "> 50 Optimal", ">= 60")
        self._greater_than_pattern = re.compile(
            r'^\s*>(=)?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            re.IGNORECASE
        )

        # Regex for two-sided interval (e.g., "12.0 - 15.5", "4000 - 11000", "13.0 to 17.0")
        self._range_pattern = re.compile(
            r'^\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(?:-|to)\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)',
            re.IGNORECASE
        )

    def parse_reference_range(
        self,
        range_str: str | None
    ) -> tuple[float | None, float | None, str]:
        """
        Parse reference range string into lower_bound, upper_bound, and reference_type.
        
        Args:
            range_str: Raw reference range string from laboratory report.
            
        Returns:
            tuple of (lower_bound, upper_bound, reference_type)
            reference_type is one of: 'RANGE', 'LESS_THAN', 'GREATER_THAN', 'UNKNOWN'
        """
        if not range_str or not isinstance(range_str, str):
            return None, None, "UNKNOWN"

        clean_str = range_str.strip()

        # 1. Check '<' or '<=' upper threshold
        m_lt = self._less_than_pattern.match(clean_str)
        if m_lt:
            upper_val = float(m_lt.group(2).replace(",", ""))
            return None, upper_val, "LESS_THAN"

        # 2. Check '>' or '>=' lower threshold
        m_gt = self._greater_than_pattern.match(clean_str)
        if m_gt:
            lower_val = float(m_gt.group(2).replace(",", ""))
            return lower_val, None, "GREATER_THAN"

        # 3. Check two-sided range "min - max"
        m_range = self._range_pattern.match(clean_str)
        if m_range:
            lower_val = float(m_range.group(1).replace(",", ""))
            upper_val = float(m_range.group(2).replace(",", ""))
            return lower_val, upper_val, "RANGE"

        # Unparseable range
        return None, None, "UNKNOWN"

    def evaluate_status(
        self,
        value: float | int | None,
        lower_bound: float | None,
        upper_bound: float | None,
        reference_type: str
    ) -> str:
        """
        Determine validation status (NORMAL, LOW, HIGH, UNKNOWN).
        
        Logic:
        - RANGE:
            value < lower_bound -> LOW
            value > upper_bound -> HIGH
            otherwise           -> NORMAL
        - LESS_THAN:
            value < upper_bound  -> NORMAL
            value >= upper_bound -> HIGH
        - GREATER_THAN:
            value > lower_bound  -> NORMAL
            value <= lower_bound -> LOW
        - UNKNOWN or value is None:
            -> UNKNOWN
        """
        if value is None or reference_type == "UNKNOWN":
            return "UNKNOWN"

        try:
            num_val = float(value)
        except (ValueError, TypeError):
            return "UNKNOWN"

        if reference_type == "RANGE":
            if lower_bound is not None and num_val < lower_bound:
                return "LOW"
            if upper_bound is not None and num_val > upper_bound:
                return "HIGH"
            return "NORMAL"

        elif reference_type == "LESS_THAN":
            if upper_bound is not None:
                if num_val < upper_bound:
                    return "NORMAL"
                else:
                    return "HIGH"
            return "UNKNOWN"

        elif reference_type == "GREATER_THAN":
            if lower_bound is not None:
                if num_val > lower_bound:
                    return "NORMAL"
                else:
                    return "LOW"
            return "UNKNOWN"

        return "UNKNOWN"

    def validate_record(self, record: dict) -> dict:
        """
        Validate a single test record, preserving all original fields and adding:
        - status: 'NORMAL' | 'LOW' | 'HIGH' | 'UNKNOWN'
        - lower_bound: float | None
        - upper_bound: float | None
        - reference_type: 'RANGE' | 'LESS_THAN' | 'GREATER_THAN' | 'UNKNOWN'
        """
        validated = dict(record)

        val = validated.get("value")
        ref_str = validated.get("reference_range")

        lower, upper, ref_type = self.parse_reference_range(ref_str)
        status = self.evaluate_status(val, lower, upper, ref_type)

        validated["status"] = status
        validated["lower_bound"] = lower
        validated["upper_bound"] = upper
        validated["reference_type"] = ref_type

        return validated

    def validate_records(self, records: list[dict]) -> list[dict]:
        """Validate a list of test records."""
        return [self.validate_record(r) for r in records]

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate a pandas DataFrame of test records.
        Returns a new DataFrame with validation columns added.
        """
        if df.empty:
            empty_cols = list(df.columns) + ["status", "lower_bound", "upper_bound", "reference_type"]
            return pd.DataFrame(columns=empty_cols)

        records = df.to_dict(orient="records")
        validated_records = self.validate_records(records)
        return pd.DataFrame(validated_records)

    def save_to_json(
        self,
        validated_records: list[dict],
        output_path: str | Path,
        metadata: dict | None = None
    ) -> Path:
        """
        Save validated records to JSON in data/processed/.
        
        Args:
            validated_records: List of validated test dictionaries.
            output_path: Output filename or destination Path.
            metadata: Optional report metadata dictionary.
        """
        dest = Path(output_path)
        if not dest.is_absolute():
            dest = PROCESSED_DATA_DIR / dest

        dest.parent.mkdir(parents=True, exist_ok=True)

        summary_counts = {
            "NORMAL": sum(1 for r in validated_records if r.get("status") == "NORMAL"),
            "LOW": sum(1 for r in validated_records if r.get("status") == "LOW"),
            "HIGH": sum(1 for r in validated_records if r.get("status") == "HIGH"),
            "UNKNOWN": sum(1 for r in validated_records if r.get("status") == "UNKNOWN"),
        }

        report_meta = metadata or {}
        report_meta.update({
            "total_tests": len(validated_records),
            "status_summary": summary_counts,
        })

        payload = {
            "metadata": report_meta,
            "tests": validated_records,
        }

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return dest


def validate_medical_tests(
    input_data: list[dict] | pd.DataFrame | str | Path,
    save_to_json: bool = True,
    output_filename: str = "sample_medical_report_validated.json",
    metadata: dict | None = None
) -> tuple[list[dict], pd.DataFrame, Path | None]:
    """
    Convenience function to validate medical test records and optionally save to JSON.
    
    Args:
        input_data: List of dicts, DataFrame, or path to structured JSON file from Stage 3.
        save_to_json: Whether to persist validated results to data/processed/.
        output_filename: Target JSON filename.
        metadata: Optional metadata dictionary to bundle in JSON.
        
    Returns:
        tuple of (validated_records_list, validated_dataframe, saved_json_path_or_none)
    """
    validator = MedicalTestValidator()
    extracted_meta = dict(metadata or {})

    # Handle file path input (load from JSON)
    if isinstance(input_data, (str, Path)):
        p = Path(input_data)
        if not p.is_absolute():
            p = PROCESSED_DATA_DIR / p
        with open(p, "r", encoding="utf-8") as f:
            loaded_json = json.load(f)
            if isinstance(loaded_json, dict) and "tests" in loaded_json:
                records = loaded_json["tests"]
                extracted_meta.update(loaded_json.get("metadata", {}))
            elif isinstance(loaded_json, list):
                records = loaded_json
            else:
                raise ValueError(f"Unrecognized JSON structure in '{p}'")
    elif isinstance(input_data, pd.DataFrame):
        records = input_data.to_dict(orient="records")
    elif isinstance(input_data, list):
        records = input_data
    else:
        raise TypeError(f"Unsupported input type for validation: {type(input_data)}")

    validated_records = validator.validate_records(records)
    validated_df = pd.DataFrame(validated_records)
    saved_path = None

    if save_to_json:
        saved_path = validator.save_to_json(
            validated_records=validated_records,
            output_path=output_filename,
            metadata=extracted_meta
        )

    return validated_records, validated_df, saved_path
