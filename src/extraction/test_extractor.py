"""
HealthLens - Stage 3: Medical Test Information Extraction
Module: src/extraction/test_extractor.py

Converts raw, unstructured report text (from digital PDFs or plain text) into
structured laboratory test records containing:
- test_name
- value (numeric: int or float)
- unit
- reference_range
- category / section (when present)
- report_date (when present)

STRICT RULE:
No medical interpretations, clinical diagnostics, or abnormal status classifications
(e.g., 'low', 'high', 'anemia', 'disease') are performed in this stage.
Extraction is strictly deterministic and ground-truth preserving.
"""

from pathlib import Path
import re
import json
import pandas as pd

from src.config import PROCESSED_DATA_DIR


class MedicalTestExtractor:
    """
    Deterministic extractor for medical laboratory test records.
    
    Parses both single-line tabular rows and multi-line block formats (from PDF cell streams),
    differentiates clinical tests from patient demographic/report metadata,
    and returns structured Python dictionaries, pandas DataFrames, and JSON files.
    """

    # Comprehensive set of recognized medical measurement units
    STANDARD_UNITS = {
        # Concentration / Mass
        "g/dl", "mg/dl", "g/l", "mg/l", "ug/dl", "µg/dl", "ng/ml", "pg/ml",
        "mmol/l", "meq/l", "umol/l", "µmol/l",
        # Blood cell & particle counts
        "million/mcl", "million/ul", "million/µl",
        "cells/mcl", "cells/ul", "cells/µl",
        "/mcl", "/ul", "/µl", "/cumm", "10^3/ul", "10^6/ul",
        # Volume & Mass per cell
        "fl", "pg",
        # Ratios & Percentages
        "%", "ratio",
        # Enzymatic & Hormonal Activity
        "u/l", "iu/l", "miu/l", "mu/l",
        # Erythrocyte sedimentation & time
        "mm/hr", "sec", "seconds"
    }

    # Common metadata keywords to exclude from test names
    METADATA_EXCLUSION_KEYWORDS = [
        "patient", "name", "age", "gender", "sex", "id", "sample",
        "referred", "doctor", "dr.", "collection", "reporting",
        "date", "time", "investigation", "observed", "biological",
        "reference", "range", "page", "hospital", "laboratory",
        "diagnostic", "services", "clinical", "impression", "note",
        "disclaimer", "end of report", "department", "test name"
    ]

    def __init__(self, custom_units: set[str] | None = None):
        """Initialize extractor with standard and optional custom units."""
        self.known_units = set(self.STANDARD_UNITS)
        if custom_units:
            self.known_units.update({u.lower() for u in custom_units})

        # Regex for single-line tabular row (2 or more whitespace/tab separators)
        self._row_pattern = re.compile(
            r'^(?P<name>[A-Za-z0-9][A-Za-z0-9\s\(\)/_\-\.]+?)\s{2,}'
            r'(?P<val>[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s{2,}'
            r'(?P<unit>[^\s]+)\s{2,}'
            r'(?P<range>.+)$'
        )

        # Regex for reference ranges (e.g. "12.0 - 15.5", "< 200", "> 50", "< 150 Normal")
        self._range_pattern = re.compile(
            r'^(?:[<>]|<=|>=)?\s*[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?'
            r'(?:\s*(?:-|to)\s*[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)?'
            r'(?:\s+[A-Za-z]+)?$',
            re.IGNORECASE
        )

        # Regex for report dates
        self._date_pattern = re.compile(
            r'(?:Reporting|Collection)?\s*Date\s*[:]\s*'
            r'([0-9]{1,2}[-/\s][A-Za-z]{3,}[-/\s][0-9]{2,4}|[0-9]{4}[-/\s][0-9]{2}[-/\s][0-9]{2})',
            re.IGNORECASE
        )

    def _extract_report_date(self, text: str) -> str | None:
        """Extract report or collection date from document header."""
        match = self._date_pattern.search(text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_category(self, line: str) -> str | None:
        """Identify laboratory department or profile section from line."""
        line_clean = line.strip()
        
        # Pattern 1: "DEPARTMENT OF HEMATOLOGY: COMPLETE BLOOD COUNT (CBC)"
        match = re.search(r'DEPARTMENT\s+OF\s+[^:]+:\s*(.+)', line_clean, re.IGNORECASE)
        if match:
            return self._format_category_name(match.group(1))

        # Pattern 2: "TEST NAME: COMPLETE BLOOD COUNT (CBC)"
        match = re.search(r'TEST\s+NAME\s*:\s*(.+)', line_clean, re.IGNORECASE)
        if match:
            return self._format_category_name(match.group(1))

        # Pattern 3: "DEPARTMENT OF BIOCHEMISTRY" or "LIPID PROFILE"
        match = re.search(r'DEPARTMENT\s+OF\s+(.+)', line_clean, re.IGNORECASE)
        if match:
            return self._format_category_name(match.group(1))

        return None

    def _format_category_name(self, raw_cat: str) -> str:
        """Clean and normalize category title while preserving medical acronyms."""
        title = raw_cat.strip()
        # Keep acronyms capitalized
        acronyms = ["CBC", "RBC", "WBC", "PCV", "MCV", "MCH", "MCHC", "RDW", "HDL", "LDL", "VLDL", "TSH"]
        words = title.split()
        normalized_words = []
        for w in words:
            clean_w = w.strip("():,")
            if clean_w.upper() in acronyms:
                normalized_words.append(w.upper())
            else:
                normalized_words.append(w.capitalize())
        return " ".join(normalized_words)

    def _is_numeric(self, s: str) -> bool:
        """Check if string represents an integer or float (supports comma separators)."""
        cleaned = s.replace(",", "").strip()
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _parse_numeric_value(self, s: str) -> float | int:
        """Convert clean numeric string to float or integer."""
        cleaned = s.replace(",", "").strip()
        val = float(cleaned)
        # Preserve integer type when there are no decimal places
        if val.is_integer() and "." not in cleaned:
            return int(val)
        return val

    def _is_unit(self, s: str) -> bool:
        """Verify whether token matches recognized laboratory measurement units."""
        s_clean = s.strip().lower()
        if s_clean in self.known_units:
            return True
        # Common structural patterns for composite units (e.g. 'million/mcL', 'mg/dL')
        if "/" in s_clean or s_clean == "%" or s_clean == "ratio":
            return True
        return False

    def _is_reference_range(self, s: str) -> bool:
        """Verify whether string matches a valid laboratory reference range pattern."""
        s_clean = s.strip()
        if not s_clean:
            return False
        return bool(self._range_pattern.match(s_clean))

    def _is_metadata_or_header(self, s: str) -> bool:
        """Check if candidate test name is a document header, demographic, or page marker."""
        s_lower = s.strip().lower()
        if not s_lower:
            return True
        if s_lower.startswith("--- page") or s_lower.startswith("page "):
            return True
        if ":" in s and any(k in s_lower for k in ["patient", "sample", "doctor", "dr.", "date", "age", "referred"]):
            return True
        for kw in self.METADATA_EXCLUSION_KEYWORDS:
            if s_lower == kw or s_lower.startswith(f"{kw}:") or s_lower.startswith(f"{kw} /"):
                return True
        return False

    def extract(self, raw_text: str, default_category: str = "General") -> list[dict]:
        """
        Extract structured medical test records from report text.
        
        Args:
            raw_text: Raw text string (from digital PDF or plain text).
            default_category: Default test category if section headers are omitted.
            
        Returns:
            list of dicts, each with keys:
                - test_name (str)
                - value (float or int)
                - unit (str)
                - reference_range (str)
                - category (str)
                - report_date (str or None)
        """
        if not raw_text or not raw_text.strip():
            return []

        report_date = self._extract_report_date(raw_text)
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        records = []
        current_category = default_category

        i = 0
        while i < len(lines):
            line = lines[i]

            # 1. Check for category / section header change
            detected_cat = self._extract_category(line)
            if detected_cat:
                current_category = detected_cat
                i += 1
                continue

            # 2. Strategy A: Check for single-line horizontal table row
            m_row = self._row_pattern.match(line)
            if m_row:
                name = m_row.group("name").strip()
                val_str = m_row.group("val").strip()
                unit = m_row.group("unit").strip()
                ref_range = m_row.group("range").strip()

                if not self._is_metadata_or_header(name) and self._is_unit(unit):
                    records.append({
                        "test_name": name,
                        "value": self._parse_numeric_value(val_str),
                        "unit": unit,
                        "reference_range": ref_range,
                        "category": current_category,
                        "report_date": report_date,
                    })
                i += 1
                continue

            # 3. Strategy B: Check for 4-line sequential block (common in PDF table streams)
            # Line i  : Test Name
            # Line i+1: Value
            # Line i+2: Unit
            # Line i+3: Reference Range
            if i <= len(lines) - 4:
                c_name = lines[i]
                c_val = lines[i + 1]
                c_unit = lines[i + 2]
                c_range = lines[i + 3]

                # Validate each element strictly to eliminate false positives
                if (
                    not self._is_metadata_or_header(c_name)
                    and self._is_numeric(c_val)
                    and self._is_unit(c_unit)
                    and self._is_reference_range(c_range)
                ):
                    records.append({
                        "test_name": c_name,
                        "value": self._parse_numeric_value(c_val),
                        "unit": c_unit,
                        "reference_range": c_range,
                        "category": current_category,
                        "report_date": report_date,
                    })
                    i += 4  # Advance past the 4-line block
                    continue

            i += 1

        return records

    def extract_to_dataframe(self, raw_text: str, default_category: str = "General") -> pd.DataFrame:
        """
        Extract tests and return as a clean pandas DataFrame.
        
        Args:
            raw_text: Raw medical report text.
            default_category: Fallback category name.
            
        Returns:
            pd.DataFrame with columns:
                ['test_name', 'value', 'unit', 'reference_range', 'category', 'report_date']
        """
        records = self.extract(raw_text, default_category=default_category)
        columns = ["test_name", "value", "unit", "reference_range", "category", "report_date"]
        if not records:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(records)[columns]

    def save_to_json(
        self,
        records: list[dict],
        output_path: str | Path,
        source_file: str | None = None
    ) -> Path:
        """
        Save extracted structured records to a standardized JSON file.
        
        This JSON format serves as the clean integration contract for Person 2's
        RAG and LLM module.
        
        Args:
            records: List of extracted test dictionaries.
            output_path: Target path or filename inside data/processed/.
            source_file: Optional name of the original medical report file.
            
        Returns:
            Path to the saved JSON file.
        """
        dest = Path(output_path)
        if not dest.is_absolute():
            dest = PROCESSED_DATA_DIR / dest

        dest.parent.mkdir(parents=True, exist_ok=True)

        categories = sorted(list({r.get("category", "General") for r in records}))
        report_date = records[0].get("report_date") if records else None

        payload = {
            "metadata": {
                "source_file": source_file or "unknown",
                "report_date": report_date,
                "total_tests_extracted": len(records),
                "categories": categories,
            },
            "tests": records,
        }

        with open(dest, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return dest


def extract_medical_tests(
    raw_text: str,
    save_to_json: bool = False,
    output_filename: str = "sample_medical_report_structured.json",
    source_file: str | None = None
) -> tuple[list[dict], pd.DataFrame, Path | None]:
    """
    Convenience function to extract medical tests into records, a DataFrame, and optional JSON.
    
    Args:
        raw_text: Extracted report text.
        save_to_json: If True, writes structured records to data/processed/.
        output_filename: Name of the JSON output file.
        source_file: Name of the source report document.
        
    Returns:
        tuple of (records_list, dataframe, saved_json_path_or_none)
    """
    extractor = MedicalTestExtractor()
    records = extractor.extract(raw_text)
    df = extractor.extract_to_dataframe(raw_text)
    saved_path = None

    if save_to_json:
        saved_path = extractor.save_to_json(
            records=records,
            output_path=output_filename,
            source_file=source_file
        )

    return records, df, saved_path
