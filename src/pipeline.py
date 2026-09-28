"""
HealthLens - Person 1 Unified Pipeline Orchestrator
Module: src/pipeline.py

Integrates all Person 1 components into a single, cohesive, production-ready pipeline:
Medical Report
  ↓
File Type Detection (Digital PDF / Scanned PDF / Image)
  ↓
PDF / Image Ingestion
  ↓
OCR if required (Pillow + pytesseract / pypdfium2)
  ↓
Raw Text Extraction
  ↓
Medical Test Extraction (Deterministic parsing, regex)
  ↓
Structured Test Records (Zero PII)
  ↓
Medical Test Data Validation (Biological reference intervals)
  ↓
Validated Structured Data (NORMAL / LOW / HIGH / UNKNOWN)
  ↓
Optional Report Comparison (Longitudinal mathematical deltas)
  ↓
Optional Trend Analysis & Visualization (Chronological trends + single-panel Matplotlib charts)
  ↓
Standardized Ground-Truth JSON Output (for Person 2 RAG integration)

STRICT CONSTRAINTS:
1. No RAG, no LLM calls, and no frontend components (Person 2 integrates RAG later).
2. Deterministic, explainable, viva-ready Python architecture.
3. No medical diagnoses or subjective clinical claims.
4. Safe handling of missing dates, values, and incompatible units.
"""

from pathlib import Path
import json
from typing import Any
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    CHARTS_DIR,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.ingestion.ocr_loader import (
    load_medical_document,
    is_tesseract_available,
    resolve_tesseract_binary,
)
from src.extraction.test_extractor import MedicalTestExtractor
from src.validation.validator import MedicalTestValidator
from src.comparison.report_comparator import MedicalReportComparator
from src.comparison.trend_analyzer import MedicalTrendAnalyzer
from src.comparison.visualizer import MedicalTrendVisualizer


class MedicalDataPipeline:
    """
    Unified orchestrator for Person 1's Document Processing and Medical Data Pipeline.
    Coordinates multimodal ingestion, OCR fallback, structured parsing, validation,
    longitudinal comparison, and trend visualization.
    """

    def __init__(
        self,
        output_dir: Path | str | None = None,
        charts_dir: Path | str | None = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else PROCESSED_DATA_DIR
        self.charts_dir = Path(charts_dir) if charts_dir else CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        self.extractor = MedicalTestExtractor()
        self.validator = MedicalTestValidator()
        self.comparator = MedicalReportComparator()
        self.trend_analyzer = MedicalTrendAnalyzer()
        self.visualizer = MedicalTrendVisualizer(output_dir=self.charts_dir)

    def process_medical_report(
        self,
        file_path: str | Path,
        save_to_json: bool = True,
        output_filename: str | Path | None = None,
    ) -> dict:
        """
        Processes a medical report file through the complete ingestion -> extraction -> validation pipeline.

        Parameters:
            file_path: Path to the target medical document (PDF, PNG, JPG, or TXT).
            save_to_json: Whether to persist the standardized output to disk.
            output_filename: Optional custom filename for the saved JSON file.

        Returns:
            Standardized ground-truth dictionary:
            {
                "metadata": {
                    "source_file": "...",
                    "report_date": "...",
                    "total_tests": ...,
                    "categories": [...],
                    "status_summary": {"NORMAL": ..., "LOW": ..., "HIGH": ..., "UNKNOWN": ...}
                },
                "tests": [
                    {
                        "test_name": "...",
                        "value": ...,
                        "unit": "...",
                        "reference_range": "...",
                        "category": "...",
                        "report_date": "...",
                        "status": "NORMAL/LOW/HIGH/UNKNOWN"
                    },
                    ...
                ],
                "processing": {
                    "source_type": "...",
                    "ocr_used": true/false,
                    "total_pages": ...,
                    "characters_extracted": ...,
                    "saved_json_path": "..." (if save_to_json=True)
                }
            }
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Medical report file not found: '{path}'")

        # Step 1: Multimodal Ingestion (Digital PDF, Scanned PDF, Image with OCR, or Text)
        if path.suffix.lower() == ".txt":
            raw_text = path.read_text(encoding="utf-8")
            source_type = "plain_text"
            ocr_used = False
            total_pages = 1
        else:
            ingestion_result = load_medical_document(path, save_to_processed=False)
            raw_text = ingestion_result.get("combined_text", "")
            source_type = ingestion_result.get("source_type", "unknown")
            ocr_used = ingestion_result.get("ocr_used", False)
            total_pages = ingestion_result.get("total_pages", 1)

        # Step 2: Deterministic Structured Test Extraction (Zero PII)
        extracted_records = self.extractor.extract(raw_text=raw_text)

        # Resolve document-level report date
        doc_report_date = None
        if extracted_records and extracted_records[0].get("report_date"):
            doc_report_date = extracted_records[0]["report_date"]

        # Step 3: Medical Test Data Validation (Strictly using printed lab reference ranges)
        validated_records = self.validator.validate_records(extracted_records)

        # Step 4: Standardize test records (ensure exact schema without PII)
        standardized_tests = []
        for t in validated_records:
            test_entry = {
                "test_name": t.get("test_name", "Unknown Test"),
                "value": t.get("value"),
                "unit": t.get("unit", "").strip(),
                "reference_range": t.get("reference_range", "Unknown"),
                "category": t.get("category", "General"),
                "report_date": t.get("report_date") or doc_report_date or "Unknown Date",
                "status": t.get("status", "UNKNOWN"),
                "lower_bound": t.get("lower_bound"),
                "upper_bound": t.get("upper_bound"),
                "reference_type": t.get("reference_type"),
            }
            standardized_tests.append(test_entry)

        # Status summary statistics
        status_counts = {
            "NORMAL": sum(1 for t in standardized_tests if t["status"] == "NORMAL"),
            "LOW": sum(1 for t in standardized_tests if t["status"] == "LOW"),
            "HIGH": sum(1 for t in standardized_tests if t["status"] == "HIGH"),
            "UNKNOWN": sum(1 for t in standardized_tests if t["status"] == "UNKNOWN"),
        }
        categories = sorted(list({t["category"] for t in standardized_tests if t.get("category")}))

        # Step 5: Assemble standardized result contract
        result = {
            "metadata": {
                "source_file": path.name,
                "report_date": doc_report_date,
                "total_tests": len(standardized_tests),
                "categories": categories,
                "status_summary": status_counts,
            },
            "tests": standardized_tests,
            "processing": {
                "source_type": source_type,
                "ocr_used": ocr_used,
                "total_pages": total_pages,
                "characters_extracted": len(raw_text),
                "saved_json_path": None,
            },
        }

        # Step 6: Persist standardized JSON to disk
        if save_to_json:
            if output_filename:
                out_path = Path(output_filename)
                if not out_path.is_absolute():
                    out_path = self.output_dir / out_path
            else:
                out_path = self.output_dir / f"{path.stem}_validated.json"

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            result["processing"]["saved_json_path"] = str(out_path)

        return result

    def compare_reports(
        self,
        report_a: dict | list | str | Path,
        report_b: dict | list | str | Path,
        save_to_json: bool = True,
        output_filename: str | Path = "report_comparison_summary.json",
    ) -> dict:
        """
        Compares two validated medical reports chronologically.
        Calculates absolute and percentage changes with unit incompatibility protection.
        """
        comparison_result = self.comparator.compare_two_reports(report_a, report_b)
        if save_to_json:
            dest = self.comparator.save_comparison_to_json(
                comparison_result=comparison_result,
                output_filename=output_filename,
            )
            comparison_result["saved_path"] = str(dest)
        return comparison_result

    def analyze_trends(
        self,
        reports: list[dict | list | str | Path],
        generate_charts: bool = True,
        save_to_json: bool = True,
        output_filename: str | Path = "longitudinal_trends_summary.json",
        charts_dir: Path | str | None = None,
    ) -> dict:
        """
        Performs longitudinal trend analysis across multiple reports and renders single-panel charts.
        """
        target_charts_dir = Path(charts_dir) if charts_dir else self.charts_dir
        trend_results = self.trend_analyzer.analyze_trends(reports)

        if save_to_json:
            dest = self.trend_analyzer.save_trends_to_json(
                trend_results=trend_results,
                output_filename=output_filename,
            )
            trend_results["saved_path"] = str(dest)

        if generate_charts:
            chart_visualizer = MedicalTrendVisualizer(output_dir=target_charts_dir)
            chart_paths = chart_visualizer.generate_all_charts(trend_results)
            trend_results["charts_generated"] = [str(p) for p in chart_paths]

        return trend_results


# Module-level singleton instance for convenient functional usage
_default_pipeline = MedicalDataPipeline()


def process_medical_report(
    file_path: str | Path,
    save_to_json: bool = True,
    output_filename: str | Path | None = None,
) -> dict:
    """
    High-level entry point to process a single medical report.
    Returns standardized ground-truth dictionary for downstream RAG consumption.
    """
    return _default_pipeline.process_medical_report(
        file_path=file_path,
        save_to_json=save_to_json,
        output_filename=output_filename,
    )


def compare_reports(
    report_a: dict | list | str | Path,
    report_b: dict | list | str | Path,
    save_to_json: bool = True,
    output_filename: str | Path = "report_comparison_summary.json",
) -> dict:
    """
    High-level entry point to compare two medical reports chronologically.
    """
    return _default_pipeline.compare_reports(
        report_a=report_a,
        report_b=report_b,
        save_to_json=save_to_json,
        output_filename=output_filename,
    )


def analyze_trends(
    reports: list[dict | list | str | Path],
    generate_charts: bool = True,
    save_to_json: bool = True,
    output_filename: str | Path = "longitudinal_trends_summary.json",
    charts_dir: Path | str | None = None,
) -> dict:
    """
    High-level entry point for longitudinal multi-report trend analysis and visualization.
    """
    return _default_pipeline.analyze_trends(
        reports=reports,
        generate_charts=generate_charts,
        save_to_json=save_to_json,
        output_filename=output_filename,
        charts_dir=charts_dir,
    )
