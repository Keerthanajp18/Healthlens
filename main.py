"""
HealthLens - Person 1 Module Entry Point
Project: HealthLens: A Multimodal RAG Framework for Simplifying and Comparing Medical Reports
Module : Document Processing & Medical Data Pipeline (Person 1)

Unified Pipeline Execution:
Medical Report (PDF / Scanned PDF / Image / Text)
  ↓
File Type Detection
  ↓
PDF/Image Ingestion & OCR fallback
  ↓
Raw Text
  ↓
Deterministic Medical Test Extraction (Zero PII)
  ↓
Structured Test Records
  ↓
Validation Against Biological Reference Ranges
  ↓
Validated Structured Data (NORMAL / LOW / HIGH / UNKNOWN)
  ↓
Optional Longitudinal Report Comparison (Mathematical Deltas)
  ↓
Optional Historical Trend Analysis & Single-Panel Visualization
  ↓
Standardized Ground-Truth JSON Output (for Person 2 RAG Integration)
"""

import sys
from pathlib import Path
import importlib.metadata
import pandas as pd

# Add project root to sys.path to ensure src imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    DATA_DIR,
    SAMPLE_REPORTS_DIR,
    PROCESSED_DATA_DIR,
    CHARTS_DIR,
)
from src.ingestion.ocr_loader import resolve_tesseract_binary
from src.pipeline import (
    process_medical_report,
    compare_reports,
    analyze_trends,
    MedicalDataPipeline,
)


def verify_dependencies() -> dict[str, str]:
    """Check that all required Person 1 libraries are installed and detect native OCR."""
    installed = {}

    try:
        import pypdf
        installed["pypdf"] = pypdf.__version__
    except ImportError as e:
        installed["pypdf"] = f"MISSING ({e})"

    try:
        import PIL
        installed["Pillow"] = PIL.__version__
    except ImportError as e:
        installed["Pillow"] = f"MISSING ({e})"

    try:
        import pytesseract
        installed["pytesseract"] = pytesseract.__version__
    except ImportError as e:
        installed["pytesseract"] = f"MISSING ({e})"

    try:
        import pypdfium2
        installed["pypdfium2"] = getattr(pypdfium2, "V_PYPDFIUM2", importlib.metadata.version("pypdfium2"))
    except ImportError as e:
        installed["pypdfium2"] = f"MISSING ({e})"

    try:
        installed["pandas"] = pd.__version__
    except ImportError as e:
        installed["pandas"] = f"MISSING ({e})"

    try:
        import matplotlib
        installed["matplotlib"] = matplotlib.__version__
    except ImportError as e:
        installed["matplotlib"] = f"MISSING ({e})"

    return installed


def run_pipeline_demo():
    """Execute end-to-end pipeline through the unified orchestration module (src/pipeline.py)."""
    sample_pdf = SAMPLE_REPORTS_DIR / "sample_medical_report.pdf"

    print("\n" + "=" * 75)
    print("  HEALTHLENS UNIFIED PERSON 1 PIPELINE DEMONSTRATION")
    print("=" * 75)
    print(f"Target Document: {sample_pdf.name}")

    if not sample_pdf.exists():
        print(f"[!] Warning: Sample PDF not found at {sample_pdf}.")
        print("    Run 'python scripts/create_sample_pdf.py' to generate it.")
        return

    # -------------------------------------------------------------------------
    # STEP 1: INGESTION + EXTRACTION + VALIDATION (process_medical_report)
    # -------------------------------------------------------------------------
    print("\n[PIPELINE STEP 1] Processing Medical Report (Ingest -> Extract -> Validate)...")
    report_data = process_medical_report(
        file_path=sample_pdf,
        save_to_json=True,
        output_filename="sample_medical_report_validated.json",
    )

    proc = report_data["processing"]
    meta = report_data["metadata"]
    summary = meta["status_summary"]

    print(f"  -> Ingestion Type    : {proc['source_type']}")
    print(f"  -> OCR Invoked       : {proc['ocr_used']}")
    print(f"  -> Total Pages       : {proc['total_pages']} ({proc['characters_extracted']} chars extracted)")
    print(f"  -> Report Date       : {meta['report_date']}")
    print(f"  -> Tests Parsed      : {meta['total_tests']} across categories: {meta['categories']}")
    print(f"  -> Validation Status : {summary['NORMAL']} NORMAL | {summary['LOW']} LOW | {summary['HIGH']} HIGH | {summary['UNKNOWN']} UNKNOWN")
    print(f"  -> Standardized JSON : {proc['saved_json_path']}")

    # Preview extracted test records
    print("\n" + "-" * 75)
    print("EXTRACTED & VALIDATED TESTS PREVIEW (Top 8 tests, zero PII):")
    print("-" * 75)
    tests_df = pd.DataFrame(report_data["tests"])
    preview_cols = ["test_name", "value", "unit", "reference_range", "status"]
    print(tests_df[preview_cols].head(8).to_string(index=False))
    print("-" * 75)

    # -------------------------------------------------------------------------
    # STEP 2: LONGITUDINAL REPORT COMPARISON (compare_reports)
    # -------------------------------------------------------------------------
    print("\n[PIPELINE STEP 2] Comparing Longitudinal Reports (January vs. April 2024)...")
    path_jan = PROCESSED_DATA_DIR / "sample_patient_report_2024_01_validated.json"
    path_apr = PROCESSED_DATA_DIR / "sample_patient_report_2024_04_validated.json"

    if not path_jan.exists() or not path_apr.exists():
        from scripts.create_comparison_sample_reports import generate_comparison_sample_reports
        generate_comparison_sample_reports()

    comp_result = compare_reports(
        report_a=path_jan,
        report_b=path_apr,
        save_to_json=True,
        output_filename="report_comparison_summary.json",
    )

    c_meta = comp_result["metadata"]
    print(f"  -> Baseline Report   : {c_meta['previous_report']['report_date']} ({c_meta['previous_report']['total_tests']} tests)")
    print(f"  -> Follow-up Report  : {c_meta['current_report']['report_date']} ({c_meta['current_report']['total_tests']} tests)")
    print(f"  -> Changes Detected  : {c_meta['total_increased']} INCREASED | {c_meta['total_decreased']} DECREASED | {c_meta['total_unchanged']} UNCHANGED")
    print(f"  -> Comparison JSON   : {comp_result.get('saved_path')}")

    # Preview comparison table
    print("\n" + "-" * 75)
    print("LONGITUDINAL COMPARISON TABLE (pandas DataFrame):")
    print("-" * 75)
    comp_df = pd.DataFrame(comp_result["comparisons"])
    comp_cols = [
        "test_name", "previous_value", "current_value", "absolute_change",
        "percentage_change", "change_direction", "unit"
    ]
    print(comp_df[comp_cols].to_string(index=False))
    print("-" * 75)

    # -------------------------------------------------------------------------
    # STEP 3: LONGITUDINAL TREND ANALYSIS & CHARTS (analyze_trends)
    # -------------------------------------------------------------------------
    print("\n[PIPELINE STEP 3] Analyzing Trends & Generating Charts Across 3 Reports...")
    path_jul = PROCESSED_DATA_DIR / "sample_patient_report_2024_07_validated.json"
    if not path_jul.exists():
        from scripts.create_comparison_sample_reports import generate_comparison_sample_reports
        generate_comparison_sample_reports()

    trend_result = analyze_trends(
        reports=[path_jan, path_apr, path_jul],
        generate_charts=True,
        save_to_json=True,
        output_filename="longitudinal_trends_summary.json",
        charts_dir=CHARTS_DIR,
    )

    t_meta = trend_result["metadata"]
    print(f"  -> Analyzed Reports  : {t_meta['total_reports']} ({', '.join(t_meta['report_dates'])})")
    print(f"  -> Unique Biomarkers : {t_meta['total_unique_tests']}")
    print(f"  -> Trend JSON Export : {trend_result.get('saved_path')}")

    print("\n" + "-" * 75)
    print("OBJECTIVE NUMERICAL OBSERVATIONS (Zero Clinical Diagnosis):")
    print("-" * 75)
    for obs in trend_result["summary_observations"][:5]:
        print(f"  * {obs}")
    if len(trend_result["summary_observations"]) > 5:
        print(f"  * ... ({len(trend_result['summary_observations']) - 5} more observations in JSON)")
    print("-" * 75)

    charts = trend_result.get("charts_generated", [])
    print(f"\nGenerated {len(charts)} Single-Panel Trend Chart(s) in {CHARTS_DIR}:")
    for c_path in charts[:4]:
        print(f"  - {Path(c_path).name}")
    if len(charts) > 4:
        print(f"  - ... ({len(charts) - 4} more charts)")

    print("\n" + "=" * 75)
    print("  PERSON 1 COMPLETE PIPELINE READY FOR PERSON 2 RAG INTEGRATION!")
    print("=" * 75)


def main():
    print("=" * 75)
    print("  HealthLens: Multimodal Medical Report Analysis System")
    print("  Person 1: Document Processing + Medical Data Pipeline")
    print("=" * 75)
    print(f"Python Executable : {sys.executable}")
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"Project Root      : {BASE_DIR}")
    print("-" * 75)

    # 1. Dependency Verification
    deps = verify_dependencies()
    all_ok = True
    for name, version in deps.items():
        status = "[OK]" if "MISSING" not in version else "[FAIL]"
        print(f"  - {name:<14}: {version:<15} {status}")
        if "MISSING" in version:
            all_ok = False

    # Check native Tesseract binary
    tess_path = resolve_tesseract_binary()
    if tess_path:
        print(f"  - {'Tesseract Binary':<14}: {tess_path} [OK]")
    else:
        print(f"  - {'Tesseract Binary':<14}: NOT DETECTED (See README to install for scanned reports)")

    if not all_ok:
        print("[WARNING] Missing dependencies. Run: pip install -r requirements.txt")
        return

    # 2. Run Pipeline Demo
    run_pipeline_demo()


if __name__ == "__main__":
    main()
