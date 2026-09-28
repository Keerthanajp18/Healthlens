# HealthLens: A Multimodal RAG Framework for Simplifying and Comparing Medical Reports

## Capstone Project Overview

**HealthLens** is a Generative AI-based healthcare report explanation and comparison system. Healthcare laboratory reports (such as Complete Blood Counts, lipid profiles, thyroid panels, and metabolic panels) are often packed with medical jargon, numerical values, and reference ranges that patients struggle to interpret.

HealthLens addresses this problem using **Multimodal Document Processing**, **Retrieval-Augmented Generation (RAG)**, and **Data Analytics**:
1. It ingests medical reports in PDF, image, and scanned formats.
2. It extracts and structures laboratory test data (test names, observed values, units, biological reference intervals).
3. It validates observed values against the laboratory's reported reference intervals.
4. It compares longitudinal reports across multiple dates to track changes over time.
5. It retrieves verified medical context from a trusted knowledge base.
6. It generates grounded, patient-friendly explanations without hallucination.
7. It visualizes health trends clearly across multiple visits.

---

## Team Division of Responsibilities

The project is developed by a two-member team:

| Team Member | Module Scope | Key Responsibilities |
|---|---|---|
| **Person 1** *(Current Module)* | **Document Processing + Medical Data Pipeline** | Ingestion (Digital PDF, Scanned PDF, Images), OCR, structured data extraction, value validation, longitudinal report comparison, trend analysis, and visualization. |
| **Person 2** *(Partner Module)* | **RAG + Knowledge Base + LLM Generation** | Medical knowledge base construction, vector indexing, retrieval-augmented prompt engineering, LLM generation, multi-level/multilingual explanations, and conversational QA. |

> **Integration Point:** Person 1 delivers a clean, validated, structured data representation (`.json` / `pd.DataFrame`) containing test names, values, units, reference intervals, status flags (`NORMAL`, `LOW`, `HIGH`), and longitudinal comparison deltas. Person 2 consumes this structured data as ground-truth context for RAG retrieval and LLM prompt generation.

---

## Person 1 Module Scope & Roadmap

Person 1's module is structured into the following sequential components:

1. **Medical Report Upload / Ingestion:** Handling file uploads (`.pdf`, `.txt`, `.png`, `.jpg`, `.jpeg`).
2. **Digital PDF Processing (Implemented - Stage 2):** Text extraction from native digital PDF laboratory reports.
3. **Medical Test Information Extraction (Implemented - Stage 3):** Converting raw report text into structured records (`test_name`, `value`, `unit`, `reference_range`, `category`, `report_date`).
4. **Medical Test Data Validation (Implemented - Stage 4):** Deterministic evaluation of observed values against biological intervals (`status`, `lower_bound`, `upper_bound`, `reference_type`).
5. **Image and Scanned Report Processing (Implemented - Stage 5):** Unified multimodal ingestion supporting digital PDFs, scanned PDFs (rasterized via `pypdfium2`), and image files (`.png`, `.jpg`, `.jpeg` via Pillow + `pytesseract`).
6. **Medical Report Comparison (Implemented - Stage 6):** Longitudinal comparison across multiple report dates, calculating absolute changes, percentage variations, and change directions without clinical diagnostic bias.
7. **Historical Trend Analysis & Visualization (Implemented - Stage 7):** Computing multi-point historical trajectories, non-clinical numerical trend summaries, and rendering single-panel Matplotlib line charts.

---

## Stage 2: Digital PDF Ingestion & Text Extraction

### What PDF Ingestion Does
PDF Ingestion serves as the entry point of Person 1's data pipeline for documents in Portable Document Format (PDF). It reads electronic laboratory reports, validates their authenticity and structure, extracts character streams page-by-page, preserves pagination boundaries, and delivers raw combined text to downstream extraction algorithms.

### How `pypdf` is Being Used
- **Document Loading:** `pypdf.PdfReader` parses the PDF object hierarchy, trailers, cross-reference tables (`xref`), and page dictionaries.
- **Page Traversal:** Pages are iterated sequentially (`reader.pages`).
- **Text Extraction:** `page.extract_text()` decodes textual content streams, fonts, and character encodings.
- **Page Boundaries:** Each page's text is tagged with boundary markers (`--- Page X of Y ---`) to preserve context locality for multi-page reports.
- **Error Detection:** Catches `pypdf.errors.PdfReadError` for encrypted or damaged files and checks for blank/empty pages.

---

## Stage 3: Medical Test Information Extraction

### Why Structured Extraction is Needed
Raw medical reports are formatted for human visual inspection, not algorithmic computation. If raw unstructured text is passed directly to an LLM:
1. LLMs frequently hallucinate or misalign numbers when multiple tests and reference ranges are in proximity.
2. Numeric calculations (e.g. percentage change over time, comparing values to intervals) are unreliable inside large language models.
3. Converting the report into structured records (`test_name`, `value`, `unit`, `reference_range`) creates a clean, verifiable ground truth for data analytics and RAG retrieval.

### Why Deterministic Extraction is Used
- **Zero Hallucination:** Deterministic regular expressions and grammar parsers guarantee that numerical values match the source report exactly.
- **Zero Latency & Cost:** Parsing occurs locally in milliseconds without API costs or internet dependencies.
- **Privacy Preservation:** Patient identifiers (names, IDs, addresses, phone numbers) are filtered out *before* data is formatted, ensuring no PII leaks to LLMs or vector databases.
- **Explainability in Viva:** The algorithm's rules, patterns, and state transitions can be explained clearly to academic examiners.

---

## Stage 4: Medical Test Data Validation

### What Data Validation Does
Medical Test Data Validation (`src/validation/validator.py`) inspects each extracted test value and compares it strictly against the biological reference interval provided on the laboratory report.

### Ground-Truth Principle (Why Laboratory Ranges Matter)
Biological reference ranges vary widely based on analytical instrument calibrators, reagent manufacturers, geographic demographics, age groups, and sex. HealthLens **strictly relies on the laboratory's reported reference interval** rather than hardcoding arbitrary internet ranges.

### Validation Decision Logic
The `MedicalTestValidator` evaluates values using deterministic mathematical boundaries:
1. **Two-Sided Interval (`RANGE`: `lower_bound - upper_bound`):**
   - $\text{value} < \text{lower\_bound} \implies \textbf{LOW}$
   - $\text{value} > \text{upper\_bound} \implies \textbf{HIGH}$
   - $\text{lower\_bound} \le \text{value} \le \text{upper\_bound} \implies \textbf{NORMAL}$
2. **Upper Bound Threshold (`LESS_THAN`: ` < threshold` or `<= threshold`):**
   - $\text{value} < \text{upper\_bound} \implies \textbf{NORMAL}$
   - $\text{value} \ge \text{upper\_bound} \implies \textbf{HIGH}$
3. **Lower Bound Threshold (`GREATER_THAN`: `> threshold` or `>= threshold`):**
   - $\text{value} > \text{lower\_bound} \implies \textbf{NORMAL}$
   - $\text{value} \le \text{lower\_bound} \implies \textbf{LOW}$
4. **Unparseable or Qualitative Findings:**
   - If a range cannot be parsed numerically, status is assigned $\textbf{UNKNOWN}$.

---

## Stage 5: Image & Scanned Medical Report Processing

### Digital PDF vs. Scanned PDF vs. Image Reports
Laboratory test reports arrive in various formats:
1. **Digital PDF (`digital_pdf`):** Generated directly by laboratory Information Management Systems (LIMS). Contains native computer fonts and vector text streams. Fast and lossless extraction via `pypdf`.
2. **Scanned PDF (`scanned_pdf`):** A physical paper document scanned by a scanner and saved as a `.pdf`. It contains only a bitmap photograph wrapped inside a PDF container. `pypdf` extracts 0 characters. It must be rasterized to an image (via `pypdfium2`) and processed through OCR.
3. **Image Reports (`image`):** Photos of paper reports taken with smartphone cameras (`.png`, `.jpg`, `.jpeg`). Processed via Pillow image enhancement and OCR.

### The Hybrid Routing Architecture (`MultimodalReportLoader`)
To avoid unnecessary CPU overhead and potential OCR degradation on clean files, `MultimodalReportLoader` implements a fallback architecture:
- Digital PDFs are extracted directly via `pypdf`.
- Scanned PDF pages and images invoke OCR via `pytesseract`.
- Output is unified with page boundaries for Stage 3 extraction.

---

## Stage 6: Longitudinal Medical Report Comparison

### What Report Comparison Does
Medical Report Comparison (`src/comparison/report_comparator.py`) aligns and compares laboratory test results across multiple dates for the same patient. It calculates absolute and percentage variations to quantify changes over time.

### Mathematical Change Formulas
For any test measured at a baseline date ($v_{\text{prev}}$) and follow-up date ($v_{\text{curr}}$):

$$\text{absolute\_change} = v_{\text{curr}} - v_{\text{prev}}$$

$$\text{percentage\_change} = \left(\frac{v_{\text{curr}} - v_{\text{prev}}}{|v_{\text{prev}}|}\right) \times 100$$

> **Zero-Division Protection:** If $v_{\text{prev}} = 0$, $\text{percentage\_change}$ is set to `None` (`null` in JSON) rather than raising an uncaught `ZeroDivisionError`.

### Objective Numerical Direction (No Diagnostic Bias)
In clinical diagnostics, whether an increase is "good" or "bad" depends entirely on the specific biomarker:
- An increase in Hemoglobin towards normal is clinically desirable.
- An increase in LDL Cholesterol is clinically undesirable.

To prevent dangerous or unwarranted diagnostic claims, the comparator **strictly avoids subjective labels like "improved" or "worsened"**. It uses purely objective, mathematical directions:
- `INCREASE` ($\text{absolute\_change} > 0$)
- `DECREASE` ($\text{absolute\_change} < 0$)
- `UNCHANGED` ($\text{absolute\_change} = 0$)
- `ONLY_IN_PREVIOUS` (test discontinued or omitted in follow-up)
- `ONLY_IN_CURRENT` (new test ordered in follow-up)
- `UNIT_MISMATCH` (units differed between labs; deltas suppressed to prevent false math)

### Chronological Auto-Sorting
Regardless of the order files are provided (e.g. passing April report first and January second), the comparator extracts the report dates, parses them into date objects, and guarantees that:
- `previous_date` is the earlier date.
- `current_date` is the later date.

### Comparison Output Schema (JSON Integration Contract)
Comparison summaries are exported to `data/processed/report_comparison_summary.json`:
```json
{
  "metadata": {
    "previous_report": {"report_date": "15-Jan-2024", "total_tests": 7},
    "current_report": {"report_date": "20-Apr-2024", "total_tests": 7},
    "total_comparable_tests": 6,
    "total_increased": 2,
    "total_decreased": 3,
    "total_unchanged": 1,
    "only_in_previous": 1,
    "only_in_current": 1
  },
  "comparisons": [
    {
      "test_name": "Hemoglobin (Hb)",
      "previous_date": "15-Jan-2024",
      "previous_value": 10.2,
      "current_date": "20-Apr-2024",
      "current_value": 11.5,
      "absolute_change": 1.3,
      "percentage_change": 12.75,
      "change_direction": "INCREASE",
      "unit": "g/dL",
      "status_previous": "LOW",
      "status_current": "LOW"
    }
  ]
}
```

---

## Stage 7: Medical Report Trend Analysis & Visualization

### What Trend Analysis & Visualization Does
Stage 7 (`src/comparison/trend_analyzer.py` and `src/comparison/visualizer.py`) expands two-report comparison into multi-point longitudinal trend analysis across $N \ge 2$ dated medical reports:
1. **Multi-Report Aggregation:** Accepts multiple validated report JSON files or structured dictionaries.
2. **Chronological Auto-Sorting:** Resolves report dates across all documents and orders records sequentially from earliest to latest regardless of input order.
3. **Structured Trend DataFrame:** Generates a standardized pandas DataFrame with columns:
   - `date`: Report date of the observation.
   - `test_name`: Name of the laboratory test.
   - `value`: Numeric observed value (`float`).
   - `unit`: Laboratory measurement unit.
   - `status`: Clinical reference classification (`NORMAL`, `LOW`, `HIGH`, `UNKNOWN`).
4. **Unit Safety & Incompatibility Protection:** Strictly refuses to compute deltas or draw continuous connecting lines across incompatible units (e.g. `g/dL` vs `mg/dL`).
5. **Objective Numerical Summaries:** Produces neutral, non-clinical summary observations such as:
   `"Hemoglobin (Hb) changed from 10.2 to 12.8."`
6. **Single-Panel Matplotlib Visualization:** Generates focused, clean line charts saved as PNGs in `data/processed/charts/` featuring:
   - Shaded green normal reference range bands (when reference bounds are available).
   - Markers color-coded by clinical status (`NORMAL` = green, `LOW` = blue, `HIGH` = red, `UNKNOWN` = gray).
   - Clean numeric value callouts on data points.
   - Non-clinical educational watermark disclaimer.
   - Headless execution (`matplotlib.use('Agg')`) and explicit memory management (`plt.close(fig)`).

### Strict Safety & Capstone Constraints
- **Zero Medical Diagnoses:** Does not diagnose anemia, hypercholesterolemia, diabetes, or any other disease.
- **No Subjective Interpretations:** Never claims that a trend is clinically "good", "bad", "improved", or "worsened".
- **Zero LLM Dependency:** 100% deterministic Python analytics and Matplotlib rendering, fully explainable in a student viva.
- **Incompatible Units Protection:** If a test is measured in different units across dates, points are not joined by a connecting line, and numerical deltas are suppressed.

---

## Person 1 Final Pipeline Architecture & Integration Contract

All Person 1 processing stages are consolidated into a unified, reusable orchestrator in [`src/pipeline.py`](file:///c:/Users/kiran/OneDrive/Desktop/GenAI_capstone/src/pipeline.py):

```
Medical Report (Digital PDF / Scanned PDF / Image / Text)
  ↓
File Type Detection
  ↓
Multimodal Ingestion (Digital PDF / Image Preprocessing)
  ↓
OCR Engine Fallback (pypdfium2 + pytesseract if required)
  ↓
Raw Text Stream (Page-boundary preserved)
  ↓
Deterministic Medical Test Extraction (Regex, decimals, bounds, zero PII)
  ↓
Structured Test Records
  ↓
Data Validation (Strict evaluation against printed laboratory intervals)
  ↓
Validated Structured Data (NORMAL / LOW / HIGH / UNKNOWN)
  ↓
Optional Longitudinal Comparison (Chronological deltas, unit mismatch protection)
  ↓
Optional Historical Trend Analysis (Multi-report tables + single-panel Matplotlib charts)
  ↓
Standardized Ground-Truth JSON Output (Contract for Person 2 RAG)
```

### Reusable High-Level API (`src/pipeline.py`)

```python
from src.pipeline import process_medical_report, compare_reports, analyze_trends

# 1. Process a single medical report (PDF, PNG, JPG, or TXT)
result = process_medical_report("data/sample_reports/sample_medical_report.pdf")

# 2. Compare two dated reports for the same patient
comparison = compare_reports("report_jan.json", "report_apr.json")

# 3. Analyze multi-visit trends and render Matplotlib line charts
trends = analyze_trends(["report_jan.json", "report_apr.json", "report_jul.json"])
```

### Standardized JSON Output Contract (Ground Truth for Person 2 RAG)

```json
{
  "metadata": {
    "source_file": "sample_medical_report.pdf",
    "report_date": "10-Feb-2024",
    "total_tests": 19,
    "categories": ["Complete Blood Count (CBC)", "Lipid Profile"],
    "status_summary": {
      "NORMAL": 9,
      "LOW": 3,
      "HIGH": 7,
      "UNKNOWN": 0
    }
  },
  "tests": [
    {
      "test_name": "Hemoglobin (Hb)",
      "value": 11.5,
      "unit": "g/dL",
      "reference_range": "12.0 - 15.5",
      "category": "Complete Blood Count (CBC)",
      "report_date": "10-Feb-2024",
      "status": "LOW",
      "lower_bound": 12.0,
      "upper_bound": 15.5,
      "reference_type": "RANGE"
    }
  ],
  "processing": {
    "source_type": "digital_pdf",
    "ocr_used": false,
    "total_pages": 2,
    "characters_extracted": 1957,
    "saved_json_path": "data/processed/sample_medical_report_validated.json"
  }
}
```

> **Zero PII Guarantee:** Patient demographics (names, age, gender, IDs, contact details, physician names) are filtered out *before* structured JSON records are created. Person 2 receives strictly anonymized biomarker measurements.

---

## Project Folder Structure

```
GenAI_capstone/
├── data/
│   ├── sample_reports/                         # Input reports for testing
│   │   ├── sample_cbc_report.txt               # Plain text CBC reference report
│   │   ├── sample_medical_report.pdf           # Digital 2-Page synthetic PDF report
│   │   ├── sample_scanned_report.pdf           # Scanned PDF (image-only container)
│   │   ├── sample_medical_report_image.png     # Synthetic report image (PNG)
│   │   └── sample_medical_report_image.jpg     # Synthetic report image (JPG)
│   └── processed/                              # Output directory for structured extractions
│       ├── sample_medical_report_extracted_text.txt # Stage 2/5: Extracted raw text
│       ├── sample_medical_report_structured.json    # Stage 3: Standardized structured JSON
│       ├── sample_medical_report_validated.json     # Stage 4: Enriched validated JSON
│       ├── sample_patient_report_2024_01_validated.json # Stage 6: Baseline report
│       ├── sample_patient_report_2024_04_validated.json # Stage 6: Follow-up report 1
│       ├── sample_patient_report_2024_07_validated.json # Stage 7: Follow-up report 2
│       ├── report_comparison_summary.json          # Stage 6: Longitudinal comparison output
│       ├── longitudinal_trends_summary.json        # Stage 7: Multi-report trend analysis JSON
│       └── charts/                                 # Stage 7: Matplotlib single-panel PNG charts
│           ├── hemoglobin_hb_trend.png
│           ├── rbc_count_trend.png
│           ├── total_cholesterol_trend.png
│           └── ...
├── docs/
│   └── PERSON2_INTEGRATION.md                  # Comprehensive interface contract for Person 2
├── scripts/
│   ├── create_sample_pdf.py                    # Generates synthetic digital PDFs
│   ├── create_sample_image.py                  # Generates synthetic image/scanned reports
│   └── create_comparison_sample_reports.py     # Generates longitudinal comparison reports (Jan, Apr, Jul 2024)
├── src/
│   ├── __init__.py                             # Exports pipeline functions & package version
│   ├── config.py                               # Centralized file paths (DATA_DIR, PROCESSED_DATA_DIR, CHARTS_DIR)
│   ├── pipeline.py                             # Reusable Person 1 unified orchestrator
│   ├── ingestion/                              # Multimodal Document Ingestion
│   │   ├── __init__.py                         # Exports loaders & OCR helpers
│   │   ├── pdf_loader.py                       # Digital PDF loader (pypdf)
│   │   └── ocr_loader.py                       # Multimodal loader & OCR pipeline
│   ├── extraction/                             # Stage 3: Test Information Extraction
│   │   ├── __init__.py                         # Exports MedicalTestExtractor
│   │   └── test_extractor.py                   # Deterministic parser (decimals, ranges, units)
│   ├── validation/                             # Stage 4: Test Value Validation
│   │   ├── __init__.py                         # Exports MedicalTestValidator
│   │   └── validator.py                        # Deterministic range evaluator & status classifier
│   ├── comparison/                             # Stages 6 & 7: Comparison, Trends & Visualization
│   │   ├── __init__.py                         # Exports MedicalReportComparator, MedicalTrendAnalyzer, MedicalTrendVisualizer
│   │   ├── report_comparator.py                # Chronological comparison & delta calculator (Stage 6)
│   │   ├── trend_analyzer.py                   # Longitudinal trend analyzer & summary generator (Stage 7)
│   │   └── visualizer.py                       # Single-panel Matplotlib chart renderer (Stage 7)
│   └── integration/                            # Person 2 RAG & LLM Integration Interface
│       ├── __init__.py                         # Exports get_rag_context, get_comparison_context, get_trend_context
│       └── person2_interface.py                # Zero-PII ground-truth context extractor
├── tests/
│   ├── test_pdf_loader.py                      # Stage 2 test suite
│   ├── test_test_extractor.py                  # Stage 3 test suite
│   ├── test_validator.py                       # Stage 4 test suite
│   ├── test_ocr_loader.py                      # Stage 5 test suite
│   ├── test_comparison.py                      # Stage 6 test suite
│   ├── test_visualizer.py                      # Stage 7 test suite (ordering, charts, units, summaries)
│   ├── test_pipeline.py                        # Unified end-to-end integration test suite
│   └── test_person2_interface.py               # Person 2 interface contract & zero-PII test suite
├── main.py                                     # Live pipeline demo (Stages 2 through 7)
├── requirements.txt                            # Project dependencies
├── .gitignore                                  # Git ignore rules for venv, cache, and temp files
└── README.md                                   # Comprehensive documentation & pipeline roadmap
```

---

## Getting Started & Verification

### 1. Activate Virtual Environment
```powershell
# PowerShell:
.\.venv\Scripts\Activate.ps1

# Or Command Prompt (cmd):
.\.venv\Scripts\activate.bat
```

### 2. Run All Verification Test Suites
```powershell
# Run Stage 2 test suite (Digital PDF):
python tests/test_pdf_loader.py

# Run Stage 3 test suite (Medical Test Extraction):
python tests/test_test_extractor.py

# Run Stage 4 test suite (Medical Test Validation):
python tests/test_validator.py

# Run Stage 5 test suite (Image & OCR Loader):
python tests/test_ocr_loader.py

# Run Stage 6 test suite (Report Comparison):
python tests/test_comparison.py

# Run Stage 7 test suite (Trend Analysis & Visualization):
python tests/test_visualizer.py

# Run Unified Pipeline Integration test suite:
python tests/test_pipeline.py

# Run Person 2 Interface Integration test suite:
python tests/test_person2_interface.py
```

### 3. Run the Full Live Pipeline Demo
```powershell
python main.py
```
This executes the end-to-end pipeline:
1. Ingests document via `MultimodalReportLoader`.
2. Extracts structured test records deterministically.
3. Validates each observed value against the laboratory reference range.
4. Compares longitudinal reports (January 2024 vs. April 2024) and computes deltas.
5. Performs longitudinal trend analysis across 3 reports (January, April, July 2024).
6. Displays the comparison DataFrame and non-clinical summary observations.
7. Generates single-panel Matplotlib line charts for all unique tests in `data/processed/charts/`.
8. Persists structured JSON summaries to `data/processed/longitudinal_trends_summary.json`.
