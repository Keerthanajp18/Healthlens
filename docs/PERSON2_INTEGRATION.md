# HealthLens: Person 1 to Person 2 Integration Specification

## Document Overview
This document specifies the technical interface and data contract between **Person 1** (Document Processing + Medical Data Pipeline) and **Person 2** (RAG + Knowledge Base + LLM Generation) in the **HealthLens** capstone project.

---

## 1. System Architecture & Separation of Responsibilities

```
========================================================================================
PERSON 1: DOCUMENT PROCESSING & DATA PIPELINE
========================================================================================
Medical Report (Digital PDF / Scanned PDF / Image / Text)
  │
  ▼
Multimodal Document Ingestion & File Type Detection
  │
  ▼
OCR Fallback (pypdfium2 + pytesseract if required)
  │
  ▼
Deterministic Structured Extraction (Regex & grammar parsing, zero PII)
  │
  ▼
Data Validation (Strictly using printed laboratory reference ranges)
  │
  ▼
Longitudinal Comparison & Historical Trend Analysis
  │
  ▼
Standardized Ground-Truth JSON Output (Zero PII, exact numbers)
=========================================│==============================================
                                         │  (JSON Contract / get_rag_context())
=========================================▼==============================================
PERSON 2: RAG + KNOWLEDGE BASE + LLM GENERATION
========================================================================================
Structured JSON Context Ingestion
  │
  ▼
Biomarker Vector Query & Knowledge Retrieval (Vector DB / ChromaDB / FAISS)
  │
  ▼
Medical Knowledge Retrieval (Trusted clinical references, Mayo Clinic / MedlinePlus)
  │
  ▼
Prompt Synthesis (Ground-truth values + retrieved medical knowledge)
  │
  ▼
LLM Generation (Multilingual, plain-language patient explanations)
  │
  ▼
Conversational Interactive QA
========================================================================================
```

### Clear Division of Responsibilities

| Responsibility Area | Person 1 (Document Pipeline) | Person 2 (RAG & LLM) |
|---|:---:|:---:|
| **File Formats Handled** | PDF, Scanned PDF, JPG, PNG, TXT | Pure Python Dictionaries / JSON |
| **OCR & Text Extraction** | Yes (Pillow, pypdf, pytesseract) | No |
| **Numerical Value Extraction** | Yes (exact decimals, comma integers) | No |
| **Reference Range Validation** | Yes (Deterministic laboratory rules) | No |
| **PII Anonymization** | Yes (Strips names, IDs, phones, addresses) | No (Assumes zero PII input) |
| **Longitudinal Math Deltas** | Yes (absolute & percentage change) | No |
| **Trend Line Chart Rendering**| Yes (Matplotlib single-panel PNGs) | No |
| **Vector Embeddings & Indexing**| No | Yes (ChromaDB, FAISS, SentenceTransformers) |
| **Knowledge Base Retrieval** | No | Yes (Medical guidelines, biomarker primers) |
| **LLM Inference & Explanations**| No | Yes (Gemini / Claude / OpenAI / Local LLMs) |
| **Patient Chat & Conversational QA**| No | Yes |

---

## 2. What Person 1 Provides

Person 1 provides Person 2 with clean, validated, deterministic medical ground truth through the module:
`src/integration/person2_interface.py`

Functions available for Person 2:
1. **`get_rag_context(processed_report)`**: Returns the core biomarker measurements and reference statuses needed for single-report patient explanations.
2. **`get_comparison_context(comparison_result)`**: Returns delta measurements across two dates for longitudinal change explanations.
3. **`get_trend_context(trend_result)`**: Returns multi-date trajectory data for multi-visit trend summaries.

---

## 3. Input Formats Accepted by Person 1's Interface

Person 2 can pass any of the following to `get_rag_context()`:
1. **Processed Report Dictionary:** The object returned directly by `src.pipeline.process_medical_report()`.
2. **Path to a Validated JSON File:** e.g., `"data/processed/sample_medical_report_validated.json"`.
3. **Path to a Raw Medical File:** e.g., `"data/sample_reports/sample_medical_report.pdf"` (Person 1's pipeline will process it automatically).
4. **List of Test Records:** e.g., `[{"test_name": "Hemoglobin", "value": 11.5, ...}]`.

---

## 4. Output Format Contract (RAG Context)

`get_rag_context()` returns a strictly structured, JSON-serializable Python dictionary:

```json
{
  "report_date": "10-Feb-2024",
  "tests": [
    {
      "test_name": "Hemoglobin (Hb)",
      "value": 11.5,
      "unit": "g/dL",
      "reference_range": "12.0 - 15.5",
      "status": "LOW",
      "category": "Complete Blood Count (CBC)",
      "lower_bound": 12.0,
      "upper_bound": 15.5
    },
    {
      "test_name": "RBC Count",
      "value": 3.95,
      "unit": "million/mcL",
      "reference_range": "3.80 - 5.10",
      "status": "NORMAL",
      "category": "Complete Blood Count (CBC)",
      "lower_bound": 3.8,
      "upper_bound": 5.1
    },
    {
      "test_name": "Total Leukocyte Count (WBC)",
      "value": 11200.0,
      "unit": "cells/mcL",
      "reference_range": "4000 - 11000",
      "status": "HIGH",
      "category": "Complete Blood Count (CBC)",
      "lower_bound": 4000.0,
      "upper_bound": 11000.0
    },
    {
      "test_name": "Total Cholesterol",
      "value": 215.0,
      "unit": "mg/dL",
      "reference_range": "< 200 Desirable",
      "status": "HIGH",
      "category": "Lipid Profile",
      "lower_bound": null,
      "upper_bound": 200.0
    },
    {
      "test_name": "HDL Cholesterol",
      "value": 48.0,
      "unit": "mg/dL",
      "reference_range": "> 50 Optimal",
      "status": "LOW",
      "category": "Lipid Profile",
      "lower_bound": 50.0,
      "upper_bound": null
    }
  ]
}
```

### Field Definitions for Person 2
* `report_date` *(str)*: Standardized date when the test specimen was collected or reported.
* `test_name` *(str)*: Normalized laboratory biomarker name.
* `value` *(float | int | None)*: Exact numerical measurement extracted from the report.
* `unit` *(str)*: Laboratory measurement unit (e.g. `g/dL`, `cells/mcL`, `mg/dL`, `%`).
* `reference_range` *(str)*: Laboratory-printed biological reference range.
* `status` *(str)*: One of `NORMAL`, `LOW`, `HIGH`, or `UNKNOWN`. Determined strictly by comparing `value` against `reference_range`.
* `category` *(str)*: Diagnostic panel (e.g. `Complete Blood Count (CBC)`, `Lipid Profile`).
* `lower_bound` *(float | None)*: Parsed numerical lower bound (if applicable).
* `upper_bound` *(float | None)*: Parsed numerical upper bound (if applicable).

---

## 5. How Person 2 Consumes This Interface

### Step-by-Step Code Example for Person 2

```python
# In Person 2's module (e.g., src/rag/pipeline.py):

from src.integration.person2_interface import get_rag_context, get_comparison_context

# Step 1: Ingest ground truth from Person 1
rag_context = get_rag_context("data/processed/sample_medical_report_validated.json")

report_date = rag_context["report_date"]
tests = rag_context["tests"]

# Step 2: Separate normal findings from flagged findings
abnormal_tests = [t for t in tests if t["status"] in ["LOW", "HIGH"]]
normal_tests = [t for t in tests if t["status"] == "NORMAL"]

# Step 3: Person 2 retrieves knowledge from Vector DB for flagged tests
retrieved_knowledge = []
for t in abnormal_tests:
    query = f"What causes {t['status']} {t['test_name']} and what does reference range {t['reference_range']} {t['unit']} mean?"
    # medical_info = vector_db.similarity_search(query)
    # retrieved_knowledge.append(medical_info)

# Step 4: Synthesize LLM prompt
prompt = f"""
You are a compassionate medical report assistant.
Explain the following patient laboratory results in simple, easy-to-understand terms.
Do NOT diagnose diseases. Do NOT provide prescriptive treatment.

Report Date: {report_date}

Flagged Results:
{abnormal_tests}

Normal Results:
{normal_tests}

Retrieved Medical Knowledge:
{retrieved_knowledge}
"""

# Step 5: Send prompt to LLM (Gemini, Claude, or local model)
# response = llm.generate(prompt)
```

---

## 6. What Person 1 Does NOT Provide (Guaranteed Boundaries)

To prevent duplication and maintain architectural hygiene:
1. **NO Patient PII:** Person 1 intentionally strips patient names, IDs, phone numbers, addresses, physician names, and hospital names. Person 2's LLM will never receive or hallucinate personal identity information.
2. **NO Medical Diagnoses:** Person 1 does not state that a patient has "anemia", "infection", or "diabetes". Statuses are strictly mathematical bounds classifications (`LOW`, `HIGH`, `NORMAL`).
3. **NO Subjective Improvement Claims:** In longitudinal comparisons, Person 1 reports mathematical directions (`INCREASE`, `DECREASE`, `UNCHANGED`), never "clinically improved" or "worsened".
4. **NO Vector Embeddings or Indexing:** Embeddings creation and vector storage are entirely within Person 2's purview.
5. **NO LLM API Keys or Prompts:** Person 1 runs 100% deterministically and offline with zero LLM dependencies.

---

## 7. Known Limitations & Technical Constraints

1. **Native Tesseract Requirement for Scanned PDFs/Images:**
   - Digital PDFs are extracted natively via `pypdf` with zero dependencies.
   - For scanned images and image-based PDFs, the host machine must have the native `tesseract.exe` binary installed (on Windows: `winget install UB-Mannheim.TesseractOCR`).
2. **Tabular Regular Expression Variations:**
   - The deterministic parser expects standard hospital/diagnostic lab layouts (test name, observed value, unit, reference interval). Highly irregular or non-standard handwritten notes are outside the current deterministic scope.
3. **Incompatible Unit Protection:**
   - When longitudinal reports use different units (e.g. `mg/dL` in January and `mmol/L` in April), Person 1 intentionally suppresses mathematical delta calculations and does not join points on charts to prevent incorrect arithmetic.
