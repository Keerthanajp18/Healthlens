# HealthLens - Person 2 Module: Medical RAG System (Phase 1)

HealthLens Person 2 is an intelligent clinical laboratory question-answering and report-interpretation engine built using Retrieval-Augmented Generation (RAG).

**Phase 1** establishes the foundational medical knowledge base, text processing, chunking, dense semantic embeddings, FAISS vector indexing, similarity retrieval, and CLI question-answering pipeline.

---

## 1. Architecture Overview (Phase 1)

```
[ Medical Knowledge Base (.json, .md) ]
                   │
                   ▼
       [ DataLoader (src/data_loader.py) ]
                   │
                   ▼
       [ TextCleaner (src/preprocessor.py) ]
                   │
                   ▼
     [ MedicalChunker (src/chunker.py) ]
                   │
                   ▼
    [ EmbeddingManager (src/embeddings.py) ]  <── SentenceTransformers / Gemini
                   │
                   ▼
    [ FaissVectorStore (src/vector_store.py) ] ──> Persisted to vector_db/
                   │
                   ▼
   [ MedicalRetriever (src/retriever.py) ]
                   │
                   ▼
   [ MedicalGenerator (src/generator.py) ]   <── Gemini 2.5 Flash / Local Synthesizer
                   │
                   ▼
         [ CLI Interface (main.py) ]
```

---

## 2. Workspace & Folder Structure

```
HealthLens/
└── person2_rag/
    ├── data/
    │   └── medical_knowledge/
    │       ├── cbc_blood_tests.json             # Complete Blood Count (Hb, WBC, RBC, PLT, Indices)
    │       ├── metabolic_panel_tests.json       # CMP/BMP (Glucose, HbA1c, Creatinine, eGFR, BUN)
    │       ├── lipid_panel_tests.json           # Lipid Profile (Cholesterol, LDL, HDL, Triglycerides)
    │       ├── liver_thyroid_tests.json         # LFT & Thyroid (ALT, AST, Bilirubin, TSH, FT4)
    │       └── inflammatory_and_clinical_guide.md # Clinical Guide (CRP, hs-CRP, ESR, Ferritin)
    ├── src/
    │   ├── __init__.py                          # Package definition
    │   ├── config.py                            # Central configuration & hyperparameters
    │   ├── data_loader.py                       # Structured JSON & Markdown knowledge loader
    │   ├── preprocessor.py                      # Clinical text cleaning & medical unit normalizer
    │   ├── chunker.py                           # Section-aware & semantic medical chunker
    │   ├── embeddings.py                        # SentenceTransformers & Gemini embedding manager
    │   ├── vector_store.py                      # FAISS vector database (Cosine similarity & persistence)
    │   ├── retriever.py                         # Top-k similarity retrieval & prompt formatting
    │   ├── generator.py                         # Medical answer generation (Gemini / local synthesis)
    │   └── rag_pipeline.py                      # End-to-end orchestrator for indexing & querying
    ├── tests/
    │   └── test_phase1.py                       # Automated verification test suite
    ├── vector_db/
    │   ├── faiss_index.bin                      # Serialized binary FAISS IndexFlatIP
    │   └── metadata.json                        # Chunk texts and clinical metadata store
    ├── .env.example                             # Environment variable template
    ├── requirements.txt                         # Python dependencies
    ├── main.py                                  # Modular CLI entrypoint
    └── README.md                                # Phase 1 Documentation
```

---

## 3. What Each File Does

| File | Role / Functionality |
| :--- | :--- |
| `src/config.py` | Centralized paths, model configurations, chunking sizes, similarity thresholds, and medical system prompts. |
| `src/data_loader.py` | Parses raw medical knowledge documents (`.json`, `.md`, `.txt`) into standardized `MedicalDocument` objects. |
| `src/preprocessor.py` | Normalizes whitespace, cleans unicode, and standardizes clinical units (`mg/dL`, `g/dL`, `µL`, `cells/mcL`) and reference ranges. |
| `src/chunker.py` | Splits medical documents into semantically coherent overlapping chunks with contextual document headers. |
| `src/embeddings.py` | Encodes text chunks into L2-normalized dense semantic vectors using `sentence-transformers` (`all-MiniLM-L6-v2`) or Google Gemini (`text-embedding-004`). |
| `src/vector_store.py` | Manages the FAISS `IndexFlatIP` index (exact cosine similarity), adding vectors, querying, and disk serialization. |
| `src/retriever.py` | Coordinates query embedding, vector search, score filtering, and context structuring with source citations. |
| `src/generator.py` | Generates clinical answers using Google Gemini (`gemini-2.5-flash`) or an extractive clinical synthesizer when offline. |
| `src/rag_pipeline.py` | High-level controller wiring ingestion, indexing, retrieval, and generation into a unified interface. |
| `main.py` | CLI tool supporting `--index`, `--query`, `--interactive`, `--status`, and `--test`. |
| `tests/test_phase1.py` | Comprehensive test suite validating all 10 Phase 1 requirements. |

---

## 4. Setup & Installation

### Step 1: Install Dependencies
```bash
cd HealthLens/person2_rag
python -m pip install -r requirements.txt
```

### Step 2: Configure Environment Variables (Optional for Gemini)
Create a `.env` file in `person2_rag/` (copied from `.env.example`):
```bash
# Optional: Set your Gemini API key to enable generative LLM answers
GEMINI_API_KEY=your_gemini_api_key_here

# Embedding Backend: "sentence-transformers" (default, runs 100% locally)
EMBEDDING_BACKEND=sentence-transformers
```
> *Note:* HealthLens works **100% out of the box** even without an API key using the local SentenceTransformers model and grounded clinical synthesis mode.

---

## 5. Running the Pipeline (Commands)

### 1. Build / Rebuild the FAISS Vector Database
```bash
python main.py --index --force
```

### 2. Query a Medical Question
```bash
python main.py --query "What is hemoglobin?"
python main.py --query "What does WBC measure?"
python main.py --query "What causes high ALT and AST?"
```

### 3. Launch the Interactive Medical CLI Chat
```bash
python main.py --interactive
```

### 4. Check Pipeline & Vector Index Status
```bash
python main.py --status
```

### 5. Run the Automated Verification Test Suite
```bash
python tests/test_phase1.py
# or
python main.py --test
```

---

## 6. Verification Test Results

### Test Query 1: *"What is hemoglobin?"*
```text
[?] Query: What is hemoglobin?
----------------------------------------------------------------------
[+] Answer (Model: HealthLens Clinical Synthesizer (Local)):
### Clinical Overview: Hemoglobin
Category: Complete Blood Count (CBC) | Standard Units: g/dL (grams per deciliter)

Description: Hemoglobin is an iron-containing metalloprotein found inside red blood
cells (erythrocytes). Its primary physiological role is to transport oxygen from the
lungs to all peripheral body tissues and return carbon dioxide from tissues back to the
lungs for expiration. Hemoglobin accounts for the characteristic red color of blood and
represents about 96% of the red blood cells' dry content by weight.

What It Measures: Measures the concentration of the oxygen-carrying hemoglobin protein
in whole blood, directly reflecting the oxygen-carrying capacity of the blood.

Reference Ranges:
 - Adult males: 13.8 - 17.2 g/dL
 - Adult females: 12.1 - 15.1 g/dL
 - Pregnant females: 11.0 - 14.0 g/dL
 - Children: 11.0 - 16.0 g/dL (varies by age)
 - Newborns: 14.0 - 24.0 g/dL

Key Clinical Takeaways:
- Reference intervals may vary slightly depending on specific laboratory methodology
  and patient demographics.
- Critical or abnormal results should always be correlated with clinical symptoms and
  reviewed by a licensed healthcare provider.

> Medical Disclaimer: HealthLens provides informational guidance only and does not
substitute for professional medical diagnosis or clinical consultation.

[i] Sources Cited: Hemoglobin
[i] Top Retrieval Confidence: 77%
```

### Test Query 2: *"What does WBC measure?"*
```text
[?] Query: What does WBC measure?
----------------------------------------------------------------------
[+] Answer (Model: HealthLens Clinical Synthesizer (Local)):
### Clinical Overview: White Blood Cell Count
Category: Complete Blood Count (CBC) | Standard Units: cells/mcL (or cells/µL, or 10^3/µL, or x10^9/L)

What It Measures: Measures the total number of circulating white blood cells per
microliter of blood. It evaluates the body's immune competence, capacity to respond to
infections, inflammatory activity, and bone marrow cellular proliferation.

WBC Differential Overview:
 - Neutrophils: 50% - 70% of total WBC (acute bacterial infection defense)
 - Lymphocytes: 20% - 40% of total WBC (viral defense and antibody production)
 - Monocytes: 2% - 8% of total WBC (debris phagocytosis & antigen presentation)
 - Eosinophils: 1% - 4% of total WBC (parasitic defense and allergic response)
 - Basophils: 0.5% - 1% of total WBC (histamine & heparin release)

Standard Units: cells/mcL (or cells/µL, or 10^3/µL, or x10^9/L)

Key Clinical Takeaways:
- Normal adult reference range: 4,000 - 11,000 cells/mcL.
- Values < 2,000 cells/mcL carry critical infection risk; values > 30,000 cells/mcL
  warrant immediate urgent hematological workup.

> Medical Disclaimer: HealthLens provides informational guidance only and does not
substitute for professional medical diagnosis or clinical consultation.

[i] Sources Cited: White Blood Cell Count
[i] Top Retrieval Confidence: 63%
```

---

## 7. API Keys and Environment Variables

| Variable | Required? | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | Optional | `None` | Unlocks dynamic Gemini 2.5 Flash clinical reasoning and generation. |
| `EMBEDDING_BACKEND` | No | `sentence-transformers` | Options: `sentence-transformers`, `gemini`, `fallback`. |
| `SENTENCE_TRANSFORMER_MODEL` | No | `all-MiniLM-L6-v2` | HuggingFace embedding model name. |
| `GEMINI_GENERATION_MODEL` | No | `gemini-2.5-flash` | Gemini model for answer synthesis. |
| `RETRIEVAL_TOP_K` | No | `3` | Number of context chunks retrieved per query. |
| `SIMILARITY_THRESHOLD` | No | `0.20` | Minimum cosine similarity score threshold. |

---

## 8. What We Should Do Next (Phase 2 Preview)

Once you approve Phase 1, we can proceed to **Phase 2**:
1. **Patient Report Context Injection**: Ingest extracted lab report data (from Person 1's OCR pipeline) into the retrieval prompt.
2. **Flagged Values Analysis**: Correlate patient values against reference ranges (High / Low flags) and retrieve targeted clinical explanations.
3. **Advanced Medical Guardrails**: Clinical disclaimer enforcement, red-flag urgent symptom detection (e.g. panic lab values), and refusal of formal diagnosis requests.
4. **Interactive Web UI**: Streamlit or web dashboard with side-by-side report analysis and conversational RAG chat.
