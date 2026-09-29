"""
HealthLens - Combined Configuration

Includes:
- Person 1: Document Processing & Medical Data Pipeline
- Person 2: Medical RAG System
"""

import os
from pathlib import Path
from dotenv import load_dotenv


# ============================================================
# Base Project Directory
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# Environment Variables
# ============================================================

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


# ============================================================
# Data Directories - Person 1
# ============================================================

DATA_DIR = BASE_DIR / "data"

SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHARTS_DIR = PROCESSED_DATA_DIR / "charts"


# ============================================================
# Medical Knowledge Base - Person 2
# ============================================================

KNOWLEDGE_BASE_DIR = DATA_DIR / "medical_knowledge"

VECTOR_DB_DIR = BASE_DIR / "vector_db"

FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_DB_DIR / "metadata.json"


# Ensure essential directories exist

SAMPLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Supported Input Extensions - Person 1
# ============================================================

SUPPORTED_DOCUMENT_EXTENSIONS = [
    ".pdf",
    ".txt",
]

SUPPORTED_IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".bmp",
]


# ============================================================
# Chunking Configuration - Person 2
# ============================================================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))


# ============================================================
# Embedding Configuration - Person 2
# ============================================================

# Options:
# "sentence-transformers"
# "gemini"
# "fallback"

EMBEDDING_BACKEND = os.getenv(
    "EMBEDDING_BACKEND",
    "sentence-transformers"
)

SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "all-MiniLM-L6-v2"
)

GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "text-embedding-004"
)


# ============================================================
# Vector Search Configuration - Person 2
# ============================================================

RETRIEVAL_TOP_K = int(
    os.getenv("RETRIEVAL_TOP_K", "3")
)

SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.20")
)


# ============================================================
# Generation Configuration - Person 2
# ============================================================

GEMINI_API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)

GEMINI_GENERATION_MODEL = os.getenv(
    "GEMINI_GENERATION_MODEL",
    "gemini-2.5-flash"
)

FALLBACK_GENERATION_MODEL = "gemini-1.5-flash"


# ============================================================
# Medical RAG System Prompt - Person 2
# ============================================================

MEDICAL_SYSTEM_PROMPT = """
You are HealthLens AI, an expert, empathetic, and clinically rigorous
medical laboratory assistant.

Your goal is to explain laboratory tests, biomarkers, and clinical
findings clearly to users using ONLY the provided retrieved medical context.

Core Safety and Clinical Grounding Guidelines:

1. Strict Context Grounding:
   Base your answer strictly on the provided retrieved medical context.
   Do not invent reference ranges, medical conditions, or clinical
   guidance not present in the context.

2. Focus Directly on the User's Inquiry:

   - When the user asks a general educational question such as
     "What is hemoglobin?" or "What does WBC measure?", focus your
     response directly on answering that specific question.

   - Explain what the biomarker/test is, its physiological function,
     and what it measures.

   - Do NOT unnecessarily include critical/panic values, alarming
     disease claims, or treatment-related information unless directly
     relevant to the user's specific question and explicitly supported
     by the retrieved context.

3. Reference Ranges as General Educational Information:

   - Clearly label any general reference-range information as
     "GENERAL EDUCATIONAL INFORMATION".

   - Never imply or state that a general reference range is the
     patient's personal laboratory reference range.

   - Explicitly clarify that reference intervals vary across individual
     clinical laboratories, instruments, methodologies, and patient
     demographics.

   - When a patient's lab report is reviewed, the specific
     laboratory-provided reference range on that report must always
     take priority.

4. Keep explanations accessible to patients while maintaining
   scientific precision.

5. Always include the standard medical disclaimer:

   "Disclaimer: HealthLens provides informational guidance only and
   does not substitute for professional medical diagnosis or clinical
   consultation."
"""