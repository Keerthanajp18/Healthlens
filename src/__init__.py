"""
HealthLens - Multimodal RAG Framework for Simplifying and Comparing Medical Reports

Includes:
- Person 1: Document Processing & Medical Data Pipeline
- Person 2: Medical RAG (Retrieval-Augmented Generation) System
"""

__version__ = "1.0.0"

from src.pipeline import (
    MedicalDataPipeline,
    process_medical_report,
    compare_reports,
    analyze_trends,
)

__all__ = [
    "MedicalDataPipeline",
    "process_medical_report",
    "compare_reports",
    "analyze_trends",
]