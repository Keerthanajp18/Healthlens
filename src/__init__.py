"""
HealthLens - Multimodal RAG Framework for Simplifying and Comparing Medical Reports
Module: Document Processing & Medical Data Pipeline (Person 1)
"""

__version__ = "0.1.0"

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

