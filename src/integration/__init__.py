"""
HealthLens - Integration Package for Person 2 (RAG + LLM Generation)
Module: src/integration/__init__.py

Exposes clean, zero-PII data extraction helpers for Person 2's knowledge retrieval
and prompt grounding pipeline.
"""

from src.integration.person2_interface import (
    get_rag_context,
    get_comparison_context,
    get_trend_context,
)

__all__ = [
    "get_rag_context",
    "get_comparison_context",
    "get_trend_context",
]
