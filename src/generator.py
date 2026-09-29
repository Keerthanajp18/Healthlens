import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.config import (
    GEMINI_API_KEY,
    GEMINI_GENERATION_MODEL,
    FALLBACK_GENERATION_MODEL,
    MEDICAL_SYSTEM_PROMPT,
)
from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """Encapsulates the final RAG answer and supporting metadata."""
    query: str
    answer: str
    sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    model_used: str = "Extractive Clinical Synthesizer"


class MedicalGenerator:
    """Generates medically accurate responses grounded in retrieved knowledge."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or GEMINI_GENERATION_MODEL
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the Gemini client if API key is available."""
        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Gemini client initialized successfully with model {self.model_name}.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client ({e}). Fallback synthesizer will be used.")
                self.client = None
        else:
            logger.info("No GEMINI_API_KEY detected. Running in Grounded Clinical Synthesis mode.")

    def generate(self, retrieval_result: RetrievalResult) -> GeneratedAnswer:
        """Generates a complete medical answer based on retrieved context."""
        query = retrieval_result.query

        # If no context was retrieved
        if not retrieval_result.has_results:
            return GeneratedAnswer(
                query=query,
                answer=(
                    "I could not locate specific clinical reference information regarding this inquiry in the "
                    "current HealthLens medical knowledge base. Please consult a qualified healthcare provider "
                    "or clinical laboratory specialist.\n\n"
                    "Disclaimer: HealthLens provides informational guidance only and does not substitute for "
                    "professional medical diagnosis or clinical consultation."
                ),
                sources=[],
                confidence_score=0.0,
                model_used="None",
            )

        top_score = retrieval_result.scored_chunks[0].score
        sources = list(
            dict.fromkeys(
                [c.chunk.metadata.get("title", c.chunk.metadata.get("source", "Reference"))
                 for c in retrieval_result.scored_chunks]
            )
        )

        # 1. Try Gemini generation if client is active
        if self.client:
            try:
                answer_text = self._call_gemini(query, retrieval_result.formatted_context)
                return GeneratedAnswer(
                    query=query,
                    answer=answer_text,
                    sources=sources,
                    confidence_score=top_score,
                    model_used=self.model_name,
                )
            except Exception as e:
                logger.error(f"Gemini API call failed ({e}). Falling back to local clinical synthesis.")

        # 2. Local Grounded Clinical Synthesis (Offline / Pre-API Key Mode)
        synthesized_text = self._synthesize_offline(query, retrieval_result)
        return GeneratedAnswer(
            query=query,
            answer=synthesized_text,
            sources=sources,
            confidence_score=top_score,
            model_used="HealthLens Clinical Synthesizer (Local)",
        )

    def _call_gemini(self, query: str, context: str) -> str:
        """Invokes Gemini using google-genai SDK."""
        prompt = (
            f"{MEDICAL_SYSTEM_PROMPT}\n\n"
            f"=== RETRIEVED MEDICAL CONTEXT ===\n"
            f"{context}\n\n"
            f"=== USER QUERY ===\n"
            f"{query}\n\n"
            f"=== CLINICAL ANSWER ==="
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            # Try fallback model if primary fails (e.g. rate limit or model name)
            logger.warning(f"Primary model {self.model_name} failed: {e}. Trying {FALLBACK_GENERATION_MODEL}")
            response = self.client.models.generate_content(
                model=FALLBACK_GENERATION_MODEL,
                contents=prompt,
            )
            return response.text.strip()

    def _format_reference_range_block(self, block_text: str) -> str:
        """Formats reference ranges clearly labeled as general educational information."""
        lines = block_text.split("\n")
        range_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("reference range") or stripped.lower().startswith("- **reference range"):
                continue
            if stripped:
                range_lines.append(stripped)

        header = (
            "**Reference Ranges (GENERAL EDUCATIONAL INFORMATION):**\n"
            "*(General educational reference only — not personalized to any patient. Reference intervals "
            "vary across individual laboratories, testing instruments, and patient demographics. When reviewing "
            "a patient's laboratory report, the specific reference range provided by that laboratory must always take priority.)*"
        )
        if range_lines:
            return header + "\n" + "\n".join(range_lines)
        return header + "\n" + block_text

    def _synthesize_offline(self, query: str, result: RetrievalResult) -> str:
        """Produces a structured, formatted clinical explanation directly from retrieved chunks

        adhering to safety, grounding, and general educational requirements.
        """
        query_lower = query.lower()

        # Query intent detection
        asks_critical = any(
            kw in query_lower for kw in [
                "critical", "panic", "emergency", "urgent", "danger", "transfusion", "threshold", "life-threatening"
            ]
        )
        asks_causes_or_disease = any(
            kw in query_lower for kw in [
                "cause", "disease", "why is", "why are", "high", "low", "abnormal", "elevat", "decreas",
                "symptom", "condition", "risk", "disorder", "diagnos", "significance"
            ]
        )
        asks_treatment = any(
            kw in query_lower for kw in [
                "treat", "medication", "cure", "therapy", "management", "drug"
            ]
        )

        top_chunk = result.scored_chunks[0].chunk
        primary_title = top_chunk.metadata.get("title", "Clinical Lab Test")
        primary_category = top_chunk.metadata.get("category", "Laboratory Medicine")
        units = top_chunk.metadata.get("units", "")

        sections: List[str] = [
            f"### Clinical Overview: {primary_title}",
            f"**Category:** {primary_category}" + (f" | **Standard Units:** {units}" if units else ""),
            "",
        ]

        # Extract and group blocks by topic title
        topic_blocks: Dict[str, List[str]] = {}
        seen_block_keys = set()

        for scored in result.scored_chunks:
            title = scored.chunk.metadata.get("title", "Reference")
            text = scored.chunk.text.strip()

            # Filter out top metadata/header markers
            lines = []
            for l in text.split("\n"):
                stripped = l.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    continue
                if stripped.startswith("Test Name:") or stripped.startswith("Category:"):
                    continue
                lines.append(l)

            clean_body = "\n".join(lines).strip()
            if not clean_body:
                continue

            raw_blocks = [b.strip() for b in clean_body.split("\n\n") if b.strip()]

            for block in raw_blocks:
                block_lower = block.lower()

                # 1. Critical / panic values: exclude unless specifically queried
                if "critical / panic values" in block_lower or "panic values" in block_lower or block_lower.startswith("critical values:"):
                    if not asks_critical:
                        continue

                # 2. Disease claims / clinical causes / abnormal levels:
                #    In a general educational inquiry (e.g. "what is hemoglobin?", "what does WBC measure?"),
                #    do not unnecessarily dump disease claims or causes unless asked.
                if block_lower.startswith("clinical significance:") or block_lower.startswith("- **clinical significance"):
                    if not (asks_causes_or_disease or asks_treatment):
                        continue

                # 3. Reference ranges: clearly label as GENERAL EDUCATIONAL INFORMATION
                if block_lower.startswith("reference range") or block_lower.startswith("- **reference range"):
                    formatted_block = self._format_reference_range_block(block)
                else:
                    formatted_block = block

                # Deduplicate repeated blocks across overlapping chunks
                dedup_key = formatted_block[:60].strip().lower()
                if dedup_key in seen_block_keys:
                    continue
                seen_block_keys.add(dedup_key)

                if title not in topic_blocks:
                    topic_blocks[title] = []
                topic_blocks[title].append(formatted_block)

        # Append primary topic blocks
        if primary_title in topic_blocks:
            for b in topic_blocks[primary_title]:
                sections.append(b)

        # Append closely related secondary topic blocks (if any)
        for title, blocks in topic_blocks.items():
            if title != primary_title and blocks:
                sections.append(f"\n#### Related Reference: {title}")
                for b in blocks:
                    sections.append(b)

        # Key Educational Notes & Disclaimer
        sections.append("\n**Key Educational Notes:**")
        sections.append(
            "- **General Educational Information:** Reference ranges and clinical descriptions provided here "
            "are for general educational purposes only. They are not personalized to any patient and must never "
            "be interpreted as an individual patient's personal laboratory reference range."
        )
        sections.append(
            "- **Priority of Patient Lab Reports:** When a patient's laboratory report is available, the specific "
            "reference range established by that testing laboratory always takes priority."
        )
        sections.append(
            "- Any abnormal or concerning lab results should always be evaluated in clinical context by a licensed healthcare provider."
        )
        sections.append(
            "\n> **Medical Disclaimer:** HealthLens provides informational guidance only and does not substitute for "
            "professional medical diagnosis, treatment, or clinical consultation."
        )

        return "\n\n".join(sections)
