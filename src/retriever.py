import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from src.chunker import MedicalChunk
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.config import RETRIEVAL_TOP_K, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight applied to the metadata-match bonus during re-ranking.
# A value of 0.15 means a perfect metadata match adds up to 0.15 to the
# cosine score.  Tuned so that a chunk about the *correct* test out-ranks
# a chunk from a different test that happens to share similar clinical prose.
# ---------------------------------------------------------------------------
_METADATA_BOOST_WEIGHT = 0.15


@dataclass
class ScoredChunk:
    """A retrieved chunk paired with its cosine similarity score."""
    chunk: MedicalChunk
    score: float


@dataclass
class RetrievalResult:
    """Encapsulates retrieval output and formatted prompt context."""
    query: str
    scored_chunks: List[ScoredChunk] = field(default_factory=list)
    formatted_context: str = ""

    @property
    def has_results(self) -> bool:
        return len(self.scored_chunks) > 0


class MedicalRetriever:
    """Performs semantic similarity retrieval with metadata-aware re-ranking."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedding_manager: EmbeddingManager,
        top_k: int = RETRIEVAL_TOP_K,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        target_tests: Optional[List[str]] = None,
    ) -> RetrievalResult:
        """Retrieves top matching medical knowledge chunks for a given query

        with test-aware metadata ranking and safety filtering.
        """
        k = top_k or self.top_k
        min_thresh = threshold if threshold is not None else self.similarity_threshold

        query_cleaned = query.strip()
        if not query_cleaned:
            return RetrievalResult(query=query, scored_chunks=[], formatted_context="")

        # 1. Determine target tests & critical query intent
        active_targets = list(target_tests) if target_tests else self._extract_query_targets(query_cleaned)
        allow_critical = self._is_critical_query(query_cleaned)

        # 2. Embed query and search FAISS index with enlarged candidate pool
        query_vector = self.embedding_manager.embed_query(query_cleaned)
        fetch_k = max(k * 4, 15)
        raw_results = self.vector_store.search(query_vector, top_k=fetch_k)

        # 3. Test-aware & safety-aware re-ranking
        query_tokens = self._tokenize(query_cleaned)
        target_matches: List[ScoredChunk] = []
        other_candidates: List[ScoredChunk] = []

        for chunk, cosine_score in raw_results:
            # Rule: Suppress critical/panic chunks unless explicitly queried
            if not allow_critical and self._is_critical_chunk(chunk):
                continue

            # Determine test match status if active targets exist
            is_target_match = False
            is_conflicting_test = False

            if active_targets:
                for target in active_targets:
                    if self._check_test_match(target, chunk.metadata):
                        is_target_match = True
                        break
                    elif self._is_conflicting_test(target, chunk.metadata):
                        is_conflicting_test = True

            # Calculate bonus
            if is_target_match:
                # Strong test-specific boost to prioritize target test chunks
                meta_bonus = 0.35 + self._metadata_relevance(query_tokens, chunk.metadata) * _METADATA_BOOST_WEIGHT
            elif is_conflicting_test:
                # Conflicting different test (e.g. HbA1c when target is Hemoglobin)
                meta_bonus = -0.40
            else:
                meta_bonus = self._metadata_relevance(query_tokens, chunk.metadata) * _METADATA_BOOST_WEIGHT

            combined_score = round(cosine_score + meta_bonus, 4)
            scored = ScoredChunk(chunk=chunk, score=combined_score)

            if is_target_match:
                target_matches.append(scored)
            elif not is_conflicting_test:
                other_candidates.append(scored)

        # Sort both lists descending by combined score
        target_matches.sort(key=lambda sc: sc.score, reverse=True)
        other_candidates.sort(key=lambda sc: sc.score, reverse=True)

        # Combine results: target test chunks take absolute precedence
        if len(target_matches) >= k:
            selected_chunks = target_matches[:k]
        else:
            needed = k - len(target_matches)
            filler = [sc for sc in other_candidates if sc.score >= min_thresh][:needed]
            selected_chunks = target_matches + filler

        # Apply minimum similarity threshold
        scored_chunks = [sc for sc in selected_chunks if sc.score >= min_thresh]

        # Format context for generator
        formatted_context = self._format_context(scored_chunks)

        logger.info(
            f"Query: '{query}' (Targets: {active_targets}) -> Retrieved {len(scored_chunks)} chunks "
            f"(top score: {scored_chunks[0].score if scored_chunks else 0.0})"
        )

        return RetrievalResult(
            query=query,
            scored_chunks=scored_chunks,
            formatted_context=formatted_context,
        )

    # ------------------------------------------------------------------
    # Test-awareness & Disambiguation Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_critical_query(query: str) -> bool:
        """Detects if user is explicitly asking for critical or emergency threshold values."""
        q_lower = query.lower()
        return any(
            kw in q_lower for kw in [
                "critical", "panic", "emergency", "urgent", "danger",
                "threshold", "life-threatening", "transfusion", "cutoff"
            ]
        )

    @staticmethod
    def _is_critical_chunk(chunk: MedicalChunk) -> bool:
        """Detects if a knowledge chunk consists of critical / panic value alerts."""
        t_lower = chunk.text.lower()
        return (
            "critical / panic values" in t_lower
            or "panic values:" in t_lower
            or t_lower.startswith("critical values:")
            or "\ncritical / panic values:" in t_lower
        )

    @staticmethod
    def _extract_query_targets(query: str) -> List[str]:
        """Infers primary target lab tests from a query string."""
        q_lower = query.lower()
        targets = []

        # Disambiguate HbA1c vs standard Hemoglobin
        if any(k in q_lower for k in ["a1c", "hba1c", "glycated hemoglobin", "glycohemoglobin"]):
            targets.append("Hemoglobin A1c")
        elif any(k in q_lower for k in ["hemoglobin", "haemoglobin"]) or re.search(r"\b(hb|hgb)\b", q_lower):
            targets.append("Hemoglobin")

        if any(k in q_lower for k in ["white blood cell", "leukocyte"]) or re.search(r"\bwbc\b", q_lower):
            targets.append("White Blood Cell Count")
        if any(k in q_lower for k in ["platelet", "platelets", "thrombocyte"]) or re.search(r"\bplt\b", q_lower):
            targets.append("Platelet Count")
        if any(k in q_lower for k in ["red blood cell", "erythrocyte"]) or re.search(r"\brbc\b", q_lower):
            targets.append("Red Blood Cell Count")
        if "glucose" in q_lower or "blood sugar" in q_lower:
            targets.append("Blood Glucose (Fasting)")
        if "creatinine" in q_lower:
            targets.append("Serum Creatinine")
        if "crp" in q_lower:
            targets.append("C-Reactive Protein (CRP)")
        if "esr" in q_lower:
            targets.append("Erythrocyte Sedimentation Rate (ESR)")
        return targets

    @classmethod
    def _check_test_match(cls, target_test: str, metadata: dict) -> bool:
        """Determines if chunk metadata matches target_test, strictly distinguishing distinct tests."""
        t_clean = target_test.strip().lower()
        title = metadata.get("title", "").strip().lower()
        test_id = metadata.get("test_id", "").strip().lower()
        aliases = [a.strip().lower() for a in metadata.get("aliases", [])]
        category = metadata.get("category", "").strip().lower()

        # Strict Disambiguation: Standard Hemoglobin vs Hemoglobin A1c (HbA1c)
        target_is_a1c = any(k in t_clean for k in ["a1c", "glycated", "glycohemoglobin"])
        chunk_is_a1c = (
            "a1c" in title
            or any("a1c" in a for a in aliases)
            or "hba1c" in test_id
            or "glycemic" in category
        )
        target_is_hgb = ("hemoglobin" in t_clean or t_clean in ["hb", "hgb", "haemoglobin"]) and not target_is_a1c

        if target_is_hgb and chunk_is_a1c:
            return False
        if target_is_a1c and not chunk_is_a1c:
            return False

        # Match by test_id
        if t_clean == "hemoglobin" and test_id == "cbc_hgb":
            return True
        if target_is_a1c and test_id == "cmp_hba1c":
            return True

        # Match by title
        if t_clean == title or title in t_clean or t_clean in title:
            return True

        # Match by aliases
        for alias in aliases:
            if t_clean == alias or alias in t_clean or t_clean in alias:
                return True

        return False

    @classmethod
    def _is_conflicting_test(cls, target_test: str, metadata: dict) -> bool:
        """Detects if a chunk belongs to a semantically similar but distinct test (e.g. HbA1c vs Hemoglobin)."""
        t_clean = target_test.strip().lower()
        title = metadata.get("title", "").strip().lower()

        # HbA1c is a conflicting test when searching for standard Hemoglobin
        target_is_hgb = ("hemoglobin" in t_clean or t_clean in ["hb", "hgb", "haemoglobin"]) and "a1c" not in t_clean
        if target_is_hgb and ("a1c" in title or "cmp_hba1c" in metadata.get("test_id", "").lower()):
            return True

        # Standard Hemoglobin is a conflicting test when searching for HbA1c
        if "a1c" in t_clean and ("cbc_hgb" in metadata.get("test_id", "").lower() and "a1c" not in title):
            return True

        return False

    # ------------------------------------------------------------------
    # Context formatting
    # ------------------------------------------------------------------

    def _format_context(self, scored_chunks: List[ScoredChunk]) -> str:
        """Constructs an organized context string with source citations."""
        if not scored_chunks:
            return "No relevant clinical knowledge found for this query."

        context_blocks = []
        for i, item in enumerate(scored_chunks, start=1):
            chunk = item.chunk
            title = chunk.metadata.get("title", "Clinical Guide")
            category = chunk.metadata.get("category", "General")
            source = chunk.metadata.get("source", "Medical Reference")
            score_pct = int(item.score * 100)

            block = (
                f"--- [Document Reference {i}] ---\n"
                f"Test/Topic: {title} | Category: {category} | Source: {source} (Relevance: {score_pct}%)\n\n"
                f"{chunk.text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

