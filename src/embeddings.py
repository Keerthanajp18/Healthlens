import logging
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import (
    EMBEDDING_BACKEND,
    SENTENCE_TRANSFORMER_MODEL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_API_KEY,
)

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embedding generation across different model providers."""

    def __init__(self, backend: Optional[str] = None):
        self.backend = (backend or EMBEDDING_BACKEND).lower()
        self._model = None
        self._dimension: Optional[int] = None
        self._init_backend()

    def _init_backend(self):
        """Initializes the selected embedding engine."""
        if self.backend == "sentence-transformers":
            try:
                import os
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer: {SENTENCE_TRANSFORMER_MODEL}...")
                self._model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
                if hasattr(self._model, "get_embedding_dimension"):
                    self._dimension = self._model.get_embedding_dimension()
                else:
                    self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"SentenceTransformer loaded successfully. Dimension: {self._dimension}")
                return
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Falling back to TF-IDF dense engine.")
                self.backend = "fallback"

        if self.backend == "gemini":
            if not GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not configured. Falling back to sentence-transformers.")
                self.backend = "sentence-transformers"
                self._init_backend()
                return
            try:
                from google import genai
                self._model = genai.Client(api_key=GEMINI_API_KEY)
                self._dimension = 768
                logger.info("Initialized Gemini Embedding client (text-embedding-004).")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client ({e}). Falling back.")
                self.backend = "fallback"

        if self.backend == "fallback":
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            self._model = {
                "tfidf": TfidfVectorizer(max_features=1000, stop_words="english"),
                "svd": TruncatedSVD(n_components=128, random_state=42),
                "fitted": False,
            }
            self._dimension = 128
            logger.info("Initialized fallback Dense TF-IDF / LSA embedding engine.")

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the generated vectors."""
        return self._dimension or 384

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates L2-normalized embeddings for a list of texts."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if self.backend == "sentence-transformers":
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalization for cosine similarity
            )
            return embeddings.astype(np.float32)

        elif self.backend == "gemini":
            vectors = []
            for text in texts:
                response = self._model.models.embed_content(
                    model=GEMINI_EMBEDDING_MODEL,
                    contents=text,
                )
                vec = response.embedding.values
                vectors.append(vec)
            arr = np.array(vectors, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms

        else:  # Fallback LSA
            tfidf = self._model["tfidf"]
            svd = self._model["svd"]
            if not self._model["fitted"]:
                tfidf_mat = tfidf.fit_transform(texts)
                n_samples, n_features = tfidf_mat.shape
                actual_components = min(self.dimension, n_samples, n_features)
                if actual_components < self.dimension:
                    svd.n_components = max(1, actual_components - 1)
                    self._dimension = svd.n_components
                dense = svd.fit_transform(tfidf_mat)
                self._model["fitted"] = True
            else:
                tfidf_mat = tfidf.transform(texts)
                dense = svd.transform(tfidf_mat)

            # L2 normalize
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (dense / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string, returning a 1D float32 vector."""
        vecs = self.embed_texts([query])
        return vecs[0]

    @staticmethod
    def build_embedding_text(text: str, metadata: Dict[str, Any]) -> str:
        """Constructs a metadata-enriched string for embedding a chunk.

        Prepends the test/topic title, aliases, and category to the chunk text
        so the embedding vector captures *which* medical entity the chunk is
        about.  This is the general mechanism that disambiguates chunks from
        different tests that otherwise share similar clinical vocabulary.
        """
        parts: List[str] = []

        title = metadata.get("title", "")
        if title:
            parts.append(f"Topic: {title}")

        aliases = metadata.get("aliases", [])
        if aliases:
            parts.append(f"Also known as: {', '.join(aliases)}")

        category = metadata.get("category", "")
        if category:
            parts.append(f"Category: {category}")

        # Separator between metadata header and content body
        if parts:
            return "\n".join(parts) + "\n\n" + text
        return text
