import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np

from src.chunker import MedicalChunk
from src.config import FAISS_INDEX_FILE, METADATA_FILE, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


class FaissVectorStore:
    """FAISS-based vector database supporting cosine similarity search and metadata storage."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        # IndexFlatIP calculates inner product. Since vectors are L2 normalized, IP == Cosine Similarity.
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[MedicalChunk] = []

    def add_chunks(self, chunks: List[MedicalChunk], embeddings: np.ndarray):
        """Adds text chunks and their corresponding embedding vectors to the FAISS index."""
        if len(chunks) == 0:
            return

        if embeddings.shape[0] != len(chunks):
            raise ValueError(f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}.")

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        self.index.add(vectors)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks to FAISS index. Total count: {self.count()}")

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> List[Tuple[MedicalChunk, float]]:
        """Searches for the most similar chunks given a query vector."""
        if self.count() == 0:
            logger.warning("Vector index is empty.")
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        # Retrieve top_k matches
        k = min(top_k, self.count())
        scores, indices = self.index.search(query_vector, k)

        results: List[Tuple[MedicalChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))

        return results

    def save(self, index_file: Optional[Path] = None, metadata_file: Optional[Path] = None):
        """Saves the FAISS index and chunk metadata to disk."""
        target_index = Path(index_file or FAISS_INDEX_FILE)
        target_meta = Path(metadata_file or METADATA_FILE)

        target_index.parent.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(target_index))

        # Save metadata
        serialized_chunks = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "text": c.text,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]

        with open(target_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dimension": self.dimension,
                    "total_chunks": len(self.chunks),
                    "chunks": serialized_chunks,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Saved FAISS index ({self.count()} items) to {target_index} and metadata to {target_meta}")

    @classmethod
    def load(cls, index_file: Optional[Path] = None, metadata_file: Optional[Path] = None) -> "FaissVectorStore":
        """Loads a persisted FAISS index and its metadata from disk."""
        target_index = Path(index_file or FAISS_INDEX_FILE)
        target_meta = Path(metadata_file or METADATA_FILE)

        if not target_index.exists() or not target_meta.exists():
            raise FileNotFoundError(f"Index or metadata file not found at {target_index}, {target_meta}")

        with open(target_meta, "r", encoding="utf-8") as f:
            meta_data = json.load(f)

        dim = meta_data.get("dimension", 384)
        store = cls(dimension=dim)
        store.index = faiss.read_index(str(target_index))

        store.chunks = [
            MedicalChunk(
                chunk_id=item["chunk_id"],
                doc_id=item["doc_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
            )
            for item in meta_data.get("chunks", [])
        ]

        logger.info(f"Successfully loaded FAISS index with {store.count()} chunks (dim: {dim})")
        return store

    def count(self) -> int:
        """Returns the number of indexed vectors."""
        return self.index.ntotal

    def clear(self):
        """Resets the vector index."""
        self.index.reset()
        self.chunks.clear()
