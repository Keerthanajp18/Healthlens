import sys
from pathlib import Path
import pytest
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.config import KNOWLEDGE_BASE_DIR
from src.data_loader import DataLoader
from src.preprocessor import TextCleaner
from src.chunker import MedicalChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.rag_pipeline import RAGPipeline


# ============================================================================
# Pytest Fixtures (Shared across module tests for efficiency)
# ============================================================================

@pytest.fixture(scope="module")
def raw_documents():
    loader = DataLoader(KNOWLEDGE_BASE_DIR)
    docs = loader.load_from_directory()
    return docs


@pytest.fixture(scope="module")
def cleaned_documents(raw_documents):
    cleaner = TextCleaner()
    return cleaner.clean_documents(raw_documents)


@pytest.fixture(scope="module")
def chunks(cleaned_documents):
    chunker = MedicalChunker(chunk_size=400, chunk_overlap=80)
    return chunker.chunk_documents(cleaned_documents)


@pytest.fixture(scope="module")
def embedding_manager():
    return EmbeddingManager()


@pytest.fixture(scope="module")
def vector_store(chunks, embedding_manager):
    dim = embedding_manager.dimension
    store = FaissVectorStore(dimension=dim)
    sample_chunks = chunks[:10]
    vecs = embedding_manager.embed_texts([c.text for c in sample_chunks])
    store.add_chunks(sample_chunks, vecs)
    return store


@pytest.fixture(scope="module")
def pipeline():
    p = RAGPipeline()
    p.build_index(force_rebuild=True)
    return p


# ============================================================================
# Test Cases
# ============================================================================

def test_document_loading(raw_documents):
    """Test 1: Verifies medical documents load from knowledge base directory."""
    assert len(raw_documents) > 0, "No documents loaded from medical knowledge directory"
    titles = [d.title for d in raw_documents]
    assert any("Hemoglobin" in t for t in titles), "Hemoglobin document missing"
    assert any("White Blood Cell" in t for t in titles), "WBC document missing"


def test_text_cleaning(raw_documents):
    """Test 2: Verifies text cleaning and medical unit normalization."""
    cleaner = TextCleaner()
    cleaned_docs = cleaner.clean_documents(raw_documents)
    assert len(cleaned_docs) == len(raw_documents), "Cleaned docs count mismatch"

    # Unit & range standardization test
    test_raw = "Hemoglobin  Hb   12.1-15.1 g/dL \u2014 \u03bcL"
    cleaned = cleaner.clean_text(test_raw)
    assert "12.1 - 15.1" in cleaned, "Reference range normalization failed"
    assert "µL" in cleaned, "Micro symbol normalization failed"


def test_chunking(cleaned_documents):
    """Test 3: Verifies semantic chunking and metadata retention."""
    chunker = MedicalChunker(chunk_size=400, chunk_overlap=80)
    chunks_list = chunker.chunk_documents(cleaned_documents)
    assert len(chunks_list) >= len(cleaned_documents), "Chunks must be at least as numerous as documents"
    for c in chunks_list[:5]:
        assert c.chunk_id, "Missing chunk_id"
        assert c.text, "Empty chunk text"
        assert "title" in c.metadata, "Missing title in chunk metadata"
        assert "category" in c.metadata, "Missing category in chunk metadata"


def test_embedding_generation(chunks, embedding_manager):
    """Test 4: Verifies embedding generation, vector dimension, and L2 normalization."""
    sample_texts = [chunks[0].text, chunks[1].text]
    vecs = embedding_manager.embed_texts(sample_texts)

    assert vecs.shape[0] == 2, "Vector count mismatch"
    assert vecs.shape[1] == embedding_manager.dimension, (
        f"Vector dimension mismatch ({vecs.shape[1]} vs {embedding_manager.dimension})"
    )

    # Cosine similarity requires L2 normalization (unit length)
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-3, f"Vectors must be L2 normalized (got norm={norm})"


def test_faiss_vector_store(chunks, embedding_manager, vector_store):
    """Test 5: Verifies FAISS vector database insertion and cosine similarity search."""
    assert vector_store.count() >= 5, "FAISS index should contain chunks"

    q_vec = embedding_manager.embed_query("hemoglobin levels and red blood cells")
    results = vector_store.search(q_vec, top_k=2)

    assert len(results) > 0, "FAISS search returned no results"
    top_chunk, top_score = results[0]
    assert top_score > 0.3, f"Similarity score unexpectedly low: {top_score}"


def test_rag_pipeline_status(pipeline):
    """Test 6: Verifies full RAG pipeline initialization and status diagnostics."""
    status = pipeline.get_status()
    assert status["indexed_chunks"] > 0, "Pipeline has 0 indexed chunks"
    assert status["embedding_dimension"] == pipeline.embedding_manager.dimension
    assert status["faiss_index_exists"] is True


def test_query_hemoglobin(pipeline):
    """Test 7: Mandatory verification query - 'What is hemoglobin?'"""
    query = "What is hemoglobin?"
    answer = pipeline.query(query)

    assert answer.answer and len(answer.answer) > 80, f"Answer too short or empty for query: {query}"
    assert len(answer.sources) > 0, f"No sources cited for query: {query}"
    assert answer.confidence_score > 0.4, f"Low confidence score ({answer.confidence_score})"
    assert any("Hemoglobin" in s for s in answer.sources), "Hemoglobin source not cited"
    assert "g/dL" in answer.answer or "oxygen" in answer.answer.lower(), "Core clinical facts missing in answer"


def test_query_wbc(pipeline):
    """Test 8: Mandatory verification query - 'What does WBC measure?'"""
    query = "What does WBC measure?"
    answer = pipeline.query(query)

    assert answer.answer and len(answer.answer) > 80, f"Answer too short or empty for query: {query}"
    assert len(answer.sources) > 0, f"No sources cited for query: {query}"
    assert answer.confidence_score > 0.4, f"Low confidence score ({answer.confidence_score})"
    assert any("White Blood Cell" in s or "WBC" in s for s in answer.sources), "WBC source not cited"
    assert "cells" in answer.answer.lower() or "infection" in answer.answer.lower(), "Core clinical facts missing in answer"


# ============================================================================
# Standalone execution support: python tests/test_phase1.py
# ============================================================================

def run_all_tests():
    print("\nRunning test suite via pytest...")
    retcode = pytest.main(["-v", str(Path(__file__))])
    return retcode


if __name__ == "__main__":
    sys.exit(run_all_tests())
