import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union

from src.config import (
    KNOWLEDGE_BASE_DIR,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD,
)
from src.data_loader import DataLoader
from src.preprocessor import TextCleaner
from src.chunker import MedicalChunker
from src.embeddings import EmbeddingManager
from src.vector_store import FaissVectorStore
from src.retriever import MedicalRetriever
from src.generator import MedicalGenerator, GeneratedAnswer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-End Orchestrator for the HealthLens Medical RAG System."""

    def __init__(
        self,
        embedding_backend: Optional[str] = None,
        auto_load_index: bool = True,
    ):
        self.embedding_manager = EmbeddingManager(backend=embedding_backend)
        self.vector_store = self._init_vector_store(auto_load=auto_load_index)
        self.retriever = MedicalRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager,
            top_k=RETRIEVAL_TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
        )
        self.generator = MedicalGenerator()

    def _init_vector_store(self, auto_load: bool = True) -> FaissVectorStore:
        """Initializes or loads the FAISS vector database."""
        dim = self.embedding_manager.dimension
        if auto_load and FAISS_INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                store = FaissVectorStore.load(FAISS_INDEX_FILE, METADATA_FILE)
                if store.dimension == dim:
                    return store
                else:
                    logger.warning(
                        f"Stored index dimension ({store.dimension}) doesn't match current embedding dimension ({dim}). Creating new store."
                    )
            except Exception as e:
                logger.warning(f"Could not load pre-existing index ({e}). Creating fresh store.")

        return FaissVectorStore(dimension=dim)

    def build_index(
        self,
        data_dir: Optional[Path] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Loads knowledge documents, chunks, embeds, and builds the FAISS vector index."""
        kb_path = Path(data_dir or KNOWLEDGE_BASE_DIR)

        # Check if already indexed
        if not force_rebuild and self.vector_store.count() > 0:
            logger.info(f"Vector store already contains {self.vector_store.count()} indexed chunks.")
            return {
                "status": "already_indexed",
                "total_chunks": self.vector_store.count(),
                "dimension": self.vector_store.dimension,
            }

        logger.info(f"Building medical index from: {kb_path}")

        # 1. Document Loading
        loader = DataLoader(kb_path)
        raw_documents = loader.load_from_directory()
        if not raw_documents:
            raise ValueError(f"No valid medical documents found in {kb_path}")

        # 2. Text Cleaning & Normalization
        cleaner = TextCleaner()
        cleaned_documents = cleaner.clean_documents(raw_documents)

        # 3. Chunking
        chunker = MedicalChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = chunker.chunk_documents(cleaned_documents)
        logger.info(f"Generated {len(chunks)} chunks from {len(cleaned_documents)} medical documents.")

        # 4. Embeddings — use metadata-enriched text so vectors capture
        #    which medical entity each chunk belongs to (title, aliases, category).
        chunk_texts = [
            EmbeddingManager.build_embedding_text(c.text, c.metadata)
            for c in chunks
        ]
        embeddings = self.embedding_manager.embed_texts(chunk_texts)

        # 5. Vector Store Population & Persistence
        self.vector_store.clear()
        # Ensure dimension matches
        self.vector_store.dimension = embeddings.shape[1]
        self.vector_store.index = FaissVectorStore(dimension=embeddings.shape[1]).index
        self.vector_store.add_chunks(chunks, embeddings)
        self.vector_store.save(FAISS_INDEX_FILE, METADATA_FILE)

        return {
            "status": "success",
            "documents_ingested": len(cleaned_documents),
            "total_chunks_indexed": len(chunks),
            "embedding_dimension": embeddings.shape[1],
            "embedding_backend": self.embedding_manager.backend,
        }

    def query(self, query_text: str, top_k: Optional[int] = None) -> GeneratedAnswer:
        """Executes a full RAG retrieval and answer generation pipeline."""
        if self.vector_store.count() == 0:
            # Auto-build index if not yet populated
            logger.info("Vector store is empty. Triggering automatic initial indexing...")
            self.build_index()

        retrieval_result = self.retriever.retrieve(query=query_text, top_k=top_k)
        answer = self.generator.generate(retrieval_result)
        return answer

    def query_with_report(
        self,
        report_data: Union[Any, Dict[str, Any], Path, str],
        user_question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes report-aware retrieval using structured patient report and user question.

        Flow:
            Structured report JSON -> Identify relevant test(s) -> Build report-aware query
            -> Existing embedding model -> Existing FAISS index -> Relevant medical knowledge
        """
        from src.report_query import build_report_query

        if self.vector_store.count() == 0:
            logger.info("Vector store is empty. Triggering automatic initial indexing...")
            self.build_index()

        report_context = build_report_query(report_data, user_question)
        retrieval_result = self.retriever.retrieve(
            query=report_context.retrieval_query,
            top_k=top_k,
        )

        # Generate educational answer from retrieved context
        answer = self.generator.generate(retrieval_result)

        return {
            "report_context": report_context,
            "patient_report_info": report_context.format_patient_report_info(),
            "retrieval_query": report_context.retrieval_query,
            "retrieval_result": retrieval_result,
            "answer": answer,
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic metrics and system status."""
        return {
            "indexed_chunks": self.vector_store.count(),
            "embedding_backend": self.embedding_manager.backend,
            "embedding_dimension": self.embedding_manager.dimension,
            "gemini_api_key_configured": bool(self.generator.api_key),
            "generation_model": self.generator.model_name,
            "faiss_index_exists": FAISS_INDEX_FILE.exists(),
            "metadata_exists": METADATA_FILE.exists(),
        }
