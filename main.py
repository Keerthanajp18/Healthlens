import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configure clean logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# Silence verbose third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logger = logging.getLogger("HealthLensCLI")

from src.rag_pipeline import RAGPipeline
from src.config import KNOWLEDGE_BASE_DIR


def print_banner():
    banner = """
======================================================================
            HEALTHLENS MEDICAL RAG SYSTEM (PHASE 1 & 2)               
    Clinical Lab Knowledge Base & Report-Aware Semantic Pipeline      
======================================================================
"""
    print(banner)


def run_index(pipeline: RAGPipeline, force: bool = False):
    print("\n[*] Starting Medical Knowledge Ingestion and FAISS Indexing...")
    res = pipeline.build_index(force_rebuild=force)
    print("\n[+] Indexing Complete!")
    print(f"    - Ingested Documents : {res.get('documents_ingested', 'N/A')}")
    print(f"    - Total Chunks       : {res.get('total_chunks_indexed', res.get('total_chunks', 'N/A'))}")
    print(f"    - Vector Dimension   : {res.get('embedding_dimension', res.get('dimension', 'N/A'))}")
    print(f"    - Embedding Backend  : {res.get('embedding_backend', 'N/A')}")
    print(f"    - Status             : {res.get('status')}")


def run_query(pipeline: RAGPipeline, question: str, top_k: int = 3):
    print(f"\n[?] Query: {question}")
    print("-" * 70)
    answer = pipeline.query(question, top_k=top_k)
    print(f"\n[+] Answer (Model: {answer.model_used}):")
    print(answer.answer)
    print("\n[i] Sources Cited:")
    for src in answer.sources:
        print(f"    - {src}")
    print(f"[i] Top Retrieval Confidence: {int(answer.confidence_score * 100)}%")
    print("=" * 70)


def run_report_query(pipeline: RAGPipeline, report_path: str, question: str, top_k: int = 3):
    print(f"\n[?] User Question: {question}")
    print(f"[*] Loading Patient Report: {report_path}")
    result = pipeline.query_with_report(report_path, question, top_k=top_k)
    report_ctx = result["report_context"]
    matched_names = ", ".join(t.name for t in report_ctx.matched_tests) if report_ctx.matched_tests else "All tests"
    print(f"[*] Matched Test(s): {matched_names}")
    print(f"[*] Generated Retrieval Query: {result['retrieval_query']}")
    print("-" * 70)
    print(result["patient_report_info"])
    print("-" * 70)
    print("RETRIEVED KNOWLEDGE:")
    retrieval_result = result["retrieval_result"]
    for i, scored in enumerate(retrieval_result.scored_chunks, 1):
        chunk_title = scored.chunk.metadata.get("title", "Clinical Reference")
        print(f"\n--- [Source {i}: {chunk_title} | Confidence: {int(scored.score * 100)}%] ---")
        lines = [l for l in scored.chunk.text.split("\n") if not l.startswith("[") and not l.startswith("Test Name:")]
        clean_text = "\n".join(lines).strip()
        print(clean_text)
    print("=" * 70)


def run_interactive(pipeline: RAGPipeline):
    print("\n[*] Starting Interactive Medical RAG Console. (Type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            q = input("\nEnter your clinical question: ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit", "q"]:
                print("Exiting HealthLens. Goodbye!")
                break
            run_query(pipeline, q)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting HealthLens. Goodbye!")
            break


def run_status(pipeline: RAGPipeline):
    status = pipeline.get_status()
    print("\n--- HealthLens RAG Pipeline Status ---")
    for k, v in status.items():
        print(f"  {k:28} : {v}")
    print("--------------------------------------\n")


def main():
    parser = argparse.ArgumentParser(
        description="HealthLens Person 2 - Medical RAG Command Line Interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Build or rebuild the FAISS vector index from medical knowledge documents",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild the index even if one already exists",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Ask a specific clinical question (e.g. --query 'What is hemoglobin?')",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to structured patient lab report JSON file (Phase 2)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive clinical Q&A session in the terminal",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display the status of the FAISS vector database and configuration",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run automated test suite",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sentence-transformers", "gemini", "fallback"],
        default=None,
        help="Override the embedding backend to use",
    )

    args = parser.parse_args()

    print_banner()

    # Initialize RAG Pipeline
    pipeline = RAGPipeline(embedding_backend=args.backend)

    if args.status:
        run_status(pipeline)
        return

    if args.index:
        run_index(pipeline, force=args.force)
        return

    if args.report:
        question = args.query or "Explain my laboratory test results."
        run_report_query(pipeline, args.report, question)
        return

    if args.query:
        run_query(pipeline, args.query)
        return

    if args.interactive:
        run_interactive(pipeline)
        return

    if args.test:
        from tests.test_phase1 import run_all_tests
        run_all_tests()
        return

    # Default action if no args provided: run status and show help
    run_status(pipeline)
    parser.print_help()


if __name__ == "__main__":
    main()
