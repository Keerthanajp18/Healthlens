import re
from dataclasses import dataclass, field
from typing import Any, Dict, List
from src.data_loader import MedicalDocument


@dataclass
class MedicalChunk:
    """Represents a discrete text chunk suitable for vector embedding and retrieval."""
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MedicalChunker:
    """Splits medical documents into semantically coherent overlapping chunks."""

    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 80):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: MedicalDocument) -> List[MedicalChunk]:
        """Splits a single MedicalDocument into MedicalChunk objects."""
        text = doc.content.strip()
        if not text:
            return []

        # If document is small enough, keep as single chunk
        if len(text) <= self.chunk_size:
            return [
                MedicalChunk(
                    chunk_id=f"{doc.doc_id}_0",
                    doc_id=doc.doc_id,
                    text=text,
                    metadata={
                        **doc.metadata,
                        "title": doc.title,
                        "category": doc.category,
                        "source": doc.source,
                        "chunk_index": 0,
                        "total_chunks": 1,
                        "char_count": len(text),
                    },
                )
            ]

        # Multi-stage split: first try paragraphs
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        raw_chunks: List[str] = []
        current_chunk = ""

        header_prefix = f"[{doc.title} - {doc.category}]\n"

        for para in paragraphs:
            # If paragraph itself is larger than chunk_size, split by sentences
            if len(para) > self.chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                        current_chunk = f"{current_chunk} {sent}".strip()
                    else:
                        if current_chunk:
                            raw_chunks.append(current_chunk)
                        current_chunk = sent
            else:
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                    current_chunk = f"{current_chunk}\n\n{para}".strip()
                else:
                    if current_chunk:
                        raw_chunks.append(current_chunk)
                    current_chunk = para

        if current_chunk:
            raw_chunks.append(current_chunk)

        # Now apply overlap if chunks are strictly partitioned
        final_chunks: List[MedicalChunk] = []
        total = len(raw_chunks)

        for i, chunk_text in enumerate(raw_chunks):
            # Prepend context header if not already containing the title
            enriched_text = chunk_text
            if doc.title.lower() not in chunk_text.lower()[:80]:
                enriched_text = f"{header_prefix}{chunk_text}"

            chunk_id = f"{doc.doc_id}_{i}"
            meta = {
                **doc.metadata,
                "title": doc.title,
                "category": doc.category,
                "source": doc.source,
                "chunk_index": i,
                "total_chunks": total,
                "char_count": len(enriched_text),
            }

            final_chunks.append(
                MedicalChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    text=enriched_text,
                    metadata=meta,
                )
            )

        return final_chunks

    def chunk_documents(self, documents: List[MedicalDocument]) -> List[MedicalChunk]:
        """Chunks a collection of MedicalDocuments."""
        all_chunks: List[MedicalChunk] = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        return all_chunks
