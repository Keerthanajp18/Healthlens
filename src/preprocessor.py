import re
import unicodedata
from typing import List
from src.data_loader import MedicalDocument


class TextCleaner:
    """Preprocesses and normalizes clinical medical text."""

    def __init__(self):
        # Common unicode normalization replacements
        self.char_map = {
            "\u00a0": " ",      # non-breaking space
            "\u2013": "-",      # en dash
            "\u2014": " - ",    # em dash
            "\u2018": "'",      # left single quote
            "\u2019": "'",      # right single quote
            "\u201c": '"',      # left double quote
            "\u201d": '"',      # right double quote
            "\u03bc": "µ",      # greek small mu to micro sign
            "\u00b2": "^2",     # superscript 2
            "\u00b3": "^3",     # superscript 3
        }

    def clean_text(self, text: str) -> str:
        """Cleans and standardizes raw text."""
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        for src, target in self.char_map.items():
            text = text.replace(src, target)

        # Replace excessive whitespace within lines
        text = re.sub(r"[ \t]+", " ", text)

        # Replace 3 or more newlines with double newline to preserve paragraph separation
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

        # Standardize medical reference ranges: e.g., "12.1 - 15.1"
        text = re.sub(r"(\d+(\.\d+)?)\s*-\s*(\d+(\.\d+)?)", r"\1 - \3", text)

        return text.strip()

    def clean_document(self, doc: MedicalDocument) -> MedicalDocument:
        """Applies text cleaning to a MedicalDocument instance."""
        cleaned_content = self.clean_text(doc.content)
        cleaned_title = self.clean_text(doc.title)
        return MedicalDocument(
            doc_id=doc.doc_id,
            title=cleaned_title,
            category=doc.category,
            source=doc.source,
            content=cleaned_content,
            metadata=doc.metadata.copy(),
        )

    def clean_documents(self, docs: List[MedicalDocument]) -> List[MedicalDocument]:
        """Cleans a list of documents."""
        return [self.clean_document(doc) for doc in docs]
