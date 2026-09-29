import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MedicalDocument:
    """Represents an ingested medical document or test entry."""
    doc_id: str
    title: str
    category: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLoader:
    """Loads medical documents from files and directories."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None

    def load_from_directory(self, dir_path: Optional[Path] = None) -> List[MedicalDocument]:
        """Loads all supported documents from a directory."""
        target_dir = Path(dir_path) if dir_path else self.data_dir
        if not target_dir or not target_dir.exists():
            raise FileNotFoundError(f"Directory not found: {target_dir}")

        documents: List[MedicalDocument] = []
        supported_extensions = [".json", ".md", ".txt"]

        files = sorted(
            [f for f in target_dir.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions]
        )

        for file_path in files:
            docs = self.load_file(file_path)
            documents.extend(docs)

        logger.info(f"Loaded {len(documents)} document entries from {len(files)} files in {target_dir}")
        return documents

    def load_file(self, file_path: Path) -> List[MedicalDocument]:
        """Loads documents from a single file based on its extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            return self._load_json(file_path)
        elif suffix in [".md", ".markdown"]:
            return self._load_markdown(file_path)
        elif suffix == ".txt":
            return self._load_text(file_path)
        else:
            logger.warning(f"Unsupported file format skipped: {file_path.name}")
            return []

    def _load_json(self, file_path: Path) -> List[MedicalDocument]:
        """Parses structured medical knowledge JSON files into documents."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: List[MedicalDocument] = []

        # If data is a list of test definitions
        if isinstance(data, list):
            for idx, entry in enumerate(data):
                doc_id = entry.get("test_id", f"{file_path.stem}_{idx}")
                title = entry.get("test_name", entry.get("title", f"Entry {idx}"))
                category = entry.get("category", "Laboratory Tests")

                # Compose rich content string representing this medical test
                content_parts = [
                    f"Test Name: {title}",
                    f"Category: {category}",
                ]
                if "aliases" in entry and entry["aliases"]:
                    content_parts.append(f"Aliases / Abbreviations: {', '.join(entry['aliases'])}")

                if "description" in entry:
                    content_parts.append(f"Description: {entry['description']}")

                if "what_it_measures" in entry:
                    content_parts.append(f"What It Measures: {entry['what_it_measures']}")

                if "standard_units" in entry:
                    content_parts.append(f"Standard Units: {entry['standard_units']}")

                if "reference_ranges" in entry:
                    ref_lines = []
                    if isinstance(entry["reference_ranges"], dict):
                        for group, r_range in entry["reference_ranges"].items():
                            ref_lines.append(f"  - {group.replace('_', ' ').capitalize()}: {r_range}")
                        content_parts.append("Reference Ranges:\n" + "\n".join(ref_lines))
                    else:
                        content_parts.append(f"Reference Range: {entry['reference_ranges']}")

                if "clinical_significance" in entry:
                    cs = entry["clinical_significance"]
                    if isinstance(cs, dict):
                        cs_lines = ["Clinical Significance:"]
                        for cond_key, details in cs.items():
                            cond_title = cond_key.replace('_', ' ').capitalize()
                            if isinstance(details, dict):
                                c_name = details.get("condition_name", cond_title)
                                cs_lines.append(f"  * {cond_title} ({c_name}):")
                                if "causes" in details:
                                    cs_lines.append(f"    - Causes: {'; '.join(details['causes'])}")
                                if "common_symptoms" in details:
                                    cs_lines.append(f"    - Symptoms: {'; '.join(details['common_symptoms'])}")
                                if "risks" in details:
                                    cs_lines.append(f"    - Risks: {'; '.join(details['risks'])}")
                            elif isinstance(details, str):
                                cs_lines.append(f"  * {cond_title}: {details}")
                        content_parts.append("\n".join(cs_lines))
                    elif isinstance(cs, str):
                        content_parts.append(f"Clinical Significance: {cs}")

                if "wbc_differential_overview" in entry:
                    diff_lines = ["WBC Differential Overview:"]
                    for cell, explanation in entry["wbc_differential_overview"].items():
                        diff_lines.append(f"  - {cell.capitalize()}: {explanation}")
                    content_parts.append("\n".join(diff_lines))

                if "components" in entry and isinstance(entry["components"], dict):
                    comp_lines = ["Sub-Components:"]
                    for comp_name, comp_info in entry["components"].items():
                        if isinstance(comp_info, dict):
                            fname = comp_info.get("full_name", comp_name)
                            rr = comp_info.get("reference_range", "N/A")
                            sig = comp_info.get("significance", "")
                            comp_lines.append(f"  * {comp_name} ({fname}): Range: {rr}. {sig}")
                    content_parts.append("\n".join(comp_lines))

                if "critical_values" in entry:
                    content_parts.append(f"Critical / Panic Values: {entry['critical_values']}")

                full_content = "\n\n".join(content_parts)

                metadata = {
                    "test_id": doc_id,
                    "title": title,
                    "category": category,
                    "aliases": entry.get("aliases", []),
                    "units": entry.get("standard_units", ""),
                    "source_file": file_path.name,
                }

                documents.append(
                    MedicalDocument(
                        doc_id=doc_id,
                        title=title,
                        category=category,
                        source=file_path.name,
                        content=full_content,
                        metadata=metadata,
                    )
                )

        return documents

    def _load_markdown(self, file_path: Path) -> List[MedicalDocument]:
        """Parses Markdown guides into distinct section documents."""
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        documents: List[MedicalDocument] = []
        # Split by level 2 headings '## '
        sections = text.split("\n## ")

        first_section = sections[0].strip()
        overall_title = file_path.stem.replace("_", " ").title()
        if first_section.startswith("# "):
            header_lines = first_section.split("\n", 1)
            overall_title = header_lines[0].replace("#", "").strip()

        for idx, sec in enumerate(sections[1:], start=1):
            lines = sec.strip().split("\n", 1)
            sec_title = lines[0].strip()
            sec_body = lines[1].strip() if len(lines) > 1 else ""

            full_text = f"Guide: {overall_title}\nSection: {sec_title}\n\n{sec_body}"
            doc_id = f"{file_path.stem}_sec_{idx}"

            documents.append(
                MedicalDocument(
                    doc_id=doc_id,
                    title=f"{sec_title} ({overall_title})",
                    category="Clinical Guidelines",
                    source=file_path.name,
                    content=full_text,
                    metadata={"source_file": file_path.name, "section": sec_title},
                )
            )

        # Fallback if no ## sections found
        if not documents:
            documents.append(
                MedicalDocument(
                    doc_id=file_path.stem,
                    title=overall_title,
                    category="Clinical Guidelines",
                    source=file_path.name,
                    content=text,
                    metadata={"source_file": file_path.name},
                )
            )

        return documents

    def _load_text(self, file_path: Path) -> List[MedicalDocument]:
        """Parses plain text files."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        return [
            MedicalDocument(
                doc_id=file_path.stem,
                title=file_path.stem.replace("_", " ").title(),
                category="General Knowledge",
                source=file_path.name,
                content=content,
                metadata={"source_file": file_path.name},
            )
        ]
