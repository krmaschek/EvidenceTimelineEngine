"""Load one case's Markdown documents, keeping their original line numbers."""

import hashlib
from pathlib import Path

from evidence_timeline.models import SourceDocument


def load_documents(case_dir: Path) -> list[SourceDocument]:
    """Load every .md file in the folder, sorted by name. `A01.md` gets the ID `A01`."""
    paths = sorted(case_dir.glob("*.md"))
    if not paths:
        raise ValueError(f"No Markdown documents found in {case_dir}")

    documents = []
    for path in paths:
        content = path.read_bytes()
        documents.append(
            SourceDocument(
                document_id=path.stem,
                sha256=hashlib.sha256(content).hexdigest(),
                # splitlines() numbers lines the same way as the dataset's validator.
                lines=content.decode("utf-8").splitlines(),
            )
        )
    return documents
