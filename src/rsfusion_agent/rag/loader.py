"""Offline loaders for project Markdown and JSON knowledge files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceDocument:
    text: str
    source: str
    metadata: dict[str, Any]


def load_documents(root: str | Path) -> list[SourceDocument]:
    """Load supported files below ``root`` without network access."""
    directory = Path(root).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Knowledge directory does not exist: {directory}")
    documents: list[SourceDocument] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".json"}:
            continue
        if path.suffix.lower() == ".md":
            text = path.read_text(encoding="utf-8")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        if text.strip():
            documents.append(
                SourceDocument(
                    text=text,
                    source=str(path.relative_to(directory)),
                    metadata={"suffix": path.suffix.lower(), "source": str(path)},
                )
            )
    return documents
