"""Offline retrieval of known error solutions."""

from __future__ import annotations

from pathlib import Path

from rsfusion_agent.rag.retriever import LocalKnowledgeRetriever, RetrievalResult


def search_error_solutions(
    knowledge_dir: str | Path, error: str, *, top_k: int = 4
) -> list[RetrievalResult]:
    if not error.strip():
        raise ValueError("Error query must not be empty")
    error_dir = Path(knowledge_dir).expanduser().resolve() / "errors"
    if not error_dir.exists():
        return []
    return LocalKnowledgeRetriever.from_directory(error_dir).search(error, top_k=top_k)
