"""Dependency-light lexical retriever for an offline RAG MVP."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rsfusion_agent.rag.loader import load_documents
from rsfusion_agent.rag.splitter import KnowledgeChunk, split_documents

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    source: str
    score: float
    chunk_index: int


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class LocalKnowledgeRetriever:
    """Search Markdown/JSON chunks locally; no embedding model or network required."""

    def __init__(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            raise ValueError("Knowledge base is empty")
        self.chunks = chunks
        self._token_cache = [_tokens(chunk.text) for chunk in chunks]

    @classmethod
    def from_directory(
        cls, root: str | Path, *, chunk_size: int = 900, overlap: int = 120
    ) -> "LocalKnowledgeRetriever":
        return cls(split_documents(load_documents(root), chunk_size=chunk_size, overlap=overlap))

    def search(self, query: str, *, top_k: int = 4) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Knowledge query must not be empty")
        if top_k < 1 or top_k > 20:
            raise ValueError("top_k must be between 1 and 20")
        query_tokens = _tokens(query)
        scored = []
        for chunk, tokens in zip(self.chunks, self._token_cache, strict=True):
            overlap = len(query_tokens & tokens)
            if overlap:
                scored.append(
                    RetrievalResult(
                        text=chunk.text,
                        source=chunk.source,
                        score=overlap / max(len(query_tokens), 1),
                        chunk_index=chunk.chunk_index,
                    )
                )
        return sorted(scored, key=lambda item: (-item.score, item.source, item.chunk_index))[:top_k]
