"""Deterministic character-window splitter for the local knowledge base."""

from __future__ import annotations

from dataclasses import dataclass

from rsfusion_agent.rag.loader import SourceDocument


@dataclass(frozen=True)
class KnowledgeChunk:
    text: str
    source: str
    chunk_index: int
    metadata: dict[str, object]


def split_documents(
    documents: list[SourceDocument], *, chunk_size: int = 900, overlap: int = 120
) -> list[KnowledgeChunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    chunks: list[KnowledgeChunk] = []
    step = chunk_size - overlap
    for document in documents:
        text = document.text.strip()
        for index, start in enumerate(range(0, len(text), step)):
            chunk = text[start : start + chunk_size].strip()
            if not chunk:
                continue
            chunks.append(
                KnowledgeChunk(
                    text=chunk,
                    source=document.source,
                    chunk_index=index,
                    metadata=dict(document.metadata),
                )
            )
            if start + chunk_size >= len(text):
                break
    return chunks
