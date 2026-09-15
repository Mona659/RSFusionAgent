"""Deterministic source-hit evaluation for the offline lexical RAG layer."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from rsfusion_agent.rag.retriever import LocalKnowledgeRetriever


class RagEvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_sources: list[str] = Field(min_length=1)


class RagEvaluationCaseResult(BaseModel):
    id: str
    query: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    source_hit: bool


class RagEvaluationReport(BaseModel):
    case_count: int
    source_hit_count: int
    source_hit_rate: float
    top_k: int
    cases: list[RagEvaluationCaseResult]


def _normalized_source(source: str) -> str:
    return Path(source).as_posix()


def load_rag_evaluation_cases(path: str | Path) -> list[RagEvaluationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("RAG evaluation file must contain a JSON list")
    cases = [RagEvaluationCase.model_validate(item) for item in payload]
    if not cases:
        raise ValueError("RAG evaluation file must not be empty")
    return cases


def evaluate_retriever(
    retriever: LocalKnowledgeRetriever,
    cases: list[RagEvaluationCase],
    *,
    top_k: int = 3,
) -> RagEvaluationReport:
    if top_k < 1 or top_k > 20:
        raise ValueError("top_k must be between 1 and 20")
    results: list[RagEvaluationCaseResult] = []
    for case in cases:
        retrieved = [_normalized_source(item.source) for item in retriever.search(case.query, top_k=top_k)]
        expected = [_normalized_source(item) for item in case.expected_sources]
        results.append(
            RagEvaluationCaseResult(
                id=case.id,
                query=case.query,
                expected_sources=expected,
                retrieved_sources=retrieved,
                source_hit=bool(set(expected).intersection(retrieved)),
            )
        )
    hit_count = sum(item.source_hit for item in results)
    return RagEvaluationReport(
        case_count=len(results),
        source_hit_count=hit_count,
        source_hit_rate=hit_count / len(results),
        top_k=top_k,
        cases=results,
    )


def evaluate_knowledge_directory(
    knowledge_dir: str | Path, evaluation_file: str | Path, *, top_k: int = 3
) -> RagEvaluationReport:
    return evaluate_retriever(
        LocalKnowledgeRetriever.from_directory(knowledge_dir),
        load_rag_evaluation_cases(evaluation_file),
        top_k=top_k,
    )


def write_rag_evaluation_report(path: str | Path, report: RagEvaluationReport) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


__all__ = [
    "RagEvaluationCase",
    "RagEvaluationCaseResult",
    "RagEvaluationReport",
    "evaluate_knowledge_directory",
    "evaluate_retriever",
    "load_rag_evaluation_cases",
    "write_rag_evaluation_report",
]
