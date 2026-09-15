"""Small local RAG building blocks used by the V2 agent."""

from rsfusion_agent.rag.evaluation import RagEvaluationReport, evaluate_knowledge_directory
from rsfusion_agent.rag.retriever import LocalKnowledgeRetriever, RetrievalResult

__all__ = [
    "LocalKnowledgeRetriever",
    "RagEvaluationReport",
    "RetrievalResult",
    "evaluate_knowledge_directory",
]
