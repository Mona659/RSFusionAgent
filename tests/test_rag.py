from __future__ import annotations

import json

from rsfusion_agent.rag.retriever import LocalKnowledgeRetriever


def test_local_retriever_returns_source_for_markdown_and_json(tmp_path) -> None:
    (tmp_path / "workflow.md").write_text("模拟实验目标 HS 作为高光谱真值。", encoding="utf-8")
    (tmp_path / "config.json").write_text(
        json.dumps({"scale": 3, "model": "YRE-151"}), encoding="utf-8"
    )
    retriever = LocalKnowledgeRetriever.from_directory(tmp_path)
    result = retriever.search("模拟实验 HS 真值", top_k=2)
    assert result
    assert result[0].source == "workflow.md"


def test_empty_query_is_rejected(tmp_path) -> None:
    (tmp_path / "note.md").write_text("内容", encoding="utf-8")
    retriever = LocalKnowledgeRetriever.from_directory(tmp_path)
    try:
        retriever.search(" ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty query should fail")
