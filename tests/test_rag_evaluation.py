import json
from pathlib import Path

from rsfusion_agent import cli
from rsfusion_agent.rag.evaluation import (
    RagEvaluationCase,
    evaluate_knowledge_directory,
    evaluate_retriever,
    write_rag_evaluation_report,
)
from rsfusion_agent.rag.retriever import LocalKnowledgeRetriever


def test_rag_evaluation_normalizes_windows_and_posix_source_separators(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    data = knowledge / "data"
    data.mkdir(parents=True)
    (data / "contract.md").write_text("真实实验需要 T1 MS T1 HS T2 MS。", encoding="utf-8")
    retriever = LocalKnowledgeRetriever.from_directory(knowledge)

    report = evaluate_retriever(
        retriever,
        [
            RagEvaluationCase(
                id="real-input", query="真实实验输入", expected_sources=["data\\contract.md"]
            )
        ],
    )

    assert report.source_hit_rate == 1.0


def test_knowledge_directory_evaluation_writes_json_report(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "runtime.md").write_text("模型权重仅在本地处理。", encoding="utf-8")
    cases = knowledge / "cases.json"
    cases.write_text(
        json.dumps(
            [{"id": "local", "query": "模型权重", "expected_sources": ["runtime.md"]}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = evaluate_knowledge_directory(knowledge, cases)
    output = write_rag_evaluation_report(tmp_path / "rag_report.json", report)

    assert report.source_hit_count == 1
    assert output.is_file()


def test_evaluation_prompt_files_are_not_included_as_rag_evidence(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    evaluation_dir = knowledge / "evaluations"
    evaluation_dir.mkdir(parents=True)
    (knowledge / "runtime.md").write_text("模型权重仅在本地处理。", encoding="utf-8")
    (evaluation_dir / "rag_eval.json").write_text(
        '[{"query": "模型权重", "expected_sources": ["runtime.md"]}]', encoding="utf-8"
    )

    result = LocalKnowledgeRetriever.from_directory(knowledge).search("模型权重", top_k=3)

    assert [item.source for item in result] == ["runtime.md"]


def test_cli_evaluate_rag_writes_requested_report(tmp_path: Path) -> None:
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "runtime.md").write_text("模型权重仅在本地处理。", encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(
        '[{"id": "local", "query": "模型权重", "expected_sources": ["runtime.md"]}]',
        encoding="utf-8",
    )
    report = tmp_path / "report.json"

    exit_code = cli.main(
        [
            "evaluate-rag",
            "--knowledge-dir",
            str(knowledge),
            "--evaluation-file",
            str(cases),
            "--output",
            str(report),
        ]
    )

    assert exit_code == 0
    assert json.loads(report.read_text(encoding="utf-8"))["source_hit_rate"] == 1.0
