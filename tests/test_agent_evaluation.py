import json
from pathlib import Path

import pytest

from rsfusion_agent import cli
from rsfusion_agent.agent.evaluation import (
    AgentEvaluationCase,
    evaluate_agent_cases,
    evaluate_agent_file,
    load_agent_evaluation_cases,
    write_agent_evaluation_report,
)
from rsfusion_agent.agent.llm_workflow import _explicitly_requests_inference


def test_public_agent_replay_suite_is_deterministic() -> None:
    cases_path = Path(__file__).parents[1] / "knowledge" / "evaluations" / "agent_eval.json"

    report = evaluate_agent_file(cases_path)

    assert report.case_count >= 30
    assert report.metrics.task_success_rate == 1.0
    assert report.metrics.intent_accuracy == 1.0
    assert report.metrics.unauthorized_inference_execution_count == 0
    assert report.metrics.security_pass_rate == 1.0
    assert report.metrics.recovery_success_rate == 1.0


def test_inference_authorization_respects_negation() -> None:
    assert _explicitly_requests_inference("执行融合") is True
    assert _explicitly_requests_inference("检查输入，不要执行融合") is False
    assert _explicitly_requests_inference("检查输入，不要执行推理") is False


def test_agent_evaluation_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    payload = [
        {
            "id": "duplicate",
            "request": "知识库查询",
            "expected_intent": "knowledge_query",
            "turns": [{"response_id": "one", "output_text": "完成"}],
        },
        {
            "id": "duplicate",
            "request": "知识库查询",
            "expected_intent": "knowledge_query",
            "turns": [{"response_id": "two", "output_text": "完成"}],
        },
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_agent_evaluation_cases(path)


def test_agent_evaluation_cli_writes_json_and_markdown_report(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "id": "cli-case",
                    "request": "知识库查询",
                    "expected_intent": "knowledge_query",
                    "turns": [
                        {
                            "response_id": "cli-answer",
                            "output_text": "完成",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "agent_eval.json"

    assert (
        cli.main(
            [
                "evaluate-agent",
                "--evaluation-file",
                str(cases_path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert output.is_file()
    assert output.with_suffix(".md").is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["case_count"] == 1


def test_agent_evaluation_report_writer_uses_same_directory(tmp_path: Path) -> None:
    case = AgentEvaluationCase(
        id="writer-case",
        request="知识库查询",
        expected_intent="knowledge_query",
        turns=[{"response_id": "writer-answer", "output_text": "完成"}],
    )
    report = evaluate_agent_cases([case])
    json_path, markdown_path = write_agent_evaluation_report(tmp_path / "report.json", report)

    assert json_path == tmp_path / "report.json"
    assert markdown_path == tmp_path / "report.md"
    assert "Task success rate" in markdown_path.read_text(encoding="utf-8")
