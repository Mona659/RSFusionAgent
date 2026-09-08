from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rsfusion_agent import cli
from rsfusion_agent.agent.llm_client import (
    FunctionCall,
    ModelTurn,
    OpenAIResponsesClient,
    resolve_provider_settings,
)
from rsfusion_agent.agent.llm_state import LLMTokenUsage
from rsfusion_agent.agent.llm_tools import AgentToolbox, LLMToolContext
from rsfusion_agent.agent.llm_workflow import LLMFusionAgent
from rsfusion_agent.cli import _write_json_file
from rsfusion_agent.tools.model_runtime import RuntimePreflightResult


class FakeResponsesClient:
    model = "fake-tool-model"
    provider = "custom"

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = turns
        self.inputs: list[list[Any]] = []

    def respond(self, **kwargs: Any) -> ModelTurn:
        self.inputs.append(list(kwargs["input_items"]))
        return self.turns.pop(0)


class FakeToolbox:
    latest_result = None

    def __init__(self, *, fail_run: bool = False) -> None:
        self.fail_run = fail_run
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @staticmethod
    def definitions() -> list[dict[str, Any]]:
        return []

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "run_yre151_fusion" and self.fail_run:
            raise RuntimeError("synthetic runtime failure")
        return {"ok": True, "tool": name}

    @staticmethod
    def sanitize_error(error: Exception) -> dict[str, Any]:
        return {"ok": False, "error_type": type(error).__name__, "error": str(error)}


def _tool_turn(response_id: str, call_id: str, name: str) -> ModelTurn:
    arguments = {"patch_index": 0}
    return ModelTurn(
        response_id=response_id,
        output_items=[
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": '{"patch_index": 0}',
            }
        ],
        tool_calls=[FunctionCall(call_id=call_id, name=name, arguments=arguments)],
    )


def test_llm_agent_runs_inspect_then_inference_then_answers() -> None:
    client = FakeResponsesClient(
        [
            _tool_turn("response-1", "call-1", "inspect_yre151_h5"),
            _tool_turn("response-2", "call-2", "run_yre151_fusion"),
            ModelTurn(response_id="response-3", output_text="融合完成。"),
        ]
    )
    toolbox = FakeToolbox()

    result = LLMFusionAgent(client=client, toolbox=toolbox).run("检查后融合第 0 个 patch")

    assert result.status == "completed"
    assert result.answer == "融合完成。"
    assert result.turns == 3
    assert [item.name for item in result.trace] == [
        "inspect_yre151_h5",
        "run_yre151_fusion",
    ]
    assert all(item.status == "completed" for item in result.trace)
    assert client.inputs[1][-1]["type"] == "function_call_output"
    assert client.inputs[1][-1]["call_id"] == "call-1"


def test_llm_agent_returns_tool_error_to_model() -> None:
    client = FakeResponsesClient(
        [
            _tool_turn("response-1", "call-1", "run_yre151_fusion"),
            ModelTurn(response_id="response-2", output_text="模型运行失败，请检查环境。"),
        ]
    )
    toolbox = FakeToolbox(fail_run=True)

    result = LLMFusionAgent(client=client, toolbox=toolbox).run("运行融合")

    assert result.status == "completed_with_tool_errors"
    assert result.trace[0].status == "failed"
    assert "synthetic runtime failure" in client.inputs[1][-1]["output"]


def test_llm_agent_records_usage_and_qwen_cost_estimate() -> None:
    client = FakeResponsesClient(
        [
            ModelTurn(
                response_id="response-1",
                output_text="融合完成。",
                usage=LLMTokenUsage(input_tokens=12_000, output_tokens=2_000),
            )
        ]
    )
    client.provider = "qwen"
    client.model = "qwen3.7-flash"

    result = LLMFusionAgent(client=client, toolbox=FakeToolbox()).run("汇报结果")

    assert result.usage.input_tokens == 12_000
    assert result.usage.output_tokens == 2_000
    assert result.model_trace[0].response_id == "response-1"
    assert result.estimated_cost is not None
    assert result.estimated_cost.currency == "CNY"
    assert result.estimated_cost.estimated_cost == pytest.approx(0.004)


def test_llm_agent_records_completed_runtime_preflight() -> None:
    preflight = RuntimePreflightResult(
        python_executable="C:/env/python.exe",
        python_version="3.10.18",
        torch_version="2.5.1+cu121",
        requested_device="cuda",
        resolved_device="cuda",
        cuda_available=True,
        cuda_device_name="Synthetic GPU",
        cuda_total_memory_bytes=8_000_000_000,
        checkpoint_epoch=200,
        checkpoint_tensor_count=42,
    )
    client = FakeResponsesClient([ModelTurn(response_id="response-1", output_text="环境正常。")])

    result = LLMFusionAgent(
        client=client,
        toolbox=FakeToolbox(),
        runtime_preflight=preflight,
    ).run("汇报运行环境")

    assert result.runtime_preflight == preflight


def test_qwen_settings_accept_generic_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RSFUSION_LLM_MODEL", "qwen3.7-flash")
    monkeypatch.setenv("RSFUSION_LLM_BASE_URL", "https://example.invalid/v1")

    settings = resolve_provider_settings(provider="qwen")

    assert settings.provider == "qwen"
    assert settings.model == "qwen3.7-flash"
    assert settings.base_url == "https://example.invalid/v1"


def test_write_json_file_uses_utf8(tmp_path: Path) -> None:
    path = tmp_path / "agent_result.json"

    _write_json_file(path, {"answer": "融合完成。"}, indent=2)

    assert path.read_text(encoding="utf-8") == '{\n  "answer": "融合完成。"\n}\n'


def test_agent_does_not_construct_llm_client_when_preflight_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    attempted_client_construction = False

    def failing_preflight(**kwargs: object) -> None:
        raise RuntimeError("synthetic preflight failure")

    def forbidden_client(**kwargs: object) -> None:
        nonlocal attempted_client_construction
        attempted_client_construction = True
        raise AssertionError("LLM client must not be created after a failed preflight")

    monkeypatch.setattr(cli, "preflight_yre151_runtime", failing_preflight)
    monkeypatch.setattr(cli, "CompatibleResponsesClient", forbidden_client)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "agent",
                "--request",
                "运行融合",
                "--aux-h5",
                str(tmp_path / "aux.h5"),
                "--target-h5",
                str(tmp_path / "target.h5"),
                "--checkpoint",
                str(tmp_path / "checkpoint.pth"),
                "--model-python",
                str(tmp_path / "python.exe"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )

    assert exc_info.value.code == 2
    assert not attempted_client_construction


def test_toolbox_blocks_inference_before_inspection(tmp_path: Path) -> None:
    context = LLMToolContext(
        auxiliary_h5_path=tmp_path / "aux.h5",
        target_h5_path=tmp_path / "target.h5",
        checkpoint_path=tmp_path / "model.pth",
        model_python=tmp_path / "python.exe",
        output_dir=tmp_path / "output",
        patch_index=0,
    )
    toolbox = AgentToolbox(context)

    with pytest.raises(ValueError, match="Safety gate"):
        toolbox.execute("run_yre151_fusion", {"patch_index": 0})


def test_toolbox_redacts_configured_paths_from_errors(tmp_path: Path) -> None:
    auxiliary_path = tmp_path / "private" / "aux.h5"
    toolbox = AgentToolbox(
        LLMToolContext(
            auxiliary_h5_path=auxiliary_path,
            target_h5_path=tmp_path / "target.h5",
            checkpoint_path=tmp_path / "model.pth",
            model_python=tmp_path / "python.exe",
            output_dir=tmp_path / "output",
        )
    )

    output = toolbox.sanitize_error(FileNotFoundError(f"Missing {auxiliary_path}"))

    assert str(auxiliary_path) not in output["error"]
    assert "<auxiliary_h5_path>" in output["error"]
    assert output["diagnosis"]["category"] == "configuration"


def test_openai_adapter_normalizes_function_calls_without_network() -> None:
    class FakeResponsesAPI:
        def __init__(self) -> None:
            self.kwargs: dict[str, Any] = {}

        def create(self, **kwargs: Any) -> SimpleNamespace:
            self.kwargs = kwargs
            item = SimpleNamespace(
                type="function_call",
                call_id="call-1",
                name="inspect_yre151_h5",
                arguments='{"patch_index": 0}',
            )
            usage = SimpleNamespace(
                input_tokens=9,
                output_tokens=4,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
                output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            )
            return SimpleNamespace(id="response-1", output_text="", output=[item], usage=usage)

    responses_api = FakeResponsesAPI()
    client = OpenAIResponsesClient.__new__(OpenAIResponsesClient)
    client.model = "fake-openai-model"
    client.provider = "openai"
    client._client = SimpleNamespace(responses=responses_api)

    turn = client.respond(
        input_items=[{"role": "user", "content": "检查数据"}],
        tools=[],
        instructions="Use tools.",
    )

    assert turn.tool_calls[0].name == "inspect_yre151_h5"
    assert turn.tool_calls[0].arguments == {"patch_index": 0}
    assert responses_api.kwargs["store"] is False
    assert responses_api.kwargs["parallel_tool_calls"] is False
    assert responses_api.kwargs["include"] == ["reasoning.encrypted_content"]
    assert turn.usage == LLMTokenUsage(
        input_tokens=9,
        output_tokens=4,
        cached_input_tokens=2,
        reasoning_tokens=1,
    )
