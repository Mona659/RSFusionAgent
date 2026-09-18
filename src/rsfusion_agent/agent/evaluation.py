"""Offline, replayable task-level evaluation for the bounded Agent loop.

The evaluator deliberately runs the production ``LLMFusionAgent`` with a
scripted Responses client and a constrained in-memory toolbox.  It never
opens user rasters, starts a model process, or contacts an LLM provider.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from rsfusion_agent.agent.intent_router import IntentKind, IntentRouter
from rsfusion_agent.agent.llm_client import FunctionCall, ModelTurn
from rsfusion_agent.agent.llm_state import LLMTokenUsage
from rsfusion_agent.agent.llm_workflow import LLMFusionAgent
from rsfusion_agent.agent.task_state import TaskState


class ReplayFunctionCall(BaseModel):
    """One deterministic function call emitted by a replayed model turn."""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReplayTurn(BaseModel):
    """A serializable subset of a model response used by offline replay."""

    response_id: str = Field(min_length=1)
    output_text: str = ""
    tool_calls: list[ReplayFunctionCall] = Field(default_factory=list)
    usage: LLMTokenUsage | None = None

    def as_model_turn(self) -> ModelTurn:
        output_items = [
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False, sort_keys=True),
            }
            for call in self.tool_calls
        ]
        return ModelTurn(
            response_id=self.response_id,
            output_text=self.output_text,
            output_items=output_items,
            tool_calls=[
                FunctionCall(call_id=call.call_id, name=call.name, arguments=call.arguments)
                for call in self.tool_calls
            ],
            usage=self.usage,
        )


class ExpectedToolCall(BaseModel):
    """The ordered, exact call expected from one replay case."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluationCase(BaseModel):
    """Versioned, public input for a deterministic Agent evaluation case."""

    id: str = Field(min_length=1)
    request: str = Field(min_length=1)
    expected_intent: IntentKind
    intent_replay: ReplayTurn | None = None
    turns: list[ReplayTurn] = Field(min_length=1)
    allowed_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    prohibited_execution_tools: list[str] = Field(default_factory=list)
    expected_tool_sequence: list[ExpectedToolCall] = Field(default_factory=list)
    expected_status: Literal["completed", "completed_with_tool_errors", "agent_error"] = "completed"
    expected_error_type: str | None = None
    expected_final_stage: str | None = None
    fail_tools: dict[str, str] = Field(default_factory=dict)
    security_expectations: list[
        Literal[
            "inference_blocked",
            "path_injection_blocked",
            "duplicate_call_id_rejected",
            "stale_result_blocked",
        ]
    ] = Field(default_factory=list)
    expects_recovery: bool = False
    provider: str = "qwen"
    model: str = "qwen3.7-flash"

    @model_validator(mode="after")
    def validate_expected_error(self) -> AgentEvaluationCase:
        if self.expected_status == "agent_error" and not self.expected_error_type:
            raise ValueError("agent_error cases require expected_error_type")
        return self


class AgentEvaluationCaseResult(BaseModel):
    id: str
    request: str
    expected_intent: IntentKind
    actual_intent: IntentKind
    intent_source: str
    expected_status: str
    actual_status: str
    error_type: str | None = None
    expected_tool_sequence: list[ExpectedToolCall]
    actual_tool_sequence: list[ExpectedToolCall]
    executed_tools: list[str]
    tool_errors: list[str]
    intent_correct: bool
    required_tools_satisfied: bool
    tool_selection_valid: bool
    arguments_valid: bool
    dependency_order_valid: bool
    prohibited_execution_free: bool
    final_stage_matches: bool | None = None
    security_expectations_met: bool
    recovery_matches: bool | None = None
    task_success: bool
    elapsed_seconds: float = Field(ge=0)
    usage: LLMTokenUsage = Field(default_factory=LLMTokenUsage)
    estimated_cost: dict[str, Any] | None = None


class AgentEvaluationMetrics(BaseModel):
    case_count: int
    task_success_count: int
    task_success_rate: float
    intent_correct_count: int
    intent_accuracy: float
    tool_selection_precision: float
    tool_selection_recall: float
    argument_valid_rate: float
    dependency_order_violation_count: int
    dependency_order_violation_rate: float
    prohibited_execution_count: int
    unauthorized_inference_execution_count: int
    unauthorized_inference_execution_rate: float
    security_case_count: int
    security_pass_count: int
    security_pass_rate: float | None = None
    recovery_case_count: int
    recovery_success_count: int
    recovery_success_rate: float | None = None
    total_elapsed_seconds: float
    mean_elapsed_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_by_currency: dict[str, float] = Field(default_factory=dict)


class AgentEvaluationReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    case_count: int
    metrics: AgentEvaluationMetrics
    cases: list[AgentEvaluationCaseResult]


class ReplayResponsesClient:
    """ResponsesClient-compatible deterministic replay without network access."""

    def __init__(self, turns: list[ReplayTurn], *, provider: str, model: str) -> None:
        self._turns = [turn.as_model_turn() for turn in turns]
        self.provider = provider
        self.model = model

    def respond(
        self,
        *,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn:
        del input_items, tools, instructions
        if not self._turns:
            raise RuntimeError("Replay exhausted before the Agent completed")
        return self._turns.pop(0).model_copy(deep=True)


class _EvaluationToolbox:
    """Small deterministic toolbox used only to score control-plane behavior."""

    latest_result: None = None

    def __init__(self, case: AgentEvaluationCase) -> None:
        self.case = case
        self.task_state = TaskState()
        self.executed_tools: list[str] = []

    def definitions(self) -> list[dict[str, Any]]:
        return []

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in self.case.allowed_tools:
            raise PermissionError(f"Tool {name!r} is not allowed for this evaluation case")
        if any(key.lower() in {"path", "file", "filepath", "directory"} for key in arguments):
            raise PermissionError("Evaluation toolbox rejected an untrusted path argument")
        self.executed_tools.append(name)
        if name in self.case.fail_tools:
            raise RuntimeError(self.case.fail_tools[name])
        return {"ok": True, "tool": name}

    def record_tool_success(self, name: str) -> None:
        self.task_state.record(name)

    @staticmethod
    def sanitize_error(error: Exception) -> dict[str, Any]:
        return {
            "ok": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def load_agent_evaluation_cases(path: str | Path) -> list[AgentEvaluationCase]:
    """Load and validate a non-empty, uniquely identified JSON case list."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Agent evaluation file must contain a JSON list")
    cases = [AgentEvaluationCase.model_validate(item) for item in payload]
    if not cases:
        raise ValueError("Agent evaluation file must not be empty")
    duplicate_ids = [case_id for case_id, count in Counter(case.id for case in cases).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Agent evaluation case ids must be unique: {', '.join(sorted(duplicate_ids))}")
    return cases


def _route_case(case: AgentEvaluationCase) -> tuple[IntentKind, str]:
    client = ReplayResponsesClient(
        [case.intent_replay] if case.intent_replay is not None else [],
        provider=case.provider,
        model=case.model,
    )
    result = IntentRouter(client).route(case.request)
    return result.decision.intent, result.source


def _tool_sequence(trace: list[Any]) -> list[ExpectedToolCall]:
    return [ExpectedToolCall(name=item.name, arguments=item.arguments) for item in trace]


def _security_expectations_met(
    case: AgentEvaluationCase,
    *,
    result_status: str,
    error_type: str | None,
    trace: list[Any],
    executed_tools: list[str],
) -> bool:
    for expectation in case.security_expectations:
        if expectation == "inference_blocked":
            inference_attempts = [
                item
                for item in trace
                if item.name in {"run_yre151_fusion", "run_yre151_tiff_fusion"}
            ]
            if not inference_attempts or any(item.status != "failed" for item in inference_attempts):
                return False
            if any(
                name in executed_tools
                for name in ("run_yre151_fusion", "run_yre151_tiff_fusion")
            ):
                return False
        elif expectation == "path_injection_blocked":
            if not any(
                item.status == "failed" and item.output.get("error_type") == "PermissionError"
                for item in trace
            ):
                return False
        elif expectation == "duplicate_call_id_rejected":
            if result_status != "agent_error" or error_type != "RuntimeError":
                return False
        elif expectation == "stale_result_blocked":
            if not any(
                item.name in {"get_latest_fusion_result", "get_latest_tiff_fusion_result"}
                and item.status == "failed"
                for item in trace
            ):
                return False
    return True


def evaluate_agent_case(case: AgentEvaluationCase) -> AgentEvaluationCaseResult:
    """Replay one case through routing and the production bounded Agent loop."""

    actual_intent, intent_source = _route_case(case)
    toolbox = _EvaluationToolbox(case)
    started_at = time.perf_counter()
    run_result = None
    error_type: str | None = None
    try:
        run_result = LLMFusionAgent(
            client=ReplayResponsesClient(case.turns, provider=case.provider, model=case.model),
            toolbox=toolbox,  # type: ignore[arg-type]
        ).run(case.request)
        actual_status = run_result.status
        trace = run_result.trace
        usage = run_result.usage
        estimated_cost = (
            run_result.estimated_cost.model_dump(mode="json")
            if run_result.estimated_cost is not None
            else None
        )
    except Exception as exc:
        actual_status = "agent_error"
        error_type = type(exc).__name__
        trace = []
        usage = LLMTokenUsage()
        estimated_cost = None
    elapsed_seconds = time.perf_counter() - started_at

    actual_sequence = _tool_sequence(trace)
    expected_sequence = case.expected_tool_sequence
    actual_names = [item.name for item in actual_sequence]
    expected_names = [item.name for item in expected_sequence]
    required_tools_satisfied = set(case.required_tools).issubset(actual_names)
    tool_selection_valid = all(name in case.allowed_tools for name in actual_names)
    arguments_valid = actual_sequence == expected_sequence
    dependency_order_valid = actual_names == expected_names
    prohibited_execution_free = not set(case.prohibited_execution_tools).intersection(
        toolbox.executed_tools
    )
    final_stage_matches = (
        toolbox.task_state.stage.value == case.expected_final_stage
        if case.expected_final_stage is not None
        else None
    )
    recovery_observed = any(item.status == "failed" for item in trace) and any(
        item.status == "completed" for item in trace[1:]
    )
    recovery_matches = recovery_observed == case.expects_recovery if case.expects_recovery else None
    security_expectations_met = _security_expectations_met(
        case,
        result_status=actual_status,
        error_type=error_type,
        trace=trace,
        executed_tools=toolbox.executed_tools,
    )
    status_matches = actual_status == case.expected_status and (
        case.expected_error_type is None or error_type == case.expected_error_type
    )
    checks = [
        actual_intent == case.expected_intent,
        status_matches,
        required_tools_satisfied,
        tool_selection_valid,
        arguments_valid,
        dependency_order_valid,
        prohibited_execution_free,
        security_expectations_met,
    ]
    if final_stage_matches is not None:
        checks.append(final_stage_matches)
    if recovery_matches is not None:
        checks.append(recovery_matches)
    return AgentEvaluationCaseResult(
        id=case.id,
        request=case.request,
        expected_intent=case.expected_intent,
        actual_intent=actual_intent,
        intent_source=intent_source,
        expected_status=case.expected_status,
        actual_status=actual_status,
        error_type=error_type,
        expected_tool_sequence=expected_sequence,
        actual_tool_sequence=actual_sequence,
        executed_tools=toolbox.executed_tools,
        tool_errors=[item.output.get("error", "") for item in trace if item.status == "failed"],
        intent_correct=actual_intent == case.expected_intent,
        required_tools_satisfied=required_tools_satisfied,
        tool_selection_valid=tool_selection_valid,
        arguments_valid=arguments_valid,
        dependency_order_valid=dependency_order_valid,
        prohibited_execution_free=prohibited_execution_free,
        final_stage_matches=final_stage_matches,
        security_expectations_met=security_expectations_met,
        recovery_matches=recovery_matches,
        task_success=all(checks),
        elapsed_seconds=elapsed_seconds,
        usage=usage,
        estimated_cost=estimated_cost,
    )


def evaluate_agent_cases(cases: list[AgentEvaluationCase]) -> AgentEvaluationReport:
    """Evaluate a non-empty case collection and calculate stable aggregate metrics."""

    if not cases:
        raise ValueError("At least one Agent evaluation case is required")
    results = [evaluate_agent_case(case) for case in cases]
    expected_counter = Counter(
        item.name for result in results for item in result.expected_tool_sequence
    )
    actual_counter = Counter(item.name for result in results for item in result.actual_tool_sequence)
    true_positive = sum((expected_counter & actual_counter).values())
    expected_total = sum(expected_counter.values())
    actual_total = sum(actual_counter.values())
    security_cases = [case for case in cases if case.security_expectations]
    recovery_cases = [case for case in cases if case.expects_recovery]
    cost_by_currency: dict[str, float] = {}
    for result in results:
        if result.estimated_cost is None:
            continue
        currency = result.estimated_cost.get("currency")
        value = result.estimated_cost.get("estimated_cost")
        if isinstance(currency, str) and isinstance(value, (int, float)):
            cost_by_currency[currency] = cost_by_currency.get(currency, 0.0) + float(value)
    count = len(results)
    unauthorized_executions = sum(
        sum(name in {"run_yre151_fusion", "run_yre151_tiff_fusion"} for name in result.executed_tools)
        for result in results
        if "融合" not in result.request and "推理" not in result.request and "infer" not in result.request.lower()
    )
    metrics = AgentEvaluationMetrics(
        case_count=count,
        task_success_count=sum(result.task_success for result in results),
        task_success_rate=sum(result.task_success for result in results) / count,
        intent_correct_count=sum(result.intent_correct for result in results),
        intent_accuracy=sum(result.intent_correct for result in results) / count,
        tool_selection_precision=true_positive / actual_total if actual_total else 1.0,
        tool_selection_recall=true_positive / expected_total if expected_total else 1.0,
        argument_valid_rate=sum(result.arguments_valid for result in results) / count,
        dependency_order_violation_count=sum(not result.dependency_order_valid for result in results),
        dependency_order_violation_rate=sum(not result.dependency_order_valid for result in results)
        / count,
        prohibited_execution_count=sum(not result.prohibited_execution_free for result in results),
        unauthorized_inference_execution_count=unauthorized_executions,
        unauthorized_inference_execution_rate=unauthorized_executions / count,
        security_case_count=len(security_cases),
        security_pass_count=sum(
            result.security_expectations_met
            for result, case in zip(results, cases, strict=True)
            if case.security_expectations
        ),
        security_pass_rate=(
            sum(
                result.security_expectations_met
                for result, case in zip(results, cases, strict=True)
                if case.security_expectations
            )
            / len(security_cases)
            if security_cases
            else None
        ),
        recovery_case_count=len(recovery_cases),
        recovery_success_count=sum(
            result.recovery_matches is True
            for result, case in zip(results, cases, strict=True)
            if case.expects_recovery
        ),
        recovery_success_rate=(
            sum(
                result.recovery_matches is True
                for result, case in zip(results, cases, strict=True)
                if case.expects_recovery
            )
            / len(recovery_cases)
            if recovery_cases
            else None
        ),
        total_elapsed_seconds=sum(result.elapsed_seconds for result in results),
        mean_elapsed_seconds=sum(result.elapsed_seconds for result in results) / count,
        total_input_tokens=sum(result.usage.input_tokens for result in results),
        total_output_tokens=sum(result.usage.output_tokens for result in results),
        estimated_cost_by_currency={
            currency: round(value, 10) for currency, value in sorted(cost_by_currency.items())
        },
    )
    return AgentEvaluationReport(case_count=count, metrics=metrics, cases=results)


def evaluate_agent_file(path: str | Path) -> AgentEvaluationReport:
    """Load and evaluate a versioned Agent replay file."""

    return evaluate_agent_cases(load_agent_evaluation_cases(path))


def _markdown_report(report: AgentEvaluationReport) -> str:
    metrics = report.metrics
    rows = [
        "| Case | Success | Intent | Status | Tools |",
        "|---|---:|---:|---|---|",
    ]
    for case in report.cases:
        rows.append(
            f"| {case.id} | {case.task_success} | {case.intent_correct} | "
            f"{case.actual_status} | {', '.join(item.name for item in case.actual_tool_sequence) or '-'} |"
        )
    return (
        "# RSFusionAgent Agent evaluation report\n\n"
        f"- Schema version: `{report.schema_version}`\n"
        f"- Cases: {metrics.case_count}\n"
        f"- Task success rate: {metrics.task_success_rate:.3f}\n"
        f"- Intent accuracy: {metrics.intent_accuracy:.3f}\n"
        f"- Tool precision / recall: {metrics.tool_selection_precision:.3f} / "
        f"{metrics.tool_selection_recall:.3f}\n"
        f"- Argument valid rate: {metrics.argument_valid_rate:.3f}\n"
        f"- Dependency-order violations: {metrics.dependency_order_violation_count}\n"
        f"- Unauthorized inference executions: {metrics.unauthorized_inference_execution_count}\n"
        f"- Security pass rate: {metrics.security_pass_rate if metrics.security_pass_rate is not None else 'n/a'}\n"
        f"- Recovery success rate: {metrics.recovery_success_rate if metrics.recovery_success_rate is not None else 'n/a'}\n"
        f"- Replay elapsed seconds: {metrics.total_elapsed_seconds:.6f}\n"
        f"- Replayed tokens: input={metrics.total_input_tokens}, output={metrics.total_output_tokens}\n"
        f"- Estimated cost by currency: {metrics.estimated_cost_by_currency or 'n/a'}\n\n"
        "## Cases\n\n"
        + "\n".join(rows)
        + "\n"
    )


def write_agent_evaluation_report(path: str | Path, report: AgentEvaluationReport) -> tuple[Path, Path]:
    """Write JSON and a same-directory Markdown summary for one evaluation run."""

    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = json_path.with_suffix(".md")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


__all__ = [
    "AgentEvaluationCase",
    "AgentEvaluationCaseResult",
    "AgentEvaluationMetrics",
    "AgentEvaluationReport",
    "ExpectedToolCall",
    "ReplayFunctionCall",
    "ReplayTurn",
    "evaluate_agent_case",
    "evaluate_agent_cases",
    "evaluate_agent_file",
    "load_agent_evaluation_cases",
    "write_agent_evaluation_report",
]
