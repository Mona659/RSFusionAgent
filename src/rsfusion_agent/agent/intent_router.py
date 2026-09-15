"""Structured, safety-first routing for natural-language UI requests."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rsfusion_agent.agent.llm_client import ResponsesClient
from rsfusion_agent.agent.llm_state import LLMTokenUsage


class IntentKind(str, Enum):
    """Small, explicit set of user intents supported by the local application."""

    KNOWLEDGE_QUERY = "knowledge_query"
    TASK_STATUS = "task_status"
    INSPECT_INPUTS = "inspect_inputs"
    INSPECT_CROP = "inspect_crop"
    PREPARE_CROP = "prepare_crop"
    RUN_FUSION = "run_fusion"
    CLARIFY = "clarify"


class IntentDecision(BaseModel):
    """Validated routing decision returned by rules or the LLM fallback."""

    intent: IntentKind
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=280)

    @property
    def requires_input_paths(self) -> bool:
        return self.intent in {
            IntentKind.INSPECT_INPUTS,
            IntentKind.INSPECT_CROP,
            IntentKind.PREPARE_CROP,
            IntentKind.RUN_FUSION,
        }


class IntentRoutingResult(BaseModel):
    """Observable result of one routing decision."""

    decision: IntentDecision
    source: str = Field(pattern="^(rule|llm|fallback)$")
    usage: LLMTokenUsage | None = None
    fallback_error: str | None = None


_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [item.value for item in IntentKind],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string", "minLength": 1, "maxLength": 280},
    },
    "required": ["intent", "confidence", "reason"],
    "additionalProperties": False,
}

_ROUTER_INSTRUCTIONS = """You classify a user's RSFusionAgent request. Do not answer it.
Call decide_user_intent exactly once. Choose knowledge_query for project concepts,
experimental rules, data requirements, error explanations, history questions, or any
read-only question. Choose task_status only when the user asks the current task state.
Choose inspect_inputs, inspect_crop, prepare_crop, or run_fusion only for an affirmative request to
perform that local operation. A phrase such as 'do not crop or fuse' is read-only, not
an execution request. Choose clarify only when the request is genuinely ambiguous.
Never infer an operation from the presence of a remote-sensing noun alone."""


def _normalized_execution_text(request: str) -> str:
    text = request.lower()
    for prefix in ("不要执行", "不执行", "无需执行", "不用执行", "do not run", "don't run"):
        text = text.replace(prefix, "")
    return text


def rule_based_intent(request: str) -> IntentDecision | None:
    """Handle clear requests locally, reserving LLM calls for real ambiguity."""

    normalized = request.lower()
    execution_text = _normalized_execution_text(request)
    if any(word in normalized for word in ("任务状态", "当前状态", "状态如何", "task state")):
        return IntentDecision(
            intent=IntentKind.TASK_STATUS,
            confidence=0.99,
            reason="请求只读取当前任务状态。",
        )
    if any(
        word in execution_text
        for word in (
            "执行融合", "开始融合", "运行融合", "继续融合", "执行推理", "开始推理",
            "继续推理", "run fusion", "infer",
        )
    ):
        return IntentDecision(
            intent=IntentKind.RUN_FUSION,
            confidence=0.99,
            reason="请求明确要求执行本地融合或推理。",
        )
    if any(
        word in execution_text
        for word in ("执行裁剪", "重新裁剪", "开始裁剪", "进行裁剪", "帮我裁剪", "剪裁", "run crop")
    ):
        return IntentDecision(
            intent=IntentKind.PREPARE_CROP,
            confidence=0.99,
            reason="请求明确要求执行本地裁剪。",
        )
    if any(
        word in execution_text
        for word in ("检查裁剪清单", "查看裁剪清单", "验证裁剪清单", "inspect crop manifest")
    ):
        return IntentDecision(
            intent=IntentKind.INSPECT_CROP,
            confidence=0.99,
            reason="请求明确要求检查已有裁剪清单。",
        )
    if any(word in execution_text for word in ("检查输入", "检查原始", "检查tiff", "检查 h5", "inspect input")):
        return IntentDecision(
            intent=IntentKind.INSPECT_INPUTS,
            confidence=0.99,
            reason="请求明确要求检查本地输入数据。",
        )
    if any(word in normalized for word in ("不要执行", "只查看", "仅查看", "历史实验", "错误解决", "知识库")):
        return IntentDecision(
            intent=IntentKind.KNOWLEDGE_QUERY,
            confidence=0.98,
            reason="请求属于只读项目问答或检索。",
        )
    return None


class IntentRouter:
    """Rules-first intent router with one schema-constrained LLM fallback."""

    def __init__(self, client: ResponsesClient) -> None:
        self.client = client

    @staticmethod
    def tool_definition() -> dict[str, Any]:
        return {
            "type": "function",
            "name": "decide_user_intent",
            "description": "Return the one safe route for this user request.",
            "parameters": _INTENT_SCHEMA,
            "strict": True,
        }

    def route(self, request: str) -> IntentRoutingResult:
        if not request.strip():
            raise ValueError("Natural-language request must not be empty")
        deterministic = rule_based_intent(request)
        if deterministic is not None:
            return IntentRoutingResult(decision=deterministic, source="rule")

        try:
            turn = self.client.respond(
                input_items=[{"role": "user", "content": request}],
                tools=[self.tool_definition()],
                instructions=_ROUTER_INSTRUCTIONS,
            )
            calls = [call for call in turn.tool_calls if call.name == "decide_user_intent"]
            if len(calls) != 1:
                raise RuntimeError("Intent model did not return exactly one routing decision")
            return IntentRoutingResult(
                decision=IntentDecision.model_validate(calls[0].arguments),
                source="llm",
                usage=turn.usage,
            )
        except Exception as exc:
            return IntentRoutingResult(
                decision=IntentDecision(
                    intent=IntentKind.KNOWLEDGE_QUERY,
                    confidence=0.0,
                    reason="意图识别服务不可用，已安全降级为只读知识问答。",
                ),
                source="fallback",
                fallback_error=f"{type(exc).__name__}: {exc}",
            )


__all__ = [
    "IntentDecision",
    "IntentKind",
    "IntentRouter",
    "IntentRoutingResult",
    "rule_based_intent",
]
