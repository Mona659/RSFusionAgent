from typing import Any

from rsfusion_agent.agent.intent_router import IntentKind, IntentRouter, rule_based_intent
from rsfusion_agent.agent.llm_client import FunctionCall, ModelTurn
from rsfusion_agent.agent.llm_state import LLMTokenUsage


class FakeIntentClient:
    model = "fake-intent-model"
    provider = "custom"

    def __init__(self, turn: ModelTurn) -> None:
        self.turn = turn
        self.calls: list[dict[str, Any]] = []

    def respond(self, **kwargs: Any) -> ModelTurn:
        self.calls.append(kwargs)
        return self.turn


def test_clear_fusion_request_uses_free_rule_route() -> None:
    decision = rule_based_intent("请执行融合第 0 个 patch")

    assert decision is not None
    assert decision.intent is IntentKind.RUN_FUSION
    assert decision.confidence == 0.99


def test_natural_language_crop_request_uses_free_rule_route() -> None:
    decision = rule_based_intent("帮我进行剪裁")

    assert decision is not None
    assert decision.intent is IntentKind.PREPARE_CROP


def test_continue_fusion_uses_free_rule_route() -> None:
    decision = rule_based_intent("继续融合")

    assert decision is not None
    assert decision.intent is IntentKind.RUN_FUSION


def test_crop_manifest_inspection_uses_free_rule_route() -> None:
    decision = rule_based_intent("请检查裁剪清单")

    assert decision is not None
    assert decision.intent is IntentKind.INSPECT_CROP


def test_negated_operation_is_read_only_rule_route() -> None:
    decision = rule_based_intent("请查看当前任务状态，不要执行裁剪或融合")

    assert decision is not None
    assert decision.intent is IntentKind.TASK_STATUS


def test_ambiguous_wording_uses_schema_constrained_llm_fallback() -> None:
    client = FakeIntentClient(
        ModelTurn(
            response_id="intent-response",
            output_items=[],
            tool_calls=[
                FunctionCall(
                    call_id="intent-call",
                    name="decide_user_intent",
                    arguments={
                        "intent": "knowledge_query",
                        "confidence": 0.92,
                        "reason": "The request asks about the real-experiment input contract.",
                    },
                )
            ],
            usage=LLMTokenUsage(input_tokens=10, output_tokens=6),
        )
    )

    routed = IntentRouter(client).route("真实实验的输入数据要求是怎样的")

    assert routed.source == "llm"
    assert routed.decision.intent is IntentKind.KNOWLEDGE_QUERY
    assert routed.usage is not None
    assert client.calls[0]["tools"][0]["name"] == "decide_user_intent"


def test_router_safely_falls_back_when_llm_output_is_not_a_decision() -> None:
    client = FakeIntentClient(ModelTurn(response_id="bad-intent", output_text="I cannot decide."))

    routed = IntentRouter(client).route("说一下这个遥感任务")

    assert routed.source == "fallback"
    assert routed.decision.intent is IntentKind.KNOWLEDGE_QUERY
    assert routed.fallback_error is not None
