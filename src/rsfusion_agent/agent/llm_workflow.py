"""Natural-language control plane using allowlisted function tools."""

from __future__ import annotations

import json
import time
from typing import Any

from rsfusion_agent.agent.llm_client import ResponsesClient, estimate_cost
from rsfusion_agent.agent.llm_state import (
    LLMModelTrace,
    LLMTokenUsage,
    LLMToolTrace,
    NaturalLanguageRunResult,
)
from rsfusion_agent.agent.llm_tools import AgentToolbox
from rsfusion_agent.tools.model_runtime import RuntimePreflightResult

SYSTEM_INSTRUCTIONS = """You are the control plane for RSFusionAgent.
Use only the supplied function tools. Never claim a file was inspected or a model was run
unless the corresponding tool returned ok=true. Do not request or manipulate image arrays.
The CLI has already authorized all paths, crop settings and patch index.

Follow the user's requested scope exactly. For an original-TIFF inspection or RGB-preview-only
request, call inspect_raw_tiff_inputs and then answer; do not crop or run fusion. For a crop-only
request, inspect raw TIFF inputs first and then call prepare_tiff_crop. For TIFF inference, use
an existing crop manifest when available; otherwise inspect raw inputs, prepare the crop, inspect
the crop manifest, and then run fusion. For any inference request, call the relevant inspect tool
before the run tool. Never run fusion unless the user explicitly asks to fuse, infer, test, or
generate a fusion result. The runtime preflight is already computed locally; call
get_runtime_preflight only when the user explicitly asks about the runtime environment.
Do not ask the user to reply with confirmation: every request is a self-contained execution.
If a tool returns an error, explain it or choose a safe recovery tool. Answer in the user's language.
"""


def _explicitly_requests_inference(request: str) -> bool:
    """Keep an RGB/input-inspection request from silently becoming a costly model run."""

    normalized = request.lower()
    return any(
        keyword in normalized
        for keyword in ("融合", "推理", "inference", "infer", "运行模型", "执行模型", "测试当前patch")
    )


class LLMFusionAgent:
    """Run a bounded Responses API function-calling loop."""

    def __init__(
        self,
        *,
        client: ResponsesClient,
        toolbox: AgentToolbox,
        max_turns: int = 6,
        runtime_preflight: RuntimePreflightResult | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self.client = client
        self.toolbox = toolbox
        self.max_turns = max_turns
        self.runtime_preflight = runtime_preflight
        if runtime_preflight is not None:
            self.toolbox.runtime_preflight = runtime_preflight

    def run(self, request: str) -> NaturalLanguageRunResult:
        if not request.strip():
            raise ValueError("Natural-language request must not be empty")

        provider = getattr(self.client, "provider", "custom")
        input_items: list[Any] = [{"role": "user", "content": request}]
        trace: list[LLMToolTrace] = []
        model_trace: list[LLMModelTrace] = []
        usage = LLMTokenUsage()
        seen_call_ids: set[str] = set()
        inference_authorized = _explicitly_requests_inference(request)

        for round_index in range(1, self.max_turns + 1):
            turn = self.client.respond(
                input_items=input_items,
                tools=self.toolbox.definitions(),
                instructions=SYSTEM_INSTRUCTIONS,
            )
            if turn.usage is not None:
                usage = usage.combined_with(turn.usage)
                model_trace.append(
                    LLMModelTrace(
                        round_index=round_index,
                        response_id=turn.response_id,
                        usage=turn.usage,
                    )
                )
            input_items.extend(turn.output_items)

            if not turn.tool_calls:
                answer = turn.output_text.strip()
                if not answer:
                    raise RuntimeError("The model returned neither text nor function calls")
                status = (
                    "completed_with_tool_errors"
                    if any(item.status == "failed" for item in trace)
                    else "completed"
                )
                return NaturalLanguageRunResult(
                    status=status,
                    model=self.client.model,
                    request=request,
                    answer=answer,
                    turns=round_index,
                    trace=trace,
                    provider=provider,
                    usage=usage,
                    model_trace=model_trace,
                    estimated_cost=estimate_cost(
                        provider=provider,
                        model=self.client.model,
                        usage=usage,
                    ),
                    runtime_preflight=self.runtime_preflight,
                    fusion_result=self.toolbox.latest_result,
                )

            for call in turn.tool_calls:
                if not call.call_id or call.call_id in seen_call_ids:
                    raise RuntimeError(f"Invalid or duplicate function call id: {call.call_id!r}")
                seen_call_ids.add(call.call_id)
                started_at = time.perf_counter()
                try:
                    if call.name in {"run_yre151_fusion", "run_yre151_tiff_fusion"} and not inference_authorized:
                        raise PermissionError(
                            "Inference was not authorized by this request. Ask explicitly to run fusion or inference."
                        )
                    output = self.toolbox.execute(call.name, call.arguments)
                    status = "completed"
                except Exception as exc:
                    output = self.toolbox.sanitize_error(exc)
                    status = "failed"
                trace.append(
                    LLMToolTrace(
                        round_index=round_index,
                        call_id=call.call_id,
                        name=call.name,
                        arguments=call.arguments,
                        status=status,
                        elapsed_seconds=time.perf_counter() - started_at,
                        output=output,
                    )
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": json.dumps(output, ensure_ascii=False),
                    }
                )

        raise RuntimeError(f"LLM tool loop exceeded the {self.max_turns}-turn limit")
