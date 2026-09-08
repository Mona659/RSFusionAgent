"""Small adapter around the OpenAI Responses API."""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """Provider-neutral function call requested by an LLM."""

    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ModelTurn(BaseModel):
    """Normalized subset of a Responses API turn used by the agent loop."""

    response_id: str
    output_text: str = ""
    output_items: list[Any] = Field(default_factory=list)
    tool_calls: list[FunctionCall] = Field(default_factory=list)


class ResponsesClient(Protocol):
    """Interface that keeps the orchestration loop testable without network calls."""

    model: str

    def respond(
        self,
        *,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn: ...


class OpenAIResponsesClient:
    """Translate OpenAI SDK response objects into provider-neutral model turns."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Configure it in the environment; "
                "do not pass API keys in CLI arguments."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                'OpenAI SDK is not installed. Run: python -m pip install -e ".[dev,llm]"'
            ) from exc

        options: dict[str, Any] = {"api_key": resolved_key}
        if base_url:
            options["base_url"] = base_url
        self._client = OpenAI(**options)
        self.model = model

    def respond(
        self,
        *,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn:
        response = self._client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            tools=tools,
            store=False,
            parallel_tool_calls=False,
            include=["reasoning.encrypted_content"],
        )

        output_items: list[Any] = []
        tool_calls: list[FunctionCall] = []
        for item in response.output:
            serialized = item if isinstance(item, dict) else {}
            output_items.append(item)
            if getattr(item, "type", serialized.get("type")) != "function_call":
                continue
            raw_arguments = getattr(item, "arguments", serialized.get("arguments", "{}"))
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, dict):
                raise TypeError("Function-call arguments must decode to a JSON object")
            tool_calls.append(
                FunctionCall(
                    call_id=getattr(item, "call_id", serialized.get("call_id", "")),
                    name=getattr(item, "name", serialized.get("name", "")),
                    arguments=arguments,
                )
            )

        return ModelTurn(
            response_id=response.id,
            output_text=response.output_text or "",
            output_items=output_items,
            tool_calls=tool_calls,
        )
