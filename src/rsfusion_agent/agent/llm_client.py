"""Provider-neutral adapter around OpenAI-compatible Responses APIs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from rsfusion_agent.agent.llm_state import LLMCostEstimate, LLMTokenUsage

ProviderName = Literal["openai", "qwen", "deepseek", "custom"]


@dataclass(frozen=True)
class ProviderSettings:
    """Resolved non-secret provider configuration used by the CLI and client."""

    provider: ProviderName
    model: str
    base_url: str | None


_DEFAULT_MODELS: dict[ProviderName, str] = {
    "openai": "gpt-5.4-mini",
    "qwen": "qwen3.7-flash",
    "deepseek": "deepseek-v4-flash",
    "custom": "",
}

_KEY_ENVIRONMENT_NAMES: dict[ProviderName, tuple[str, ...]] = {
    "openai": ("RSFUSION_LLM_API_KEY", "OPENAI_API_KEY"),
    "qwen": ("RSFUSION_LLM_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    "deepseek": ("RSFUSION_LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "custom": ("RSFUSION_LLM_API_KEY", "OPENAI_API_KEY"),
}


def normalize_provider(provider: str) -> ProviderName:
    """Validate a configured provider name without guessing API compatibility."""

    normalized = provider.strip().lower()
    if normalized not in _DEFAULT_MODELS:
        supported = ", ".join(_DEFAULT_MODELS)
        raise ValueError(f"Unsupported LLM provider {provider!r}. Choose one of: {supported}.")
    return normalized  # type: ignore[return-value]


def default_model_for_provider(provider: str) -> str:
    """Return the documented default model for a configured provider."""

    return _DEFAULT_MODELS[normalize_provider(provider)]


def resolve_provider_settings(
    *,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
) -> ProviderSettings:
    """Resolve provider settings while preserving legacy OpenAI-named variables."""

    normalized_provider = normalize_provider(provider)
    resolved_model = model or os.environ.get("RSFUSION_LLM_MODEL") or os.environ.get("OPENAI_MODEL")
    resolved_model = resolved_model or default_model_for_provider(normalized_provider)
    if not resolved_model:
        raise ValueError("A model is required for the custom provider.")
    resolved_base_url = (
        base_url
        or os.environ.get("RSFUSION_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or None
    )
    return ProviderSettings(
        provider=normalized_provider,
        model=resolved_model,
        base_url=resolved_base_url,
    )


def _resolve_api_key(provider: ProviderName, api_key: str | None) -> str:
    if api_key:
        return api_key
    for environment_name in _KEY_ENVIRONMENT_NAMES[provider]:
        value = os.environ.get(environment_name)
        if value:
            return value
    names = " or ".join(_KEY_ENVIRONMENT_NAMES[provider])
    raise ValueError(
        f"No API key was configured for provider {provider!r}. Set {names} in the environment; "
        "do not pass API keys in CLI arguments."
    )


def _usage_value(source: Any, name: str) -> int:
    if isinstance(source, dict):
        value = source.get(name, 0)
    else:
        value = getattr(source, name, 0)
    return value if isinstance(value, int) and value >= 0 else 0


def normalize_usage(raw_usage: Any) -> LLMTokenUsage | None:
    """Normalize OpenAI-, Qwen-, and DeepSeek-compatible usage fields."""

    if raw_usage is None:
        return None
    input_details = (
        raw_usage.get("input_tokens_details", {})
        if isinstance(raw_usage, dict)
        else getattr(raw_usage, "input_tokens_details", None)
    )
    output_details = (
        raw_usage.get("output_tokens_details", {})
        if isinstance(raw_usage, dict)
        else getattr(raw_usage, "output_tokens_details", None)
    )
    return LLMTokenUsage(
        input_tokens=_usage_value(raw_usage, "input_tokens") or _usage_value(raw_usage, "prompt_tokens"),
        output_tokens=_usage_value(raw_usage, "output_tokens")
        or _usage_value(raw_usage, "completion_tokens"),
        cached_input_tokens=_usage_value(input_details, "cached_tokens")
        or _usage_value(input_details, "cached_token_count"),
        reasoning_tokens=_usage_value(output_details, "reasoning_tokens"),
    )


def estimate_cost(
    *,
    provider: str,
    model: str,
    usage: LLMTokenUsage,
) -> LLMCostEstimate | None:
    """Estimate cost for models explicitly documented in this project.

    Prices are only an observability aid and must be checked against provider billing.
    """

    normalized_provider = normalize_provider(provider)
    normalized_model = model.lower()
    if normalized_provider == "qwen" and normalized_model.startswith("qwen3.7-flash"):
        input_price, output_price, currency = 0.2, 0.8, "CNY"
        note = "Qwen3.7 Flash price table; requests up to 32K input tokens."
    elif normalized_provider == "deepseek" and normalized_model.startswith("deepseek-v4-flash"):
        input_price, output_price, currency = 0.14, 0.28, "USD"
        note = "DeepSeek V4 Flash cache-miss input price; actual cache billing may be lower."
    else:
        return None
    estimated_cost = (
        usage.input_tokens * input_price + usage.output_tokens * output_price
    ) / 1_000_000
    return LLMCostEstimate(
        currency=currency,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        estimated_cost=estimated_cost,
        note=note,
    )


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
    usage: LLMTokenUsage | None = None


class ResponsesClient(Protocol):
    """Interface that keeps the orchestration loop testable without network calls."""

    model: str
    provider: str

    def respond(
        self,
        *,
        input_items: list[Any],
        tools: list[dict[str, Any]],
        instructions: str,
    ) -> ModelTurn: ...


class CompatibleResponsesClient:
    """Translate OpenAI-compatible Responses API objects into model turns."""

    def __init__(
        self,
        *,
        model: str,
        provider: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.provider = normalize_provider(provider)
        resolved_key = _resolve_api_key(self.provider, api_key)
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
            usage=normalize_usage(getattr(response, "usage", None)),
        )


# Kept as a source-compatible name for existing callers and notebooks.
OpenAIResponsesClient = CompatibleResponsesClient
