"""Typed state for natural-language tool orchestration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rsfusion_agent.agent.state import FusionRunResult
from rsfusion_agent.agent.tiff_workflow import TiffFusionResult
from rsfusion_agent.tools.model_runtime import RuntimePreflightResult


class LLMToolTrace(BaseModel):
    round_index: int = Field(ge=1)
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["completed", "failed"]
    elapsed_seconds: float = Field(ge=0)
    output: dict[str, Any]


class LLMTokenUsage(BaseModel):
    """Normalized token usage returned by an LLM provider."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

    def combined_with(self, other: "LLMTokenUsage") -> "LLMTokenUsage":
        return LLMTokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


class LLMModelTrace(BaseModel):
    """Observable metadata for one provider response in the agent loop."""

    round_index: int = Field(ge=1)
    response_id: str
    usage: LLMTokenUsage


class LLMCostEstimate(BaseModel):
    """Best-effort estimate using a versioned, local price table."""

    currency: Literal["CNY", "USD"]
    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    estimated_cost: float = Field(ge=0)
    note: str


class NaturalLanguageRunResult(BaseModel):
    status: Literal["completed", "completed_with_tool_errors"]
    model: str
    request: str
    answer: str
    turns: int = Field(ge=1)
    trace: list[LLMToolTrace]
    provider: str = "unknown"
    usage: LLMTokenUsage = Field(default_factory=LLMTokenUsage)
    model_trace: list[LLMModelTrace] = Field(default_factory=list)
    estimated_cost: LLMCostEstimate | None = None
    runtime_preflight: RuntimePreflightResult | None = None
    fusion_result: FusionRunResult | TiffFusionResult | None = None
