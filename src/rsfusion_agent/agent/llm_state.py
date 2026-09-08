"""Typed state for natural-language tool orchestration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from rsfusion_agent.agent.state import FusionRunResult


class LLMToolTrace(BaseModel):
    round_index: int = Field(ge=1)
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: Literal["completed", "failed"]
    elapsed_seconds: float = Field(ge=0)
    output: dict[str, Any]


class NaturalLanguageRunResult(BaseModel):
    status: Literal["completed", "completed_with_tool_errors"]
    model: str
    request: str
    answer: str
    turns: int = Field(ge=1)
    trace: list[LLMToolTrace]
    fusion_result: FusionRunResult | None = None
