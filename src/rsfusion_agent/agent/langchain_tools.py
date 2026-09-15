"""LangChain Core adapters for the existing RSFusionAgent toolboxes.

The adapter is intentionally thin: all authorization, Pydantic validation and local
execution remain in :mod:`llm_tools`.  LangChain is therefore optional and can be
introduced without changing the V1 Responses API control plane.
"""

from __future__ import annotations

import json
from typing import Any

from rsfusion_agent.agent.llm_tools import AgentToolbox, TiffAgentToolbox

Toolbox = AgentToolbox | TiffAgentToolbox


def _require_langchain_core() -> Any:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            'LangChain Core is not installed. Run: python -m pip install -e ".[dev,langchain]"'
        ) from exc
    return StructuredTool


def _schema_for(name: str) -> Any:
    """Return a small Pydantic schema matching the existing tool definition."""

    from pydantic import BaseModel, ConfigDict, Field

    class EmptyArguments(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class PatchArguments(BaseModel):
        model_config = ConfigDict(extra="forbid")
        patch_index: int = Field(ge=0)

    class KnowledgeArguments(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1)
        top_k: int = Field(default=4, ge=1, le=20)

    if name in {"inspect_yre151_h5", "run_yre151_fusion"}:
        return PatchArguments
    if name == "search_knowledge" or name == "search_similar_experiments":
        return KnowledgeArguments
    if name == "search_error_solution":
        class ErrorArguments(BaseModel):
            model_config = ConfigDict(extra="forbid")
            error: str = Field(min_length=1)
            top_k: int = Field(default=4, ge=1, le=20)
        return ErrorArguments
    return EmptyArguments


def build_langchain_tools(toolbox: Toolbox) -> list[Any]:
    """Convert an existing allowlisted toolbox to LangChain ``StructuredTool`` objects.

    The returned tools preserve the toolbox's exact names and schemas.  Results are
    JSON strings because LangChain tool messages are text-oriented, while the JSON
    payload itself remains structured and can be fed back to an LLM or UI.
    """

    StructuredTool = _require_langchain_core()
    tools: list[Any] = []
    for definition in toolbox.definitions():
        name = str(definition["name"])
        description = str(definition["description"])
        schema = _schema_for(name)

        def invoke(*, _name: str = name, **arguments: Any) -> str:
            payload = arguments
            try:
                result = toolbox.execute(_name, payload)
            except Exception as exc:
                result = toolbox.sanitize_error(exc)
            return json.dumps(result, ensure_ascii=False, default=str)

        tools.append(
            StructuredTool.from_function(
                func=invoke,
                name=name,
                description=description,
                args_schema=schema,
            )
        )
    return tools


__all__ = ["build_langchain_tools"]
