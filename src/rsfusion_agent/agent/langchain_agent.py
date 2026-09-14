"""Small LangChain-compatible execution facade for V2 day one.

This class deliberately does not replace the tested V1 LLM loop.  It supplies a
stable entry point for LangChain runtimes and makes local tool execution observable
without requiring a network model.
"""

from __future__ import annotations

from typing import Any

from rsfusion_agent.agent.langchain_tools import Toolbox, build_langchain_tools


class LangChainToolAgent:
    """Expose an allowlisted RSFusionAgent toolbox as LangChain tools.

    ``invoke`` is useful for smoke tests and deterministic local workflows.  A
    LangChain model/graph can consume ``tools`` directly for model-driven calls.
    """

    def __init__(self, toolbox: Toolbox) -> None:
        self.toolbox = toolbox
        self.tools = build_langchain_tools(toolbox)
        self._tools_by_name = {tool.name: tool for tool in self.tools}

    def invoke(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Invoke one registered tool by name and return its JSON payload."""

        try:
            tool = self._tools_by_name[name]
        except KeyError as exc:
            raise ValueError(f"Tool is not allowlisted: {name}") from exc
        return tool.invoke(arguments or {})


__all__ = ["LangChainToolAgent"]
