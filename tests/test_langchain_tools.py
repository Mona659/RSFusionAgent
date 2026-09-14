from __future__ import annotations

import json

import pytest

from rsfusion_agent.agent.langchain_agent import LangChainToolAgent


class FakeToolbox:
    @staticmethod
    def definitions() -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "name": "inspect_demo",
                "description": "Inspect demo data.",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def execute(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        assert name == "inspect_demo"
        assert arguments == {}
        return {"ok": True, "tool": name}

    @staticmethod
    def sanitize_error(error: Exception) -> dict[str, object]:
        return {"ok": False, "error": str(error)}


def test_langchain_adapter_invokes_allowlisted_tool() -> None:
    pytest.importorskip("langchain_core")
    result = LangChainToolAgent(FakeToolbox()).invoke("inspect_demo")
    assert json.loads(result) == {"ok": True, "tool": "inspect_demo"}


def test_langchain_adapter_keeps_unknown_tool_blocked() -> None:
    pytest.importorskip("langchain_core")
    agent = LangChainToolAgent(FakeToolbox())
    with pytest.raises(ValueError, match="not allowlisted"):
        agent.invoke("delete_everything")
