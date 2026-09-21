"""The table of tools the bridge can dispatch to.

One name, one callable, one dictionary of arguments. The dispatch layer adds
the rules around a call: one at a time per session, the busy policy, the undo
group, the receipt. This module only says which names exist and what they run,
so a tool can be added without touching the transport.

The one tool here touches no scene and no `hou`. It exists so the whole path
from request to reply can be exercised before there is anything to drive.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

ToolHandler = Callable[[Mapping[str, Any]], Any]


class UnknownTool(KeyError):
    """No tool is registered under that name."""


class ToolRegistry:
    """Tool names in the order they were added."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def add(self, name: str, handler: ToolHandler) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name} is already registered")
        self._tools[name] = handler

    def get(self, name: str) -> ToolHandler:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownTool(name) from None

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


def ping(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Answer without reading the scene. Whatever is sent as `echo` comes back."""
    return {"pong": True, "echo": arguments.get("echo")}


def default_registry() -> ToolRegistry:
    """The tools every bridge starts with."""
    registry = ToolRegistry()
    registry.add("bridge.ping", ping)
    return registry
