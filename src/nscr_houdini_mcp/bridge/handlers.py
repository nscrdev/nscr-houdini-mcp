"""The table of tools the bridge can dispatch to.

One name, one callable, and what the dispatch layer needs to know about it:
whether it changes the scene, which arguments it takes, and what to call the
undo entry. The dispatch layer adds the rules around a call: one at a time per
session, the order calls are served in, the busy policy, the undo group, the
receipt. This module only says which names exist and what they run, so a tool
can be added without touching the transport.

Declaring the argument names here is what lets a mistyped one come back as a
named mistake with the closest names, instead of whatever the tool happened to
raise when it read a key that was not there.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nscr_houdini_mcp.bridge import tools as tool_module
from nscr_houdini_mcp.bridge.tools import ToolContext

ToolHandler = Callable[..., Any]


class UnknownTool(KeyError):
    """No tool is registered under that name."""


@dataclass(frozen=True)
class Tool:
    """One tool, and the rules that apply to running it."""

    name: str
    handler: ToolHandler
    # Whether it changes the scene. A mutating tool runs on the main thread in
    # a graphical session, and always inside one undo group.
    mutating: bool = False
    # The argument names it takes. `None` means it takes whatever it is given.
    arguments: tuple[str, ...] | None = None
    required: tuple[str, ...] = ()
    # Whether it is handed the call context as a second argument.
    context: bool = False
    # What the undo entry is called. The tool name when nothing is given.
    label: str | None = None
    # Its own run budget, when it needs one other than the bridge default.
    timeout_s: float | None = None
    summary: str = ""

    def undo_label(self) -> str:
        return self.label or self.name

    def run(self, arguments: Mapping[str, Any], context: ToolContext) -> Any:
        return self.handler(arguments, context) if self.context else self.handler(arguments)


class ToolRegistry:
    """Tool names in the order they were added."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(
        self,
        name: str,
        handler: ToolHandler,
        *,
        mutating: bool = False,
        arguments: Sequence[str] | None = None,
        required: Sequence[str] = (),
        context: bool = False,
        label: str | None = None,
        timeout_s: float | None = None,
        summary: str = "",
    ) -> Tool:
        if name in self._tools:
            raise ValueError(f"tool {name} is already registered")
        tool = Tool(
            name=name,
            handler=handler,
            mutating=mutating,
            arguments=None if arguments is None else tuple(arguments),
            required=tuple(required),
            context=context,
            label=label,
            timeout_s=timeout_s,
            summary=summary,
        )
        self._tools[name] = tool
        return tool

    def get(self, name: str) -> Tool:
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


@dataclass
class Call:
    """One tool call as the dispatch layer passes it around."""

    tool: Tool
    arguments: Mapping[str, Any] = field(default_factory=dict)
    context: ToolContext = field(default_factory=ToolContext)


def ping(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Answer without reading the scene. Whatever is sent as `echo` comes back."""
    return {"pong": True, "echo": arguments.get("echo")}


def default_registry() -> ToolRegistry:
    """The tools every bridge starts with."""
    registry = ToolRegistry()
    registry.add(
        "bridge.ping",
        ping,
        arguments=("echo",),
        summary="answer without touching the scene",
    )
    registry.add(
        "bridge.selfcheck",
        tool_module.selfcheck,
        mutating=True,
        arguments=("sleep_s", "creates", "fail_at", "parent"),
        context=True,
        label="self check",
        summary="take a while, make nodes, fail where asked",
    )
    registry.add(
        "scene.info",
        tool_module.scene_info,
        arguments=(),
        context=True,
        summary="what is open, and how big it is",
    )
    registry.add(
        "node.create",
        tool_module.create_node,
        mutating=True,
        arguments=("parent", "type", "name", "parms"),
        required=("parent", "type"),
        context=True,
        label="create node",
        summary="make one node and set its parameters",
    )
    return registry
