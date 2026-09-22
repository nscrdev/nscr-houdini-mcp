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
from dataclasses import dataclass
from typing import Any

from nscr_houdini_mcp.bridge import tools as tool_module
from nscr_houdini_mcp.bridge.tools import ToolContext

ToolHandler = Callable[..., Any]

# The longest undo entry name a call may give its own work.
MAX_LABEL = 200


class UnknownTool(KeyError):
    """No tool is registered under that name."""


@dataclass(frozen=True)
class Tool:
    """One tool, and the rules that apply to running it."""

    name: str
    handler: ToolHandler
    # Whether it changes the scene: it then runs inside one undo group. In a
    # graphical session every tool runs on the main thread regardless.
    mutating: bool = False
    # Whether its change can be undone. A change that cannot, such as loading
    # a scene or writing a file, still takes a receipt, but no undo group is
    # opened around it and the reply says there is nothing to undo.
    undoable: bool = True
    # The argument names it takes. `None` means it takes whatever it is given.
    arguments: tuple[str, ...] | None = None
    required: tuple[str, ...] = ()
    # Whether it is handed the call context as a second argument.
    context: bool = False
    # Whether it runs without taking the session at all, on the thread that
    # took the request. Only for tools that touch no scene and no `hou`: they
    # are the ones that can answer while another call is running.
    immediate: bool = False
    # What the undo entry is called. The tool name when nothing is given.
    label: str | None = None
    # The argument that names this call's undo entry, for a tool whose calls
    # each want a name of their own. The label above when it is not sent.
    label_argument: str | None = None
    # Arguments the receipt digest leaves out: ones the sender fills in for a
    # caller that named none, which a retry from another sender fills in
    # differently for the same call.
    digest_ignores: tuple[str, ...] = ()
    # Its own run budget, when it needs one other than the bridge default.
    timeout_s: float | None = None
    summary: str = ""
    # How much of its answer is carried, when that is more than the default
    # caps in `encoding`: a keyword for each cap `encoding.convert` takes.
    caps: Mapping[str, int] | None = None

    def undo_label(self, arguments: Mapping[str, Any] | None = None) -> str:
        if self.label_argument and arguments:
            named = arguments.get(self.label_argument)
            if isinstance(named, str) and named.strip():
                return named.strip()[:MAX_LABEL]
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
        undoable: bool = True,
        arguments: Sequence[str] | None = None,
        required: Sequence[str] = (),
        context: bool = False,
        immediate: bool = False,
        label: str | None = None,
        label_argument: str | None = None,
        digest_ignores: Sequence[str] = (),
        timeout_s: float | None = None,
        summary: str = "",
        caps: Mapping[str, int] | None = None,
    ) -> Tool:
        if name in self._tools:
            raise ValueError(f"tool {name} is already registered")
        tool = Tool(
            name=name,
            handler=handler,
            mutating=mutating,
            undoable=undoable,
            arguments=None if arguments is None else tuple(arguments),
            required=tuple(required),
            context=context,
            immediate=immediate,
            label=label,
            label_argument=label_argument,
            digest_ignores=tuple(digest_ignores),
            timeout_s=timeout_s,
            summary=summary,
            caps=dict(caps) if caps else None,
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


# What `node.inspect` takes. `path` is the one a tree or a search looks under,
# `paths` the ones a node or parameter read reads. The server turns its
# caller's page token into `after`, and says with `batch` whether a missing
# path is one entry's error or the whole call's.
INSPECT_ARGUMENTS = (
    "mode",
    "path",
    "paths",
    "batch",
    "evaluate",
    "depth",
    "pattern",
    "type",
    "parm_filter",
    "include",
    "detail",
    "limit",
    "after",
)

# What `python.run` takes. The server always names the namespace and the undo
# entry, so a retry under the same operation id sends exactly the same call.
PYTHON_ARGUMENTS = ("code", "namespace", "reset", "undo_label")

# A page can hold two thousand rows, and a full read of fifty nodes a great
# many values. The server spills an answer that large to a file rather than
# cut it, so the bridge carries all of it.
INSPECT_CAPS = {"max_items": 4096, "max_values": 500_000}


def ping(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Answer without reading the scene. Whatever is sent as `echo` comes back."""
    return {"pong": True, "echo": arguments.get("echo")}


def default_registry(
    *, selfcheck: bool = False, python: tool_module.Namespaces | None = None
) -> ToolRegistry:
    """The tools every bridge starts with.

    The self check is left out unless it is asked for. It makes nodes and can
    park the session for a minute, which is wanted in a worker that was
    started to be tested against and unwanted in a session somebody is using.
    A session carrying it also lets `python.run` take `drop_reply`, so a lost
    answer to running code can be tried end to end.

    `python` holds the namespaces `python.run` keeps between calls, one set per
    bridge. A test hands in its own to move the clock.
    """
    registry = ToolRegistry()
    namespaces = python if python is not None else tool_module.Namespaces()
    registry.add(
        "bridge.ping",
        ping,
        arguments=("echo",),
        summary="answer without touching the scene",
    )
    if selfcheck:
        registry.add(
            "bridge.selfcheck",
            tool_module.selfcheck,
            mutating=True,
            arguments=(
                "sleep_s",
                "creates",
                "fail_at",
                "parent",
                "save_hip",
                "new_scene",
                "load_hip",
                "drop_reply",
            ),
            context=True,
            label="self check",
            summary="take a while, make nodes, replace the scene or fail where asked",
        )
    registry.add(
        "bridge.capabilities",
        tool_module.capabilities,
        arguments=(),
        context=True,
        summary="what this Houdini can do, read once when it comes up",
    )
    registry.add(
        "scene.info",
        tool_module.scene_info,
        arguments=("dependencies",),
        context=True,
        summary="what is open, and how big it is",
    )
    registry.add(
        "scene.open",
        tool_module.scene_open,
        mutating=True,
        undoable=False,
        arguments=("path", "discard_unsaved"),
        required=("path",),
        context=True,
        label="open scene",
        summary="load a scene file and say what it could not find",
    )
    registry.add(
        "scene.save",
        tool_module.scene_save,
        mutating=True,
        undoable=False,
        arguments=(),
        context=True,
        label="save scene",
        summary="save the scene where it already is",
    )
    registry.add(
        "scene.save_as",
        tool_module.scene_save_as,
        mutating=True,
        undoable=False,
        arguments=("path",),
        required=("path",),
        context=True,
        label="save scene as",
        summary="save the scene to a new file, never over one that is there",
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
    registry.add(
        "node.inspect",
        tool_module.inspect,
        arguments=INSPECT_ARGUMENTS,
        context=True,
        summary="read nodes, networks and parameters a page at a time",
        caps=INSPECT_CAPS,
    )
    registry.add(
        "python.run",
        namespaces.run,
        mutating=True,
        arguments=PYTHON_ARGUMENTS + (("drop_reply",) if selfcheck else ()),
        required=("code", "namespace"),
        context=True,
        label="hou_python",
        label_argument="undo_label",
        summary="run Python with hou in a namespace kept between calls",
    )
    return registry
