"""The first tools that touch a scene.

Two real ones and one for exercising the rules around them. They are bridge
level tools, named the way `bridge.ping` is named. The tool surface a client
sees is a separate, smaller set built on top of these.

- `scene.info` reads. It cooks nothing and changes nothing.
- `node.create` mutates, so it runs on the main thread in a graphical session
  and inside one undo group in every session.
- `bridge.selfcheck` mutates on purpose, and can be asked to take its time or
  to fail part way, so the queue, the timeout and the rollback can be tried
  against a real Houdini rather than only against a stand in.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nscr_houdini_mcp.bridge.errors import BridgeError, did_you_mean

# The networks a scene summary counts, in the order a person reads them.
CONTEXTS = ("/obj", "/out", "/stage", "/mat", "/ch", "/shop", "/img", "/tasks")

# Bounds on what the self check will do, so a mistyped argument cannot park
# the session for an hour or fill a scene.
MAX_SLEEP_S = 600.0
MAX_CREATES = 64


@dataclass(frozen=True)
class ToolContext:
    """What a tool is told about the call it is running under."""

    hou: Any | None = None
    kind: str = "hython"
    session_id: str = ""
    scene_epoch: int = 0
    operation_id: str = ""
    label: str = ""


# Section: reads


def scene_info(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """What is open, where it is, and how big it is. Nothing is cooked."""
    hou = _houdini(context)
    counts: dict[str, int] = {}
    for path in CONTEXTS:
        node = _quiet(lambda path=path: hou.node(path))
        if node is not None:
            children = _quiet(node.children)
            if children is not None:
                counts[path] = len(children)
    return {
        "hip_path": _quiet(hou.hipFile.path),
        "houdini_version": _quiet(hou.applicationVersionString),
        "frame": _quiet(hou.frame),
        "fps": _quiet(hou.fps),
        "frame_range": _range(hou),
        "nodes": counts,
        "undo_entries": _undo_entries(hou),
        "unsaved": _unsaved(hou, context),
        "scene_epoch": context.scene_epoch,
        "session_id": context.session_id,
        "kind": context.kind,
    }


def _unsaved(hou: Any, context: ToolContext) -> bool | None:
    """Whether the scene has edits that are not on disk, where that is known.

    A headless session answers yes to this even straight after a save, so the
    answer is only meaningful with a user interface. Nothing is better than a
    value that is always the same.
    """
    if context.kind != "gui":
        return None
    return _quiet(hou.hipFile.hasUnsavedChanges)


def _undo_entries(hou: Any) -> int | None:
    """How many entries are on the undo stack, where the build will say.

    One call that changes the scene should add one. It is here so a caller can
    check that for itself rather than take the reply's word for it.
    """
    labels = _quiet(hou.undos.undoLabels)
    return None if labels is None else len(labels)


def _range(hou: Any) -> list[float] | None:
    frames = _quiet(lambda: hou.playbar.frameRange())
    return None if frames is None else [float(value) for value in frames]


# Section: mutations


def create_node(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Make one node under a parent, and set the parameters that were given."""
    hou = _houdini(context)
    parent_path = str(arguments["parent"])
    type_name = str(arguments["type"])
    wanted_name = arguments.get("name")
    parms = arguments.get("parms") or {}
    if not isinstance(parms, Mapping):
        raise BridgeError("BAD_ARGUMENTS", "parms must be an object of name to value")

    parent = _node(hou, parent_path)
    try:
        node = (
            parent.createNode(type_name, str(wanted_name))
            if wanted_name
            else parent.createNode(type_name)
        )
    except Exception as error:  # noqa: BLE001 - a wrong type is the caller's mistake
        if _is_hou_error(error):
            raise BridgeError(
                "BAD_ARGUMENTS",
                f"no node type {type_name} can be made in that network",
                {
                    "parent": parent_path,
                    "type": type_name,
                    "did_you_mean": _types(parent, type_name),
                },
                hint="ask for the parent's child types and use one of those names",
            ) from None
        raise

    for name, value in parms.items():
        _set_parm(node, str(name), value)

    return {
        "path": node.path(),
        "name": node.name(),
        "type": node.type().name(),
        "parent": parent_path,
        "parms_set": sorted(str(name) for name in parms),
    }


def _set_parm(node: Any, name: str, value: Any) -> None:
    parm = node.parm(name) or node.parmTuple(name)
    if parm is None:
        raise BridgeError(
            "PARM_NOT_FOUND",
            f"the node has no parameter named {name}",
            {"node": node.path(), "parm": name, "did_you_mean": _parms(node, name)},
            hint="ask for the node's parameters and use one of those names",
        )
    parm.set(value)


def selfcheck(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Take a while, make a few nodes, and fail where asked to.

    It exists so the rules around a tool can be tried against a real Houdini:
    the queue, the wait, the timeout, one undo entry per call, and the
    rollback when a call fails after it has already changed the graph.
    """
    hou = _houdini(context)
    sleep_s = _number(arguments.get("sleep_s"), "sleep_s", MAX_SLEEP_S)
    creates = int(_number(arguments.get("creates"), "creates", MAX_CREATES))
    fail_at = arguments.get("fail_at")
    parent_path = str(arguments.get("parent") or "/obj")

    if sleep_s:
        time.sleep(sleep_s)

    made: list[str] = []
    parent = _node(hou, parent_path) if creates or fail_at else None
    for index in range(1, creates + 1):
        if fail_at is not None and int(fail_at) == index:
            raise BridgeError(
                "TOOL_FAILED",
                "the self check was asked to fail here",
                {"failed_at": index, "created_before_failing": list(made)},
            )
        made.append(parent.createNode("geo").path())
    return {"created": made, "slept_s": sleep_s, "operation_id": context.operation_id}


# Section: shared helpers


def _houdini(context: ToolContext) -> Any:
    if context.hou is None:
        raise BridgeError(
            "TOOL_FAILED",
            "this tool needs a Houdini and this process has none",
            hint="send the call to a session, not to a bare bridge",
        )
    return context.hou


def _node(hou: Any, path: str) -> Any:
    """The node at a path, or a not found error with the closest paths."""
    node = _quiet(lambda: hou.node(path))
    if node is None:
        raise BridgeError(
            "NODE_NOT_FOUND",
            f"no node at {path}",
            {"path": path, "did_you_mean": _near(hou, path)},
            hint="read the scene and use a path that is there",
        )
    return node


def _near(hou: Any, path: str) -> list[str]:
    """Paths close to one that is not there, from the deepest parent that is."""
    parts = [part for part in str(path).split("/") if part]
    while parts:
        parts.pop()
        above = "/" + "/".join(parts)
        found = _quiet(lambda above=above: hou.node(above))
        if found is None:
            continue
        children = _quiet(found.children) or ()
        paths = [child.path() for child in children]
        near = did_you_mean(path, paths)
        return near or paths[:3]
    return []


def _types(parent: Any, wanted: str) -> list[str]:
    """Child type names close to one the network does not have."""
    category = _quiet(parent.childTypeCategory)
    if category is None:
        return []
    types = _quiet(category.nodeTypes) or {}
    return did_you_mean(wanted, list(types))


def _parms(node: Any, wanted: str) -> list[str]:
    parms = _quiet(node.parms) or ()
    return did_you_mean(wanted, [parm.name() for parm in parms])


def _number(value: Any, name: str, cap: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeError("BAD_ARGUMENTS", f"{name} must be a number")
    if value < 0 or value > cap:
        raise BridgeError("BAD_ARGUMENTS", f"{name} must be between 0 and {cap:g}")
    return float(value)


def _is_hou_error(error: BaseException) -> bool:
    return type(error).__module__.split(".")[0] == "hou"


def _quiet(read: Any) -> Any:
    """Read one fact, or nothing when this build or session will not say."""
    try:
        return read()
    except Exception:  # noqa: BLE001 - a fact we cannot read is a fact we do not have
        return None
