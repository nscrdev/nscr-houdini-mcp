"""Managed outputs on the Houdini side: frozen parameters and the lint.

The ruling on output parameters. While a run is in progress, the output
node's real output parameter holds the frozen absolute path for that run, so
a rename halfway through moves nothing and the record and the scene agree.
When the run is over, whether it ended done, failed or cancelled, the `$HIP`
template goes back on the node, so a scene saved afterwards never carries a
path from this machine.

Each freeze is recorded in the coordination store before the parameter is
set: the session, the node, the parameter, the template it is owed back, the
path it was given and the scene it is in. The record goes when the template
is back. Three things put it back:

- the call that froze it, when it ends: `python.run` restores whatever its
  code froze through `mcp.freeze_parm`, and a tool that freezes through
  `outputs.freeze_parm` restores through `outputs.restore_parm` when its run
  is over;
- the session that opens the scene next, when the one that froze it has gone,
  asked by the server with `outputs.restore_parm` and the owner named;
- nobody, for a scene that was never saved and whose session has gone: that
  record is dropped, because no session can open the scene again.

A parameter is only given back its template when it still holds the path it
was frozen to. One changed since is somebody's own edit and is left as it is.

`outputs.lint` reads every output parameter under a node: the string file
parameters a node type marks as written to, read from the type rather than
from a list of names. What it reports, per parameter, is any of
`absolute_path` (the value starts at a root or a drive), `outside_hip` (the
path is outside the folders this server manages for the scene),
`unversioned` (no `v<number>` anywhere in it), `missing_on_disk` (nothing is
there, for a sequence at the current frame nor at either end of the frame
range) and `frozen_after_run` (a frozen path nobody gave back).

This module never imports `hou`: it is handed it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import outputs as output_rules
from nscr_houdini_mcp.bridge import tools as tool_module
from nscr_houdini_mcp.bridge.errors import BridgeError
from nscr_houdini_mcp.bridge.tools import ToolContext

PROBLEMS = (
    "frozen_after_run",
    "absolute_path",
    "outside_hip",
    "unversioned",
    "missing_on_disk",
)

# The chooser mode a node type gives a parameter it writes to.
WRITE_MODE = "write"

DEFAULT_LIMIT = 200
MAX_LIMIT = 2000

# What `outputs.lint` takes. `after` is the node path and parameter name the
# page before ended on.
LINT_ARGUMENTS = ("scope", "after", "limit")
FREEZE_ARGUMENTS = ("node", "parm", "run_id")
RESTORE_ARGUMENTS = ("node", "parm", "owner")


# Section: the scene


def scene_path(hou: Any) -> str | None:
    """The scene file this session holds, or nothing for a scene never saved."""
    if tool_module._quiet(hou.hipFile.isNewFile):
        return None
    return tool_module._quiet(hou.hipFile.path)


def raw_value(parm: Any) -> str:
    """What a string parameter holds before anything is expanded.

    A parameter with keyframes has no single string, so its expression is
    read instead.
    """
    text = tool_module._quiet(parm.unexpandedString)
    if text is None:
        text = tool_module._quiet(parm.expression)
    return "" if text is None else str(text)


# Section: freezing and restoring


def freeze(
    hou: Any,
    open_store: Callable[[], Any],
    *,
    session_id: str,
    node_path: str,
    parm_name: str,
    run_id: str,
) -> dict[str, Any]:
    """Set one output parameter to a run's own path, after recording it.

    Only a path this server handed out can be frozen: the run names it, and
    the run's record says the template the parameter is owed back. A run
    whose line holds a machine path, such as a spill, never goes on a node.
    """
    node = tool_module._node(hou, node_path)
    parm = node.parm(parm_name)
    if parm is None:
        raise BridgeError(
            "PARM_NOT_FOUND",
            f"the node has no parameter named {parm_name}",
            {
                "node": node.path(),
                "parm": parm_name,
                "did_you_mean": tool_module._parms(node, parm_name),
            },
            hint="ask for the node's parameters and use one of those names",
        )
    kind = tool_module._quiet(lambda: parm.parmTemplate().type().name())
    if kind != "String":
        raise BridgeError(
            "BAD_ARGUMENTS",
            "only a string parameter holds a path",
            {"node": node.path(), "parm": parm.name(), "parm_type": kind},
        )
    with open_store() as store:
        run = store.get_run(run_id)
        paths = run.paths if run is not None and isinstance(run.paths, Mapping) else {}
        template = str(paths.get("template") or "")
        frozen = str(paths.get("path") or "")
        if run is None or not template or not frozen:
            raise BridgeError(
                "BAD_ARGUMENTS",
                "no run under that id handed out a path, so there is nothing to freeze",
                {"run_id": run_id},
                hint="take a path from hou_outputs resolve or mcp.output_path first",
            )
        if output_rules.is_machine_path(template):
            raise BridgeError(
                "BAD_ARGUMENTS",
                f"a {run.kind} path belongs to this machine and never goes on a node",
                {"run_id": run_id, "kind": run.kind},
            )
        store.freeze_parm(
            session_id=session_id,
            node_path=node.path(),
            parm_name=parm.name(),
            template=template,
            frozen=frozen,
            run_id=run_id,
            hip_key=output_rules.scene_key(scene_path(hou)),
        )
    previous = raw_value(parm)
    try:
        # A keyframe would carry on under the value, so the parameter holds
        # the one path and nothing else while the run is going.
        tool_module._quiet(parm.deleteAllKeyframes)
        parm.set(frozen)
    except BaseException:
        with open_store() as store:
            store.thaw_parm(session_id, node.path(), parm.name())
        raise
    return {
        "node": node.path(),
        "parm": parm.name(),
        "value": frozen,
        "template": template,
        "run_id": run_id,
        "previous": previous,
    }


def restore(
    hou: Any,
    open_store: Callable[[], Any],
    *,
    session_id: str,
    node_path: str,
    parm_name: str,
    owner: str | None = None,
) -> dict[str, Any]:
    """Give one frozen parameter its template back, and drop its record.

    `owner` is the session that froze it, when that is not this one: only a
    session that is over can have its parameters restored by another, and
    only in the scene they were frozen in.
    """
    holder = owner or session_id
    said: dict[str, Any] = {"node": node_path, "parm": parm_name, "restored": False}
    with open_store() as store:
        row = store.get_frozen_parm(holder, node_path, parm_name)
        if row is None:
            return {**said, "reason": "not_frozen"}
        if holder != session_id:
            if not store.session_is_over(holder):
                raise BridgeError(
                    "BAD_ARGUMENTS",
                    "the session that froze this parameter is still running and restores it itself",
                    {"owner": holder},
                )
            if row.hip_key != output_rules.scene_key(scene_path(hou)):
                raise BridgeError(
                    "BAD_ARGUMENTS",
                    "that parameter was frozen in another scene",
                    {"owner": holder},
                    hint="open the scene it was frozen in, then restore it there",
                )
    node = tool_module._quiet(lambda: hou.node(node_path))
    parm = None if node is None else tool_module._quiet(lambda: node.parm(parm_name))
    if parm is None:
        said["reason"] = "node_gone"
    elif raw_value(parm) == row.frozen:
        parm.set(row.template)
        said["restored"] = True
    else:
        said["reason"] = "changed_since"
    if parm is not None:
        said["value"] = raw_value(parm)
    said["template"] = row.template
    with open_store() as store:
        store.thaw_parm(holder, node_path, parm_name)
    return said


def freeze_parm(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """`outputs.freeze_parm`: hold a run's own path on an output parameter."""
    hou = tool_module._houdini(context)
    return freeze(
        hou,
        _store_opener(context),
        session_id=context.session_id,
        node_path=str(arguments["node"]),
        parm_name=str(arguments["parm"]),
        run_id=str(arguments["run_id"]),
    )


def restore_parm(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """`outputs.restore_parm`: put a frozen parameter's template back."""
    hou = tool_module._houdini(context)
    owner = arguments.get("owner")
    return restore(
        hou,
        _store_opener(context),
        session_id=context.session_id,
        node_path=str(arguments["node"]),
        parm_name=str(arguments["parm"]),
        owner=None if owner is None else str(owner),
    )


def _store_opener(context: ToolContext) -> Callable[[], Any]:
    if context.open_store is None:
        raise BridgeError(
            "STORE_UNAVAILABLE",
            "this session keeps no state folder, so it keeps no record of outputs",
        )
    return context.open_store


# Section: the lint


def lint(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """`outputs.lint`: output parameters under a node that break the conventions.

    Rows come sorted by node path and then parameter name, one per problem,
    and a page ends on a whole parameter. `more` says a later page has rows;
    `last` is where the next one starts after.
    """
    hou = tool_module._houdini(context)
    scope = str(arguments.get("scope") or "/")
    limit = int(tool_module._number(arguments.get("limit"), "limit", MAX_LIMIT) or DEFAULT_LIMIT)
    after = _after(arguments.get("after"))
    top = tool_module._node(hou, scope)

    hip = scene_path(hou)
    notes: list[str] = []
    try:
        conventions = output_rules.load_conventions(home=context.home, hip_path=hip)
    except output_rules.OutputError as error:
        conventions = output_rules.DEFAULT_CONVENTIONS_TABLE
        notes.append(
            f"the output conventions could not be read, so the defaults were used: {error}"
        )
    roots = output_rules.managed_roots(
        conventions,
        hip_path=hip,
        session_id=context.session_id,
        scratch_root=home_scratch(context.home),
    )
    frozen = _frozen_here(context, hip)

    nodes, capped = _nodes_under(top)
    kinds: dict[str, tuple[str, ...]] = {}
    candidates: list[tuple[str, str, Any]] = []
    for node in nodes:
        path = tool_module._quiet(node.path)
        if not path:
            continue
        for name in _output_parms(node, kinds):
            candidates.append((path, name, node))
    candidates.sort(key=lambda item: (item[0], item[1]))

    frames = tool_module._frames_to_try(hou)
    rows: list[dict[str, Any]] = []
    last: list[str] | None = None
    more = False
    for path, name, node in candidates:
        if after is not None and (path, name) <= after:
            continue
        parm = tool_module._quiet(lambda node=node, name=name: node.parm(name))
        if parm is None or tool_module._quiet(parm.isDisabled):
            continue
        found = _check(parm, path, name, roots, frozen, frames)
        if not found:
            continue
        if len(rows) >= limit:
            more = True
            break
        rows.extend(found)
        last = [path, name]
    said: dict[str, Any] = {
        "rows": rows,
        "more": more,
        "last": last,
        "roots": roots,
        "parms_checked": len(candidates),
    }
    if capped:
        said["truncated"] = True
    if notes:
        said["notes"] = notes
    return said


def _after(value: Any) -> tuple[str, str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(part, str) for part in value)
    ):
        raise BridgeError("BAD_ARGUMENTS", "after must be a node path and a parameter name")
    return (value[0], value[1])


def _frozen_here(context: ToolContext, hip: str | None) -> dict[tuple[str, str], Any]:
    """Frozen parameters this session holds, and those left in this scene."""
    if context.open_store is None:
        return {}
    found: dict[tuple[str, str], Any] = {}
    with context.open_store() as store:
        rows = list(store.list_frozen_parms(session_id=context.session_id))
        key = output_rules.scene_key(hip)
        if key is not None:
            rows += store.list_frozen_parms(hip_key=key)
    for row in rows:
        found.setdefault((row.node_path, row.parm_name), row)
    return found


def _nodes_under(top: Any) -> tuple[list[Any], bool]:
    """The node and everything under it, not inside locked assets, up to the cap."""
    below = tool_module._quiet(lambda: top.allSubChildren(recurse_in_locked_nodes=False))
    if below is None:
        below = tool_module._quiet(top.allSubChildren) or ()
    nodes = [top, *below]
    capped = len(nodes) > tool_module.MAX_NODES_SCANNED
    return nodes[: tool_module.MAX_NODES_SCANNED], capped


def _output_parms(node: Any, kinds: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The parameters a node's type writes to, read once per type.

    A string file parameter whose chooser the type opens for writing, and
    not a folder: a render's picture, a cache's file, a layer's output. The
    names come from the type, never from a list kept here, so a type this
    code has never seen is read the same way.
    """
    node_type = tool_module._quiet(node.type)
    key = str(
        tool_module._quiet(node_type.nameWithCategory) or tool_module._quiet(node_type.name) or ""
    )
    if key in kinds:
        return kinds[key]
    names: list[str] = []
    for parm in tool_module._quiet(node.parms) or ():
        if tool_module._quiet(parm.isSpare) or tool_module._quiet(parm.isMultiParmInstance):
            continue
        if _writes(tool_module._quiet(parm.parmTemplate)):
            names.append(parm.name())
    kinds[key] = tuple(names)
    return kinds[key]


def _writes(template: Any) -> bool:
    if template is None:
        return False
    if tool_module._quiet(lambda: template.type().name()) != "String":
        return False
    if not str(tool_module._quiet(template.stringType)).endswith("FileReference"):
        return False
    if str(tool_module._quiet(template.fileType)).endswith("Directory"):
        return False
    tags = tool_module._quiet(template.tags) or {}
    return tags.get("filechooser_mode") == WRITE_MODE


def _check(
    parm: Any,
    path: str,
    name: str,
    roots: list[str],
    frozen: Mapping[tuple[str, str], Any],
    frames: Iterable[float],
) -> list[dict[str, Any]]:
    """The problems one output parameter has, one row each."""
    raw = raw_value(parm)
    if not _names_a_place(raw):
        return []
    expanded = str(tool_module._quiet(parm.evalAsString) or "")
    problems: list[str] = []
    held = frozen.get((path, name))
    if held is not None and raw == held.frozen:
        problems.append("frozen_after_run")
    if output_rules.is_machine_path(raw):
        problems.append("absolute_path")
    if expanded and not _managed(roots, expanded):
        problems.append("outside_hip")
    if not output_rules.has_version(raw) and not output_rules.has_version(expanded):
        problems.append("unversioned")
    if not _there(parm, raw, expanded, frames):
        problems.append("missing_on_disk")
    return [
        {"node": path, "parm": name, "raw": raw, "expanded": expanded, "problem": problem}
        for problem in problems
    ]


def _names_a_place(raw: str) -> bool:
    """Whether a value is a path at all, rather than a device or a word.

    `ip` sends a render to a viewer and `__render__.usd` is a name a render
    node fills in for itself: neither is a place on disk.
    """
    text = raw.strip()
    return bool(text) and ("/" in text or "\\" in text or text.startswith("$"))


def _managed(roots: list[str], expanded: str) -> bool:
    if output_rules.inside_roots(roots, expanded):
        return True
    # A folder reached through a link is the same folder.
    try:
        real = os.path.realpath(expanded)
        return output_rules.inside_roots([os.path.realpath(root) for root in roots], real)
    except (OSError, ValueError):
        return False


def _there(parm: Any, raw: str, expanded: str, frames: Iterable[float]) -> bool:
    """Whether the output is on disk, for a sequence at any frame tried."""
    if expanded and os.path.exists(expanded):
        return True
    if "$F" not in raw and "${F" not in raw:
        return False
    for frame in frames:
        value = tool_module._quiet(lambda frame=frame: parm.evalAsStringAtFrame(frame))
        if value and os.path.exists(str(value)):
            return True
    return False


def home_scratch(home: Any) -> Path | None:
    """The scratch folder a session with no `$HOUDINI_TEMP_DIR` writes under."""
    if os.environ.get("HOUDINI_TEMP_DIR") or home is None:
        return None
    return Path(home) / "temp"
