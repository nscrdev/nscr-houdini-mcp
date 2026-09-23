"""Managed outputs on the Houdini side: frozen parameters and the lint.

The ruling on output parameters. While a run is in progress, the output
node's real output parameter holds the frozen absolute path for that run, so
a rename halfway through moves nothing and the record and the scene agree.
When the run is over, whether it ended done, failed or cancelled, the
parameter gets back what it held before, the `$HIP` line a person set or the
expression that fed it, so a scene saved afterwards never carries a path from
this machine. A save while the run is still going writes that same value and
puts the run's path back straight after, so the file never holds it either.

Each freeze is recorded in the coordination store before the parameter is
touched: the session, the node and its session id, the parameter, what it
held before, the path it is to be given, the scene it is in and a token that
says whose freeze it is. The record is `prepared` until the run's path is on
the node and `active` after, so a process that dies part way leaves a record
that says to put the old value back whatever the parameter holds. A
parameter already held under another token is `PARM_FROZEN`, naming the run
that holds it, and only the token's holder can give it back. The record goes,
compared on its token and state, when the parameter has its own value back.
Two things give it back:

- the call that froze it, when it ends. Today that is `python.run`, for
  whatever its code froze through `mcp.freeze_parm`. `outputs.freeze_parm`
  is the same freeze as a bridge operation, for a later tool that runs a job
  and gives the parameter back through `outputs.restore_parm` when the job
  is over; nothing on the server calls it yet.
- the session that loads the scene next, or saves it, when the one that
  froze it has gone, asked by the server with `outputs.restore_parm` and the
  owner and token named. Its copy of the scene may be older than the file the
  dead session saved, so the record stays for as long as the file on disk
  still holds the frozen path, and goes once a save has written it out.

A record from a scene that was never saved and whose session has gone is
dropped, because no session can open that scene again. An active record's
parameter is given its value back only when it still holds the path it was
frozen to; one changed since is somebody's own edit and is left as it is.

`outputs.lint` reads every output parameter under a node, walking the network
in path order a node at a time and stopping once its page is full or the call
is asked to stop. A parameter is an output when its node type marks it as a
file it writes to, a multiparm's instances included, or, on a render node,
when it is a file parameter the type leaves unmarked, as Alembic and USD
render nodes do; hooks run before and after a render, a renderer's own log
files, a render it reads back and a folder are not. What it reports, per
parameter, is any of `absolute_path`, `outside_hip`, `unversioned`,
`missing_on_disk` (with `empty` when a node's main output, or one a toggle
turns on, holds nothing; a spare output left empty is only unused) and
`frozen_after_run`, or `expression` for a value that only an evaluation could
give, which the lint never runs, and `unexpanded` for a variable it cannot
fill in. Nothing is cooked or evaluated: the value is expanded here from the
variables the output table knows, as the session has them, and the node's
and scene's own names.

This module never imports `hou`: it is handed it.
"""

from __future__ import annotations

import os
import re
import secrets
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import outputs as output_rules
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import tools as tool_module
from nscr_houdini_mcp.bridge.errors import BridgeError
from nscr_houdini_mcp.bridge.tools import ToolContext

PROBLEMS = (
    "expression",
    "unexpanded",
    "frozen_after_run",
    "absolute_path",
    "outside_hip",
    "unversioned",
    "missing_on_disk",
)

# The chooser mode a node type gives a parameter it writes to.
WRITE_MODE = "write"

# What a render node carries that is a file parameter and still not an output:
# scripts it runs before and after, the log files of its renderer, and a
# render it only reads back.
HOOK_PREFIXES = ("pre", "post")
LOG_PREFIX = "husk_"
NOT_OUTPUTS = frozenset({"renderexisting"})
NOT_A_PLACE_TYPES = ("Directory", "Lut")

# Values that are not places on disk: a viewer, and a name a render node fills
# in for itself.
DEVICES = frozenset({"ip", "md"})
PLACEHOLDER_PREFIX = "__render__"

# The variables the output table fills in, read from the session.
SESSION_VARIABLES = ("HIP", "JOB", "HOUDINI_TEMP_DIR")

DEFAULT_LIMIT = 200
MAX_LIMIT = 2000

# The most bytes of a scene file read at a time when looking for a path in it.
SCAN_CHUNK = 1 << 20

TOKEN_BYTES = 8

# What the bridge operations take. `after` is the node path and parameter
# name the lint page before ended on.
LINT_ARGUMENTS = ("scope", "after", "limit")
FREEZE_ARGUMENTS = ("node", "parm", "run_id", "token")
RESTORE_ARGUMENTS = ("node", "parm", "token", "owner")

_BRACED_FRAME = re.compile(r"\$\{(F\d*)\}")

# Sorts after every parameter name, which is letters, digits and underscores.
PAST_EVERY_NAME = "~"


# Section: the scene and the session


def scene_path(hou: Any) -> str | None:
    """The scene file this session holds, or nothing for a scene never saved."""
    if tool_module._quiet(hou.hipFile.isNewFile):
        return None
    return tool_module._quiet(hou.hipFile.path)


def session_variables(hou: Any) -> dict[str, str | None]:
    """`$HIP`, `$JOB` and `$HOUDINI_TEMP_DIR` as this session has them.

    Houdini's own values first, which is where a launcher or a package sets
    them, then this process's environment.
    """
    found: dict[str, str | None] = {}
    for name in SESSION_VARIABLES:
        value = tool_module._quiet(lambda name=name: hou.getenv(name))
        found[name] = str(value) if value else os.environ.get(name) or None
    return found


def variables(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """`outputs.variables`: the output table's variables as this session has them."""
    return session_variables(tool_module._houdini(context))


def raw_value(parm: Any) -> str:
    """What a string parameter holds before anything is expanded.

    A parameter with an expression has no single string, so its expression
    is read instead.
    """
    text = tool_module._quiet(parm.unexpandedString)
    if text is None:
        text = tool_module._quiet(parm.expression)
    return "" if text is None else str(text)


def file_holds(path: str | None, text: str) -> bool:
    """Whether a file on disk has this text in it, read a piece at a time."""
    if not path or not text:
        return False
    wanted = text.encode("utf-8")
    try:
        with open(path, "rb") as stream:
            tail = b""
            while True:
                piece = stream.read(SCAN_CHUNK)
                if not piece:
                    return False
                if wanted in tail + piece:
                    return True
                tail = piece[-len(wanted) :]
    except OSError:
        return False


def _find_node(hou: Any, row: Any, node_path: str, *, own: bool) -> Any:
    """The node a record is about: by its session id where that still means
    something, which follows a rename, and by its path otherwise."""
    if own and row.node_sid is not None:
        found = tool_module._quiet(lambda: hou.nodeBySessionId(int(row.node_sid)))
        if found is not None:
            return found
    return tool_module._quiet(lambda: hou.node(node_path))


def _language_value(hou: Any, language: str | None) -> Any:
    names = getattr(hou, "exprLanguage", None)
    python = (language or "").lower() == "python"
    found = getattr(names, "Python" if python else "Hscript", None) if names is not None else None
    return found if found is not None else ("python" if python else "hscript")


def put_back(hou: Any, parm: Any, row: Any) -> None:
    """Give a parameter what it held before it was frozen.

    An expression goes back over a cleared value: Houdini keeps the value an
    expression was set over as the channel's own default and writes it into
    the scene file, so setting it over the run's path would save that path.
    """
    if row.original_expression is not None:
        tool_module._quiet(parm.deleteAllKeyframes)
        parm.set(row.original if row.original is not None else "")
        parm.setExpression(row.original_expression, _language_value(hou, row.original_language))
        return
    parm.set(row.original if row.original is not None else row.template)


def hold(parm: Any, frozen: str) -> None:
    """Put a run's own path on a parameter, and nothing else under it."""
    # A keyframe or an expression would carry on under the value.
    tool_module._quiet(parm.deleteAllKeyframes)
    parm.set(frozen)


def new_token() -> str:
    return secrets.token_hex(TOKEN_BYTES)


# Section: freezing and restoring


def freeze(
    hou: Any,
    open_store: Callable[[], Any],
    *,
    session_id: str,
    node_path: str,
    parm_name: str,
    run_id: str,
    token: str | None = None,
) -> dict[str, Any]:
    """Set one output parameter to a run's own path, after recording it.

    Only a path this server handed out can be frozen: the run names it. What
    the parameter holds now, text or an expression, is read and recorded
    before anything touches it, and is what it gets back. A parameter with
    several keyframes is refused, because putting that back is not something
    a record can promise. A run whose line holds a machine path, such as a
    spill, never goes on a node. `token` is the holder's own, for a later run
    of the same call on the same parameter; the answer carries the one to
    give it back with.
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
    expression, keyed = tool_module._expression_of(parm)
    if keyed:
        raise BridgeError(
            "BAD_ARGUMENTS",
            "the parameter is animated with several keyframes, which a run will not overwrite",
            {"node": node.path(), "parm": parm.name()},
            hint="set the parameter to one value or one expression first",
        )
    held_by = token or new_token()
    _drop_stale(hou, open_store, session_id, node.path(), parm.name(), held_by)
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
        try:
            row = store.freeze_parm(
                session_id=session_id,
                node_path=node.path(),
                parm_name=parm.name(),
                template=template,
                frozen=frozen,
                run_id=run_id,
                hip_key=output_rules.scene_key(scene_path(hou)),
                node_sid=tool_module._quiet(node.sessionId),
                original=None if expression is not None else raw_value(parm),
                original_expression=expression,
                original_language=tool_module._language(parm) if expression is not None else None,
                token=held_by,
            )
        except store_module.ParmHeld as error:
            raise BridgeError(
                "PARM_FROZEN",
                "the parameter is held by another run",
                {"node": node.path(), "parm": parm.name(), "run_id": error.run_id},
            ) from None
    try:
        hold(parm, frozen)
    except BaseException:
        # The run never took hold: the parameter gets back what it had, and
        # the record, still prepared, goes.
        tool_module._quiet(lambda: put_back(hou, parm, row))
        with open_store() as store:
            store.thaw_parm(
                session_id,
                node.path(),
                parm.name(),
                token=held_by,
                state=store_module.FROZEN_PREPARED,
            )
        raise
    with open_store() as store:
        store.activate_frozen_parm(session_id, node.path(), parm.name(), token=held_by)
    return {
        "node": node.path(),
        "parm": parm.name(),
        "value": frozen,
        "template": template,
        "run_id": run_id,
        "token": held_by,
        "owed": _owed(row),
    }


def _drop_stale(
    hou: Any,
    open_store: Callable[[], Any],
    session_id: str,
    node_path: str,
    parm_name: str,
    token: str,
) -> None:
    """Let go of this session's own record for a parameter path that no longer
    means the node it was made for.

    A record kept because its node could not be found would otherwise hold
    the path for the rest of the session, so a node made again under the same
    name could never be frozen. It is stale when the node it names is gone
    from this session, or when this session now holds another scene.
    """
    with open_store() as store:
        row = store.get_frozen_parm(session_id, node_path, parm_name)
        if row is None or row.token == token:
            return
        gone = (
            row.node_sid is not None
            and tool_module._quiet(lambda: hou.nodeBySessionId(int(row.node_sid))) is None
        )
        moved = row.hip_key != output_rules.scene_key(scene_path(hou))
        if gone or moved:
            store.thaw_parm(session_id, node_path, parm_name, token=row.token, state=row.state)


def _owed(row: Any) -> dict[str, Any]:
    if row.original_expression is not None:
        return {"expression": row.original_expression, "language": row.original_language}
    return {"value": row.original if row.original is not None else row.template}


def restore(
    hou: Any,
    open_store: Callable[[], Any],
    *,
    session_id: str,
    node_path: str,
    parm_name: str,
    token: str | None,
    owner: str | None = None,
) -> dict[str, Any]:
    """Give one frozen parameter what it held before, and drop its record.

    `token` has to be the one the freeze handed out. `owner` is the session
    that froze it, when that is not this one: only a session that is over can
    have its parameters restored by another, and only in the scene they were
    frozen in. The owner finds a renamed node by its session id; a node it
    cannot find keeps its record, because the path may still be on it
    somewhere. Another session keeps the record for as long as the scene file
    on disk still holds the frozen path.
    """
    holder = owner or session_id
    own = holder == session_id
    said: dict[str, Any] = {"node": node_path, "parm": parm_name, "restored": False}
    with open_store() as store:
        row = store.get_frozen_parm(holder, node_path, parm_name)
        if row is None:
            return {**said, "reason": "not_frozen"}
        if row.token != token:
            raise BridgeError(
                "PARM_FROZEN",
                "the parameter is held by another run, under another token",
                {"node": node_path, "parm": parm_name, "run_id": row.run_id},
            )
        if not own:
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
    node = _find_node(hou, row, node_path, own=own)
    parm = None if node is None else tool_module._quiet(lambda: node.parm(parm_name))
    if parm is None:
        said["reason"] = "node_gone"
    elif row.state == store_module.FROZEN_PREPARED or raw_value(parm) == row.frozen:
        # A prepared record's run never took hold, so whatever the parameter
        # holds now is left from part way through, never somebody's edit.
        put_back(hou, parm, row)
        said["restored"] = True
    else:
        said["reason"] = "changed_since"
    if parm is not None:
        said["node"] = tool_module._quiet(node.path) or node_path
        said["value"] = raw_value(parm)
    said["owed"] = _owed(row)
    if own:
        kept = said.get("reason") == "node_gone"
    else:
        kept = file_holds(scene_path(hou), row.frozen)
    if kept:
        said["kept"] = True
    else:
        with open_store() as store:
            store.thaw_parm(holder, node_path, parm_name, token=row.token, state=row.state)
    return said


class SaveGuard:
    """While a call holds frozen parameters, a save writes their own values.

    It listens to the scene file's save events: before a save each frozen
    parameter gets what it held before, and after the save the run's path
    goes back on it, so the file on disk never holds this machine's path and
    the run carries on as it was. It listens only while it watches something,
    and `close` stops it.
    """

    def __init__(self, hou: Any, open_store: Callable[[], Any], session_id: str) -> None:
        self._hou = hou
        self._open_store = open_store
        self._session_id = session_id
        self._watched: list[tuple[str, str]] = []
        self._lent: list[tuple[Any, str]] = []
        self._listening = False

    def watch(self, node_path: str, parm_name: str) -> None:
        if (node_path, parm_name) not in self._watched:
            self._watched.append((node_path, parm_name))
        if not self._listening:
            self._hou.hipFile.addEventCallback(self._event)
            self._listening = True

    def close(self) -> None:
        if self._listening:
            tool_module._quiet(lambda: self._hou.hipFile.removeEventCallback(self._event))
            self._listening = False
        self._take_back()

    def _event(self, event_type: Any, *rest: Any) -> None:
        name = str(event_type).rsplit(".", 1)[-1]
        if name == "BeforeSave":
            self._lend()
        elif name == "AfterSave":
            self._take_back()

    def _lend(self) -> None:
        try:
            with self._open_store() as store:
                rows = [store.get_frozen_parm(self._session_id, *each) for each in self._watched]
        except Exception:  # noqa: BLE001 - a save must go on whatever the store says
            return
        for row in rows:
            if row is None:
                continue
            node = _find_node(self._hou, row, row.node_path, own=True)
            parm = (
                None
                if node is None
                else tool_module._quiet(lambda node=node, row=row: node.parm(row.parm_name))
            )
            if parm is None or raw_value(parm) != row.frozen:
                continue
            try:
                put_back(self._hou, parm, row)
            except Exception:  # noqa: BLE001 - one parameter must not stop a save
                continue
            self._lent.append((parm, row.frozen))

    def _take_back(self) -> None:
        while self._lent:
            parm, frozen = self._lent.pop()
            tool_module._quiet(lambda parm=parm, frozen=frozen: hold(parm, frozen))


def freeze_parm(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """`outputs.freeze_parm`: hold a run's own path on an output parameter."""
    hou = tool_module._houdini(context)
    token = arguments.get("token")
    return freeze(
        hou,
        _store_opener(context),
        session_id=context.session_id,
        node_path=str(arguments["node"]),
        parm_name=str(arguments["parm"]),
        run_id=str(arguments["run_id"]),
        token=None if token is None else str(token),
    )


def restore_parm(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """`outputs.restore_parm`: give a frozen parameter its own value back."""
    hou = tool_module._houdini(context)
    owner = arguments.get("owner")
    return restore(
        hou,
        _store_opener(context),
        session_id=context.session_id,
        node_path=str(arguments["node"]),
        parm_name=str(arguments["parm"]),
        token=str(arguments["token"]),
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
    and a page ends on a whole parameter. The walk goes a node at a time in
    that order and stops once the page is full and one more parameter with a
    problem has been seen, or when the call is asked to stop. `more` says a
    later page has rows; `last` is where the next one starts after.
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
    scene = _SceneNames(hou, hip)
    roots = output_rules.managed_roots(
        conventions,
        hip_path=hip,
        session_id=context.session_id,
        scratch_root=home_scratch(context.home),
        variables=scene.variables,
    )
    frozen = _frozen_here(context, hip)
    frames = [int(round(frame)) for frame in tool_module._frames_to_try(hou)]

    kinds: dict[str, tuple[tuple[tuple[str, bool], ...], bool]] = {}
    mains: dict[str, str | None] = {}
    rows: list[dict[str, Any]] = []
    last: list[str] | None = None
    more = False
    stopped = False
    checked = 0
    scanned = 0
    capped = False
    # The last node every parameter of which has been looked at: a page cut
    # short goes on after it.
    through: list[str] | None = None
    for node in _walk(top, after):
        if context.should_stop():
            stopped = more = True
            break
        scanned += 1
        if scanned > tool_module.MAX_NODES_SCANNED:
            capped = more = True
            break
        path = str(tool_module._quiet(node.path) or "")
        if not path:
            continue
        outputs = _output_parms(node, kinds)
        main = _main_output(node, kinds, mains)
        if not outputs:
            through = [path, PAST_EVERY_NAME]
            continue
        # Whether a parameter is turned off depends on the others, and a
        # node just made may not have worked that out yet.
        tool_module._quiet(node.updateParmStates)
        for name, marked, parm in outputs:
            if after is not None and (_key(path), name) <= after:
                continue
            if tool_module._quiet(parm.isDisabled):
                continue
            if not marked and tool_module._quiet(parm.isHidden):
                continue
            checked += 1
            wanted = name == main or _turned_on(node, name)
            found = _check(parm, node, path, name, marked, roots, frozen, frames, scene, wanted)
            if not found:
                continue
            if len(rows) >= limit:
                more = True
                break
            rows.extend(found)
            last = [path, name]
        if more:
            break
        through = [path, PAST_EVERY_NAME]
    if stopped or capped:
        # The page stops where the walk did, so the next one goes on from there.
        last = through if through is not None else last
    said: dict[str, Any] = {
        "rows": rows,
        "more": more and last is not None,
        "last": last,
        "roots": roots,
        "parms_checked": checked,
    }
    if capped:
        said["truncated"] = True
    if stopped:
        said["stopped"] = True
    if notes:
        said["notes"] = notes
    return said


def home_scratch(home: Any) -> Path | None:
    """The scratch folder a session with no `$HOUDINI_TEMP_DIR` writes under."""
    if os.environ.get("HOUDINI_TEMP_DIR") or home is None:
        return None
    return Path(home) / "temp"


def _key(path: str) -> tuple[str, ...]:
    """A path's sort key: its parts, so a node sorts just before its insides."""
    return tuple(part for part in str(path).split("/") if part)


def _after(value: Any) -> tuple[tuple[str, ...], str] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(part, str) for part in value)
    ):
        raise BridgeError("BAD_ARGUMENTS", "after must be a node path and a parameter name")
    return (_key(value[0]), value[1])


def _walk(top: Any, after: tuple[tuple[str, ...], str] | None) -> Iterator[Any]:
    """The node and everything under it in path order, one node at a time.

    Children are read as the walk reaches them, never all at once, and a
    subtree that sorts wholly before `after` is not entered. The insides of a
    locked asset are not a person's to change, so they are not walked.
    """
    stack = [top]
    while stack:
        node = stack.pop()
        key = _key(str(tool_module._quiet(node.path) or ""))
        if after is not None and key < after[0] and after[0][: len(key)] != key:
            continue
        yield node
        if node is not top and tool_module._quiet(node.isLockedHDA):
            continue
        children = tool_module._sorted_children(node)
        stack.extend(reversed(children))


def _frozen_here(context: ToolContext, hip: str | None) -> dict[Any, Any]:
    """Frozen parameters this session holds, by path and by node session id,
    and those left in this scene by sessions that have gone."""
    if context.open_store is None:
        return {}
    found: dict[Any, Any] = {}
    with context.open_store() as store:
        own = list(store.list_frozen_parms(session_id=context.session_id))
        key = output_rules.scene_key(hip)
        others = store.list_frozen_parms(hip_key=key) if key is not None else []
    for row in [*own, *others]:
        found.setdefault((row.node_path, row.parm_name), row)
    for row in own:
        if row.node_sid is not None:
            found.setdefault(("sid", row.node_sid, row.parm_name), row)
    return found


def _output_parms(
    node: Any, kinds: dict[str, tuple[tuple[tuple[str, bool], ...], bool]]
) -> list[tuple[str, bool, Any]]:
    """The parameters of a node that are outputs, sorted by name.

    Which ones are read from the node's type once and kept, each with
    whether the type marks it as written to; the names come from the type,
    never from a list kept here. A type with a multiparm has its instances
    read on each node, because each node has its own.
    """
    node_type = tool_module._quiet(node.type)
    key = str(
        tool_module._quiet(node_type.nameWithCategory) or tool_module._quiet(node_type.name) or ""
    )
    renders = _is_render_node(node)
    if key not in kinds:
        found: list[tuple[str, bool]] = []
        multiparm = False
        for parm in tool_module._quiet(node.parms) or ():
            template = tool_module._quiet(parm.parmTemplate)
            if _is_multiparm(template):
                multiparm = True
            if tool_module._quiet(parm.isSpare) or tool_module._quiet(parm.isMultiParmInstance):
                continue
            name = str(parm.name())
            marked = _writes(template, name, renders)
            if marked is not None:
                found.append((name, marked))
        kinds[key] = (tuple(found), multiparm)
    names, multiparm = kinds[key]
    picked: list[tuple[str, bool, Any]] = []
    for name, marked in names:
        parm = tool_module._quiet(lambda name=name: node.parm(name))
        if parm is not None:
            picked.append((name, marked, parm))
    if multiparm:
        for parm in tool_module._quiet(node.parms) or ():
            if not tool_module._quiet(parm.isMultiParmInstance):
                continue
            name = str(parm.name())
            marked = _writes(tool_module._quiet(parm.parmTemplate), name, renders)
            if marked is not None:
                picked.append((name, marked, parm))
    picked.sort(key=lambda item: item[0])
    return picked


def _main_output(
    node: Any,
    kinds: dict[str, tuple[tuple[tuple[str, bool], ...], bool]],
    mains: dict[str, str | None],
) -> str | None:
    """The node's main output: the first parameter its type marks as written
    to, in the type's own order, such as a render's picture."""
    node_type = tool_module._quiet(node.type)
    key = str(
        tool_module._quiet(node_type.nameWithCategory) or tool_module._quiet(node_type.name) or ""
    )
    if key not in mains:
        names = kinds.get(key, ((), False))[0]
        mains[key] = next((name for name, marked in names if marked), None)
    return mains[key]


# Toggles a node type turns an extra output on with, by the output's name.
TOGGLE_PREFIXES = ("use", "enable")


def _turned_on(node: Any, name: str) -> bool:
    for prefix in TOGGLE_PREFIXES:
        toggle = tool_module._quiet(lambda prefix=prefix: node.parm(f"{prefix}{name}"))
        if toggle is None:
            continue
        kind = tool_module._quiet(lambda toggle=toggle: toggle.parmTemplate().type().name())
        if kind == "Toggle" and tool_module._quiet(toggle.eval):
            return True
    return False


def _is_multiparm(template: Any) -> bool:
    if template is None or tool_module._quiet(lambda: template.type().name()) != "Folder":
        return False
    return "Multiparm" in str(tool_module._quiet(template.folderType) or "")


def _is_render_node(node: Any) -> bool:
    """A render node: one in an output network, or one with a render button."""
    category = tool_module._quiet(lambda: node.type().category().name())
    if category == "Driver":
        return True
    button = tool_module._quiet(lambda: node.parm("execute"))
    kind = (
        None if button is None else tool_module._quiet(lambda: button.parmTemplate().type().name())
    )
    return kind == "Button"


def _writes(template: Any, name: str, renders: bool) -> bool | None:
    """True for a parameter the type marks as written to, False for an unmarked
    file parameter on a render node, and nothing for anything else."""
    if template is None:
        return None
    if tool_module._quiet(lambda: template.type().name()) != "String":
        return None
    if not str(tool_module._quiet(template.stringType)).endswith("FileReference"):
        return None
    if str(tool_module._quiet(template.fileType)).endswith(NOT_A_PLACE_TYPES):
        return None
    if name.startswith(HOOK_PREFIXES) or name.startswith(LOG_PREFIX) or name in NOT_OUTPUTS:
        return None
    tags = tool_module._quiet(template.tags) or {}
    mode = tags.get("filechooser_mode")
    if mode == WRITE_MODE:
        return True
    if not mode and renders:
        return False
    return None


class _SceneNames:
    """The variables a lint fills in itself, read once, with nothing evaluated."""

    def __init__(self, hou: Any, hip: str | None) -> None:
        def env(name: str) -> str | None:
            value = tool_module._quiet(lambda: hou.getenv(name))
            return str(value) if value else os.environ.get(name) or None

        folder, stem = output_rules.split_hip(hip) if hip else (None, None)
        self.variables = session_variables(hou)
        self.hip_dir = self.variables.get("HIP") or folder or os.getcwd()
        self.names = {
            "HIPNAME": env("HIPNAME") or stem or "untitled",
            "HIPFILE": env("HIPFILE") or hip or "",
        }

    def expand(self, raw: str, node_name: str, frame: int) -> str | None:
        """The value with every variable filled in, or nothing when one cannot be."""
        text = _BRACED_FRAME.sub(r"$\1", raw)
        try:
            filled = output_rules.expand(
                text,
                hip_dir=self.hip_dir,
                temp_dir=self.variables.get("HOUDINI_TEMP_DIR"),
                job=self.variables.get("JOB") or "",
                frame=frame,
                names={**self.names, "OS": node_name},
            )
        except output_rules.OutputError:
            return None
        return None if "$" in filled else filled


def _check(
    parm: Any,
    node: Any,
    path: str,
    name: str,
    marked: bool,
    roots: list[str],
    frozen: Mapping[Any, Any],
    frames: list[int],
    scene: _SceneNames,
    wanted: bool = True,
) -> list[dict[str, Any]]:
    """The problems one output parameter has, one row each.

    `wanted` says an empty value is a problem: the node's main output, or
    one a toggle of its own turns on. An empty spare output is just unused.
    """
    raw = raw_value(parm)
    expression, keyed = tool_module._expression_of(parm)
    evaluated = (
        expression is not None
        or keyed
        or "`" in raw
        or bool(tool_module._quiet(parm.isOverrideTrackActive))
    )
    if evaluated:
        # Only an evaluation could say what it is, and a lint never runs one.
        # An unmarked parameter that is an expression is not taken for an output.
        if not marked:
            return []
        return [_row(path, name, raw, None, "expression")]
    if not raw.strip():
        # A parameter the type writes to that names nothing writes nowhere.
        if not marked or not wanted:
            return []
        return [{**_row(path, name, raw, None, "missing_on_disk"), "empty": True}]
    if not _names_a_place(raw):
        return []
    node_name = str(tool_module._quiet(node.name) or "")
    current = frames[0] if frames else 1
    expanded = scene.expand(raw, node_name, current)
    if expanded is None and not marked:
        # A variable this server does not know, on a parameter the type does
        # not mark: not an output the table manages.
        return []
    problems: list[str] = []
    held = frozen.get((path, name)) or frozen.get(("sid", tool_module._quiet(node.sessionId), name))
    if held is not None and raw == held.frozen:
        problems.append("frozen_after_run")
    if output_rules.is_machine_path(raw):
        problems.append("absolute_path")
    if not output_rules.has_version(raw) and not output_rules.has_version(expanded or ""):
        problems.append("unversioned")
    if expanded is None:
        problems.append("unexpanded")
    else:
        if not _managed(roots, expanded):
            problems.append("outside_hip")
        if not _there(raw, expanded, node_name, frames, scene):
            problems.append("missing_on_disk")
    return [_row(path, name, raw, expanded, problem) for problem in problems]


def _row(path: str, name: str, raw: str, expanded: str | None, problem: str) -> dict[str, Any]:
    return {"node": path, "parm": name, "raw": raw, "expanded": expanded, "problem": problem}


def _names_a_place(raw: str) -> bool:
    """Whether a value names a file at all.

    `ip` and `md` send a render to a viewer and `__render__.usd` is a name a
    render node fills in for itself: none is a place on disk. Anything else
    with a folder, a variable or an extension in it is, a bare `x.exr` too.
    """
    text = raw.strip()
    if not text or text.lower() in DEVICES or text.startswith(PLACEHOLDER_PREFIX):
        return False
    leaf = text.replace("\\", "/").rsplit("/", 1)[-1]
    return "/" in text or "\\" in text or "$" in text or "." in leaf


def _managed(roots: list[str], expanded: str) -> bool:
    if output_rules.inside_roots(roots, expanded):
        return True
    # A folder reached through a link is the same folder.
    try:
        real = os.path.realpath(expanded)
        return output_rules.inside_roots([os.path.realpath(root) for root in roots], real)
    except (OSError, ValueError):
        return False


def _there(
    raw: str, expanded: str, node_name: str, frames: Iterable[int], scene: _SceneNames
) -> bool:
    """Whether the output is on disk, for a sequence at any frame tried."""
    if os.path.exists(expanded):
        return True
    if "$F" not in raw and "${F" not in raw:
        return False
    for frame in frames:
        value = scene.expand(raw, node_name, frame)
        if value and os.path.exists(value):
            return True
    return False
