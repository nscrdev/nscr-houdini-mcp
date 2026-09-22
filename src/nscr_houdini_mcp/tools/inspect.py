"""`hou_inspect`: read nodes, networks and parameters, summary first.

Five modes, all read only.

- `tree` lists the nodes under a path, `depth` levels down, one compact row
  each. `standard` adds flags, wires, network boxes and sticky notes; `full`
  adds how many parameters each node has changed and where it sits.
- `node` reads one node, or a batch of up to fifty. A batch never fails for
  one path: each entry is the node or its own error.
- `parms` reads parameter tables: one node's, a batch of them, or single
  parameters by their paths. `parm_filter` keeps the ones that differ from
  their defaults (the default), all of them, or the names a glob matches.
- `find` looks under a path for nodes whose name, or whole path when the glob
  has a slash in it, matches `pattern`, and whose type matches `type`.
- `selection` lists what is selected in a session with a user interface. A
  worker has no selection, and says so rather than failing.

Rows are always sorted by path. A read with more rows than `limit` hands back
`next_page`, an opaque token holding the mode, the session, the last path
returned and the scene's state when it was taken. Sending it back as `page`
carries on after that path. When the scene was replaced, or the rows being
paged were added, removed or renamed, the next page still comes back and says
`scene_changed`. A token from another mode or session is refused with
`BAD_CURSOR`.

Nothing cooks unless `evaluate` is set. Without it, a value whose evaluation
would pull on a cook is left out and the row says `not_cooked`, and what a
node reports from its last cook, its errors included, carries `not_cooked`
when it has never cooked or `stale` with the reason when it has changed since.
With `evaluate` the read may cook, under the same `wait_s` and `timeout_s` as
any other call.

Keys are short because a large scene makes many rows: `n` a parameter's name,
`v` its value, `ev` a string as Houdini expands it, `expr` and `lang` its
expressions, `inst` a multiparm's instances, `in` the paths wired into a node,
`err` a count of errors, `auto` a node still named the way its type named it.
A field with nothing to say is left out rather than sent empty.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from typing import Any

from nscr_houdini_mcp.bridge.tools import (
    DEFAULT_LIMIT,
    INCLUDES,
    INSPECT_MODES,
    MAX_BATCH,
    MAX_LIMIT,
    MAX_TREE_DEPTH,
)
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import (
    DETAIL,
    SESSION,
    TIMEOUT_S,
    WAIT_S,
    Call,
    ToolSpec,
    inputs,
    outputs,
)

# The modes that read the paths they are given. `tree` and `find` look under
# one path instead, and `selection` takes none.
BY_PATHS = ("node", "parms")

# What a page token carries, under one short key each, and its version.
TOKEN_VERSION = 1

# What the bridge hands back that goes on to the caller unchanged.
PASSED_ON = ("rows", "nodes", "total", "boxes", "notes", "note", "truncated", "stopped")


def inspect(call: Call) -> Mapping[str, Any]:
    arguments = call.arguments
    mode = arguments.get("mode") or "tree"
    paths = given_paths(mode, arguments)
    within("depth", arguments.get("depth"), MAX_TREE_DEPTH)
    within("limit", arguments.get("limit"), MAX_LIMIT)
    if mode == "find" and not (arguments.get("pattern") or arguments.get("type")):
        raise CallError(
            "BAD_ARGUMENTS",
            "find needs pattern, type or both",
            details={"argument": "pattern"},
        )
    target = call.target()
    token = read_token(arguments.get("page"), mode=mode, session_id=target.session_id)

    sent: dict[str, Any] = {"mode": mode}
    if paths is not None:
        sent["paths"] = paths
        sent["batch"] = "paths" in arguments
    elif arguments.get("path"):
        sent["path"] = arguments["path"]
    for name in ("evaluate", "depth", "pattern", "type", "parm_filter", "include", "detail"):
        if arguments.get(name) is not None:
            sent[name] = arguments[name]
    sent["limit"] = arguments.get("limit") or DEFAULT_LIMIT
    if token is not None:
        sent["after"] = token["k"]

    reply = call.bridge("node.inspect", sent)
    data = dict(reply.get("data") or {})
    epoch = call.trace.get("scene_epoch")
    result: dict[str, Any] = {"mode": mode}
    for key in PASSED_ON:
        if key in data:
            result[key] = data[key]
    if data.get("more"):
        result["next_page"] = make_token(
            mode=mode,
            session_id=target.session_id,
            epoch=epoch,
            last=str(data.get("last") or "/"),
            digest=str(data.get("digest") or ""),
        )
    if token is not None and (token.get("e") != epoch or token.get("d") != data.get("digest")):
        result["scene_changed"] = True
    if reply.get("lossy"):
        result["lossy"] = True
        result["cut"] = reply.get("cut")
    return result


def given_paths(mode: str, arguments: Mapping[str, Any]) -> list[str] | None:
    """The paths a mode reads, checked here so a mistake never reaches a session.

    `node` and `parms` take `path` or `paths` and hand both on as a list; the
    others take at most a `path` to look under, and nothing is returned for
    them.
    """
    path = arguments.get("path")
    paths = arguments.get("paths")
    if path is not None and paths is not None:
        raise CallError(
            "BAD_ARGUMENTS",
            "pass path or paths, not both",
            details={"argument": "paths"},
        )
    wanted = [path] if path is not None else list(paths or [])
    if len(wanted) > MAX_BATCH:
        raise CallError(
            "BAD_ARGUMENTS",
            f"at most {MAX_BATCH} paths in one call",
            details={"argument": "paths", "given": len(wanted)},
        )
    for each in wanted:
        if not str(each).startswith("/"):
            raise CallError(
                "BAD_ARGUMENTS",
                f"{each} is not an absolute node path such as /obj/geo1",
                details={"argument": "paths" if paths is not None else "path", "path": each},
            )
    if mode in BY_PATHS:
        if not wanted:
            raise CallError(
                "BAD_ARGUMENTS",
                f"{mode} needs path or paths",
                details={"argument": "path"},
            )
        return [str(each) for each in wanted]
    if paths is not None:
        raise CallError(
            "BAD_ARGUMENTS",
            f"{mode} takes one path to look under, not paths",
            details={"argument": "paths"},
        )
    if mode == "selection" and path is not None:
        raise CallError(
            "BAD_ARGUMENTS",
            "selection takes no path",
            details={"argument": "path"},
        )
    return None


def within(name: str, value: Any, most: int) -> None:
    """Refuse a count outside 1 to `most`.

    Checked here rather than in the schema, which every client pays for on
    every tool list.
    """
    if value is not None and not 1 <= value <= most:
        raise CallError(
            "BAD_ARGUMENTS",
            f"{name} must be from 1 to {most}",
            details={"argument": name, "given": value},
        )


# Section: page tokens


def make_token(*, mode: str, session_id: str, epoch: Any, last: str, digest: str) -> str:
    body = {"v": TOKEN_VERSION, "m": mode, "s": session_id, "e": epoch, "k": last, "d": digest}
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=True)
    return base64.urlsafe_b64encode(text.encode("ascii")).decode("ascii").rstrip("=")


def read_token(page: Any, *, mode: str, session_id: str) -> dict[str, Any] | None:
    """The token a caller sent back, or `BAD_CURSOR` when it is not one of ours.

    A token only ever goes on in the mode and the session that made it. Scene
    edits never make a token unusable; the page that follows says the scene
    changed instead.
    """
    if page is None:
        return None
    try:
        padded = str(page) + "=" * (-len(str(page)) % 4)
        body = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (binascii.Error, UnicodeError, ValueError):
        raise bad_token("the page token could not be read") from None
    if not isinstance(body, dict) or body.get("v") != TOKEN_VERSION:
        raise bad_token("the page token could not be read")
    if not isinstance(body.get("k"), str) or not str(body["k"]).startswith("/"):
        raise bad_token("the page token could not be read")
    if body.get("m") != mode:
        raise bad_token(
            f"that page token belongs to a {body.get('m')} read, not {mode}",
            token_mode=body.get("m"),
        )
    if body.get("s") != session_id:
        raise bad_token("that page token belongs to another session")
    return body


def bad_token(message: str, **details: Any) -> CallError:
    return CallError("BAD_CURSOR", message, details={"argument": "page", **details})


# Section: the one line a long result is summed up in


def summary_line(data: Mapping[str, Any]) -> str:
    mode = data.get("mode")
    if "rows" in data:
        shown = len(data.get("rows") or [])
        line = f"hou_inspect {mode}: {shown} of {data.get('total', shown)} rows"
    else:
        entries = data.get("nodes") or []
        parms = sum(len(entry.get("parms") or []) for entry in entries)
        line = f"hou_inspect {mode}: {len(entries)} nodes, {parms} parameters"
    if data.get("next_page"):
        line += "; more with next_page"
    return line


HOU_INSPECT = ToolSpec(
    name="hou_inspect",
    description=(
        "Read nodes, networks and parameters. Modes: tree, node, parms, find, selection. "
        "Summary first; page with next_page. Cooks nothing unless evaluate."
    ),
    input_schema=inputs(
        {
            "mode": {"type": "string", "enum": list(INSPECT_MODES)},
            "session": SESSION,
            "path": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "evaluate": {"type": "boolean"},
            "depth": {"type": "integer"},
            "pattern": {"type": "string"},
            "type": {"type": "string"},
            "parm_filter": {"type": "string", "description": "non_default, all or a glob"},
            "include": {"type": "array", "items": {"enum": list(INCLUDES)}},
            "detail": DETAIL,
            "limit": {"type": "integer"},
            "page": {"type": "string"},
            "wait_s": WAIT_S,
            "timeout_s": TIMEOUT_S,
        }
    ),
    output_schema=outputs({}),
    handler=inspect,
    open_world=False,
    summary=summary_line,
)
