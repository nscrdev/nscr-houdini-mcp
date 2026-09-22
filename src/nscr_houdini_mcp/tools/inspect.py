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
  Every node asked for has one entry, even when nothing in it passes, and a
  parameter asked for twice, by its name and a component's, is one row.
- `find` looks under a path for nodes whose name, or whole path when the glob
  has a slash in it, matches `pattern`, and whose type matches `type`.
- `selection` lists what is selected in a session with a user interface. A
  worker has no selection, and says so rather than failing.

Rows are always sorted by path, and every mode pages: `tree`, `find` and
`selection` by node, `node` by entry, `parms` by parameter row. A read with
more than `limit` hands back `next_page`, an opaque token holding the mode,
the session, the last path returned, a fingerprint of the arguments that
decide the rows, and the scene's state when it was taken. Sending it back as
`page`, with the same arguments, carries on after that path. A tree or a
search walks in path order and stops once the page is full, so `total` is
there only when the first page already holds every row. When the scene was
replaced or edited in between, or the row a page goes on after has gone, the
next page still comes back and says `scene_changed`. A token from another
mode, session or query, or one that is not exactly what this server writes,
is refused with `BAD_CURSOR`.

Nothing cooks unless `evaluate` is set. Without it, a value that could only
be had by cooking is left out, and the row says why in `not_cooked`:
`override` for a number a channel operator's export may drive, `keyframes`
for a parameter with more than one key, `python` for a Python expression,
`backtick` for a string with an expression in it, `variable $NPT` for a
variable that only means something inside a cook, `calls npoints()` for an
expression that reads the scene, and a reference's own reason after its name
for a `ch()` that reaches one of those. A node that has never cooked carries
`not_cooked: true`, and one that has changed since its last cook carries
`stale` with the reason, because its errors are the ones that cook left. With
`evaluate` the read may cook, under the same `wait_s` and `timeout_s` as any
other call, and a read that runs out of time is asked to stop between rows.

Keys are short because a large scene makes many rows: `n` a parameter's name,
`v` its value, `ev` a string as Houdini expands it, `expr` and `lang` its
expressions, `inst` a multiparm's instances (at most 200 in a row, with
`inst_total` and `truncated` past that; a multiparm named on its own pages
through all of them from `inst_from`), `in` the paths wired into a node,
`err` a count of errors, `auto` a node still named the way its type named it.
A field with nothing to say is left out rather than sent empty.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
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

# What a page token carries, under one short key each, and its version. The
# token is refused unread when it is longer than a token this server makes.
TOKEN_VERSION = 1
MAX_TOKEN_CHARS = 1024
TOKEN_FIELDS = {
    "v": int,  # the version
    "m": str,  # the mode
    "s": str,  # the session id
    "e": int,  # the scene epoch when the page was made
    "k": str,  # the path the next page starts after
    "d": str,  # the scene's edit mark when the page was made
    "q": str,  # which query the page belongs to
}
TOKEN_LENGTHS = {"m": 16, "s": 128, "k": 900, "d": 64, "q": 32}

# The arguments that decide which rows a read walks through. A page token only
# goes on with the same ones.
QUERY_ARGUMENTS = ("pattern", "type", "parm_filter", "depth", "detail")

# What the bridge hands back that goes on to the caller unchanged.
PASSED_ON = (
    "rows",
    "nodes",
    "total",
    "boxes",
    "notes",
    "boxes_truncated",
    "notes_truncated",
    "note",
    "truncated",
    "stopped",
)


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
    query = query_of(mode, arguments, paths)
    token = read_token(arguments.get("page"), mode=mode, session_id=target.session_id, query=query)

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
    mark = str(data.get("mark") or "")
    if data.get("more"):
        result["next_page"] = make_token(
            mode=mode,
            session_id=target.session_id,
            epoch=epoch if isinstance(epoch, int) else -1,
            last=str(data.get("last") or "/"),
            mark=mark,
            query=query,
        )
    if token is not None and (token["e"] != epoch or token["d"] != mark or data.get("resume_gone")):
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


def query_of(mode: str, arguments: Mapping[str, Any], paths: list[str] | None) -> str:
    """A short fingerprint of the arguments that decide which rows a read has."""
    where = tidy(str(arguments.get("path") or "/")) if paths is None else None
    asked = {
        "mode": mode,
        "path": where,
        "paths": sorted({tidy(path) for path in paths}) if paths is not None else None,
        **{name: arguments.get(name) for name in QUERY_ARGUMENTS},
    }
    text = json.dumps(asked, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def tidy(path: str) -> str:
    """A node path without a trailing or doubled slash."""
    return "/" + "/".join(part for part in path.split("/") if part)


def make_token(*, mode: str, session_id: str, epoch: int, last: str, mark: str, query: str) -> str:
    body = {
        "v": TOKEN_VERSION,
        "m": mode,
        "s": session_id,
        "e": epoch,
        "k": last,
        "d": mark,
        "q": query,
    }
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=True)
    return base64.urlsafe_b64encode(text.encode("ascii")).decode("ascii").rstrip("=")


def read_token(page: Any, *, mode: str, session_id: str, query: str) -> dict[str, Any] | None:
    """The token a caller sent back, or `BAD_CURSOR` when it is not one of ours.

    A token only ever goes on in the mode, the session and the query that
    made it. Anything that is not exactly the shape this server writes is
    refused, however it fails to be read. Scene edits never make a token
    unusable; the page that follows says the scene changed instead.
    """
    if page is None:
        return None
    body = decode_token(page)
    if body["m"] != mode:
        raise bad_token(
            f"that page token belongs to a {body['m']} read, not {mode}",
            token_mode=body["m"],
        )
    if body["s"] != session_id:
        raise bad_token("that page token belongs to another session")
    if body["q"] != query:
        raise bad_token(
            "that page token belongs to a read of other paths or filters; "
            "send the same arguments as the first page"
        )
    return body


def decode_token(page: Any) -> dict[str, Any]:
    unreadable = bad_token("the page token could not be read")
    if not isinstance(page, str) or len(page) > MAX_TOKEN_CHARS:
        raise unreadable
    try:
        padded = page + "=" * (-len(page) % 4)
        raw = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        body = json.loads(raw.decode("ascii"))
    except (binascii.Error, UnicodeError, ValueError, RecursionError):
        raise unreadable from None
    if not isinstance(body, dict) or set(body) != set(TOKEN_FIELDS):
        raise unreadable
    for name, kind in TOKEN_FIELDS.items():
        value = body[name]
        # A bool is an int to Python and never a count to this token.
        if isinstance(value, bool) or not isinstance(value, kind):
            raise unreadable
        if kind is str and len(value) > TOKEN_LENGTHS[name]:
            raise unreadable
    if body["v"] != TOKEN_VERSION or body["e"] < -1 or not body["k"].startswith("/"):
        raise unreadable
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
        "Read nodes, networks, parameters. Modes: tree, node, parms, find (pattern: name or "
        "path glob; type: type glob), selection. page takes next_page. Cooks only if evaluate."
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
