"""`hou_node_type`: what a node type takes, before a node of it is made.

Two ways in, both read only, answered from the running Houdini so installed
assets and this build's own versions are the ones described.

- `type` names one type in a `context`: its inputs and outputs, its real
  parameter names with their defaults, and, at `full`, its menus, ranges,
  folders and help line. A bare name is the version Houdini makes for it, and
  `resolved_from` says so. A multiparm carries its instance template nested
  under `instances`, and a menu a script fills in says `dynamic` rather than
  running the script.
- `query` searches the types by keyword over name, label and help line,
  closest first: the exact name, then a name that starts with it, then one
  that holds it, then the label, then the help. Hidden types are left out
  unless `include` has `hidden`. A query is at most 200 characters and 16
  different words.

`summary` is who the type is, its input and output counts and how many
parameters it has. `standard` adds the inputs, the outputs and the visible
parameters with their defaults; `full` adds hidden parameters, menus, ranges,
folders and help. `include` brings `help` or `hidden` into a lower level.

Parameters, and search rows, page the way `hou_inspect` does: past `limit`
there is `next_page`, an opaque token tied to the session and to the
arguments that decide the rows, refused with `BAD_CURSOR` anywhere else. When
the rows changed between two pages, because an asset was installed or
changed, the next page still comes back and says `changed`.

Nothing is cooked or made, and an asset's definition is read only as far as
its node type exposes it.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from nscr_houdini_mcp.bridge.node_types import (
    MAX_QUERY_CHARS,
    MAX_QUERY_WORDS,
    TYPE_INCLUDES,
)
from nscr_houdini_mcp.bridge.tools import DEFAULT_LIMIT, MAX_LIMIT
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import DETAIL, SESSION, WAIT_S, Call, ToolSpec, inputs, outputs
from nscr_houdini_mcp.tools.inspect import bad_token, within

# What a page token carries, under one short key each, and its version.
TOKEN_VERSION = 1
MAX_TOKEN_CHARS = 600
TOKEN_FIELDS = {
    "v": int,  # the version
    "m": str,  # the mode: type or query
    "s": str,  # the session id
    "o": int,  # the row the next page starts at
    "d": str,  # the fingerprint of the rows when the page was made
    "q": str,  # which lookup the page belongs to
}
TOKEN_LENGTHS = {"m": 8, "s": 128, "d": 64, "q": 32}

# The arguments that decide which rows a lookup pages through.
QUERY_ARGUMENTS = ("context", "type", "query", "parm_filter", "detail", "include")

# What the bridge hands back that is only for this end.
KEPT_BACK = ("mark", "more", "next_offset")


def node_type(call: Call) -> Mapping[str, Any]:
    arguments = call.arguments
    named, query = arguments.get("type"), arguments.get("query")
    if (named is None) == (query is None):
        raise CallError(
            "BAD_ARGUMENTS",
            "pass type to read one node type, or query to search, not both",
            details={"argument": "type" if named is None else "query"},
        )
    if query is not None and arguments.get("parm_filter") is not None:
        raise CallError(
            "BAD_ARGUMENTS",
            "parm_filter is for reading a type, not for a search",
            details={"argument": "parm_filter"},
        )
    if arguments.get("parm_filter") == "non_default":
        raise CallError(
            "BAD_ARGUMENTS",
            "a type has only defaults; parm_filter takes all or a glob",
            details={"argument": "parm_filter"},
        )
    if query is not None and len(set(str(query).lower().split())) > MAX_QUERY_WORDS:
        raise CallError(
            "BAD_ARGUMENTS",
            f"query is at most {MAX_QUERY_WORDS} different words",
            details={"argument": "query"},
        )
    within("limit", arguments.get("limit"), MAX_LIMIT)
    mode = "type" if named is not None else "query"
    target = call.target()
    fingerprint = query_of(mode, arguments)
    token = read_token(
        arguments.get("page"), mode=mode, session_id=target.session_id, query=fingerprint
    )

    sent: dict[str, Any] = {}
    for name in ("context", "type", "query", "parm_filter", "detail", "include"):
        if arguments.get(name) is not None:
            sent[name] = arguments[name]
    sent["limit"] = arguments.get("limit") or DEFAULT_LIMIT
    if token is not None:
        sent["offset"] = token["o"]

    reply = call.bridge("node.type", sent)
    data = dict(reply.get("data") or {})
    result = {key: value for key, value in data.items() if key not in KEPT_BACK}
    mark = str(data.get("mark") or "")
    if data.get("more"):
        result["next_page"] = make_token(
            mode=mode,
            session_id=target.session_id,
            offset=int(data.get("next_offset") or 0),
            mark=mark,
            query=fingerprint,
        )
    if token is not None and token["d"] != mark:
        result["changed"] = True
    if reply.get("lossy"):
        result["lossy"] = True
        result["cut"] = reply.get("cut")
    return result


# Section: page tokens


def query_of(mode: str, arguments: Mapping[str, Any]) -> str:
    """A short fingerprint of the arguments that decide which rows a lookup has.

    `include` is a set, so its order does not count. A search's rows are the
    same at every level, so there `detail` does not count either.
    """
    asked = {"mode": mode, **{name: arguments.get(name) for name in QUERY_ARGUMENTS}}
    if asked["include"] is not None:
        asked["include"] = sorted(set(asked["include"]))
    if mode == "query":
        asked.pop("detail")
    text = json.dumps(asked, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_token(*, mode: str, session_id: str, offset: int, mark: str, query: str) -> str:
    body = {"v": TOKEN_VERSION, "m": mode, "s": session_id, "o": offset, "d": mark, "q": query}
    text = json.dumps(body, separators=(",", ":"), ensure_ascii=True)
    return base64.urlsafe_b64encode(text.encode("ascii")).decode("ascii").rstrip("=")


def read_token(page: Any, *, mode: str, session_id: str, query: str) -> dict[str, Any] | None:
    """The token a caller sent back, or `BAD_CURSOR` when it is not one of ours.

    A token only goes on in the mode, the session and the lookup that made
    it, and anything that is not exactly the shape this tool writes is
    refused, however it fails to be read.
    """
    if page is None:
        return None
    body = decode_token(page)
    if body["m"] != mode:
        raise bad_token(
            f"that page token belongs to a {body['m']} lookup, not {mode}", token_mode=body["m"]
        )
    if body["s"] != session_id:
        raise bad_token("that page token belongs to another session")
    if body["q"] != query:
        raise bad_token(
            "that page token belongs to another type, search or filter; "
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
    if body["v"] != TOKEN_VERSION or body["o"] < 0 or body["m"] not in ("type", "query"):
        raise unreadable
    return body


# Section: the one line a long result is summed up in


def summary_line(data: Mapping[str, Any]) -> str:
    if "rows" in data:
        shown = len(data.get("rows") or [])
        line = f"hou_node_type query {data.get('query')!r}: {shown} of {data.get('total')} types"
    else:
        parms = data.get("parms")
        count = len(parms) if isinstance(parms, list) else data.get("parm_count")
        total = data.get("total", count)
        line = (
            f"hou_node_type {data.get('category')}/{data.get('type')}: "
            f"{data.get('max_inputs')} inputs at most, {count} of {total} parameters"
        )
    if data.get("next_page"):
        line += "; more with next_page"
    return line


HOU_NODE_TYPE = ToolSpec(
    name="hou_node_type",
    description=(
        "A node type in the running Houdini: inputs, outputs, parm names, defaults, menus, "
        "help; labels_from help is approximate. type with context (sop, obj...), or query "
        "to search."
    ),
    input_schema=inputs(
        {
            "session": SESSION,
            "context": {"type": "string"},
            "type": {"type": "string"},
            "query": {"type": "string", "maxLength": MAX_QUERY_CHARS},
            "parm_filter": {"type": "string"},
            "include": {"type": "array", "items": {"enum": list(TYPE_INCLUDES)}},
            "detail": DETAIL,
            "limit": {"type": "integer"},
            "page": {"type": "string"},
            "wait_s": WAIT_S,
        }
    ),
    output_schema=outputs({}),
    handler=node_type,
    read_only=True,
    open_world=False,
    summary=summary_line,
)
