"""Request and reply shapes for the bridge endpoints.

Nothing here imports `hou` or the web server. A request arrives as plain data,
is checked against the rules below, and leaves as a `Reply`: an HTTP status and
a payload the transport only has to serialise.

The envelope never carries a secret. Proof of who is calling lives in the
signing headers, so there is nothing here worth catching.

Two payload shapes and nothing else. A call that ran has `ok` true and its
`data`. A call that did not has `ok` false and one error object with a code, a
message and optional details. The full code table and the argument rewriting
belong to the dispatch layer; this module carries the shape and the handful of
codes the envelope itself can produce.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# Headers no non-browser client sends. Their presence means a web page is
# calling, and a web page has no business here.
BROWSER_HEADERS = ("origin", "referer")

ENVELOPE_FIELDS = frozenset(
    {
        "session_id",
        "scene_epoch",
        "operation_id",
        "tool",
        "arguments",
        "wait_s",
        "timeout_s",
        "skip_if_busy",
    }
)

# How long a call may wait for the session to be free. Zero means answer now.
MAX_WAIT_S = 50.0

# How long a call may wait for work that is already running. It is a separate
# budget from `wait_s`: waiting for a turn and waiting for an answer are
# different things, and a caller may want a short first and a long second.
MAX_TIMEOUT_S = 3600.0

MAX_TOOL_NAME = 128

# `mcp.call`, `bridge.ping`: lower case words joined by dots.
TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")


class EnvelopeError(Exception):
    """The request envelope could not be read, so nothing was dispatched."""

    def __init__(self, message: str, *, code: str = "BAD_ENVELOPE", **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


@dataclass(frozen=True)
class Envelope:
    """One checked request.

    `scene_epoch` and `operation_id` are carried and echoed back but not acted
    on here: the scene guard and the receipt table are the dispatch layer's
    work. Accepting them from the first build means a caller never has to
    change its request shape later.
    """

    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    scene_epoch: int | None = None
    operation_id: str | None = None
    wait_s: float | None = None
    timeout_s: float | None = None
    skip_if_busy: bool = False


@dataclass(frozen=True)
class Reply:
    """What a handler hands back to the transport."""

    status: int
    payload: Mapping[str, Any]


def parse_envelope(payload: Any) -> Envelope:
    """Read one request envelope, or raise `EnvelopeError`.

    Unknown fields are refused rather than ignored, so a misspelled field name
    is a visible failure instead of a silently dropped argument.
    """
    if not isinstance(payload, Mapping):
        raise EnvelopeError("the envelope must be an object")
    unknown = sorted(set(payload) - ENVELOPE_FIELDS)
    if unknown:
        raise EnvelopeError(
            f"unknown envelope field: {unknown[0]}",
            unknown=unknown,
            known=sorted(ENVELOPE_FIELDS),
        )

    tool = payload.get("tool")
    if not isinstance(tool, str) or not tool:
        raise EnvelopeError("tool must be a non empty string")
    if len(tool) > MAX_TOOL_NAME or not TOOL_NAME.match(tool):
        raise EnvelopeError(f"tool is not a tool name: {tool[:MAX_TOOL_NAME]!r}")

    arguments = payload.get("arguments", {})
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise EnvelopeError("arguments must be an object")
    bad_key = next((key for key in arguments if not isinstance(key, str)), None)
    if bad_key is not None:
        raise EnvelopeError(f"argument names must be strings, got {type(bad_key).__name__}")

    scene_epoch = payload.get("scene_epoch")
    if scene_epoch is not None:
        if isinstance(scene_epoch, bool) or not isinstance(scene_epoch, int):
            raise EnvelopeError("scene_epoch must be an integer")

    skip = payload.get("skip_if_busy", False)
    if not isinstance(skip, bool):
        raise EnvelopeError("skip_if_busy must be true or false")

    return Envelope(
        tool=tool,
        arguments=dict(arguments),
        session_id=_text(payload.get("session_id"), "session_id"),
        scene_epoch=scene_epoch,
        operation_id=_text(payload.get("operation_id"), "operation_id"),
        wait_s=_seconds(payload.get("wait_s"), "wait_s", MAX_WAIT_S),
        timeout_s=_seconds(payload.get("timeout_s"), "timeout_s", MAX_TIMEOUT_S),
        skip_if_busy=skip,
    )


def _seconds(value: Any, name: str, limit: float) -> float | None:
    """One time budget, checked against the range it is allowed."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnvelopeError(f"{name} must be a number")
    if value != value or value < 0 or value > limit:
        raise EnvelopeError(f"{name} must be between 0 and {limit:g}")
    return float(value)


def _text(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise EnvelopeError(f"{name} must be a string")
    return value


def ok_payload(data: Any, *, timing_ms: float | None = None) -> dict[str, Any]:
    """Payload for a call that ran."""
    payload: dict[str, Any] = {"ok": True, "data": data}
    if timing_ms is not None:
        payload["timing_ms"] = round(timing_ms, 3)
    return payload


def error_payload(
    code: str,
    message: str,
    *,
    hint: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Payload for a call that did not run, or ran and failed."""
    error: dict[str, Any] = {"code": code, "message": message}
    if hint:
        error["hint"] = hint
    if details:
        error["details"] = dict(details)
    return {"ok": False, "error": error}


# How deeply a request body may nest. An envelope needs a handful of levels;
# thousands is not a request, and a parser handed thousands can take the whole
# process down with it before it ever returns.
MAX_DEPTH = 32


def load_json(raw: bytes, *, max_depth: int = MAX_DEPTH) -> Any:
    """Decode a request body, refusing one that nests too deeply.

    The depth is counted before the parser sees the text, because the parser
    follows the nesting as it goes and a body nested thousands deep can end
    the process rather than raise. Brackets inside strings are skipped, so a
    piece of code sent as an argument counts for nothing.
    """
    if not isinstance(raw, (bytes, bytearray)):
        raise EnvelopeError("the body must be bytes")
    depth = json_depth(raw)
    if depth > max_depth:
        raise EnvelopeError(
            f"the body nests more than {max_depth} deep",
            code="BODY_REFUSED",
            depth=depth,
            max_depth=max_depth,
        )
    try:
        text = bytes(raw).decode("utf-8")
    except UnicodeDecodeError:
        raise EnvelopeError("the body is not utf-8") from None
    try:
        return json.loads(text)
    except ValueError as error:
        raise EnvelopeError(f"the body is not JSON: {error}") from None


def json_depth(raw: bytes) -> int:
    """How deeply a JSON text nests, counting only brackets outside strings."""
    depth = 0
    deepest = 0
    in_string = False
    escaped = False
    for byte in bytes(raw):
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x5B, 0x7B):  # [ {
            depth += 1
            deepest = max(deepest, depth)
        elif byte in (0x5D, 0x7D):  # ] }
            depth -= 1
    return deepest
