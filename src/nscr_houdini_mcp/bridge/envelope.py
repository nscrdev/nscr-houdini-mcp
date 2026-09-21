"""Request and reply shapes for the bridge endpoints.

Nothing here imports `hou` or the web server. A request arrives as plain data,
is checked against the rules below, and leaves as a `Reply`: an HTTP status and
a payload the transport only has to serialise.

Two payload shapes and nothing else. A call that ran has `ok` true and its
`data`. A call that did not has `ok` false and one error object with a code, a
message and optional details. The full code table and the argument rewriting
belong to the dispatch layer; this module carries the shape and the handful of
codes the envelope itself can produce.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# The header the server's own client sends. The envelope may carry the token
# instead, for a caller that cannot set headers.
TOKEN_HEADER = "x-nscr-mcp-token"

# Headers no non-browser client sends. Their presence means a web page is
# calling, and a web page has no business here.
BROWSER_HEADERS = ("origin", "referer")

ENVELOPE_FIELDS = frozenset(
    {"token", "session_id", "scene_epoch", "operation_id", "tool", "arguments"}
)

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
    token: str | None = None
    session_id: str | None = None
    scene_epoch: int | None = None
    operation_id: str | None = None


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

    return Envelope(
        tool=tool,
        arguments=dict(arguments),
        token=_text(payload.get("token"), "token"),
        session_id=_text(payload.get("session_id"), "session_id"),
        scene_epoch=scene_epoch,
        operation_id=_text(payload.get("operation_id"), "operation_id"),
    )


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
