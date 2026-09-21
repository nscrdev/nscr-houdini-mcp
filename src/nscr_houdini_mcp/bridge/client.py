"""The smallest client that can talk to a bridge.

The web server takes one POST to `/api` with a form field holding
`[name, args, kwargs]`. This wraps that, adds the token header, and hands back
the status with the decoded body. It sends no `Origin` and no `Referer`, which
is what lets the bridge refuse anything that does.

Standard library only: the same module is imported inside Houdini.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any, NamedTuple

from nscr_houdini_mcp.bridge.envelope import TOKEN_HEADER
from nscr_houdini_mcp.bridge.net import LOOPBACK

DEFAULT_TIMEOUT_S = 10.0


class BridgeUnreachable(Exception):
    """Nothing answered on that port."""


class Answer(NamedTuple):
    """One answer: the status, the decoded body and the response headers."""

    status: int
    payload: Any
    headers: dict[str, str]


def post(
    port: int,
    function: str,
    *,
    arguments: Mapping[str, Any] | None = None,
    token: str | None = None,
    headers: Mapping[str, str] | None = None,
    address: str = LOOPBACK,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> Answer:
    """Call one registered function."""
    body = urllib.parse.urlencode(
        {"json": json.dumps([function, [], dict(arguments or {})])}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://{address}:{port}/api",
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if token is not None:
        request.add_header(TOKEN_HEADER, token)
    for name, value in (headers or {}).items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as answer:  # noqa: S310
            return Answer(answer.status, _decode(answer.read()), _headers(answer))
    except urllib.error.HTTPError as error:
        return Answer(error.code, _decode(error.read()), _headers(error))
    except (urllib.error.URLError, OSError) as error:
        raise BridgeUnreachable(f"{address}:{port} did not answer: {error}") from error


def health(port: int, *, token: str, **rest: Any) -> Answer:
    """Ask a bridge whether it is alive."""
    return post(port, "mcp.health", token=token, **rest)


def call(
    port: int,
    tool: str,
    *,
    token: str,
    arguments: Mapping[str, Any] | None = None,
    session_id: str | None = None,
    scene_epoch: int | None = None,
    operation_id: str | None = None,
    **rest: Any,
) -> Answer:
    """Send one request envelope."""
    envelope: dict[str, Any] = {"tool": tool, "arguments": dict(arguments or {})}
    if session_id is not None:
        envelope["session_id"] = session_id
    if scene_epoch is not None:
        envelope["scene_epoch"] = scene_epoch
    if operation_id is not None:
        envelope["operation_id"] = operation_id
    return post(port, "mcp.call", arguments={"envelope": envelope}, token=token, **rest)


def _headers(answer: Any) -> dict[str, str]:
    """Response headers, names lowercased so a check cannot miss one."""
    return {str(name).lower(): str(value) for name, value in answer.headers.items()}


def _decode(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text
