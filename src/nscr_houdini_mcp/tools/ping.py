"""`hou_ping`: which session a call reaches, and whether it answers.

It asks the session's health endpoint, which answers from memory even in the
middle of a cook. When health says the session is busy, running another call
or with a main thread the bridge itself calls away, the ping answers
from health at once: busy, since when, and what the session is doing. Only a
session that looks free is also sent one signed call through the same path
every other tool uses, with a short wait for its turn, so a cook that health
has not seen yet costs the ping a fraction of a second rather than a second.
A caller that names a `wait_s` above zero is sent the call whatever health
says, and it waits that long, as with any other tool.

It reads no scene and changes nothing. Pinging a worker renews its idle lease
like any other call: a worker that is being read from is in use, and keeping
it warm is the point.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from nscr_houdini_mcp.bridge import marshal
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import SESSION, WAIT_S, Call, ToolSpec, inputs, outputs

# The bridge's name for the address it listens on. Every bridge serves plain
# HTTP on loopback with signed requests and replies.
BRIDGE_TRANSPORT = "loopback http, signed"

# How long the signed call waits for its turn when the caller named no wait.
# It covers an idle session's pickup with a sleeping display, and a paced turn.
PING_WAIT_S = 0.5


def ping(call: Call) -> Mapping[str, Any]:
    target = call.target()
    health = call.health()
    busy = busy_with(health, now=time.time())
    wait_s = call.arguments.get("wait_s")
    if busy is not None and not wait_s:
        answered: dict[str, Any] = {
            "ok": False,
            "code": "SESSION_BUSY",
            "message": f"not sent, since health says the session is busy ({busy['busy_cause']})",
            "skipped": True,
        }
    else:
        answered = ask(call, PING_WAIT_S if wait_s is None else wait_s)
    said: dict[str, Any] = {
        "status": health.get("status"),
        "busy": health.get("busy"),
        "current_op": health.get("current_op"),
        "progress": health.get("current_op_progress") or None,
        "queued": health.get("queued"),
        "heartbeat_age_s": health.get("heartbeat_age_s"),
        "round_trip_ms": health.get("round_trip_ms"),
    }
    if busy is not None:
        said["busy"] = True
        said.update(busy)
    return {
        "session_id": call.trace["session_id"],
        "alias": call.trace["alias"],
        "kind": health.get("kind") or target.record.kind,
        "build": target.houdini_version,
        "transport": {
            "server": call.transport,
            "bridge": BRIDGE_TRANSPORT,
            "port": target.session.port,
            "port_answers_itself": health.get("last_self_check_ok"),
        },
        "health": said,
        "call": answered,
        "scene_epoch": call.trace["scene_epoch"],
    }


def ask(call: Call, wait_s: float) -> dict[str, Any]:
    """The signed call through the main thread path, and how it went."""
    started = time.monotonic()
    try:
        call.bridge("bridge.ping", wait_s=wait_s)
    except CallError as error:
        if error.code != "SESSION_BUSY":
            raise
        return {"ok": False, "code": error.code, "message": error.message}
    return {"ok": True, "round_trip_ms": round((time.monotonic() - started) * 1000.0, 3)}


def busy_with(health: Mapping[str, Any], *, now: float) -> dict[str, Any] | None:
    """What keeps the session busy, as health tells it, and since when.

    A call it is running comes first, since health names it and its age.
    Otherwise a main thread the bridge calls away, past its own stale limit:
    a cook, a render or a modal dialog in a session with a user interface.
    `busy_since` is on this machine's wall clock, in seconds. Nothing when health says neither.
    """
    if health.get("busy"):
        found: dict[str, Any] = {
            "busy_cause": f"running {health.get('current_op') or 'another call'}",
        }
        elapsed = _seconds(health.get("current_op_elapsed_s"))
        if elapsed is not None:
            found["busy_for_s"] = round(elapsed, 3)
            found["busy_since"] = round(now - elapsed, 3)
        return found
    thread = health.get("main_thread")
    if not isinstance(thread, Mapping) or not thread.get("installed"):
        return None
    away = _seconds(thread.get("pulse_age_s"))
    if away is None or not main_thread_away(thread, away):
        return None
    return {
        "busy_cause": "main thread busy",
        "busy_for_s": round(away, 3),
        "busy_since": round(now - away, 3),
    }


def main_thread_away(thread: Mapping[str, Any], age_s: float) -> bool:
    """The bridge's own verdict on its main thread, under its own stale limit.

    A pulse that is old but inside that limit is not a busy session: during
    playback the loop callback stops while posted work still lands, so such
    a session is asked, with the ping's short wait. A bridge that gives no
    verdict is judged by the default limit.
    """
    verdict = thread.get("away")
    if isinstance(verdict, bool):
        return verdict
    return age_s > marshal.DEFAULT_STALE_S


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


HOU_PING = ToolSpec(
    name="hou_ping",
    description=(
        "Check which Houdini session a call reaches and that it answers. Returns its "
        "session_id, alias, kind (gui or hython), Houdini build, transport, health and "
        "scene_epoch. Reads no scene. Any call to a worker keeps it warm."
    ),
    input_schema=inputs({"session": SESSION, "wait_s": WAIT_S}),
    output_schema=outputs(
        {
            "session_id": {"type": "string"},
            "alias": {"type": "string"},
            "kind": {"type": "string"},
            "build": {"type": ["string", "null"]},
            "transport": {"type": "object"},
            "health": {"type": "object"},
            "call": {"type": "object"},
            "scene_epoch": {"type": "integer"},
        }
    ),
    handler=ping,
    read_only=True,
    idempotent=True,
    open_world=False,
)
