"""`hou_ping`: which session a call reaches, and whether it answers.

It asks the session's health endpoint, which answers from memory even in the
middle of a cook, then sends one signed call through the same path every other
tool uses. A session busy with another call still pings: the health part says
what it is doing and the call part says it was busy, rather than the whole
ping failing.

It reads no scene and changes nothing. Pinging a worker renews its idle lease
like any other call: a worker that is being read from is in use, and keeping
it warm is the point.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import SESSION, WAIT_S, Call, ToolSpec, inputs, outputs

# The bridge's name for the address it listens on. Every bridge serves plain
# HTTP on loopback with signed requests and replies.
BRIDGE_TRANSPORT = "loopback http, signed"


def ping(call: Call) -> Mapping[str, Any]:
    target = call.target()
    health = call.health()
    started = time.monotonic()
    try:
        call.bridge("bridge.ping")
        answered: dict[str, Any] = {
            "ok": True,
            "round_trip_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
    except CallError as error:
        if error.code != "SESSION_BUSY":
            raise
        answered = {"ok": False, "code": error.code, "message": error.message}
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
        "health": {
            "status": health.get("status"),
            "busy": health.get("busy"),
            "current_op": health.get("current_op"),
            "queued": health.get("queued"),
            "heartbeat_age_s": health.get("heartbeat_age_s"),
            "round_trip_ms": health.get("round_trip_ms"),
        },
        "call": answered,
        "scene_epoch": call.trace["scene_epoch"],
    }


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
