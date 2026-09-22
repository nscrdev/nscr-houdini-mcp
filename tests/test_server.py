"""The MCP server end to end in one process, with the sessions stood in for.

The client is the SDK's own, connected in process, so the tool list, the
argument check, the result shapes and the trace are what a real client sees.
The store, the session files and the bridge are the stand ins from the router
tests, so nothing here needs a Houdini.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mcp.client.client import Client
from mcp.shared.inbound import find_invalid_x_mcp_header

from nscr_houdini_mcp.bridge import client as bridge_client
from nscr_houdini_mcp.config import Config, ConfigError
from nscr_houdini_mcp.router import Router
from nscr_houdini_mcp.server import INSTRUCTIONS, SERVER_NAME, build_server
from nscr_houdini_mcp.tools.base import (
    OPERATION_ID,
    SCENE_EPOCH,
    SESSION,
    TIMEOUT_S,
    WAIT_S,
    Call,
    ToolSpec,
    inputs,
    outputs,
)
from nscr_houdini_mcp.tools.registry import TOOLS
from test_router import FakeFiles, FakeStore, Sent, record

HEALTH = {
    "ok": True,
    "data": {
        "status": "ok",
        "session_id": "s-1",
        "alias": "w1",
        "kind": "hython",
        "scene_epoch": 4,
        "busy": False,
        "current_op": None,
        "queued": 0,
        "heartbeat_age_s": 1.5,
        "last_self_check_ok": True,
    },
}


def pong(epoch: int = 4) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {"pong": True, "echo": None},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": epoch,
        "operation_id": "op-bridge",
    }


class Stage:
    """Stand ins for everything past the server: store, files, bridge."""

    def __init__(self, rows: list, *, replies: tuple = (), home: Path | None = None) -> None:
        self.store = FakeStore(rows)
        self.files = FakeFiles([row.session_id for row in rows])
        self.sent = Sent(*replies)
        self.health_asked = 0
        self.home = home or Path(".")

    def health(self, session: bridge_client.Session, **rest: Any) -> bridge_client.Answer:
        self.health_asked += 1
        return bridge_client.Answer(200, HEALTH, {})

    def router(self, config: Config) -> Router:
        return Router(
            self.home,
            default_session=config.default_session,
            open_store=lambda path: self.store,
            open_session=self.files.open,
            send=self.sent,
            ask_health=self.health,
            renew_lease=lambda store, session_id: None,
        )


def serve(
    stage: Stage, *, tools: tuple = TOOLS, config: Config | None = None, loader: Any = None
) -> Any:
    settings = config or Config(path=Path("config.toml"))
    return build_server(
        tools, config_loader=loader or (lambda: settings), router_factory=stage.router
    )


async def _talk(server: Any, calls: list[tuple[str, dict]]) -> tuple[Any, list[Any]]:
    async with Client(server) as connected:
        listed = await connected.list_tools()
        results = [await connected.call_tool(name, args) for name, args in calls]
    return listed, results


def talk(server: Any, *calls: tuple[str, dict]) -> tuple[Any, list[Any]]:
    return asyncio.run(_talk(server, list(calls)))


def text_of(result: Any) -> str:
    [block] = result.content
    return block.text


# Section: the tool list


def test_the_tool_list_is_fixed_and_the_same_every_time() -> None:
    server = build_server()
    assert server.name == SERVER_NAME
    first = asyncio.run(server.list_tools())
    second = asyncio.run(build_server().list_tools())
    assert [tool.name for tool in first] == [spec.name for spec in TOOLS]
    assert [t.model_dump() for t in first] == [t.model_dump() for t in second]


def test_hou_ping_is_listed_read_only_with_plain_schemas() -> None:
    listed, _ = talk(serve(Stage([])))
    [tool] = listed.tools
    assert tool.name == "hou_ping"
    assert tool.annotations is not None and tool.annotations.read_only_hint is True
    assert tool.input_schema["additionalProperties"] is False
    assert set(tool.input_schema["properties"]) == {"session", "wait_s"}
    assert find_invalid_x_mcp_header(tool.input_schema) is None
    assert tool.output_schema is not None
    assert tool.output_schema["required"] == ["trace"]


def test_the_instructions_are_five_short_lines() -> None:
    lines = INSTRUCTIONS.splitlines()
    assert len(lines) <= 5
    assert len(INSTRUCTIONS.split()) < 120
    assert "session" in INSTRUCTIONS and "wait_s" in INSTRUCTIONS and "job" in INSTRUCTIONS


def test_the_server_never_imports_hou() -> None:
    code = (
        "import sys, nscr_houdini_mcp.cli, nscr_houdini_mcp.server;sys.exit('hou' in sys.modules)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


# Section: hou_ping through the server


def test_hou_ping_reaches_the_only_session_and_echoes_the_trace() -> None:
    stage = Stage([record("s-1", "w1", scene_epoch=4)], replies=(pong(),))
    _, [result] = talk(serve(stage), ("hou_ping", {}))
    assert not result.is_error
    body = result.structured_content
    assert body["session_id"] == "s-1"
    assert body["alias"] == "w1"
    assert body["kind"] == "hython"
    assert body["scene_epoch"] == 4
    assert body["health"]["status"] == "ok"
    assert body["call"]["ok"] is True
    assert body["transport"]["server"] == "stdio"
    assert body["transport"]["port_answers_itself"] is True
    assert body["trace"] == {"session_id": "s-1", "alias": "w1", "scene_epoch": 4}
    assert json.loads(text_of(result)) == body
    [sent] = stage.sent.calls
    assert sent["tool"] == "bridge.ping"
    # A read carries no operation id, so a lost reply is not sent twice.
    assert sent["operation_id"] is None


def test_hou_ping_of_a_busy_session_reports_busy_rather_than_failing() -> None:
    busy = {
        "ok": False,
        "error": {"code": "SESSION_BUSY", "message": "this session is running another call"},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 4,
    }
    stage = Stage([record("s-1", "w1")], replies=(busy,))
    _, [result] = talk(serve(stage), ("hou_ping", {"wait_s": 0}))
    assert not result.is_error
    assert result.structured_content["call"]["code"] == "SESSION_BUSY"
    assert stage.sent.calls[0]["wait_s"] == 0


def test_an_ambiguous_ping_is_an_error_the_text_alone_can_fix() -> None:
    stage = Stage([record("s-1", "w1"), record("s-2", "acc-1", kind="gui")])
    _, [result] = talk(serve(stage), ("hou_ping", {}))
    assert result.is_error is True
    text = text_of(result)
    assert text.startswith("SESSION_AMBIGUOUS: ")
    assert "hint: pass session as one of the candidates" in text
    assert '"alias":"acc-1"' in text and '"alias":"w1"' in text
    assert result.structured_content["trace"]["session_id"] is None


def test_the_config_default_settles_which_session_answers() -> None:
    stage = Stage([record("s-1", "w1"), record("s-2", "w2")], replies=(pong(),))
    config = Config(path=Path("config.toml"), default_session="w2")
    _, [result] = talk(serve(stage, config=config), ("hou_ping", {}))
    assert not result.is_error
    assert stage.sent.calls[0]["session"] == "s-2"


def test_a_dead_session_names_its_successor() -> None:
    rows = [record("s-old", "w1", state="gone"), record("s-new", "w1", started_at=2.0)]
    _, [result] = talk(serve(Stage(rows)), ("hou_ping", {"session": "s-old"}))
    assert result.is_error is True
    assert result.structured_content["error"]["details"]["live_session_id"] == "s-new"
    assert "s-new" in text_of(result)


def test_a_misspelled_argument_names_the_real_one() -> None:
    _, [result] = talk(serve(Stage([])), ("hou_ping", {"sesion": "w1"}))
    assert result.is_error is True
    error = result.structured_content["error"]
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["did_you_mean"] == ["session"]


def test_a_value_out_of_range_is_refused_before_anything_is_sent() -> None:
    stage = Stage([record("s-1", "w1")])
    _, [result] = talk(serve(stage), ("hou_ping", {"wait_s": 99}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "BAD_ARGUMENTS"
    assert "wait_s" in text_of(result)
    assert stage.sent.calls == []


def test_an_unknown_tool_offers_the_nearest_name() -> None:
    _, [result] = talk(serve(Stage([])), ("hou_pnig", {}))
    assert result.is_error is True
    assert result.structured_content["error"]["details"]["did_you_mean"] == ["hou_ping"]


def test_a_bad_config_is_reported_on_the_call_and_read_again_next_time() -> None:
    stage = Stage([record("s-1", "w1")], replies=(pong(),))
    attempts: list[int] = []

    def loader() -> Config:
        attempts.append(1)
        if len(attempts) == 1:
            raise ConfigError("unknown key 'poolcap'; did you mean pool_cap?", key="poolcap")
        return Config(path=Path("config.toml"))

    _, [first, second] = talk(serve(stage, loader=loader), ("hou_ping", {}), ("hou_ping", {}))
    assert first.is_error is True
    assert text_of(first).startswith("CONFIG_INVALID: unknown key 'poolcap'")
    assert not second.is_error


# Section: the plumbing a changing tool gets


def edit_twice(call: Call) -> Mapping[str, Any]:
    call.bridge("node.create", {"parent": "/obj", "type": "geo"}, mutating=True)
    call.bridge("node.create", {"parent": "/obj", "type": "null"}, mutating=True)
    return {"made": 2}


EDIT = ToolSpec(
    name="test_edit",
    description="test only",
    input_schema=inputs(
        {
            "session": SESSION,
            "wait_s": WAIT_S,
            "timeout_s": TIMEOUT_S,
            "scene_epoch": SCENE_EPOCH,
            "operation_id": OPERATION_ID,
        }
    ),
    output_schema=outputs({"made": {"type": "integer"}}),
    handler=edit_twice,
)


def made(epoch: int, operation_id: str) -> dict[str, Any]:
    return {
        "ok": True,
        "data": {"path": "/obj/geo1"},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": epoch,
        "operation_id": operation_id,
    }


def test_a_change_mints_an_operation_id_and_passes_the_budgets() -> None:
    stage = Stage([record("s-1", "w1")], replies=(made(4, "x"), made(4, "y")))
    arguments = {"scene_epoch": 4, "wait_s": 5, "timeout_s": 60}
    _, [result] = talk(serve(stage, tools=(EDIT,)), ("test_edit", arguments))
    assert not result.is_error, text_of(result)
    first, second = stage.sent.calls
    assert first["operation_id"].startswith("op-")
    assert second["operation_id"] == f"{first['operation_id']}.2"
    for sent in (first, second):
        assert sent["scene_epoch"] == 4
        assert sent["wait_s"] == 5
        assert sent["timeout_s"] == 60
    assert result.structured_content["trace"]["operation_id"] == "y"


def test_a_caller_operation_id_is_used_as_given() -> None:
    stage = Stage([record("s-1", "w1")], replies=(made(0, "op-mine"), made(0, "op-mine.2")))
    _, [result] = talk(serve(stage, tools=(EDIT,)), ("test_edit", {"operation_id": "op-mine"}))
    assert not result.is_error
    assert [c["operation_id"] for c in stage.sent.calls] == ["op-mine", "op-mine.2"]
    # No epoch was given, so none is sent: the guard is the caller's choice.
    assert [c["scene_epoch"] for c in stage.sent.calls] == [None, None]


def test_a_replaced_scene_comes_back_with_the_new_epoch_and_scene() -> None:
    refused = {
        "ok": False,
        "error": {"code": "SCENE_REPLACED", "message": "the scene changed under the call"},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 5,
        "operation_id": "op-given",
        "scene": {"hip_name": "other.hip"},
    }
    stage = Stage([record("s-1", "w1", scene_epoch=4)], replies=(refused,))
    arguments = {"scene_epoch": 4, "operation_id": "op-given"}
    _, [result] = talk(serve(stage, tools=(EDIT,)), ("test_edit", arguments))
    assert result.is_error is True
    body = result.structured_content
    assert body["error"]["code"] == "SCENE_REPLACED"
    assert body["error"]["details"]["scene"] == {"hip_name": "other.hip"}
    assert body["trace"]["scene_epoch"] == 5
    assert body["trace"]["operation_id"] == "op-given"
    assert "scene_epoch" in text_of(result)


def test_a_large_result_spills_under_the_state_home(tmp_path: Path) -> None:
    home = tmp_path / "home"

    def big(call: Call) -> Mapping[str, Any]:
        call.target()
        return {"rows": [f"/obj/node{i}" for i in range(500)]}

    spec = ToolSpec(
        name="test_big",
        description="test only",
        input_schema=inputs({"session": SESSION}),
        output_schema=outputs({"rows": {"type": "array"}}),
        handler=big,
        read_only=True,
    )
    config = Config(path=home / "config.toml", state_home=home, spill_over_bytes=2048)
    stage = Stage([record("s-1", "w1")], home=home)
    _, [result] = talk(serve(stage, tools=(spec,), config=config), ("test_big", {}))
    assert not result.is_error
    spilled = result.structured_content["spilled"]
    path = Path(spilled["path"])
    assert path.is_relative_to(home / "spill")
    assert spilled["bytes"] > 2048
    assert json.loads(path.read_text(encoding="utf-8"))["rows"][499] == "/obj/node499"
    assert str(path) in text_of(result)
