"""The MCP server end to end in one process, with the sessions stood in for.

The client is the SDK's own, connected in process, so the tool list, the
argument check, the result shapes and the trace are what a real client sees.
The store, the session files and the bridge are the stand ins from the router
tests, so nothing here needs a Houdini.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.shared.inbound import find_invalid_x_mcp_header

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client as bridge_client
from nscr_houdini_mcp.config import Config, ConfigError
from nscr_houdini_mcp.results import CallError
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
    [tool] = [tool for tool in listed.tools if tool.name == "hou_ping"]
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.idempotent_hint is True
    assert tool.annotations.open_world_hint is False
    assert tool.annotations.destructive_hint is None
    assert tool.input_schema["additionalProperties"] is False
    assert set(tool.input_schema["properties"]) == {"session", "wait_s"}
    assert find_invalid_x_mcp_header(tool.input_schema) is None
    assert tool.output_schema is not None
    assert tool.output_schema["required"] == ["trace"]


def test_the_instructions_are_four_short_lines() -> None:
    lines = INSTRUCTIONS.splitlines()
    assert len(lines) == 4
    assert len(INSTRUCTIONS.split()) < 120
    assert "session" in INSTRUCTIONS and "wait_s" in INSTRUCTIONS
    # Nothing is promised that no tool offers yet.
    assert "job" not in INSTRUCTIONS and "operation_id" not in INSTRUCTIONS


def test_only_tools_are_advertised() -> None:
    async def capabilities() -> Any:
        async with Client(serve(Stage([])), mode="legacy") as connected:
            return connected.session.server_capabilities

    advertised = asyncio.run(capabilities())
    assert advertised.tools is not None
    assert advertised.prompts is None
    assert advertised.resources is None


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


def test_hou_ping_shows_the_progress_the_running_call_reported() -> None:
    busy = {
        "ok": False,
        "error": {"code": "SESSION_BUSY", "message": "this session is running another call"},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 4,
    }
    note = {"done": 2, "total": 5, "message": "caching", "elapsed_s": 1.2}
    stage = Stage([record("s-1", "w1")], replies=(busy,))
    said = {**HEALTH["data"], "busy": True, "current_op": "python.run"}
    stage.health = lambda session, **rest: bridge_client.Answer(  # type: ignore[method-assign]
        200, {"ok": True, "data": {**said, "current_op_progress": [note]}}, {}
    )
    _, [result] = talk(serve(stage), ("hou_ping", {"wait_s": 0}))
    health = result.structured_content["health"]
    assert health["current_op"] == "python.run"
    assert health["progress"] == [note]


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


def test_a_caller_holding_a_freed_up_name_never_reaches_a_second_houdini(
    tmp_path: Path,
) -> None:
    """A name read before a rename goes on reaching the session that had it.

    The caller reads `untitled-1`, the scene loads and the session becomes
    `shot_010-1`, and a second, empty Houdini starts. The second one is not
    handed `untitled-1`, and a call by that name reaches the first, with a
    warning that names what it answers to now.
    """
    with store_module.Store(tmp_path / "coord.sqlite") as store:
        first = store.register_session(
            "s-1", kind="gui", pid=os.getpid(), alias_template="untitled-{n}"
        )
        assert first.alias == "untitled-1"
        store.rename_session("s-1", alias_template="shot_010-{n}", hip_path="/s/shot_010.hip")
        second = store.register_session(
            "s-2", kind="gui", pid=os.getpid(), alias_template="untitled-{n}"
        )
        assert second.alias == "untitled-2"
        rows = store.list_sessions(include_gone=True)
    reply = {**pong(), "alias": "shot_010-1"}
    stage = Stage(rows, replies=(reply,))

    _, [result] = talk(serve(stage), ("hou_ping", {"session": "untitled-1"}))

    assert not result.is_error, text_of(result)
    assert stage.sent.calls[0]["session"] == "s-1"
    trace = result.structured_content["trace"]
    assert trace["alias"] == "shot_010-1"
    assert [warning["code"] for warning in trace["warnings"]] == ["ALIAS_RENAMED"]
    assert "shot_010-1" in trace["warnings"][0]["message"]


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
    assert result.structured_content["error"]["details"]["did_you_mean"][0] == "hou_ping"


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


def test_a_bad_config_names_its_file_without_the_whole_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NSCR_MCP_HOME", str(home))
    (home / "config.toml").write_text("poolcap = 2\n", encoding="utf-8")
    server = build_server(router_factory=Stage([]).router)
    _, [result] = talk(server, ("hou_ping", {}))
    assert result.is_error is True
    details = result.structured_content["error"]["details"]
    assert details == {"path": "config.toml", "key": "poolcap"}
    assert str(tmp_path) not in text_of(result)


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
    assert second["operation_id"] == f"{first['operation_id']}:2"
    for sent in (first, second):
        assert sent["scene_epoch"] == 4
        assert sent["wait_s"] == 5
        assert sent["timeout_s"] == 60
    # The trace names the id the caller can send again, not a derived one.
    assert result.structured_content["trace"]["operation_id"] == first["operation_id"]


def test_a_caller_operation_id_is_used_as_given() -> None:
    stage = Stage([record("s-1", "w1")], replies=(made(0, "op-mine"), made(0, "op-mine.2")))
    _, [result] = talk(serve(stage, tools=(EDIT,)), ("test_edit", {"operation_id": "op-mine"}))
    assert not result.is_error
    assert [c["operation_id"] for c in stage.sent.calls] == ["op-mine", "op-mine:2"]
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


@pytest.mark.parametrize("given", ["", "op.2", "op:2", "op 1", "x" * 121])
def test_an_operation_id_outside_the_allowed_form_is_refused(given: str) -> None:
    stage = Stage([record("s-1", "w1")])
    _, [result] = talk(serve(stage, tools=(EDIT,)), ("test_edit", {"operation_id": given}))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "BAD_ARGUMENTS"
    assert stage.sent.calls == []


def test_a_session_given_as_an_empty_string_is_refused() -> None:
    _, [result] = talk(serve(Stage([record("s-1", "w1")])), ("hou_ping", {"session": ""}))
    assert result.structured_content["error"]["code"] == "BAD_ARGUMENTS"


def test_a_lost_reply_on_the_second_change_names_the_base_id_to_resend() -> None:
    lost = bridge_client.BridgeUnreachable("the reply was lost")
    stage = Stage(
        [record("s-1", "w1")],
        replies=(made(0, "op-base"), lost, made(0, "op-base"), made(0, "op-base:2")),
    )
    server = serve(stage, tools=(EDIT,))
    _, [first, again] = talk(
        server,
        ("test_edit", {"operation_id": "op-base"}),
        ("test_edit", {"operation_id": "op-base"}),
    )
    assert first.is_error is True
    body = first.structured_content
    assert body["error"]["code"] == "SESSION_UNREACHABLE"
    assert body["error"]["details"]["operation_id"] == "op-base"
    assert body["trace"]["operation_id"] == "op-base"
    assert "op-base:2" not in text_of(first)
    # Sending the base id again replays both steps under the same ids.
    assert not again.is_error
    sent = [call["operation_id"] for call in stage.sent.calls]
    assert sent == ["op-base", "op-base:2", "op-base", "op-base:2"]
    assert again.structured_content["trace"]["operation_id"] == "op-base"


SLOW_SERVER = """
import time
from nscr_houdini_mcp.server import build_server
from nscr_houdini_mcp.tools.base import ToolSpec, inputs

def slow(call):
    time.sleep(60)
    return {}

spec = ToolSpec(name="test_slow", description="test only", input_schema=inputs({}), handler=slow)
build_server((spec,)).run(transport="stdio")
"""


def test_the_server_exits_at_once_when_stdin_closes_during_a_long_call(tmp_path: Path) -> None:
    env = {**os.environ, "NSCR_MCP_HOME": str(tmp_path / "home")}
    child = subprocess.Popen(
        [sys.executable, "-c", SLOW_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    assert child.stdin is not None and child.stdout is not None

    def send(message: dict) -> None:
        child.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        child.stdin.flush()

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            }
        )
        assert json.loads(child.stdout.readline())["id"] == 1
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "test_slow", "arguments": {}},
            }
        )
        time.sleep(0.5)
        closed = time.monotonic()
        child.stdin.close()
        child.wait(timeout=10)
        assert time.monotonic() - closed < 2.0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_a_flood_refusal_tells_a_change_to_keep_its_id_and_a_read_nothing_more() -> None:
    flood = {
        "ok": False,
        "error": {
            "code": "FLOOD_GUARD",
            "message": "more than 20000 signed requests arrived inside 120 seconds",
            "hint": "wait 30 seconds, then call again",
        },
    }
    stage = Stage([record("s-1", "w1")], replies=(flood, flood))
    router = stage.router(Config(path=Path("config.toml")))
    spec = next(tool for tool in TOOLS if tool.name == "hou_python")
    change = Call(spec, {"code": "x = 1", "operation_id": "op-1"}, router)
    with pytest.raises(CallError) as refused:
        change.bridge("python.run", mutating=True)
    assert refused.value.hint.endswith("send the same operation_id, since the change was not made")
    read = Call(spec, {"code": "x = 1"}, router)
    with pytest.raises(CallError) as refused:
        read.bridge("python.run")
    assert refused.value.hint == "wait 30 seconds, then call again"
