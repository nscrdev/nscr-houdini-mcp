"""`hou_python` through the server, down to a bridge running the code.

Every call goes the whole way: the server checks the arguments and fits the
answer to its budget, the router sends the call, and a real dispatcher with
real receipts runs the bridge's `python.run` against the stand in for `hou`.
The lost reply is tried over a real socket against a real bridge, with the
hook that holds an answer back. What a real Houdini does with the same calls
is in the integration tests.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import support
from fake_hou import Scene
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, tools
from nscr_houdini_mcp.bridge import receipts as receipt_module
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.identity import Identity
from nscr_houdini_mcp.tools import python as python_tool
from test_bridge_app import make_bridge
from test_server import talk, text_of
from test_tools_sessions import Bench

DAY_S = 24 * 60 * 60.0


class Clock:
    """The namespaces' clock, moved by the test."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Through:
    """Sends each call to a real dispatcher, with receipts, over the stand in."""

    def __init__(self, module: Any, home: Path, namespaces: tools.Namespaces) -> None:
        store_path = home / store_module.STORE_FILE_NAME
        self.identity = Identity(session_id="s-1", kind="hython", alias="w1", hou=module)
        self.dispatcher = Dispatcher(
            default_registry(python=namespaces),
            lock=threading.Lock(),
            kind="hython",
            session_id="s-1",
            identity=self.identity,
            receipts=receipt_module.Receipts(
                lambda: store_module.Store(store_path), session_id="s-1"
            ),
            hou=module,
            wait_s=5.0,
            timeout_s=10.0,
            home=home,
            open_store=lambda: store_module.Store(store_path),
        )
        self.calls: list[dict[str, Any]] = []

    def __call__(self, session: client.Session, tool: str, **rest: Any) -> client.Answer:
        self.calls.append({"tool": tool, **rest})
        envelope = Envelope(
            tool=tool,
            arguments=rest.get("arguments") or {},
            session_id=rest.get("session_id"),
            scene_epoch=rest.get("scene_epoch"),
            operation_id=rest.get("operation_id"),
            wait_s=rest.get("wait_s"),
            timeout_s=rest.get("timeout_s"),
        )
        return client.Answer(200, dict(self.dispatcher.dispatch(envelope).payload), {})


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def module(scene: Scene) -> Any:
    made = scene.module()
    # Something a check can hold the code on, reached the way the code
    # reaches everything else.
    made.gate = threading.Event()
    return made


@pytest.fixture
def bench(tmp_path: Path, module: Any, clock: Clock) -> Bench:
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    made.sent = Through(module, home, tools.Namespaces(clock=clock))  # type: ignore[assignment]
    return made


def through(bench: Bench) -> Through:
    return bench.sent  # type: ignore[return-value]


def python(bench: Bench, **arguments: Any) -> Any:
    _, [result] = talk(bench.serve(), ("hou_python", arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    return result.structured_content


def raised(result: Any) -> dict[str, Any]:
    """The code's own exception, carried in a whole result marked as an error."""
    assert result.is_error is True
    body = result.structured_content
    assert "code" not in body["error"], body["error"]
    return body


def refused(result: Any) -> dict[str, Any]:
    assert result.is_error is True
    return result.structured_content["error"]


def sent(bench: Bench) -> list[dict[str, Any]]:
    return [call for call in through(bench).calls if call["tool"] == "python.run"]


# Section: what comes back


def test_result_is_whatever_the_code_left_in_result(bench: Bench, scene: Scene) -> None:
    scene.node("/obj").createNode("geo")
    scene.node("/obj").createNode("null")
    body = ok(python(bench, code="result = len(hou.node('/obj').children())"))
    assert body["result"] == 2
    assert body["stdout_tail"] == ""
    assert body["elided_chars"] == 0
    assert "spill_path" not in body and "error" not in body
    assert body["namespace"] == python_tool.DEFAULT_NAMESPACE
    assert body["namespace"].startswith("c_")
    assert body["scene_epoch"] == 0
    assert body["duration_ms"] >= 0
    assert body["undo_label"].startswith("hou_python ")
    assert body["trace"]["operation_id"]


def test_houdini_values_and_arrays_come_back_the_way_every_answer_does(bench: Bench) -> None:
    code = (
        "class Array:\n"
        "    shape = (3,)\n"
        "    def tolist(self):\n"
        "        return [1.5, 2.5, 3.5]\n"
        "result = {'node': hou.node('/obj'), 'points': Array(), 'v': hou.Vector3(1, 2, 3)}"
    )
    body = ok(python(bench, code=code))
    assert body["result"] == {"node": "/obj", "points": [1.5, 2.5, 3.5], "v": [1.0, 2.0, 3.0]}


def test_a_value_past_the_encoder_caps_is_cut_and_says_so(bench: Bench) -> None:
    body = ok(python(bench, code="result = list(range(5000))"))
    assert len(body["result"]) == 1024
    assert body["lossy"] is True


def test_what_the_code_prints_on_either_stream_comes_back(bench: Bench) -> None:
    code = "import sys\nprint('one')\nsys.stderr.write('two\\n')\nprint('three')"
    body = ok(python(bench, code=code))
    assert body["stdout_tail"] == "one\ntwo\nthree\n"
    assert sys.stdout is not None and type(sys.stdout).__name__ != "_Routed"


def test_another_threads_printing_is_not_taken_into_the_call(
    bench: Bench, capsys: pytest.CaptureFixture[str]
) -> None:
    code = (
        "import threading\n"
        "print('mine')\n"
        "other = threading.Thread(target=lambda: print('not mine'))\n"
        "other.start()\n"
        "other.join()\n"
    )
    body = ok(python(bench, code=code))
    assert body["stdout_tail"] == "mine\n"
    assert "not mine" in capsys.readouterr().out


def test_a_result_is_never_left_over_from_an_earlier_call(bench: Bench) -> None:
    ok(python(bench, code="result = 1"))
    assert ok(python(bench, code="x = 2"))["result"] is None


# Section: the budget and the spill


def test_printing_past_max_chars_keeps_the_end_and_spills_the_whole(bench: Bench) -> None:
    code = "print('start')\nprint('a' * 20000)\nprint('end')"
    body = ok(python(bench, code=code, max_chars=1000))
    tail = body["stdout_tail"]
    assert len(tail) == 1000
    assert tail.endswith("a\nend\n")
    whole = len("start\n") + 20001 + len("end\n")
    assert body["elided_chars"] == whole - 1000
    spilled = json.loads(Path(body["spill_path"]).read_text(encoding="utf-8"))
    assert spilled["stdout"].startswith("start\naaa")
    assert len(spilled["stdout"]) == whole


def test_the_result_takes_the_budget_first(bench: Bench) -> None:
    body = ok(python(bench, code="result = [1, 2, 3]\nprint('x' * 50)", max_chars=20))
    assert body["result"] == [1, 2, 3]
    assert body["stdout_tail"] == "x" * 12 + "\n"
    assert body["elided_chars"] == 51 - 13


def test_a_result_over_the_budget_comes_back_as_the_start_of_its_text(bench: Bench) -> None:
    body = ok(python(bench, code="result = 'b' * 5000\nprint('gone')", max_chars=100))
    assert body["result"] == '"' + "b" * 99
    assert body["stdout_tail"] == ""
    assert body["elided_chars"] == 5002 - 100 + len("gone\n")
    spilled = json.loads(Path(body["spill_path"]).read_text(encoding="utf-8"))
    assert spilled["result"] == "b" * 5000


def test_a_long_result_shows_its_output_in_the_text_block(bench: Bench) -> None:
    result = python(bench, code="print('line ' * 400)\nresult = 7")
    text = text_of(result)
    assert text.startswith("hou_python in c_")
    assert "result: 7" in text
    assert "stdout_tail:\nline line" in text


# Section: the code's own exceptions


def test_an_exception_is_data_with_the_traceback_tail_and_no_places_on_disk(
    bench: Bench,
) -> None:
    code = (
        "kept = 'before'\n"
        "def inner():\n"
        "    raise ValueError('cannot read /Users/somebody/cache/a.bgeo')\n"
        "inner()\n"
        "kept = 'after'\n"
    )
    result = python(bench, code=code, namespace="t1")
    body = raised(result)
    error = body["error"]
    assert error["type"] == "ValueError"
    assert "/Users" not in error["message"]
    assert "<path>" in error["message"]
    tail = error["traceback_tail"]
    assert "line 4" in tail and "line 3, in inner" in tail
    assert "raise ValueError" in tail
    assert "/Users" not in tail
    assert len(tail.splitlines()) <= tools.TRACEBACK_LINES
    assert "tools.py" not in tail
    assert "ValueError" in text_of(result)
    # The namespace is as it was at the raise.
    assert ok(python(bench, code="result = kept", namespace="t1"))["result"] == "before"


def test_a_deep_traceback_keeps_its_last_lines(bench: Bench) -> None:
    steps = "".join(f"def f{index}():\n    f{index + 1}()\n" for index in range(15))
    code = steps + "def f15():\n    return 1 / 0\nf0()\n"
    error = raised(python(bench, code=code))["error"]
    assert error["type"] == "ZeroDivisionError"
    lines = error["traceback_tail"].splitlines()
    assert len(lines) == tools.TRACEBACK_LINES
    assert lines[-1].startswith("ZeroDivisionError")


def test_a_syntax_error_says_the_line_and_the_offset(bench: Bench) -> None:
    error = raised(python(bench, code="x = 1\ny = (2,\n"))["error"]
    assert error["type"] == "SyntaxError"
    assert error["line"] == 2
    assert error["offset"] == 5
    assert error["message"]


def test_exit_in_the_code_is_an_exception_like_any_other(bench: Bench) -> None:
    error = raised(python(bench, code="raise SystemExit(3)"))["error"]
    assert error["type"] == "SystemExit"
    assert ok(python(bench, code="result = 'still here'"))["result"] == "still here"


# Section: namespaces


def test_a_namespace_keeps_its_variables_between_calls(bench: Bench) -> None:
    ok(python(bench, code="total = 40"))
    assert ok(python(bench, code="total += 2\nresult = total"))["result"] == 42


def test_a_new_namespace_holds_hou_and_mcp_and_nothing_else(bench: Bench) -> None:
    code = "result = sorted(k for k in globals() if k != '__builtins__')"
    assert ok(python(bench, code=code, namespace="fresh"))["result"] == ["hou", "mcp"]


def test_reset_clears_the_namespace_first(bench: Bench) -> None:
    ok(python(bench, code="left = 1", namespace="n"))
    body = ok(python(bench, code="result = 'left' in globals()", namespace="n", reset=True))
    assert body["result"] is False
    assert sent(bench)[-1]["arguments"]["reset"] is True


def test_the_default_namespace_is_this_servers_own(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok(python(bench, code="mine = 1"))
    monkeypatch.setattr(python_tool, "DEFAULT_NAMESPACE", "c_another")
    body = ok(python(bench, code="result = 'mine' in globals()"))
    assert body["namespace"] == "c_another"
    assert body["result"] is False


def test_shared_is_shared_between_two_callers(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    ok(python(bench, code="note = 'from the first'", namespace="shared"))
    # A second server, with its own default namespace, reads it.
    monkeypatch.setattr(python_tool, "DEFAULT_NAMESPACE", "c_second")
    body = ok(python(bench, code="result = note", namespace="shared"))
    assert body["result"] == "from the first"
    assert body["namespace"] == "shared"


def test_a_namespace_unused_for_a_day_is_dropped(bench: Bench, clock: Clock) -> None:
    ok(python(bench, code="kept = 1", namespace="old"))
    clock.now += DAY_S - 60.0
    assert ok(python(bench, code="result = kept", namespace="old"))["result"] == 1
    clock.now += DAY_S + 1.0
    body = ok(python(bench, code="result = 'kept' in globals()", namespace="old"))
    assert body["result"] is False


def test_separate_namespaces_share_one_scene(bench: Bench, scene: Scene) -> None:
    ok(python(bench, code="hou.node('/obj').createNode('geo', 'made_in_a')", namespace="a"))
    body = ok(python(bench, code="result = hou.node('/obj/made_in_a') is not None", namespace="b"))
    assert body["result"] is True


# Section: every call is a change


def test_each_call_is_one_undo_entry_under_its_label(bench: Bench, scene: Scene) -> None:
    code = "for name in ('a', 'b', 'c'):\n    hou.node('/obj').createNode('geo', name)"
    body = ok(python(bench, code=code, undo_label="three boxes"))
    assert body["undo_label"] == "three boxes"
    assert scene.undos.undoLabels() == ["three boxes"]
    ok(python(bench, code="hou.node('/obj').createNode('null')", operation_id="op-7"))
    assert scene.undos.undoLabels()[-1] == "hou_python op-7"


def test_a_call_that_names_no_id_is_labelled_with_the_one_it_was_given(bench: Bench) -> None:
    body = ok(python(bench, code="x = 1"))
    assert body["undo_label"] == f"hou_python {body['trace']['operation_id']}"


def test_an_exception_leaves_what_the_code_did_as_one_undo_entry(
    bench: Bench, scene: Scene
) -> None:
    code = "hou.node('/obj').createNode('geo', 'half')\nraise RuntimeError('stop')"
    raised(python(bench, code=code))
    assert scene.node("/obj/half") is not None
    assert len(scene.undos.undoLabels()) == 1
    scene.undos.performUndo()
    assert scene.node("/obj/half") is None


def test_the_same_operation_id_answers_from_the_receipt_and_runs_once(
    bench: Bench, scene: Scene
) -> None:
    code = "runs = globals().get('runs', 0) + 1\nhou.node('/obj').createNode('geo')\nresult = runs"
    first = ok(python(bench, code=code, operation_id="op-once"))
    again = ok(python(bench, code=code, operation_id="op-once"))
    assert first["result"] == again["result"] == 1
    assert len(scene.node("/obj").children()) == 1
    assert ok(python(bench, code="result = runs"))["result"] == 1


def test_the_same_operation_id_with_other_code_is_a_mismatch(bench: Bench, scene: Scene) -> None:
    ok(python(bench, code="hou.node('/obj').createNode('geo')", operation_id="op-m"))
    error = refused(python(bench, code="hou.node('/obj').createNode('cam')", operation_id="op-m"))
    assert error["code"] == "OPERATION_MISMATCH"
    assert len(scene.node("/obj").children()) == 1


def test_an_id_whose_session_died_between_the_work_and_the_receipt_is_unknown(
    bench: Bench, scene: Scene
) -> None:
    code = "hou.node('/obj').createNode('geo')"
    arguments = {
        "code": code,
        "namespace": python_tool.DEFAULT_NAMESPACE,
        "undo_label": "hou_python op-dead",
    }
    digest = receipt_module.digest_call("python.run", arguments)
    with bench.store() as store:
        store.begin_operation("op-dead", digest, session_id="s-1", scene_epoch=0, owner_pid=1 << 30)
    error = refused(python(bench, code=code, operation_id="op-dead"))
    assert error["code"] == "OUTCOME_UNKNOWN"
    assert scene.node("/obj").children() == ()


def test_a_scene_epoch_from_a_scene_since_replaced_is_refused_before_the_code_runs(
    bench: Bench,
) -> None:
    epoch = ok(python(bench, code="result = 1"))["scene_epoch"]
    through(bench).identity.bump("cleared")
    error = refused(python(bench, code="ran = True", scene_epoch=epoch))
    assert error["code"] == "SCENE_REPLACED"
    body = ok(python(bench, code="result = 'ran' in globals()", scene_epoch=epoch + 1))
    assert body["result"] is False
    assert body["scene_epoch"] == epoch + 1


# Section: time


def test_code_that_outruns_its_timeout_answers_and_keeps_running(bench: Bench, module: Any) -> None:
    code = "hou.gate.wait(10)\nresult = 'finished'"
    result = python(bench, code=code, timeout_s=0.2)
    error = refused(result)
    assert error["code"] == "TIMEOUT"
    assert error["details"]["still_running"] is True
    operation_id = result.structured_content["trace"]["operation_id"]
    assert error["details"]["operation_id"] == operation_id
    assert through(bench).dispatcher.state()["busy"] is True
    module.gate.set()
    support.wait_until(lambda: not through(bench).dispatcher.state()["busy"], timeout_s=10.0)
    # The same id fetches the answer the code came to.
    body = ok(python(bench, code=code, operation_id=operation_id))
    assert body["result"] == "finished"


def test_the_run_budget_defaults_to_a_minute_and_is_capped_by_the_config(
    bench: Bench,
) -> None:
    ok(python(bench, code="x = 1"))
    assert sent(bench)[-1]["timeout_s"] == 60.0
    bench.config = replace(bench.config, python_timeout_cap_s=5)
    ok(python(bench, code="x = 1", timeout_s=900))
    assert sent(bench)[-1]["timeout_s"] == 5


def test_progress_shows_in_health_while_the_code_runs(bench: Bench, module: Any) -> None:
    code = "mcp.progress(1, 3, 'first of three')\nhou.gate.wait(10)\nmcp.progress(3, 3)"
    done: list[Any] = []
    runner = threading.Thread(target=lambda: done.append(python(bench, code=code)))
    runner.start()
    try:
        state = through(bench).dispatcher.state
        support.wait_until(lambda: state()["current_op_progress"], timeout_s=10.0)
        [note] = state()["current_op_progress"]
        assert note["done"] == 1 and note["total"] == 3
        assert note["message"] == "first of three"
    finally:
        module.gate.set()
        runner.join(10.0)
    ok(done[0])
    assert through(bench).dispatcher.state()["last_op"]["progress"]["done"] == 3


def test_a_long_loop_can_see_it_was_asked_to_stop(bench: Bench) -> None:
    code = (
        "import time\n"
        "laps = 0\n"
        "while not mcp.cancelled() and laps < 1000:\n"
        "    laps += 1\n"
        "    time.sleep(0.01)\n"
        "result = laps"
    )
    done: list[Any] = []
    runner = threading.Thread(target=lambda: done.append(python(bench, code=code)))
    runner.start()
    dispatcher = through(bench).dispatcher
    support.wait_until(lambda: dispatcher.state()["busy"], timeout_s=10.0)
    asked = dispatcher.dispatch(Envelope(tool="bridge.cancel", arguments={}))
    assert asked.payload["data"]["asked"] is True
    runner.join(20.0)
    assert ok(done[0])["result"] < 1000


def test_the_helper_refuses_a_progress_that_is_not_a_number(bench: Bench) -> None:
    error = raised(python(bench, code="mcp.progress('half')"))["error"]
    assert error["type"] == "TypeError"


# Section: managed outputs


def test_an_output_path_comes_from_the_table_for_this_scene(
    bench: Bench, scene: Scene, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    scene.hipFile.setName(str(project / "shot.hip"))
    body = ok(python(bench, code="result = mcp.output_path('cache', 'test')"))
    expected = project / "geo" / "test" / "v001" / "test_v001.$F4.bgeo.sc"
    assert Path(body["result"]) == expected
    assert (project / "geo" / "test" / "v001").is_dir()
    again = ok(python(bench, code="result = mcp.output_path('cache', 'test')"))["result"]
    assert "v002" in again


def test_an_unsaved_scene_gets_output_paths_in_the_scratch_folder(
    bench: Bench, scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOUDINI_TEMP_DIR", str(tmp_path / "scratch"))
    scene.hipFile.setName("untitled.hip")
    result = ok(python(bench, code="result = mcp.output_path('capture', 'look', 'jpg')"))["result"]
    assert Path(result).is_relative_to(tmp_path / "scratch")
    assert result.endswith(".jpg")


def test_a_kind_the_table_does_not_have_is_an_exception_in_the_code(bench: Bench) -> None:
    error = raised(python(bench, code="mcp.output_path('texture', 'x')"))["error"]
    assert error["type"] == "UnknownKind"


# Section: arguments


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({"code": "x = 1", "namespace": "has space"}, "namespace"),
        ({"code": "x = 1", "max_chars": 0}, "max_chars"),
        ({}, "arguments"),
        ({"code": "x = 1", "justification": "because"}, "justification"),
    ],
)
def test_arguments_that_cannot_work_are_refused_before_anything_is_sent(
    bench: Bench, arguments: dict[str, Any], argument: str
) -> None:
    error = refused(python(bench, **arguments))
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == argument
    assert through(bench).calls == []


def test_the_tool_is_listed_after_hou_inspect_as_one_that_changes_things(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    names = [tool.name for tool in listed.tools]
    assert names[names.index("hou_inspect") + 1] == "hou_python"
    [tool] = [tool for tool in listed.tools if tool.name == "hou_python"]
    assert tool.annotations is None or tool.annotations.read_only_hint is None
    assert tool.input_schema["required"] == ["code"]
    assert "justification" not in tool.input_schema["properties"]
    assert "background" not in tool.input_schema["properties"]


# Section: a reply lost on the way back


def test_a_lost_reply_is_sent_again_and_the_code_runs_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The code runs, its answer is held back, and the caller sends it again.

    A real bridge on a real socket, carrying the self check so it will hold
    an answer back, and the client's own retry under the same id.
    """
    monkeypatch.setenv("NSCR_MCP_SELFCHECK", "1")
    scene = Scene()
    bridge, _ = make_bridge(tmp_path, driver="stdlib", hou=scene.module(), drop_reply_s=3.0)
    bridge.start()
    try:
        session = client.Session.open(tmp_path, bridge.session_id)
        arguments = {
            "code": "hou.node('/obj').createNode('geo')\nresult = len(hou.node('/obj').children())",
            "namespace": "lost",
            "undo_label": "hou_python op-lost",
            "drop_reply": True,
        }
        began = time.monotonic()
        answer = client.call(
            session,
            "python.run",
            arguments=arguments,
            operation_id="op-lost",
            http_timeout_s=0.5,
        )
        assert time.monotonic() - began >= 0.5
        assert answer.payload["ok"] is True, answer.payload
        assert answer.payload["replayed"] is True
        assert answer.payload["data"]["result"] == 1
        assert len(scene.node("/obj").children()) == 1
    finally:
        bridge.stop()
        scene.ui.stop()
