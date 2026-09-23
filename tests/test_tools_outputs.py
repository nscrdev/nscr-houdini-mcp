"""`hou_outputs` and the parameter ruling, through the server down to a bridge.

Every call goes the whole way: the server checks the arguments, the router
sends the call, and a real dispatcher with real receipts runs the bridge's
own operations against the stand in for `hou`. The store is real, and so are
the scene folders and the files an output is looked for on disk. A session
that dies is a real process, ended the way a crash ends one. What a real
Houdini does with the same calls is in the integration test.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp import outputs as output_rules
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, tools
from nscr_houdini_mcp.tools import outputs as output_tool
from test_server import talk, text_of
from test_tools_python import Clock, Through
from test_tools_sessions import Bench

TYPES = ("geo", "null", "cam", "karma", "rop_geometry", "filecache", "file")
SESSION = client.Session(session_id="s-1", token="token", port=18000)


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene(types=TYPES)
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def project(tmp_path: Path, scene: Scene) -> Path:
    """A scene saved in a folder of its own, so `$HIP` is a real place."""
    folder = tmp_path / "project"
    folder.mkdir()
    scene.hipFile.setName((folder / "shot_v001.hip").as_posix())
    return folder


@pytest.fixture
def bench(tmp_path: Path, scene: Scene, monkeypatch: pytest.MonkeyPatch) -> Bench:
    monkeypatch.delenv("HOUDINI_TEMP_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    made.sent = Through(scene.module(), home, tools.Namespaces(clock=Clock()))  # type: ignore[assignment]
    return made


def through(bench: Bench) -> Through:
    return bench.sent  # type: ignore[return-value]


def call(bench: Bench, tool: str = "hou_outputs", **arguments: Any) -> Any:
    _, [result] = talk(bench.serve(), (tool, arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, text_of(result)
    return result.structured_content


def refused(result: Any) -> dict[str, Any]:
    assert result.is_error is True
    return result.structured_content["error"]


def bridge(bench: Bench, tool: str, **arguments: Any) -> dict[str, Any]:
    """One bridge operation straight to the dispatcher, the way a later tool sends it."""
    answer = through(bench)(
        SESSION, tool, arguments=arguments, operation_id=client.new_operation_id()
    )
    return answer.payload


def picture(scene: Scene, node: str = "/out/karma1") -> Any:
    return scene.node(node).parm("picture")


def store(bench: Bench) -> store_module.Store:
    return bench.store()


# Section: resolve


def test_resolve_takes_a_cache_path_named_after_its_node(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/obj").createNode("geo").createNode("filecache")
    body = ok(call(bench, action="resolve", kind="cache", node="/obj/geo1/filecache1"))
    assert body["parm_string"] == "$HIP/geo/${OS}/v001/${OS}_v001.$F4.bgeo.sc"
    expected = project / "geo" / "filecache1" / "v001" / "filecache1_v001.$F4.bgeo.sc"
    assert Path(body["expanded_path"]) == expected
    assert body["version"] == 1
    assert Path(body["folder"]) == project / "geo" / "filecache1" / "v001"
    assert Path(body["folder"]).is_dir()
    assert Path(body["sidecar"]).is_file()
    with store(bench) as opened:
        run = opened.get_run(body["run_id"])
    assert run.source_node == "/obj/geo1/filecache1"
    assert run.session_id == "s-1"
    again = ok(call(bench, action="resolve", kind="cache", node="/obj/geo1/filecache1"))
    assert again["version"] == 2


@pytest.mark.parametrize(
    ("kind", "folder"),
    [("reference", ".agent/reference"), ("check", ".agent/checks")],
)
def test_references_and_checks_resolve_under_the_agent_folder(
    bench: Bench, project: Path, kind: str, folder: str
) -> None:
    body = ok(call(bench, action="resolve", kind=kind, name="look"))
    assert body["parm_string"].startswith(f"$HIP/{folder}/")
    assert Path(body["expanded_path"]).is_relative_to(project / folder)
    assert body["run_id"] in body["expanded_path"]


def test_a_spill_resolves_in_the_servers_spill_folder(bench: Bench, project: Path) -> None:
    body = ok(call(bench, action="resolve", kind="spill", name="dump"))
    assert Path(body["expanded_path"]).is_relative_to(bench.config.spill_folder)
    # The one kind whose line is this machine's own: it never goes on a node.
    assert body["parm_string"] == body["expanded_path"]


def test_a_scene_never_saved_resolves_in_the_scratch_folder(
    bench: Bench, scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOUDINI_TEMP_DIR", str(tmp_path / "scratch"))
    scene.hipFile.setName("untitled.hip")
    body = ok(call(bench, action="resolve", kind="render", name="beauty"))
    assert body["unsaved_hip"] is True
    assert body["parm_string"].startswith("$HOUDINI_TEMP_DIR/")
    assert Path(body["expanded_path"]).is_relative_to(tmp_path / "scratch")
    assert body["warnings"]


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({"action": "resolve"}, "kind"),
        ({"action": "resolve", "kind": "renders"}, "kind"),
        ({"action": "resolve", "kind": "job"}, "kind"),
        ({"action": "resolve", "kind": "render", "node": "karma1"}, "node"),
        ({"action": "list", "limit": 0}, "limit"),
        ({"action": "list", "filter": {"kinds": "render"}}, "filter.kinds"),
        ({"action": "list", "filter": {"since": "yesterday"}}, "filter.since"),
        ({"action": "list", "filter": {"kind": "texture"}}, "filter.kind"),
        ({"action": "list", "filter": "render"}, "filter"),
        ({"action": "lint", "node": "out"}, "node"),
    ],
)
def test_arguments_that_cannot_work_are_refused(
    bench: Bench, project: Path, arguments: dict[str, Any], argument: str
) -> None:
    error = refused(call(bench, **arguments))
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == argument


def test_a_misspelled_kind_says_the_closest(bench: Bench, project: Path) -> None:
    error = refused(call(bench, action="resolve", kind="renders"))
    assert "render" in error["details"]["did_you_mean"]


def test_resolve_for_a_node_that_is_not_there_takes_nothing(bench: Bench, project: Path) -> None:
    error = refused(call(bench, action="resolve", kind="cache", node="/obj/nothing"))
    assert error["code"] == "NODE_NOT_FOUND"
    with store(bench) as opened:
        assert opened.find_runs(hip_family="shot") == []


# Section: list


def test_list_reads_what_this_scene_made_newest_first(bench: Bench, project: Path) -> None:
    first = ok(call(bench, action="resolve", kind="render", name="beauty"))
    second = ok(call(bench, action="resolve", kind="cache", name="sim"))
    # A frame of the render is on disk; nothing of the cache is.
    frame = first["expanded_path"].replace("$F4", "0001")
    Path(frame).write_bytes(b"exr")
    body = ok(call(bench, action="list"))
    runs = body["runs"]
    assert [run["run_id"] for run in runs] == [second["run_id"], first["run_id"]]
    assert runs[0]["kind"] == "cache" and runs[0]["version"] == 1
    assert runs[1]["on_disk"] is True
    assert runs[0]["on_disk"] is False
    assert runs[0]["session"] == "s-1"
    assert runs[0]["template"] == second["parm_string"]
    assert runs[0]["path"] == second["expanded_path"]
    assert datetime.fromisoformat(runs[0]["created"]).tzinfo is not None
    assert body["scene_family"] == "shot"
    assert "next_page" not in body


def test_list_filters_by_kind_name_and_time(bench: Bench, project: Path) -> None:
    ok(call(bench, action="resolve", kind="render", name="beauty"))
    ok(call(bench, action="resolve", kind="render", name="shadow"))
    ok(call(bench, action="resolve", kind="cache", name="sim"))
    names = lambda body: [run["name"] for run in body["runs"]]  # noqa: E731
    assert names(ok(call(bench, action="list", filter={"kind": "render"}))) == [
        "shadow",
        "beauty",
    ]
    assert names(ok(call(bench, action="list", filter={"name": "sh*"}))) == ["shadow"]
    later = ok(call(bench, action="list", filter={"since": "2999-01-01T00:00"}))
    assert later["runs"] == []
    assert len(ok(call(bench, action="list", filter={"since": 0}))["runs"]) == 3


def test_list_pages_with_a_token_that_only_fits_its_own_query(bench: Bench, project: Path) -> None:
    made = [ok(call(bench, action="resolve", kind="capture", name=f"c{n}")) for n in range(3)]
    first = ok(call(bench, action="list", limit=2))
    assert len(first["runs"]) == 2
    token = first["next_page"]
    second = ok(call(bench, action="list", limit=2, page=token))
    seen = [run["run_id"] for run in first["runs"] + second["runs"]]
    assert sorted(seen) == sorted(body["run_id"] for body in made)
    assert "next_page" not in second
    other = refused(call(bench, action="list", limit=2, page=token, filter={"kind": "render"}))
    assert other["code"] == "BAD_CURSOR"
    assert refused(call(bench, action="lint", page=token))["code"] == "BAD_CURSOR"
    assert refused(call(bench, action="list", page="not-a-token"))["code"] == "BAD_CURSOR"


def test_list_leaves_out_what_another_scene_made(
    bench: Bench, scene: Scene, project: Path, tmp_path: Path
) -> None:
    ok(call(bench, action="resolve", kind="render", name="here"))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    # Same family name, another folder: another scene.
    scene.hipFile.setName((elsewhere / "shot_v003.hip").as_posix())
    ok(call(bench, action="resolve", kind="render", name="there"))
    assert [run["name"] for run in ok(call(bench, action="list"))["runs"]] == ["there"]
    scene.hipFile.setName((project / "shot_v002.hip").as_posix())
    # A later version of the same scene is the same family in the same place.
    assert [run["name"] for run in ok(call(bench, action="list"))["runs"]] == ["here"]


# Section: lint


def rows_of(body: dict[str, Any]) -> set[tuple[str, str, str]]:
    return {(row["node"], row["parm"], row["problem"]) for row in body["rows"]}


@pytest.fixture
def fixture_scene(scene: Scene, project: Path, tmp_path: Path) -> dict[str, str]:
    """Four render nodes, each set by hand to break one rule and only that one."""
    out = scene.node("/out")
    render = project / "render"
    render.mkdir()
    (render / "beauty_v001.exr").write_bytes(b"exr")
    (render / "beauty.exr").write_bytes(b"exr")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "beauty_v001.exr").write_bytes(b"exr")
    values = {
        "absolute_path": f"{project.as_posix()}/render/beauty_v001.exr",
        "outside_hip": "$HIP/../elsewhere/beauty_v001.exr",
        "unversioned": "$HIP/render/beauty.exr",
        "missing_on_disk": "$HIP/render/beauty_v002.exr",
    }
    made = {}
    for problem, value in values.items():
        node = out.createNode("karma")
        node.parm("picture").set(value)
        made[problem] = node.path()
    # A viewer instead of a file, a disabled output, a read parameter and a
    # node that reads a file: none of them is an output to lint.
    viewer = out.createNode("karma")
    viewer.parm("picture").set("ip")
    off = out.createNode("karma")
    off.parm("picture").set("/somewhere/else.exr")
    off.parm("picture").disabled = True
    reader = scene.node("/obj").createNode("geo").createNode("file")
    reader.parm("file").set("/somewhere/else.bgeo")
    return made


def test_lint_reports_each_hand_set_problem_and_nothing_else(
    bench: Bench, fixture_scene: dict[str, str]
) -> None:
    body = ok(call(bench, action="lint", node="/out"))
    expected = {(node, "picture", problem) for problem, node in fixture_scene.items()}
    assert rows_of(body) == expected
    [absolute] = [row for row in body["rows"] if row["problem"] == "absolute_path"]
    assert absolute["raw"] == absolute["expanded"]
    [outside] = [row for row in body["rows"] if row["problem"] == "outside_hip"]
    assert outside["raw"].startswith("$HIP/")
    assert "elsewhere" in outside["expanded"]
    assert body["parms_checked"] == 6


def test_lint_scope_is_the_node_named(bench: Bench, scene: Scene, project: Path) -> None:
    geo = scene.node("/obj").createNode("geo")
    geo.createNode("rop_geometry").parm("sopoutput").set("/abs/geo_v001.bgeo")
    out = scene.node("/out").createNode("karma")
    out.parm("picture").set("/abs/beauty_v001.exr")
    under_obj = rows_of(ok(call(bench, action="lint", node="/obj")))
    assert {node for node, _, _ in under_obj} == {"/obj/geo1/rop_geometry1"}
    everywhere = rows_of(ok(call(bench, action="lint")))
    assert {node for node, _, _ in everywhere} == {"/obj/geo1/rop_geometry1", out.path()}


def test_lint_pages_on_whole_parameters(bench: Bench, fixture_scene: dict[str, str]) -> None:
    whole = rows_of(ok(call(bench, action="lint", node="/out")))
    seen: set[tuple[str, str, str]] = set()
    page = None
    for _ in range(10):
        arguments: dict[str, Any] = {"action": "lint", "node": "/out", "limit": 1}
        if page:
            arguments["page"] = page
        body = ok(call(bench, **arguments))
        seen |= rows_of(body)
        page = body.get("next_page")
        if not page:
            break
    assert seen == whole
    first = ok(call(bench, action="lint", node="/out", limit=1))
    moved = refused(call(bench, action="lint", node="/obj", limit=1, page=first["next_page"]))
    assert moved["code"] == "BAD_CURSOR"


def test_lint_for_a_node_that_is_not_there_says_so(bench: Bench, project: Path) -> None:
    assert refused(call(bench, action="lint", node="/out/nothing"))["code"] == "NODE_NOT_FOUND"


# Section: the frozen parameter, through hou_python


FREEZE = """
node = hou.node('/out/karma1')
path = mcp.output_path('render', 'beauty')
mcp.freeze_parm(node.parm('picture'), path)
result = {'during': node.parm('picture').unexpandedString(), 'path': path}
"""


def test_a_frozen_parm_holds_the_run_path_and_gets_its_template_back_at_run_end(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    body = ok(call(bench, "hou_python", code=FREEZE))
    during = body["result"]["during"]
    assert during == body["result"]["path"]
    assert Path(during).is_relative_to(project)
    [restored] = body["restored_parms"]
    assert restored["restored"] is True
    assert restored["node"] == "/out/karma1" and restored["parm"] == "picture"
    now = picture(scene).unexpandedString()
    assert now == restored["template"]
    assert now.startswith("$HIP/renders/") and now.endswith("/beauty_v001.$F4.exr")
    with store(bench) as opened:
        assert opened.list_frozen_parms() == []


def test_code_that_raises_after_freezing_still_gives_the_template_back(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    result = call(bench, "hou_python", code=FREEZE + "raise RuntimeError('render failed')\n")
    assert result.is_error is True
    body = result.structured_content
    assert body["error"]["type"] == "RuntimeError"
    assert body["restored_parms"][0]["restored"] is True
    assert picture(scene).unexpandedString().startswith("$HIP/")


def test_a_parm_the_code_changed_after_freezing_is_left_as_the_code_left_it(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    code = FREEZE + "node.parm('picture').set('$HIP/render/mine_v009.exr')\n"
    body = ok(call(bench, "hou_python", code=code))
    [restored] = body["restored_parms"]
    assert restored["restored"] is False
    assert restored["reason"] == "changed_since"
    assert picture(scene).unexpandedString() == "$HIP/render/mine_v009.exr"
    with store(bench) as opened:
        assert opened.list_frozen_parms() == []


def test_only_a_path_this_call_was_handed_can_be_frozen(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    code = "mcp.freeze_parm('/out/karma1/picture', '/somewhere/else.exr')"
    result = call(bench, "hou_python", code=code)
    assert result.structured_content["error"]["type"] == "ValueError"
    assert picture(scene).unexpandedString() == "$HIP/render/$HIPNAME.$OS.$F4.exr"


def test_a_parm_path_works_as_well_as_the_parm(bench: Bench, scene: Scene, project: Path) -> None:
    scene.node("/out").createNode("karma")
    code = FREEZE.replace("node.parm('picture'), path", "'/out/karma1/picture', path")
    body = ok(call(bench, "hou_python", code=code))
    assert body["restored_parms"][0]["restored"] is True


# Section: the bridge operations other tools call


def test_freeze_and_restore_operations_and_lint_between_them(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    # No name, so it comes from the node and stays a variable in the line.
    run = ok(call(bench, action="resolve", kind="render", node="/out/karma1"))
    assert "karma1" in run["expanded_path"]
    frozen = bridge(
        bench, "outputs.freeze_parm", node="/out/karma1", parm="picture", run_id=run["run_id"]
    )
    assert frozen["ok"], frozen
    assert picture(scene).unexpandedString() == run["expanded_path"]
    # The run has not given it back, so as far as lint can tell it is left over.
    problems = {row["problem"] for row in ok(call(bench, action="lint", node="/out"))["rows"]}
    assert {"frozen_after_run", "absolute_path"} <= problems
    restored = bridge(bench, "outputs.restore_parm", node="/out/karma1", parm="picture")
    assert restored["data"]["restored"] is True
    assert picture(scene).unexpandedString() == run["parm_string"]
    assert "${OS}" in picture(scene).unexpandedString()
    after = {row["problem"] for row in ok(call(bench, action="lint", node="/out"))["rows"]}
    assert "frozen_after_run" not in after and "absolute_path" not in after
    again = bridge(bench, "outputs.restore_parm", node="/out/karma1", parm="picture")
    assert again["data"] == {
        "node": "/out/karma1",
        "parm": "picture",
        "restored": False,
        "reason": "not_frozen",
    }


def test_the_freeze_operation_takes_only_a_run_that_handed_out_a_path(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    unknown = bridge(
        bench, "outputs.freeze_parm", node="/out/karma1", parm="picture", run_id="run-x"
    )
    assert unknown["error"]["code"] == "BAD_ARGUMENTS"
    spill = ok(call(bench, action="resolve", kind="spill", name="dump"))
    machine = bridge(
        bench, "outputs.freeze_parm", node="/out/karma1", parm="picture", run_id=spill["run_id"]
    )
    assert machine["error"]["code"] == "BAD_ARGUMENTS"
    render = ok(call(bench, action="resolve", kind="render", name="beauty"))
    wrong = bridge(
        bench, "outputs.freeze_parm", node="/out/karma1", parm="pictur", run_id=render["run_id"]
    )
    assert wrong["error"]["code"] == "PARM_NOT_FOUND"
    assert "picture" in wrong["error"]["details"]["did_you_mean"]
    assert picture(scene).unexpandedString() == "$HIP/render/$HIPNAME.$OS.$F4.exr"
    with store(bench) as opened:
        assert opened.list_frozen_parms() == []


# Section: a session that died with a parameter frozen


@pytest.fixture
def killed() -> Iterator[subprocess.Popen]:
    """A process standing in for a session, ended the way a crash ends one."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield child
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)


def dead_session_froze(
    bench: Bench, scene: Scene, project: Path, killed: subprocess.Popen, *, node: str
) -> str:
    """A worker froze a parameter in this scene, saved it, and died."""
    stamp = store_module.process_start_stamp(killed.pid)
    with store(bench) as opened:
        opened.register_session(
            "s-dead", kind="hython", pid=killed.pid, pid_start=stamp, alias="w9"
        )
        frozen = f"{project.as_posix()}/renders/20260921_beauty/v001/beauty_v001.$F4.exr"
        opened.freeze_parm(
            session_id="s-dead",
            node_path=node,
            parm_name="picture",
            template="$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr",
            frozen=frozen,
            run_id="run-dead",
            hip_key=output_rules.scene_key(scene.hipFile.path()),
        )
    killed.kill()
    killed.wait(timeout=30)
    return frozen


def test_a_dead_workers_frozen_parm_is_restored_by_the_next_session_in_the_scene(
    bench: Bench, scene: Scene, project: Path, killed: subprocess.Popen
) -> None:
    scene.node("/out").createNode("karma")
    frozen = dead_session_froze(bench, scene, project, killed, node="/out/karma1")
    # The saved scene came back with the dead run's path in it.
    picture(scene).set(frozen)
    body = ok(call(bench, action="lint", node="/out"))
    [restored] = body["restored_parms"]
    assert restored == {
        "node": "/out/karma1",
        "parm": "picture",
        "restored": True,
        "value": "$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr",
        "template": "$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr",
        "owner": "s-dead",
    }
    assert "frozen_after_run" not in {row["problem"] for row in body["rows"]}
    with store(bench) as opened:
        assert opened.list_frozen_parms() == []


def test_a_live_sessions_frozen_parm_is_not_taken_from_it(
    bench: Bench, scene: Scene, project: Path
) -> None:
    scene.node("/out").createNode("karma")
    with store(bench) as opened:
        opened.register_session("s-other", kind="hython", pid=os.getpid(), alias="w8")
        opened.freeze_parm(
            session_id="s-other",
            node_path="/out/karma1",
            parm_name="picture",
            template="$HIP/x_v001.exr",
            frozen="/abs/x_v001.exr",
            hip_key=output_rules.scene_key(scene.hipFile.path()),
        )
    picture(scene).set("/abs/x_v001.exr")
    body = ok(call(bench, action="lint", node="/out", session="w1"))
    assert "restored_parms" not in body
    assert "frozen_after_run" in {row["problem"] for row in body["rows"]}
    refused_here = bridge(
        bench, "outputs.restore_parm", node="/out/karma1", parm="picture", owner="s-other"
    )
    assert refused_here["error"]["code"] == "BAD_ARGUMENTS"
    assert picture(scene).unexpandedString() == "/abs/x_v001.exr"


def test_opening_the_scene_is_where_a_dead_workers_leftovers_are_swept(
    bench: Bench, scene: Scene, project: Path, killed: subprocess.Popen
) -> None:
    hip = project / "shot_v001.hip"
    hip.write_bytes(b"scene")
    dead_session_froze(bench, scene, project, killed, node="/out/karma1")
    body = ok(call(bench, "hou_scene", action="open", path=str(hip)))
    # The stand in's scene is empty after a load, so the node is not there to
    # hold anything; the record goes all the same.
    [restored] = body["restored_parms"]
    assert restored["owner"] == "s-dead"
    assert restored["reason"] == "node_gone"
    with store(bench) as opened:
        assert opened.list_frozen_parms() == []


def test_a_dead_workers_record_from_a_scene_never_saved_is_dropped(
    bench: Bench, scene: Scene, project: Path, killed: subprocess.Popen
) -> None:
    stamp = store_module.process_start_stamp(killed.pid)
    with store(bench) as opened:
        opened.register_session(
            "s-dead", kind="hython", pid=killed.pid, pid_start=stamp, alias="w9"
        )
        opened.freeze_parm(
            session_id="s-dead",
            node_path="/out/karma1",
            parm_name="picture",
            template="$HOUDINI_TEMP_DIR/x.exr",
            frozen="/tmp/x.exr",
        )
    killed.kill()
    killed.wait(timeout=30)
    body = ok(call(bench, action="list"))
    assert "restored_parms" not in body
    with store(bench) as opened:
        assert opened.list_frozen_parms() == []


# Section: the tool as a client sees it


def test_the_tool_is_listed_last_and_is_not_read_only(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    names = [tool.name for tool in listed.tools]
    assert names[-1] == "hou_outputs"
    [tool] = [tool for tool in listed.tools if tool.name == "hou_outputs"]
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is None
    assert tool.annotations.open_world_hint is False
    assert set(tool.input_schema["properties"]) == {
        "action",
        "session",
        "kind",
        "name",
        "ext",
        "node",
        "filter",
        "limit",
        "page",
        "wait_s",
    }
    assert tool.input_schema["properties"]["action"]["enum"] == list(output_tool.ACTIONS)


def test_a_long_result_is_summed_up_in_one_line(bench: Bench, project: Path) -> None:
    assert output_tool.summary_line({"action": "lint", "rows": [1, 2], "next_page": "x"}) == (
        "hou_outputs lint: 2 rows; more with next_page"
    )
    assert output_tool.summary_line({"action": "list", "runs": [1]}) == "hou_outputs list: 1 runs"
