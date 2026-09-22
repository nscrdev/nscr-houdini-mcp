"""`hou_scene` through the server, with the session's answers stood in for.

The store is real, so versions are taken the way they are in use, and the
scene files the versions are checked against are real files in the test's own
folder. What the session answers is recorded and chosen by the test. What a
real Houdini does with the same calls is in the integration tests.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from nscr_houdini_mcp import outputs
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.bridge.tools import UNDO_NOTE
from test_router import Sent
from test_server import talk, text_of
from test_tools_sessions import Bench


def reply(data: dict[str, Any], *, epoch: int = 0) -> dict[str, Any]:
    return {"ok": True, "data": data, "session_id": "s-1", "alias": "w1", "scene_epoch": epoch}


def refusal(code: str, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": {"code": code, "message": message},
        "session_id": "s-1",
        "alias": "w1",
        "scene_epoch": 0,
    }


def info(hip: Path | str, *, untitled: bool = False) -> dict[str, Any]:
    return reply(
        {
            "hip_path": str(hip),
            "hip_name": Path(hip).name,
            "untitled": untitled,
            "unsaved": None,
            "frame": 1.0,
            "fps": 24.0,
            "frame_range": [1.0, 240.0],
            "nodes": {"/obj": 2},
            "houdini_version": "22.0.368",
            "undo_entries": 0,
            "kind": "hython",
        }
    )


def saved(path: str) -> dict[str, Any]:
    return reply({"hip_path": path, "bytes": 1234, "undo": UNDO_NOTE})


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    monkeypatch.delenv("HOUDINI_TEMP_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    return made


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    return folder


def scene(bench: Bench, *replies: Any, **arguments: Any) -> Any:
    bench.sent = Sent(*replies)
    _, [result] = talk(bench.serve(), ("hou_scene", arguments))
    return result


def sent_tools(bench: Bench) -> list[str]:
    return [call["tool"] for call in bench.sent.calls]


# Section: info


def test_info_is_a_summary_by_default(bench: Bench) -> None:
    result = scene(bench, info("/p/shot_v007.hip"))
    assert not result.is_error, text_of(result)
    body = result.structured_content
    assert body["hip_path"] == "/p/shot_v007.hip"
    assert body["version"] == 7
    assert body["nodes"] == {"/obj": 2}
    assert "dependencies" not in body
    assert "houdini_version" not in body
    [sent] = bench.sent.calls
    assert sent["tool"] == "scene.info"
    assert sent["arguments"] == {}
    assert sent["operation_id"] is None


def test_full_info_asks_for_the_dependency_report(bench: Bench) -> None:
    answer = info("/p/shot.hip")
    answer["data"]["dependencies"] = {"missing_files": [{"parm": "/obj/a/file", "path": "/x"}]}
    result = scene(bench, answer, detail="full")
    body = result.structured_content
    assert body["dependencies"]["missing_files"][0]["parm"] == "/obj/a/file"
    assert body["houdini_version"] == "22.0.368"
    assert body["version"] is None
    assert bench.sent.calls[0]["arguments"] == {"dependencies": True}


# Section: open


def test_open_refuses_a_path_that_is_not_there_without_asking_the_session(
    bench: Bench, project: Path
) -> None:
    result = scene(bench, action="open", path=str(project / "gone.hip"))
    assert result.is_error is True
    assert result.structured_content["error"]["code"] == "FILE_NOT_FOUND"
    assert text_of(result).startswith("FILE_NOT_FOUND: there is no scene file")
    assert bench.sent.calls == []


@pytest.mark.parametrize(
    ("path", "why"),
    [(None, "open needs path"), ("shot.hip", "absolute"), ("/p/notes.txt", ".hip")],
)
def test_open_refuses_a_path_it_cannot_use(bench: Bench, path: str | None, why: str) -> None:
    arguments = {"action": "open"} if path is None else {"action": "open", "path": path}
    result = scene(bench, **arguments)
    assert result.structured_content["error"]["code"] == "BAD_ARGUMENTS"
    assert why in text_of(result)
    assert bench.sent.calls == []


def test_open_loads_and_carries_the_new_epoch(bench: Bench, project: Path) -> None:
    hip = project / "shot_v002.hip"
    hip.write_bytes(b"scene")
    loaded = reply(
        {
            "hip_path": str(hip),
            "hip_name": hip.name,
            "nodes": {"/obj": 1},
            "discarded_unsaved": None,
            "dependencies": {"unresolved_types": [{"type": "fancy_sop", "parent": "/obj"}]},
            "undo": UNDO_NOTE,
        },
        epoch=5,
    )
    result = scene(bench, loaded, action="open", path=str(hip))
    assert not result.is_error, text_of(result)
    body = result.structured_content
    assert body["scene_epoch"] == 5
    assert body["trace"]["scene_epoch"] == 5
    assert body["version"] == 2
    assert body["dependencies"]["unresolved_types"][0]["type"] == "fancy_sop"
    [sent] = bench.sent.calls
    assert sent["tool"] == "scene.open"
    assert sent["arguments"] == {"path": str(hip), "discard_unsaved": False}
    assert sent["operation_id"] is not None
    assert body["trace"]["operation_id"] == sent["operation_id"]


def test_open_passes_on_the_sessions_refusal_of_unsaved_changes(
    bench: Bench, project: Path
) -> None:
    hip = project / "shot.hip"
    hip.write_bytes(b"scene")
    result = scene(
        bench,
        refusal("UNSAVED_CHANGES", "the scene open in this session has changes that are not saved"),
        action="open",
        path=str(hip),
    )
    assert result.structured_content["error"]["code"] == "UNSAVED_CHANGES"
    assert "discard_unsaved" in text_of(result)


# Section: save


def test_save_refuses_an_untitled_scene_and_points_at_save_increment(bench: Bench) -> None:
    result = scene(
        bench,
        refusal("SCENE_UNTITLED", "the scene has never been saved"),
        action="save",
    )
    assert result.structured_content["error"]["code"] == "SCENE_UNTITLED"
    assert "save_increment" in text_of(result)
    assert sent_tools(bench) == ["scene.save"]
    assert bench.sent.calls[0]["operation_id"] is not None


# Section: save_increment


COMMERCIAL = reply({"license": "Commercial"})


def test_save_increment_goes_on_above_the_versions_beside_the_scene(
    bench: Bench, project: Path
) -> None:
    for name in ("shot_v001.hip", "shot_v002.hip"):
        (project / name).write_bytes(name.encode())
    current = project / "shot_v002.hip"
    target = project / "shot_v003.hip"
    result = scene(bench, info(current), COMMERCIAL, saved(str(target)), action="save_increment")
    assert not result.is_error, text_of(result)
    body = result.structured_content
    assert body["version"] == 3
    assert body["hip_path"] == str(target)
    assert body["unsaved_hip"] is False
    assert body["undo"] == UNDO_NOTE
    assert sent_tools(bench) == ["scene.info", "bridge.capabilities", "scene.save_as"]
    assert bench.sent.calls[2]["arguments"] == {"path": str(target)}
    # The versions that were there are as they were.
    assert (project / "shot_v001.hip").read_bytes() == b"shot_v001.hip"
    assert (project / "shot_v002.hip").read_bytes() == b"shot_v002.hip"
    # The claim goes once the file is its own guard; the record stays.
    assert not Path(f"{target}{outputs.CLAIM_SUFFIX}").exists()
    assert Path(body["sidecar"]).is_file()


def test_save_increment_twice_takes_two_versions(bench: Bench, project: Path) -> None:
    current = project / "shot.hip"
    current.write_bytes(b"scene")
    first = scene(
        bench,
        info(current),
        COMMERCIAL,
        saved(str(project / "shot_v001.hip")),
        action="save_increment",
    )
    (project / "shot_v001.hip").write_bytes(b"one")
    second = scene(
        bench,
        info(project / "shot_v001.hip"),
        COMMERCIAL,
        saved(str(project / "shot_v002.hip")),
        action="save_increment",
    )
    assert first.structured_content["version"] == 1
    assert second.structured_content["version"] == 2
    assert bench.sent.calls[2]["arguments"] == {"path": str(project / "shot_v002.hip")}


def test_save_increment_steps_over_a_place_another_writer_claimed(
    bench: Bench, project: Path
) -> None:
    current = project / "shot.hip"
    current.write_bytes(b"scene")
    # Another machine with a store of its own took v001 and has not written it.
    Path(f"{project / 'shot_v001.hip'}{outputs.CLAIM_SUFFIX}").write_bytes(b"")
    result = scene(
        bench,
        info(current),
        COMMERCIAL,
        saved(str(project / "shot_v002.hip")),
        action="save_increment",
    )
    assert result.structured_content["version"] == 2
    assert bench.sent.calls[2]["arguments"] == {"path": str(project / "shot_v002.hip")}


def test_save_increment_of_an_untitled_scene_goes_to_the_scratch_folder(bench: Bench) -> None:
    expected = bench.home / "temp" / "nscr-houdini-mcp" / "s-1" / "untitled_v001.hip"
    result = scene(
        bench,
        info("/somewhere/untitled.hip", untitled=True),
        COMMERCIAL,
        saved(str(expected)),
        action="save_increment",
    )
    assert not result.is_error, text_of(result)
    body = result.structured_content
    assert body["hip_path"] == str(expected)
    assert body["unsaved_hip"] is True
    assert any("scratch" in warning for warning in body["warnings"])
    assert sent_tools(bench) == ["scene.info", "bridge.capabilities", "scene.save_as"]


@pytest.mark.parametrize(
    ("license_name", "hip", "suffix"),
    [
        ("Apprentice", "/somewhere/untitled.hip", "hipnc"),
        ("Indie", "PROJECT/shot.hip", "hiplc"),
        ("Commercial", "PROJECT/shot.hipnc", "hipnc"),
    ],
)
def test_the_suffix_is_the_one_houdini_will_write(
    bench: Bench, project: Path, license_name: str, hip: str, suffix: str
) -> None:
    """A license that writes one kind of file decides the name before it is claimed."""
    titled = hip.startswith("PROJECT")
    path = str(project / hip.split("/", 1)[1]) if titled else hip
    if titled:
        Path(path).write_bytes(b"scene")
    result = scene(
        bench,
        info(path, untitled=not titled),
        reply({"license": license_name}),
        saved("x"),
        action="save_increment",
    )
    assert not result.is_error, text_of(result)
    asked = bench.sent.calls[2]["arguments"]["path"]
    assert asked.endswith(f"_v001.{suffix}")
    assert Path(f"{asked}{outputs.CLAIM_SUFFIX}").exists() is False


def test_a_save_increment_sent_again_gives_the_first_answer(bench: Bench, project: Path) -> None:
    current = project / "shot.hip"
    current.write_bytes(b"scene")
    target = str(project / "shot_v001.hip")
    first = scene(
        bench,
        info(current),
        COMMERCIAL,
        saved(target),
        action="save_increment",
        operation_id="inc-1",
    )
    again = scene(bench, action="save_increment", operation_id="inc-1")
    assert not again.is_error, text_of(again)
    assert again.structured_content["replayed"] is True
    assert again.structured_content["version"] == first.structured_content["version"] == 1
    assert bench.sent.calls == []


def test_a_save_whose_reply_was_lost_asks_for_the_same_file_again(
    bench: Bench, project: Path
) -> None:
    current = project / "shot.hip"
    current.write_bytes(b"scene")
    target = str(project / "shot_v001.hip")
    lost = scene(
        bench,
        info(current),
        COMMERCIAL,
        client.BridgeUnreachable("the reply never came"),
        action="save_increment",
        operation_id="inc-2",
    )
    assert lost.structured_content["error"]["code"] == "SESSION_UNREACHABLE"
    # The claim stays: the save may have happened.
    assert Path(f"{target}{outputs.CLAIM_SUFFIX}").exists()
    again = scene(bench, saved(target), action="save_increment", operation_id="inc-2")
    assert not again.is_error, text_of(again)
    assert again.structured_content["version"] == 1
    assert sent_tools(bench) == ["scene.save_as"]
    assert bench.sent.calls[0]["arguments"] == {"path": target}
    assert bench.sent.calls[0]["operation_id"] == "inc-2"


def test_a_save_that_definitely_failed_takes_its_version_back(bench: Bench, project: Path) -> None:
    current = project / "shot.hip"
    current.write_bytes(b"scene")
    target = project / "shot_v001.hip"
    result = scene(
        bench,
        info(current),
        COMMERCIAL,
        refusal("FILE_EXISTS", "a file is already at that path"),
        action="save_increment",
    )
    assert result.structured_content["error"]["code"] == "FILE_EXISTS"
    assert not Path(f"{target}{outputs.CLAIM_SUFFIX}").exists()
    assert list(project.glob("*_run.json")) == []
    with bench.store() as store:
        rows = store._read_all("SELECT version, run_id FROM versions WHERE kind = 'hip'")
    assert [(row["version"], row["run_id"]) for row in rows] == [(1, None)]


def test_a_save_another_server_is_running_under_the_same_id_is_not_run_twice(
    bench: Bench, project: Path
) -> None:
    current = project / "shot.hip"
    current.write_bytes(b"scene")
    digest = store_module.digest_arguments({"action": "save_increment", "session_id": "s-1"})
    with bench.store() as store:
        store.begin_operation("inc-3:increment", digest, owner_pid=os.getppid())
    result = scene(bench, action="save_increment", operation_id="inc-3")
    assert result.structured_content["error"]["code"] == "OUTCOME_UNKNOWN"
    assert bench.sent.calls == []
    with bench.store() as store:
        assert store.latest_version(kind="hip", name="shot", hip_family="shot") == 0
