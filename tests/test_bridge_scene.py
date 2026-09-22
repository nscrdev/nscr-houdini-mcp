"""The scene file tools on the bridge side, against a stand in for Houdini.

What a real load reports, and whether the dependency report finds a missing
asset, a bad node type and a missing file, is checked against a real Houdini
in the integration tests. What is here is the rules: which calls are refused
and why, what a warning is read into, that none of these touch the undo
stack, and that a save never writes over a file.
"""

from __future__ import annotations

import errno
import os
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import receipts, tools
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.errors import BridgeError
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.identity import Identity

WARNING = (
    "Error loading: /somewhere/shot.hip\n"
    "Warning:     Bad node type found: fancy_sop in /obj/geo1.\n"
    "             \n"
    '             "/obj/rig" using incomplete asset definition (full definition not found).'
)


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def hip(tmp_path: Path) -> Path:
    path = tmp_path / "shot_v002.hip"
    path.write_bytes(b"scene")
    return path


def context(scene: Scene, kind: str = "hython") -> tools.ToolContext:
    return tools.ToolContext(hou=scene.module(), kind=kind, session_id="s-1")


def dispatcher(
    scene: Scene, *, identity: Identity | None = None, store_path: Path | None = None
) -> Dispatcher:
    kept = (
        receipts.Receipts(lambda: store_module.Store(store_path), session_id="s-1")
        if store_path is not None
        else None
    )
    return Dispatcher(
        default_registry(),
        lock=threading.Lock(),
        kind="hython",
        session_id="s-1",
        identity=identity,
        receipts=kept,
        hou=scene.module(),
        wait_s=5.0,
        timeout_s=5.0,
    )


def refused(action: Any) -> BridgeError:
    with pytest.raises(BridgeError) as caught:
        action()
    return caught.value


# Section: info


def test_info_names_the_file_and_says_whether_it_has_one(scene: Scene) -> None:
    info = tools.scene_info({}, context(scene))
    assert info["hip_path"] == "/Users/somebody/scenes/example.hip"
    assert info["hip_name"] == "example.hip"
    assert info["untitled"] is False
    assert "dependencies" not in info
    scene.hipFile.new = True
    assert tools.scene_info({}, context(scene))["untitled"] is True


def test_info_adds_the_dependency_report_when_asked(scene: Scene) -> None:
    info = tools.scene_info({"dependencies": True}, context(scene))
    report = info["dependencies"]
    assert report["unresolved_types"] == []
    assert report["missing_hdas"] == []
    assert report["missing_files"] == []
    assert report["truncated"] is False


# Section: open


def test_open_loads_the_file_and_says_it_cannot_be_undone(scene: Scene, hip: Path) -> None:
    scene.node("/obj").createNode("geo", "old")
    reply = dispatcher(scene).dispatch(
        Envelope(tool="scene.open", arguments={"path": str(hip)}, operation_id="op-open")
    )
    assert reply.payload["ok"] is True, reply.payload
    data = reply.payload["data"]
    assert data["hip_path"] == str(hip)
    assert data["hip_name"] == "shot_v002.hip"
    assert data["undo"] == tools.UNDO_NOTE
    assert reply.payload["undo"] == {"label": "open scene", "undoable": False}
    # No undo group was opened around the load, so the stack is as it was.
    assert scene.node("/obj/old") is None
    assert scene.undos.performed == 0


def test_open_moves_the_scene_epoch_once(scene: Scene, hip: Path) -> None:
    identity = Identity(session_id="s-1", kind="hython", hou=scene.module())
    identity.watch()
    reply = dispatcher(scene, identity=identity).dispatch(
        Envelope(tool="scene.open", arguments={"path": str(hip)})
    )
    assert reply.payload["ok"] is True
    assert reply.payload["scene_epoch"] == 1


def test_open_reads_a_load_warning_into_data(scene: Scene, hip: Path) -> None:
    scene.hipFile.load_warning = WARNING
    data = tools.scene_open({"path": str(hip)}, context(scene))
    report = data["dependencies"]
    assert report["unresolved_types"] == [{"type": "fancy_sop", "parent": "/obj/geo1"}]
    assert report["missing_hdas"] == [{"node": "/obj/rig", "type": None}]
    # The line that only repeats which file was loading is left out.
    assert report["load_warnings"][0] == "Bad node type found: fancy_sop in /obj/geo1."
    assert not any("Error loading" in line for line in report["load_warnings"])
    assert data["hip_path"] == str(hip)


def test_open_refuses_a_path_that_is_not_there(scene: Scene, tmp_path: Path) -> None:
    error = refused(lambda: tools.scene_open({"path": str(tmp_path / "gone.hip")}, context(scene)))
    assert error.code == "FILE_NOT_FOUND"
    assert scene.hipFile.path() == "/Users/somebody/scenes/example.hip"


def test_open_refuses_a_file_that_is_not_a_scene(scene: Scene, tmp_path: Path) -> None:
    other = tmp_path / "notes.txt"
    other.write_text("x")
    error = refused(lambda: tools.scene_open({"path": str(other)}, context(scene)))
    assert error.code == "BAD_ARGUMENTS"


def test_open_in_a_gui_with_unsaved_changes_asks_to_discard_them(scene: Scene, hip: Path) -> None:
    scene.hipFile.unsaved = True
    error = refused(lambda: tools.scene_open({"path": str(hip)}, context(scene, "gui")))
    assert error.code == "UNSAVED_CHANGES"
    assert "discard_unsaved" in (error.hint or "")
    # Nothing was loaded, so nothing was lost.
    assert scene.hipFile.path() == "/Users/somebody/scenes/example.hip"

    data = tools.scene_open({"path": str(hip), "discard_unsaved": True}, context(scene, "gui"))
    assert data["discarded_unsaved"] is True
    assert scene.hipFile.path() == str(hip)


def test_open_in_a_gui_with_nothing_unsaved_just_loads(scene: Scene, hip: Path) -> None:
    scene.hipFile.unsaved = False
    data = tools.scene_open({"path": str(hip)}, context(scene, "gui"))
    assert data["discarded_unsaved"] is False


def test_a_worker_is_never_refused_for_unsaved_changes(scene: Scene, hip: Path) -> None:
    # A headless session says it has unsaved changes whatever it has.
    scene.hipFile.unsaved = True
    data = tools.scene_open({"path": str(hip)}, context(scene))
    assert data["discarded_unsaved"] is None
    assert scene.hipFile.path() == str(hip)


# Section: save


def test_save_refuses_a_scene_that_has_no_file(scene: Scene) -> None:
    scene.hipFile.new = True
    error = refused(lambda: tools.scene_save({}, context(scene)))
    assert error.code == "SCENE_UNTITLED"
    assert "save_increment" in (error.hint or "")
    assert scene.hipFile.saved == []


def test_save_writes_over_its_own_file(scene: Scene) -> None:
    data = tools.scene_save({}, context(scene))
    assert scene.hipFile.saved == ["/Users/somebody/scenes/example.hip"]
    assert data["hip_path"] == "/Users/somebody/scenes/example.hip"
    assert data["undo"] == tools.UNDO_NOTE


def test_save_as_never_writes_over_a_file(scene: Scene, hip: Path) -> None:
    error = refused(lambda: tools.scene_save_as({"path": str(hip)}, context(scene)))
    assert error.code == "FILE_EXISTS"
    assert scene.hipFile.saved == []
    assert hip.read_bytes() == b"scene"


def test_save_as_refuses_a_folder_that_is_not_there(scene: Scene, tmp_path: Path) -> None:
    path = tmp_path / "missing" / "shot_v001.hip"
    error = refused(lambda: tools.scene_save_as({"path": str(path)}, context(scene)))
    assert error.code == "FILE_NOT_FOUND"


def test_save_as_refuses_a_relative_path(scene: Scene) -> None:
    error = refused(lambda: tools.scene_save_as({"path": "shot_v001.hip"}, context(scene)))
    assert error.code == "BAD_ARGUMENTS"


def test_save_as_moves_the_session_to_the_new_file(scene: Scene, tmp_path: Path) -> None:
    scene.hipFile.writes_files = True
    path = tmp_path / "shot_v003.hip"
    data = tools.scene_save_as({"path": str(path)}, context(scene))
    # Written under a private name first, then published under the one asked for.
    [written] = scene.hipFile.saved
    assert written != str(path)
    assert Path(written).parent == tmp_path
    assert not Path(written).exists()
    assert path.read_bytes() == b"a scene"
    assert scene.hipFile.path() == str(path)
    assert data["hip_path"] == str(path)
    assert data["bytes"] == len(b"a scene")
    assert data["warnings"] == []


def test_a_file_that_appears_during_the_save_is_not_written_over(
    scene: Scene, tmp_path: Path
) -> None:
    scene.hipFile.writes_files = True
    path = tmp_path / "shot_v003.hip"
    save = scene.hipFile.save

    def racing(name: str | None = None, **rest: Any) -> None:
        save(name, **rest)
        path.write_bytes(b"somebody else's scene")

    scene.hipFile.save = racing  # type: ignore[method-assign]
    error = refused(lambda: tools.scene_save_as({"path": str(path)}, context(scene)))
    assert error.code == "FILE_EXISTS"
    assert path.read_bytes() == b"somebody else's scene"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["shot_v003.hip"]
    # The session holds the file it held before.
    assert scene.hipFile.path() == "/Users/somebody/scenes/example.hip"


def failing_link(number: int, winerror: int | None = None) -> Any:
    def link(source: str, target: str) -> None:
        error = OSError(number, os.strerror(number))
        if winerror is not None:
            error.winerror = winerror  # type: ignore[attr-defined]
        raise error

    return link


@pytest.mark.parametrize("name", ["EPERM", "ENOTSUP", "EOPNOTSUPP", "ENOSYS", "EXDEV"])
def test_a_folder_that_cannot_link_publishes_by_a_rename_and_says_so(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    if not hasattr(errno, name):
        pytest.skip(f"{name} is not an error this system has")
    monkeypatch.setattr(tools.os, "link", failing_link(getattr(errno, name)))
    scene.hipFile.writes_files = True
    path = tmp_path / "shot_v003.hip"
    data = tools.scene_save_as({"path": str(path)}, context(scene))
    assert path.read_bytes() == b"a scene"
    assert data["warnings"] and "cannot link" in data["warnings"][0]
    assert sorted(item.name for item in tmp_path.iterdir()) == ["shot_v003.hip"]


@pytest.mark.parametrize("name", ["ENOSPC", "EACCES", "ENOENT", "EIO"])
def test_any_other_link_failure_is_the_saves_own_failure(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.setattr(tools.os, "link", failing_link(getattr(errno, name)))
    scene.hipFile.writes_files = True
    path = tmp_path / "shot_v003.hip"
    error = refused(lambda: tools.scene_save_as({"path": str(path)}, context(scene)))
    assert error.code == "TOOL_FAILED"
    assert error.details["errno"] == name
    assert list(tmp_path.iterdir()) == []
    assert scene.hipFile.path() == "/Users/somebody/scenes/example.hip"


def test_windows_says_it_cannot_link_in_its_own_numbers() -> None:
    unsupported = OSError(errno.EINVAL, "not supported")
    unsupported.winerror = 50  # type: ignore[attr-defined]
    assert tools.links_unsupported(unsupported) is True
    assert tools.links_unsupported(OSError(errno.ENOSPC, "full")) is False


def test_the_rename_never_writes_over_a_file_that_appeared(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools.os, "link", failing_link(errno.EPERM))
    scene.hipFile.writes_files = True
    path = tmp_path / "shot_v003.hip"
    save = scene.hipFile.save

    def racing(name: str | None = None, **rest: Any) -> None:
        save(name, **rest)
        path.write_bytes(b"somebody else's scene")

    scene.hipFile.save = racing  # type: ignore[method-assign]
    error = refused(lambda: tools.scene_save_as({"path": str(path)}, context(scene)))
    assert error.code == "FILE_EXISTS"
    assert path.read_bytes() == b"somebody else's scene"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["shot_v003.hip"]


def test_a_save_that_fails_part_way_leaves_no_file_and_the_name_it_had(
    scene: Scene, tmp_path: Path
) -> None:
    scene.hipFile.writes_files = True
    scene.hipFile.save_error = OSError(errno.ENOSPC, "no space left")
    with pytest.raises(OSError):
        tools.scene_save_as({"path": str(tmp_path / "shot_v003.hip")}, context(scene))
    assert list(tmp_path.iterdir()) == []
    assert scene.hipFile.path() == "/Users/somebody/scenes/example.hip"


def test_an_untitled_scene_whose_save_failed_is_untitled_again(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools.os, "link", failing_link(errno.EIO))
    scene.hipFile.writes_files = True
    scene.hipFile.new = True
    target = str(tmp_path / "untitled_v001.hip")
    refused(lambda: tools.scene_save_as({"path": target}, context(scene)))
    assert scene.hipFile.isNewFile() is True
    assert ".part" not in scene.hipFile.path()
    assert tools.scene_info({}, context(scene))["untitled"] is True


def test_the_private_save_stays_out_of_the_recent_files(scene: Scene, tmp_path: Path) -> None:
    scene.hipFile.writes_files = True
    tools.scene_save_as({"path": str(tmp_path / "shot_v003.hip")}, context(scene))
    assert scene.hipFile.recent == [False]


def test_private_files_left_by_an_attempt_that_died_are_swept(scene: Scene, tmp_path: Path) -> None:
    scene.hipFile.writes_files = True
    old = time.time() - 2 * tools.PRIVATE_KEEP_S
    stale = tmp_path / "shot_v002.part4242.hip"
    fresh = tmp_path / "shot_v003.part4343.hip"
    other = tmp_path / "prop_v001.part4242.hip"
    for left in (stale, fresh, other):
        left.write_bytes(b"left")
        os.utime(left, (old, old))
    os.utime(fresh, None)
    tools.scene_save_as({"path": str(tmp_path / "shot_v004.hip")}, context(scene))
    assert not stale.exists()
    assert fresh.exists()
    assert other.exists()


def licensed(scene: Scene, name: str) -> Any:
    module = scene.module()
    module.licenseCategory = lambda: SimpleNamespace(name=lambda: name)
    return tools.ToolContext(hou=module, kind="hython", session_id="s-1")


def test_a_save_refuses_a_name_the_license_would_not_write(scene: Scene, tmp_path: Path) -> None:
    scene.hipFile.writes_files = True
    error = refused(
        lambda: tools.scene_save_as(
            {"path": str(tmp_path / "shot_v001.hip")}, licensed(scene, "Apprentice")
        )
    )
    assert error.code == "BAD_ARGUMENTS"
    assert error.details["license_suffix"] == ".hipnc"
    data = tools.scene_save_as(
        {"path": str(tmp_path / "shot_v001.hipnc")}, licensed(scene, "Apprentice")
    )
    assert data["hip_path"].endswith("shot_v001.hipnc")
    assert tools.license_suffix(licensed(scene, "Indie").hou) == ".hiplc"
    assert tools.license_suffix(licensed(scene, "Commercial").hou) is None


def test_a_gui_that_will_not_say_whether_it_has_changes_is_taken_to_have_them(
    scene: Scene, hip: Path
) -> None:
    scene.hipFile.unsaved = None
    error = refused(lambda: tools.scene_open({"path": str(hip)}, context(scene, "gui")))
    assert error.code == "UNSAVED_CHANGES"
    assert scene.hipFile.path() == "/Users/somebody/scenes/example.hip"


def test_a_save_sent_twice_under_one_id_is_done_once(scene: Scene, tmp_path: Path) -> None:
    scene.hipFile.writes_files = True
    running = dispatcher(scene, store_path=tmp_path / "coord.sqlite")
    path = str(tmp_path / "shot_v003.hip")
    first = running.dispatch(
        Envelope(tool="scene.save_as", arguments={"path": path}, operation_id="op-save")
    )
    # The file is there now, so a second real save would be refused: the
    # receipt is what answers the same id with the first answer.
    second = running.dispatch(
        Envelope(tool="scene.save_as", arguments={"path": path}, operation_id="op-save")
    )
    assert first.payload["ok"] is True, first.payload
    assert second.payload["ok"] is True
    assert second.payload["data"] == first.payload["data"]
    assert len(scene.hipFile.saved) == 1
