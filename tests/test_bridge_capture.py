"""`capture.image` on the bridge side, against a stand in for Houdini.

The routes are tried in their order here, the restore after a viewport
capture is checked value by value, and so is what a render node capture
leaves behind: nothing. The stand in draws its pictures with Pillow, so every
file checked is a real PNG. What a real Houdini draws is in the integration
test; what a real user interface does with the viewport and pane routes is
checked by hand with a person present, and nothing here stands for that.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from fake_hou import (
    Desktop,
    FlipbookSettings,
    NetworkEditorTab,
    OperationFailed,
    Pane,
    Rect,
    Scene,
    SceneViewerTab,
    Tab,
    Vector3,
    Window,
)
from nscr_houdini_mcp import outputs
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import capture, receipts, tools
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.errors import BridgeError
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.identity import Identity

TYPES = (
    "geo",
    "null",
    "cam",
    "box",
    "flipbook",
    "copnet",
    "fractalnoise",
    "img",
    "file",
    "subnet",
    "hlight::2.0",
    "control",
    "instance",
    "dopnet",
    "pathcv",
    "path",
    "handle",
    "muscle",
    "merge",
    "convert",
)
# The box every geometry node in the stand in draws, as framing reads it.
UNIT = ((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


@pytest.fixture
def scene(tmp_path: Path) -> Iterator[Scene]:
    made = Scene(types=TYPES)
    made.hipFile.setName(str(tmp_path / "shot.hip"))
    geo = made.node("/obj").createNode("geo", "boxgeo")
    geo.createNode("box", "box1").setDisplayFlag(True)
    made.undos.labels.clear()
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture(autouse=True)
def no_named_screen_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each check says which Qt screen plugin it has; the shell's is not one of them."""
    monkeypatch.delenv(capture.QT_PLATFORM_ENV_VAR, raising=False)


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


class Run:
    """One call's context, with its progress notes and its cancel flag."""

    def __init__(self, scene: Scene, home: Path, kind: str) -> None:
        self.notes: list[dict[str, Any]] = []
        self.cancel = threading.Event()
        store_path = home / store_module.STORE_FILE_NAME
        self.context = tools.ToolContext(
            hou=scene.module(),
            kind=kind,
            session_id="s-1",
            home=home,
            open_store=lambda: store_module.Store(store_path),
            progress=self.notes.append,
            cancel=self.cancel,
        )


def take(scene: Scene, home: Path, *, kind: str = "hython", **arguments: Any) -> dict[str, Any]:
    return capture.capture_image(arguments, Run(scene, home, kind).context)


def refused(scene: Scene, home: Path, *, kind: str = "hython", **arguments: Any) -> BridgeError:
    with pytest.raises(BridgeError) as caught:
        take(scene, home, kind=kind, **arguments)
    return caught.value


def picture(path: str) -> Image.Image:
    with Image.open(path) as opened:
        return opened.copy()


def names(scene: Scene, network: str) -> list[str]:
    return [child.name() for child in scene.node(network)._children]


def desktop(scene: Scene, *tabs: Any) -> Pane:
    pane = Pane()
    for tab in tabs:
        pane.add(tab)
    made = Desktop()
    made.tabs = list(tabs)
    scene.ui.desktop = made
    return pane


# Section: a session with no user interface


def test_the_viewport_in_hython_is_a_fitted_camera_through_the_flipbook_rop(
    scene: Scene, home: Path, tmp_path: Path
) -> None:
    said = take(scene, home, resolution=[640, 360])
    [shot] = said["views"]
    assert shot["route"] == capture.FLIPBOOK_ROP
    assert "framing_unverified" not in shot
    assert shot["camera"]["kind"] == "fitted"
    assert shot["camera"]["view"] == "persp"
    [path] = shot["files"]
    folder = Path(path).parent
    assert folder.parent == tmp_path / ".agent" / "captures"
    assert Path(path).name.endswith(f"_viewport_{shot['run_id']}.png")
    image = picture(path)
    assert image.size == (640, 360)
    assert image.getextrema()[3] == (0, 255)
    [seen] = scene.capture.seen
    assert seen["camera"].startswith("/obj/nscr_capture_cam")
    assert seen["r"] == (-25.0, 45.0, 0.0)
    assert seen["sopsource"] == "display"
    assert seen["trange"] == "normal"
    assert seen["undo_enabled"] is False
    # Nothing the capture made is left, and none of it is on the undo stack.
    assert names(scene, "/out") == []
    assert names(scene, "/obj") == ["boxgeo"]
    assert scene.undos.undoLabels() == []
    assert said["sheet"] is None
    assert said["unsaved_hip"] is False


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("kind", ["hython", "gui"])
def test_only_windows_workers_release_rop_materials_before_removing_the_node(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch, platform: str, kind: str
) -> None:
    monkeypatch.setattr(capture, "sys", SimpleNamespace(platform=platform))
    if kind == "gui":
        gui_without_viewer(scene)
    said = take(scene, home, kind=kind, frames=[1, 3, 1], resolution=[64, 64])
    assert [seen["frame"] for seen in scene.capture.seen] == [1, 2, 3]
    if platform == "win32" and kind == "hython":
        [cleanup] = scene.capture.cleanups
        assert cleanup["size"] == (1, 1)
        assert cleanup["frame"] == 3
        assert cleanup["vobjects"] == cleanup["forceobjects"] == ""
        assert cleanup["displayed"] == {}
        assert cleanup["undo_enabled"] is False
        assert "$F4" not in cleanup["picture"]
        assert not Path(cleanup["picture"]).exists()
    else:
        assert scene.capture.cleanups == []
    assert len(said["views"][0]["files"]) == 3
    assert names(scene, "/out") == []
    assert names(scene, "/obj") == ["boxgeo"]
    assert scene.undos.undoLabels() == []


def test_a_material_release_failure_does_not_prevent_other_cleanup(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "sys", SimpleNamespace(platform="win32"))

    def fail_cleanup(frame: float) -> None:
        if scene.capture.cleanups:
            raise OperationFailed("material release failed")

    scene.capture.after_frame = fail_cleanup
    error = refused(scene, home)
    assert error.code == "CLEANUP_FAILED"
    assert "material release failed" in str(error.details)
    [cleanup] = scene.capture.cleanups
    assert not Path(cleanup["picture"]).exists()
    assert names(scene, "/out") == []
    assert names(scene, "/obj") == ["boxgeo"]


def test_material_release_is_attempted_after_a_render_raises(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "sys", SimpleNamespace(platform="win32"))
    scene.capture.fail_at_frame = 1
    refused(scene, home, frame=1)
    [cleanup] = scene.capture.cleanups
    assert cleanup["frame"] == 1
    assert not Path(cleanup["picture"]).exists()
    assert names(scene, "/out") == []
    assert names(scene, "/obj") == ["boxgeo"]


def camera_state(camera: Any) -> list[tuple[Any, ...]]:
    return [
        (parm.name(), parm.value, parm._expression, tuple(parm.keys), parm.locked)
        for parm in camera.parms()
    ]


def test_a_named_camera_is_followed_and_never_written(scene: Scene, home: Path) -> None:
    shot_cam = scene.node("/obj").createNode("cam", "shotcam")
    shot_cam.parm("focal").set(35.0)
    shot_cam.parm("tx").setKeyframes([(1.0, "3", "hscript")])
    shot_cam.parm("winx").locked = True
    scene.undos.labels.clear()
    before = camera_state(shot_cam)
    said = take(scene, home, camera="/obj/shotcam")
    [seen] = scene.capture.seen
    assert seen["camera"].startswith("/obj/nscr_capture_cam")
    assert seen["follows"] == "/obj/shotcam"
    assert seen["focal"] == 35.0
    assert said["views"][0]["camera"] == {"kind": "node", "path": "/obj/shotcam"}
    assert camera_state(shot_cam) == before
    assert names(scene, "/obj") == ["boxgeo", "shotcam"]
    assert scene.undos.undoLabels() == []


def test_quad_writes_four_views_and_names_a_sheet(scene: Scene, home: Path) -> None:
    said = take(scene, home, views="quad", resolution=[320, 180])
    assert [view["view"] for view in said["views"]] == ["persp", "top", "front", "right"]
    rotations = [seen["r"] for seen in scene.capture.seen]
    assert rotations == [(-25.0, 45.0, 0.0), (-90.0, 0.0, 0.0), (-0.0, 0.0, 0.0), (-0.0, 90.0, 0.0)]
    projections = [seen["projection"] for seen in scene.capture.seen]
    assert projections == ["perspective", "ortho", "ortho", "ortho"]
    for view in said["views"]:
        assert Path(view["files"][0]).is_file()
        assert f"_viewport_{view['view']}_" in Path(view["files"][0]).name
    assert "_viewport_sheet_" in said["sheet"]["path"]
    # The sheet is the server's to draw; the session only names it.
    assert not Path(said["sheet"]["path"]).exists()
    assert names(scene, "/obj") == ["boxgeo"]


def test_turntable_orbits_a_quarter_turn_at_a_time(scene: Scene, home: Path) -> None:
    said = take(scene, home, views="turntable4", camera={"orbit": 10, "elevation": 40})
    assert [view["view"] for view in said["views"]] == ["orbit0", "orbit90", "orbit180", "orbit270"]
    assert [seen["r"] for seen in scene.capture.seen] == [
        (-40.0, 0.0, 0.0),
        (-40.0, 90.0, 0.0),
        (-40.0, 180.0, 0.0),
        (-40.0, 270.0, 0.0),
    ]


def test_a_node_is_shown_alone_and_the_display_flag_goes_back(scene: Scene, home: Path) -> None:
    geo = scene.node("/obj/boxgeo")
    other = geo.createNode("box", "box2")
    other.bounds = ((10.0, 10.0, 10.0), (12.0, 12.0, 12.0))
    scene.node("/obj").createNode("geo", "elsewhere").createNode("box").setDisplayFlag(True)
    scene.undos.labels.clear()
    said = take(scene, home, source="node", path="/obj/boxgeo/box2")
    [seen] = scene.capture.seen
    assert seen["vobjects"] == "/obj/boxgeo"
    assert seen["forceobjects"] == "/obj/boxgeo"
    assert seen["displayed"] == {"/obj/boxgeo": "/obj/boxgeo/box2"}
    # The camera was fitted to the node's own box, far from the origin.
    assert all(value > 5 for value in seen["t"])
    assert said["views"][0]["camera"]["target"] == "/obj/boxgeo/box2"
    assert geo.displayNode().path() == "/obj/boxgeo/box1"
    assert scene.undos.undoLabels() == []
    assert names(scene, "/obj") == ["boxgeo", "elsewhere"]


def test_a_sequence_writes_a_numbered_file_a_frame_and_reports_progress(
    scene: Scene, home: Path
) -> None:
    run = Run(scene, home, "hython")
    said = capture.capture_image({"frames": [1, 3, 1]}, run.context)
    [shot] = said["views"]
    assert said["sequence"] is True
    assert shot["frames"] == [1.0, 2.0, 3.0]
    assert [Path(item).name[-8:] for item in shot["files"]] == [
        "0001.png",
        "0002.png",
        "0003.png",
    ]
    assert all(Path(item).is_file() for item in shot["files"])
    assert [note["done"] for note in run.notes] == [1, 2, 3]
    assert run.notes[-1]["total"] == 3


def test_a_cancelled_sequence_stops_between_frames(scene: Scene, home: Path) -> None:
    run = Run(scene, home, "hython")
    context = run.context

    def note(said: dict[str, Any]) -> None:
        run.notes.append(said)
        run.cancel.set()

    context = tools.ToolContext(**{**context.__dict__, "progress": note})
    said = capture.capture_image({"frames": [1, 5, 1]}, context)
    [shot] = said["views"]
    assert shot["frames"] == [1.0]
    assert said["stopped_early"] is True
    assert names(scene, "/out") == []


@pytest.mark.parametrize(("source", "word"), [("network", "hou_inspect"), ("pane", "interface")])
def test_a_pane_in_hython_is_ui_unavailable(
    scene: Scene, home: Path, source: str, word: str
) -> None:
    error = refused(scene, home, source=source, path="panetab1" if source == "pane" else None)
    assert error.code == "UI_UNAVAILABLE"
    assert word in error.hint
    assert error.details["tried"] == []


def test_a_route_that_writes_nothing_is_capture_empty(scene: Scene, home: Path) -> None:
    scene.capture.writes = False
    error = refused(scene, home)
    assert error.code == "CAPTURE_EMPTY"
    assert error.details["tried"] == [{"route": capture.FLIPBOOK_ROP, "reason": capture.NO_FILE}]
    assert names(scene, "/out") == []


def test_no_flipbook_type_means_no_route(scene: Scene, home: Path) -> None:
    scene.types = tuple(name for name in TYPES if name != "flipbook")
    error = refused(scene, home)
    assert error.code == "UI_UNAVAILABLE"
    assert error.details["tried"][0]["route"] == capture.FLIPBOOK_ROP


def test_an_empty_scene_is_framed_at_the_origin_with_a_warning(scene: Scene, home: Path) -> None:
    scene.node("/obj/boxgeo").hidden = True
    said = take(scene, home)
    assert any("nothing to frame" in item for item in said["warnings"])


# Section: the viewport routes, in a session with a user interface


def viewer_scene(scene: Scene) -> SceneViewerTab:
    viewer = SceneViewerTab(scene)
    desktop(scene, viewer)
    return viewer


def view_state(viewport: Any) -> tuple[Any, ...]:
    return (
        viewport.type(),
        viewport.camera(),
        viewport._default.state(),
        viewport.viewTransform().asTuple(),
        viewport.shading(),
    )


def test_the_showing_viewport_is_flipbooked_with_settings_of_its_own(
    scene: Scene, home: Path
) -> None:
    viewer = viewer_scene(scene)
    said = take(scene, home, kind="gui", resolution=[800, 450], frame=12)
    [shot] = said["views"]
    assert shot["route"] == capture.VIEWPORT
    assert shot["camera"] == {"kind": "viewport"}
    [seen] = scene.capture.seen
    settings = seen["settings"]
    assert settings["outputToMPlay"] is False
    assert settings["beautyPassOnly"] is True
    assert settings["useResolution"] is True
    assert settings["resolution"] == (800, 450)
    assert settings["frameRange"] == (12.0, 12.0)
    assert settings["output"] == shot["files"][0]
    # The viewer's own settings are not the ones changed.
    assert viewer.settings.values["outputToMPlay"] is True
    assert picture(shot["files"][0]).size == (800, 450)
    # Nothing was framed: the view is what the artist sees.
    assert viewer.viewport.framed == []


def test_guides_leave_the_beauty_pass_off(scene: Scene, home: Path) -> None:
    viewer_scene(scene)
    take(scene, home, kind="gui", guides=True)
    assert scene.capture.seen[0]["settings"]["beautyPassOnly"] is False


def test_a_view_applied_for_the_capture_is_put_back_exactly(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    viewport = viewer.viewport
    viewport._default.setTranslation((1.5, 2.5, 3.5))
    viewport._default.setOrthoWidth(3.25)
    before = view_state(viewport)
    take(scene, home, kind="gui", camera="top", display="wire")
    [seen] = scene.capture.seen
    assert seen["type"] == "Top"
    assert seen["shading"] == {"SceneObject": "Wire", "DisplayModel": "Wire"}
    assert viewport.framed == [UNIT]
    after = view_state(viewport)
    assert after == before
    translation, rotation, pivot, ortho_width = viewport._default.state()
    assert translation == (1.5, 2.5, 3.5)
    assert ortho_width == 3.25
    assert pivot == (0.0, 0.0, 0.0)
    assert rotation == (0.0, 0.0, 0.0)


def test_a_camera_node_is_looked_through_and_let_go_after(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    scene.node("/obj").createNode("cam", "shotcam")
    before = view_state(viewer.viewport)
    said = take(scene, home, kind="gui", camera="/obj/shotcam", frame_target="all")
    [seen] = scene.capture.seen
    assert seen["camera"] == "/obj/shotcam"
    assert viewer.viewport.framed == []
    assert any("frame_target was not applied" in item for item in said["warnings"])
    assert view_state(viewer.viewport) == before


def test_an_orbit_turns_the_view_and_frames_the_target(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    before = view_state(viewer.viewport)
    take(
        scene,
        home,
        kind="gui",
        camera={"orbit": 90, "elevation": 30},
        frame_target="/obj/boxgeo",
    )
    [seen] = scene.capture.seen
    assert seen["view"][1] == (-30.0, 90.0, 0.0)
    assert viewer.viewport.framed == [((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))]
    assert view_state(viewer.viewport) == before


def test_a_failed_flipbook_puts_the_view_back_and_moves_on(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    before = view_state(viewer.viewport)
    scene.capture.flipbook_error = OperationFailed("the flipbook failed")
    said = take(scene, home, kind="gui", camera="front", display="matcap")
    [shot] = said["views"]
    assert view_state(viewer.viewport) == before
    assert shot["route"] == capture.FLIPBOOK_ROP
    assert shot["framing_unverified"] is True
    assert shot["tried"][0]["route"] == capture.VIEWPORT
    assert shot["tried"][0]["reason"].startswith("OperationFailed")
    # The route that ran is the render node, with the view asked for fitted.
    [seen] = scene.capture.seen
    assert seen["route"] == "rop"
    assert seen["shadingmode"] == "matcap"


def test_a_hidden_viewer_is_shown_for_the_capture_and_hidden_again(
    scene: Scene, home: Path
) -> None:
    viewer = SceneViewerTab(scene, "viewer")
    other = Tab(scene, "Parm", "parms")
    pane = desktop(scene, other, viewer)
    assert pane.currentTab() is other
    said = take(scene, home, kind="gui")
    [shot] = said["views"]
    assert shot["route"] == capture.VIEWPORT_TAB
    assert shot["tried"] == [{"route": capture.VIEWPORT, "reason": "no Scene Viewer is showing"}]
    assert scene.capture.seen[0]["current"] is True
    assert pane.currentTab() is other


def test_no_viewer_at_all_falls_to_the_render_node_with_framing_unverified(
    scene: Scene, home: Path
) -> None:
    desktop(scene)
    said = take(scene, home, kind="gui")
    [shot] = said["views"]
    assert shot["route"] == capture.FLIPBOOK_ROP
    assert shot["framing_unverified"] is True
    assert [item["route"] for item in shot["tried"]] == [capture.VIEWPORT, capture.VIEWPORT_TAB]


# Section: the network editor and panes, grabbed through Qt


def editor_scene(scene: Scene, ratio: float = 2.0) -> NetworkEditorTab:
    editor = NetworkEditorTab(scene)
    window = Window(100, 50, 800, 600, ratio)
    editor.window = window
    editor.geometry = Rect(300, 250, 400, 200)
    window.painted.append((editor.geometry, (255, 0, 0, 255)))
    desktop(scene, editor)
    return editor


def test_the_network_editor_is_its_own_window_cropped_to_the_pane(scene: Scene, home: Path) -> None:
    editor = editor_scene(scene)
    said = take(scene, home, kind="gui", source="network")
    [shot] = said["views"]
    assert shot["route"] == capture.NETWORK_GRAB
    assert shot["native"] == [800, 400]
    image = picture(shot["files"][0])
    assert image.size == (800, 400)
    assert image.convert("RGB").getextrema() == ((255, 255), (0, 0), (0, 0))
    assert editor.window.grabs == 1


def test_a_network_asked_for_is_shown_then_the_editor_goes_back(scene: Scene, home: Path) -> None:
    editor = editor_scene(scene, ratio=1.0)
    said = take(scene, home, kind="gui", source="network", path="/obj/boxgeo")
    [shot] = said["views"]
    assert shot["framing_unverified"] is True
    assert editor.pwd().path() == "/obj"


def test_a_pane_is_found_by_name(scene: Scene, home: Path) -> None:
    editor_scene(scene, ratio=1.0)
    said = take(scene, home, kind="gui", source="pane", path="panetab2")
    assert said["views"][0]["route"] == capture.PANE_GRAB
    error = refused(scene, home, kind="gui", source="pane", path="panetab9")
    assert error.code == "BAD_ARGUMENTS"
    assert "panetab2" in error.details["did_you_mean"]


def test_a_pane_with_no_name_is_the_scene_viewer(scene: Scene, home: Path) -> None:
    viewer = SceneViewerTab(scene)
    viewer.window = Window(0, 0, 400, 300, 1.0)
    viewer.geometry = Rect(0, 0, 400, 300)
    viewer.window.painted.append((Rect(10, 10, 20, 20), (0, 0, 255, 255)))
    editor = NetworkEditorTab(scene)
    desktop(scene, editor, viewer)
    said = take(scene, home, kind="gui", source="pane")
    [shot] = said["views"]
    assert shot["route"] == capture.PANE_GRAB
    assert viewer.window.grabs == 1
    assert editor.isCurrentTab()


def test_a_pane_with_no_name_and_no_scene_viewer_is_refused(scene: Scene, home: Path) -> None:
    editor_scene(scene)
    error = refused(scene, home, kind="gui", source="pane")
    assert error.code == "UI_UNAVAILABLE"
    assert "no Scene Viewer" in error.details["tried"][0]["reason"]


@pytest.mark.parametrize(
    ("pane", "origin", "size", "ratio", "box"),
    [
        ((300, 250, 400, 200), (100, 50), (1600, 1200), 2.0, (400, 400, 1200, 800)),
        ((100, 50, 800, 600), (100, 50), (800, 600), 1.0, (0, 0, 800, 600)),
        # A pane that runs off the window is cut at the window's edge.
        ((50, 40, 200, 100), (100, 50), (800, 600), 1.0, (0, 0, 150, 90)),
        ((10, 10, 101, 51), (0, 0), (300, 200), 1.5, (15, 15, 167, 92)),
    ],
)
def test_crop_maths(
    pane: tuple[int, ...], origin: tuple[int, int], size: tuple[int, int], ratio: float, box: Any
) -> None:
    assert capture.crop_box(pane, origin, size, ratio) == box


def test_a_pane_off_its_window_has_no_crop() -> None:
    assert capture.crop_box((900, 700, 50, 50), (0, 0), (800, 600), 1.0) is None


# Section: COP images


def test_a_copernicus_layer_is_written_the_right_way_up(scene: Scene, home: Path) -> None:
    net = scene.node("/obj").createNode("copnet", "cops")
    net.createNode("fractalnoise", "noise")
    said = take(scene, home, source="cop", path="/obj/cops/noise", frame=24)
    [shot] = said["views"]
    assert shot["route"] == capture.COP_LAYER
    assert shot["native"] == [64, 32]
    image = picture(shot["files"][0])
    assert image.size == (64, 32)
    assert image.getpixel((0, 0)) == (255, 255, 255, 255)
    assert image.getpixel((0, 31)) == (0, 0, 0, 255)
    assert scene.capture.cop_frames == [24.0]


def test_an_older_cop_saves_its_own_image(scene: Scene, home: Path) -> None:
    img = scene.root.createNode("img", "img")
    img.createNode("file", "plate")
    said = take(scene, home, source="cop", path="/img/plate")
    [shot] = said["views"]
    assert shot["route"] == capture.COP2_SAVE
    assert shot["tried"] == [
        {"route": capture.COP_LAYER, "reason": "the node is not a Copernicus COP"}
    ]
    assert Path(shot["files"][0]).is_file()


# Section: what is refused before anything is made


@pytest.mark.parametrize(
    ("arguments", "code", "argument"),
    [
        ({"camera": "/obj/boxgeo"}, "BAD_ARGUMENTS", "camera"),
        ({"camera": "/obj/nothing"}, "NODE_NOT_FOUND", "camera"),
        ({"camera": "side"}, "BAD_ARGUMENTS", "camera"),
        ({"camera": {"orbit": 1, "tilt": 2}}, "BAD_ARGUMENTS", "camera"),
        ({"frame_target": "/obj/gone"}, "NODE_NOT_FOUND", "frame_target"),
        ({"frames": [1, 4, 1], "views": "quad"}, "BAD_ARGUMENTS", "views"),
        ({"frames": [1, 4, 1], "frame": 2}, "BAD_ARGUMENTS", "frames"),
        ({"frames": [1.5, 4, 1]}, "BAD_ARGUMENTS", "frames"),
        ({"source": "cop", "path": "/obj/boxgeo"}, "BAD_ARGUMENTS", "path"),
        ({"source": "cop", "path": "/obj/cops", "camera": "top"}, "BAD_ARGUMENTS", "camera"),
        ({"source": "node"}, "BAD_ARGUMENTS", "path"),
        ({"source": "node", "path": "/out"}, "BAD_ARGUMENTS", "path"),
        ({"resolution": [0, 10]}, "BAD_ARGUMENTS", "resolution"),
    ],
)
def test_what_is_refused(
    scene: Scene, home: Path, arguments: dict[str, Any], code: str, argument: str
) -> None:
    error = refused(scene, home, **arguments)
    assert error.code == code
    assert error.details["argument"] == argument
    assert scene.capture.seen == []


# Section: the numbers and the file


def project(fit: dict[str, Any], point: tuple[float, ...], aspect: float) -> tuple[float, float]:
    """Where a point lands on a fitted camera's frame, from -1 to 1 across and up."""
    rx, ry, _ = (math.radians(value) for value in fit["r"])
    offset = [p - t for p, t in zip(point, fit["t"], strict=True)]
    # Undo the orbit about Y, then the tilt about X.
    x = offset[0] * math.cos(-ry) + offset[2] * math.sin(-ry)
    z = -offset[0] * math.sin(-ry) + offset[2] * math.cos(-ry)
    y = offset[1]
    y, z = y * math.cos(-rx) - z * math.sin(-rx), y * math.sin(-rx) + z * math.cos(-rx)
    assert z < 0, "the point is behind the camera"
    if fit["orthowidth"]:
        return x / (fit["orthowidth"] / 2.0), y / (fit["orthowidth"] / 2.0 / aspect)
    tan_across = (41.4214 / 2.0) / 50.0
    return x / (-z * tan_across), y / (-z * tan_across / aspect)


@pytest.mark.parametrize("view", ["persp", "top", "front", "right"])
def test_a_fitted_camera_sees_every_corner(view: str) -> None:
    bounds = ((-1.0, 0.0, -3.0), (5.0, 2.0, 1.0))
    orbit, elevation, ortho = capture.FITTED[view]
    fit = capture.fit_camera(
        bounds,
        orbit=orbit,
        elevation=elevation,
        ortho=ortho,
        aspect=16 / 9,
        focal=50.0,
        aperture=41.4214,
    )
    reach = 0.0
    for corner in capture._corners(*bounds):
        x, y = project(fit, corner, 16 / 9)
        assert abs(x) <= 1.0 and abs(y) <= 1.0, (view, corner, x, y)
        reach = max(reach, abs(x), abs(y))
    # And it is not so far back that the box is a speck.
    assert reach > 0.5


def test_write_png_turns_a_bottom_up_image_the_right_way_up(tmp_path: Path) -> None:
    rows = [bytes((value, value, value, 255)) * 3 for value in (10, 20)]
    path = tmp_path / "out.png"
    capture.write_png(str(path), 3, 2, b"".join(rows), bottom_up=True)
    image = picture(str(path))
    assert image.getpixel((0, 0)) == (20, 20, 20, 255)
    assert image.getpixel((2, 1)) == (10, 10, 10, 255)
    assert not (tmp_path / "out.png.part").exists()


def test_a_sequence_path_puts_the_frame_before_the_extension() -> None:
    assert outputs.sequence_path("/a/b/c_run-1.png") == "/a/b/c_run-1.$F4.png"
    assert outputs.sequence_path("c.png", "$F") == "c.$F.png"
    assert outputs.sequence_path("/a/noext") == "/a/noext.$F4"


def test_the_capability_probe_names_the_routes(scene: Scene) -> None:
    module = scene.module()
    assert capture.routes(module) == [
        capture.VIEWPORT,
        capture.VIEWPORT_TAB,
        capture.FLIPBOOK_ROP,
        capture.COP_LAYER,
        capture.COP2_SAVE,
        capture.NETWORK_GRAB,
        capture.PANE_GRAB,
    ]
    del module.ui
    assert capture.routes(module) == [capture.FLIPBOOK_ROP, capture.COP_LAYER, capture.COP2_SAVE]


# Section: through the dispatcher


def test_through_the_dispatcher_nothing_lands_on_undo_and_a_retry_is_the_receipt(
    scene: Scene, home: Path
) -> None:
    store_path = home / store_module.STORE_FILE_NAME
    module = scene.module()
    identity = Identity(session_id="s-1", kind="hython", alias="w1", hou=module)
    dispatcher = Dispatcher(
        default_registry(),
        lock=threading.Lock(),
        kind="hython",
        session_id="s-1",
        identity=identity,
        receipts=receipts.Receipts(lambda: store_module.Store(store_path), session_id="s-1"),
        hou=module,
        wait_s=5.0,
        timeout_s=10.0,
        home=home,
        open_store=lambda: store_module.Store(store_path),
    )
    mark = identity.dirty.state_name
    envelope = Envelope(
        tool="capture.image",
        arguments={"resolution": [100, 50]},
        operation_id="cap-1",
    )
    first = dispatcher.dispatch(envelope).payload
    assert first["ok"] is True, first
    assert first["undo"]["recorded"] is False
    assert first["job_id"] == "job-cap-1"
    assert scene.undos.undoLabels() == []
    assert identity.dirty.state_name == mark
    again = dispatcher.dispatch(envelope).payload
    assert again["replayed"] is True
    assert again["data"]["views"][0]["files"] == first["data"]["views"][0]["files"]
    assert len(scene.capture.seen) == 1
    with store_module.Store(store_path) as store:
        job = store.get_job("job-cap-1")
    assert job.kind == "capture"
    assert job.state == "done"
    assert job.spec == {"source": "viewport", "views": "single", "frames": None}


# Section: a render node that draws larger than asked


def gui_without_viewer(scene: Scene) -> None:
    """A session with a user interface whose only route is the render node."""
    desktop(scene)


def test_a_dense_screen_is_worked_out_and_the_made_camera_makes_up_for_it(
    scene: Scene, home: Path
) -> None:
    gui_without_viewer(scene)
    scene.capture.backing = 2.0
    said = take(scene, home, kind="gui", resolution=[320, 160])
    [shot] = said["views"]
    assert shot["camera"]["window_scaled"] == 2.0
    [seen] = scene.capture.seen
    assert seen["window"] == (0.5, 0.5, 2.0, 2.0)
    # The box sits in the middle of the frame, where it would with no scale.
    assert picture(shot["files"][0]).getchannel("A").getbbox() == (80, 40, 240, 120)
    assert len(scene.capture.probes) == 1
    assert list(Path(shot["files"][0]).parent.glob("*.probe.png")) == []
    assert names(scene, "/obj") == ["boxgeo"]
    assert names(scene, "/out") == []


def test_a_hython_on_the_offscreen_plugin_never_draws_a_calibration(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(capture.QT_PLATFORM_ENV_VAR, "offscreen")
    scene.ui.ratio = 2.0
    take(scene, home)
    assert scene.capture.probes == []


def test_a_hython_on_a_one_to_one_screen_draws_no_calibration(scene: Scene, home: Path) -> None:
    take(scene, home)
    assert scene.capture.probes == []


def test_a_hython_on_another_plugin_and_a_dense_screen_is_calibrated(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(capture.QT_PLATFORM_ENV_VAR, "cocoa")
    scene.ui.ratio = 2.0
    scene.capture.backing = 2.0
    said = take(scene, home, resolution=[320, 160])
    [shot] = said["views"]
    assert len(scene.capture.probes) == 1
    assert "framing_unverified" not in shot
    assert picture(shot["files"][0]).getchannel("A").getbbox() == (80, 40, 240, 120)


def test_a_hython_whose_screen_will_not_say_is_calibrated_or_unverified(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = scene.module()
    del module.ui
    monkeypatch.setattr(capture, "alpha_box", lambda path: None)
    context = tools.ToolContext(**{**Run(scene, home, "hython").context.__dict__, "hou": module})
    said = capture.capture_image({}, context)
    assert len(scene.capture.probes) == 1
    assert said["views"][0]["framing_unverified"] is True
    assert any("drawing scale" in item for item in said["warnings"])


def test_the_scale_is_read_again_when_the_screen_ratio_changes(scene: Scene, home: Path) -> None:
    gui_without_viewer(scene)
    run = Run(scene, home, "gui")
    capture.capture_image({}, run.context)
    capture.capture_image({}, run.context)
    assert len(scene.capture.probes) == 1
    scene.ui.ratio = 2.0
    capture.capture_image({}, run.context)
    assert len(scene.capture.probes) == 2


def test_a_sequence_first_still_reads_the_scale_under_a_plain_name(
    scene: Scene, home: Path
) -> None:
    gui_without_viewer(scene)
    scene.capture.backing = 2.0
    said = take(scene, home, kind="gui", frames=[1, 2, 1], resolution=[320, 160])
    [probe] = scene.capture.probes
    assert "$F" not in probe["picture"]
    [shot] = said["views"]
    for item in shot["files"]:
        assert picture(item).getchannel("A").getbbox() == (80, 40, 240, 120)
    assert list(Path(shot["files"][0]).parent.glob("*.probe.png")) == []


def test_a_named_camera_on_a_dense_screen_is_followed_with_a_wider_window(
    scene: Scene, home: Path
) -> None:
    gui_without_viewer(scene)
    scene.capture.backing = 2.0
    shot_cam = scene.node("/obj").createNode("cam", "shotcam")
    shot_cam.parmTuple("win").set((0.1, 0.0))
    before = camera_state(shot_cam)
    take(scene, home, kind="gui", camera="/obj/shotcam")
    [seen] = scene.capture.seen
    assert seen["window"] == pytest.approx((0.6, 0.5, 2.0, 2.0))
    assert camera_state(shot_cam) == before


def test_a_scale_that_cannot_be_read_leaves_the_framing_unverified(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gui_without_viewer(scene)
    monkeypatch.setattr(capture, "alpha_box", lambda path: None)
    said = take(scene, home, kind="gui")
    [shot] = said["views"]
    assert shot["framing_unverified"] is True
    assert any("drawing scale" in item for item in said["warnings"])


@pytest.mark.parametrize(
    ("box", "scale"),
    [
        ((16, 16, 48, 48), 1.0),
        ((32, 0, 64, 32), 2.0),
        ((24, 0, 64, 40), 1.5),
        # A scale half way between quarters goes up, not to the even one.
        ((18, 10, 54, 46), 1.25),
        # Edges that disagree are no reading at all.
        ((16, 0, 48, 20), None),
        ((0, 0, 64, 64), None),
        (None, None),
    ],
)
def test_the_scale_from_the_calibration_box(box: Any, scale: float | None) -> None:
    assert capture.scale_from_box(box, 64) == scale


def test_alpha_box_reads_what_a_render_node_writes(tmp_path: Path) -> None:
    image = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
    image.paste((9, 9, 9, 200), (3, 2, 7, 9))
    image.save(tmp_path / "rgba.png")
    assert capture.alpha_box(str(tmp_path / "rgba.png")) == (3, 2, 7, 9)
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(tmp_path / "clear.png")
    assert capture.alpha_box(str(tmp_path / "clear.png")) is None
    Image.new("RGB", (4, 4), (1, 2, 3)).save(tmp_path / "rgb.png")
    with pytest.raises(ValueError):
        capture.alpha_box(str(tmp_path / "rgb.png"))


# Section: what a failed or stopped capture leaves


def sidecar_of(path: str) -> dict[str, Any]:
    [record] = Path(path).parent.glob("*_run.json")
    import json

    return json.loads(record.read_text(encoding="utf-8"))


def test_the_run_record_lists_the_numbered_frames(scene: Scene, home: Path) -> None:
    said = take(scene, home, frames=[1, 3, 1])
    [shot] = said["views"]
    record = sidecar_of(shot["files"][0])
    assert record["paths"]["files"] == shot["files"]
    with store_module.Store(home / store_module.STORE_FILE_NAME) as store:
        assert store.get_run(shot["run_id"]).paths["files"] == shot["files"]


def test_a_render_that_fails_part_way_leaves_no_frames(scene: Scene, home: Path) -> None:
    scene.capture.fail_at_frame = 2.0
    error = refused(scene, home, frames=[1, 3, 1])
    assert error.code == "CAPTURE_FAILED"
    assert error.details["error"] == "OperationFailed: the render stopped with an error"
    assert "camera" in error.hint
    assert error.details["tried"][0]["reason"].startswith("OperationFailed")
    folder = next((home.parent / ".agent" / "captures").iterdir())
    assert list(folder.iterdir()) == []
    assert names(scene, "/out") == []


def test_a_stopped_sequence_keeps_and_lists_its_frames(scene: Scene, home: Path) -> None:
    run = Run(scene, home, "hython")

    def note(said: dict[str, Any]) -> None:
        run.notes.append(said)
        run.cancel.set()

    context = tools.ToolContext(**{**run.context.__dict__, "progress": note})
    said = capture.capture_image({"frames": [1, 4, 1]}, context)
    [shot] = said["views"]
    assert len(shot["files"]) == 1 and Path(shot["files"][0]).is_file()
    assert any("before the stop are kept" in item for item in said["warnings"])
    assert sidecar_of(shot["files"][0])["paths"]["files"] == shot["files"]


# Section: the settings a viewport capture sets for itself


def test_every_flipbook_setting_is_set_not_carried(scene: Scene, home: Path) -> None:
    viewer_scene(scene)
    module = scene.module()
    module.flipbookObjectType = SimpleNamespace(Visible="Visible")
    module.flipbookAntialias = SimpleNamespace(
        Fast="Fast",
        Good="Good",
        HighQuality="HighQuality",
        Off="Off",
        UseViewportSetting="UseViewportSetting",
    )
    context = tools.ToolContext(**{**Run(scene, home, "gui").context.__dict__, "hou": module})
    said = capture.capture_image({"resolution": [64, 36]}, context)
    [seen] = scene.capture.seen
    settings = seen["settings"]
    carried = {
        name: value
        for name, value in settings.items()
        if value == FlipbookSettings.ARTIST[name] and name != "output"
    }
    assert carried == {}
    assert settings["visibleObjects"] == "*"
    assert settings["visibleTypes"] == "Visible"
    assert settings["useMotionBlur"] is False
    assert settings["backgroundImage"] == ""
    assert not any("could not be set" in item for item in said["warnings"])


def test_a_setting_this_build_does_not_name_is_reported(scene: Scene, home: Path) -> None:
    viewer_scene(scene)
    said = take(scene, home, kind="gui")
    [note] = [item for item in said["warnings"] if "could not be set" in item]
    assert "visibleTypes" in note and "antialias" in note


def test_framing_while_looking_through_a_camera_leaves_the_camera_alone(
    scene: Scene, home: Path
) -> None:
    viewer = viewer_scene(scene)
    shot_cam = scene.node("/obj").createNode("cam", "shotcam")
    viewer.viewport.setCamera(shot_cam)
    before = view_state(viewer.viewport)
    said = take(scene, home, kind="gui", frame_target="all")
    [seen] = scene.capture.seen
    assert seen["camera"] is None
    assert viewer.viewport.framed == [UNIT]
    assert said["views"][0]["camera"]["left_camera"] is True
    assert view_state(viewer.viewport) == before
    assert viewer.viewport.camera() is shot_cam


# Section: what framing everything goes around, and what is drawn

# The shape a stock guide object draws, by its type, as a real 22.0 makes it.
STOCK = {"null": "control", "cam": "box", "hlight::2.0": "box"}


def guide_shape(scene: Scene, parent: str, type_name: str, name: str, at: tuple) -> Any:
    """An object that draws a shape of its own, as a real camera, light or null does."""
    made = scene.node(parent).createNode(type_name, name)
    made.parmTuple("t").set(at)
    shape = made.createNode(STOCK.get(type_name, "box"), "shape")
    shape.bounds = UNIT
    shape.setDisplayFlag(True)
    return made


def busy_scene(scene: Scene) -> None:
    """The box, with a shown camera, light and null far from it, a hidden object,
    and a second box inside a subnet beside a camera."""
    guide_shape(scene, "/obj", "cam", "shotcam", (20.0, 10.0, 40.0))
    guide_shape(scene, "/obj", "hlight::2.0", "key", (-30.0, 15.0, 0.0))
    guide_shape(scene, "/obj", "null", "handle", (0.0, -25.0, 0.0))
    guide_shape(scene, "/obj", "geo", "hidden", (50.0, 0.0, 0.0)).hidden = True
    group = scene.node("/obj").createNode("subnet", "group")
    guide_shape(scene, "/obj/group", "geo", "inner", (2.0, 0.0, 0.0))
    guide_shape(scene, "/obj/group", "cam", "rigcam", (0.0, 0.0, -60.0))
    group.createNode("subnet", "off").hidden = True
    guide_shape(scene, "/obj/group/off", "geo", "tucked", (0.0, 80.0, 0.0))
    scene.undos.labels.clear()


GUIDES_LEFT_OUT = "* ^/obj/shotcam ^/obj/group/rigcam ^/obj/key ^/obj/handle"


def test_framing_all_goes_around_geometry_not_cameras_lights_or_nulls(
    scene: Scene, home: Path
) -> None:
    busy_scene(scene)
    viewer = viewer_scene(scene)
    said = take(scene, home, kind="gui", camera={"orbit": 90, "elevation": 20})
    assert viewer.viewport.framed == [((-0.5, -0.5, -0.5), (2.5, 0.5, 0.5))]
    assert said["views"][0]["camera"]["target"] == "all"
    assert not any("nothing to frame" in item for item in said["warnings"])
    [seen] = scene.capture.seen
    # Everything is drawn but the guides, which are left out by name.
    assert seen["settings"]["visibleObjects"] == GUIDES_LEFT_OUT


def test_a_fitted_camera_frames_the_same_geometry_whatever_else_is_shown(
    scene: Scene, home: Path
) -> None:
    take(scene, home, resolution=[320, 180], frame_target="all")
    alone = scene.capture.seen[-1]
    busy_scene(scene)
    scene.node("/obj/group/inner").hidden = True
    take(scene, home, resolution=[320, 180], frame_target="all")
    crowded = scene.capture.seen[-1]
    assert crowded["t"] == alone["t"]
    assert crowded["vobjects"] == GUIDES_LEFT_OUT
    assert alone["vobjects"] == "*"


def test_a_null_with_geometry_of_its_own_is_framed_and_drawn(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    null = scene.node("/obj").createNode("null", "holder")
    null.parmTuple("t").set((3.0, 0.0, 0.0))
    null.createNode("box", "mine").setDisplayFlag(True)
    take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [((-0.5, -0.5, -0.5), (3.5, 0.5, 0.5))]
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == "*"


def test_framing_reads_the_frames_captured_not_the_scene_s(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    geo = scene.node("/obj/boxgeo")
    geo.moves = lambda frame: (frame, 0.0, 0.0)
    late = guide_shape(scene, "/obj", "geo", "late", (0.0, 10.0, 0.0))
    late.shown_at = {20.0}
    take(scene, home, kind="gui", frame_target="all", frame=10)
    assert viewer.viewport.framed[-1] == ((9.5, -0.5, -0.5), (10.5, 0.5, 0.5))
    # A sequence is framed on its first and last frames together.
    take(scene, home, kind="gui", frame_target="all", frames=[10, 20, 5])
    assert viewer.viewport.framed[-1] == ((-0.5, -0.5, -0.5), (20.5, 10.5, 0.5))


def test_a_simulation_is_drawn_but_not_framed(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    sim = scene.node("/obj").createNode("geo", "sim")
    sim.parmTuple("t").set((40.0, 0.0, 0.0))
    # A simulation network's display node is not a geometry node: it has no
    # geometry to read.
    sim.displayNode = lambda: SimpleNamespace(path=lambda: "/obj/sim/output")
    said = take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [UNIT]
    assert not any("nothing to frame" in item for item in said["warnings"])
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == "*"


def test_an_instance_object_is_framed_around_what_it_copies(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    scene.node("/obj/boxgeo").hidden = True
    placed = scene.node("/obj").createNode("instance", "copies")
    placed.createNode("box", "points").setDisplayFlag(True)
    placed.children()[0].bounds = ((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))
    placed.parm = lambda name: SimpleNamespace(
        evalAsNodeAtFrame=lambda frame: scene.node("/obj/boxgeo")
    )
    take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [((-0.5, -0.5, -0.5), (4.5, 0.5, 0.5))]


def test_a_viewer_away_from_objects_keeps_houdini_s_frame_all(scene: Scene, home: Path) -> None:
    busy_scene(scene)
    viewer = viewer_scene(scene)
    viewer.network = scene.node("/obj/boxgeo")
    take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == ["all"]
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == "*"


def test_guides_draw_every_object(scene: Scene, home: Path) -> None:
    busy_scene(scene)
    viewer_scene(scene)
    take(scene, home, kind="gui", guides=True)
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == "*"
    take(scene, home, guides=True)
    assert scene.capture.seen[-1]["vobjects"] == "*"


def test_a_scene_with_no_geometry_frames_the_origin_and_hides_the_guides(
    scene: Scene, home: Path
) -> None:
    scene.node("/obj/boxgeo").destroy()
    guide_shape(scene, "/obj", "cam", "shotcam", (20.0, 10.0, 40.0))
    guide_shape(scene, "/obj", "null", "handle", (0.0, -25.0, 0.0))
    viewer = viewer_scene(scene)
    said = take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [UNIT]
    assert "nothing to frame was found, so the view frames the origin" in said["warnings"]
    mask = "* ^/obj/shotcam ^/obj/handle"
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == mask
    said = take(scene, home, frame_target="all")
    assert "nothing to frame was found, so the camera frames the origin" in said["warnings"]
    assert scene.capture.seen[-1]["vobjects"] == mask
    assert scene.capture.seen[-1]["t"] == pytest.approx(
        capture.fit_camera(
            UNIT,
            orbit=45.0,
            elevation=25.0,
            ortho=False,
            aspect=1280 / 720,
            focal=50.0,
            aperture=41.4214,
        )["t"]
    )


def test_framing_stops_at_the_cap_and_says_so(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    busy_scene(scene)
    monkeypatch.setattr(capture, "MAX_FRAMED", 1)
    viewer = viewer_scene(scene)
    said = take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [UNIT]
    assert capture.TOO_MANY_TO_FRAME in said["warnings"]


def test_bounds_read_vectors_by_index(scene: Scene, home: Path) -> None:
    busy_scene(scene)
    viewer_scene(scene)
    Vector3.overreads = 0
    take(scene, home, kind="gui", frame_target="all")
    take(scene, home, frame_target="all")
    assert Vector3.overreads == 0


def test_framing_fits_the_picture_s_shape_not_the_viewport_s(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    viewer.viewport.box = (0, 0, 950, 653)
    take(scene, home, kind="gui", camera="top", resolution=[1280, 720])
    grow = (1280 / 720) / (950 / 653)
    [(low, high)] = viewer.viewport.framed
    assert low == pytest.approx((-0.5 * grow,) * 3)
    assert high == pytest.approx((0.5 * grow,) * 3)


def test_guide_types_that_draw_their_stock_shape_are_left_out(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    for type_name, shape in (
        ("pathcv", "control"),
        ("path", "convert"),
        ("handle", "merge"),
        ("muscle", "muscle"),
    ):
        made = scene.node("/obj").createNode(type_name, f"stock_{type_name}")
        made.parmTuple("t").set((30.0, 0.0, 0.0))
        drawn = made.createNode(shape, "shape")
        drawn.bounds = UNIT
        drawn.setDisplayFlag(True)
    # A path whose display flag is on geometry of its own is geometry.
    own = scene.node("/obj").createNode("path", "own_path")
    own.parmTuple("t").set((3.0, 0.0, 0.0))
    own.createNode("box", "mine").setDisplayFlag(True)
    take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [((-0.5, -0.5, -0.5), (3.5, 0.5, 0.5))]
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == (
        "* ^/obj/stock_pathcv ^/obj/stock_path ^/obj/stock_handle ^/obj/stock_muscle"
    )


def test_the_walk_stays_in_object_networks_and_is_made_once(scene: Scene, home: Path) -> None:
    viewer = viewer_scene(scene)
    sim = scene.node("/obj").createNode("dopnet", "f_dopnet")
    sim.hidden = True
    # A node inside a simulation of a type Houdini counts as geometry.
    guide_shape(scene, "/obj/f_dopnet", "geo", "inside", (40.0, 0.0, 0.0))
    guide_shape(scene, "/obj/f_dopnet", "null", "marker", (40.0, 0.0, 0.0))
    scene.globbed = 0
    take(scene, home, kind="gui", frame_target="all")
    assert viewer.viewport.framed == [UNIT]
    assert scene.capture.seen[-1]["settings"]["visibleObjects"] == "*"
    # One filtered walk for each of geometry, cameras and lights.
    assert scene.globbed == 3
    scene.globbed = 0
    take(scene, home, frame_target="all")
    assert scene.globbed == 3


def test_the_ortho_width_is_put_back_after_framing_a_perspective_view(
    scene: Scene, home: Path
) -> None:
    viewer = viewer_scene(scene)
    viewport = viewer.viewport
    viewport.keeps_ortho_width = True
    viewport._default.setOrthoWidth(7.7)
    take(scene, home, kind="gui", frame_target="all")
    assert viewport.framed == [UNIT]
    assert viewport._default.orthoWidth() == 7.7


# Section: grabbing a pane that is not the current tab


def test_a_pane_behind_another_is_brought_forward_then_put_back(
    scene: Scene, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor = NetworkEditorTab(scene)
    editor.window = Window(0, 0, 400, 300, 1.0)
    editor.geometry = Rect(0, 0, 400, 300)
    editor.window.painted.append((Rect(10, 10, 20, 20), (255, 0, 0, 255)))
    other = Tab(scene, "Parm", "parms")
    pane = desktop(scene, other, editor)
    order: list[str] = []
    monkeypatch.setattr(
        capture,
        "process_events",
        lambda: order.append("paint" if editor.isCurrentTab() else "paint behind"),
    )
    grab = editor.window.grab

    def grabbed() -> Any:
        order.append("grab" if editor.isCurrentTab() else "grab behind")
        return grab()

    editor.window.grab = grabbed
    take(scene, home, kind="gui", source="network")
    assert order == ["paint", "grab"]
    assert pane.currentTab() is other


# Section: a hython started by hand


class _Stopped(Exception):
    pass


def _no_bridge(config: Any) -> Any:
    raise _Stopped


@pytest.mark.parametrize(("given", "expected"), [(None, "offscreen"), ("cocoa", "cocoa")])
def test_a_bridge_started_by_hand_draws_offscreen_unless_told(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, given: str | None, expected: str
) -> None:
    import os

    from nscr_houdini_mcp.bridge import main as bridge_main

    if given is not None:
        monkeypatch.setenv(bridge_main.QT_PLATFORM_ENV_VAR, given)
    monkeypatch.setattr(bridge_main, "Bridge", _no_bridge)
    with pytest.raises(_Stopped):
        bridge_main.main(["--home", str(tmp_path)])
    assert os.environ[bridge_main.QT_PLATFORM_ENV_VAR] == expected


# Section: the display flag a node capture borrows


def test_a_node_whose_object_showed_nothing_gives_the_flag_up_after(
    scene: Scene, home: Path
) -> None:
    geo = scene.node("/obj").createNode("geo", "bare")
    lone = geo.createNode("box", "lone")
    scene.undos.labels.clear()
    assert geo.displayNode() is None
    take(scene, home, source="node", path="/obj/bare/lone")
    [seen] = scene.capture.seen
    assert seen["displayed"] == {"/obj/bare": "/obj/bare/lone"}
    assert geo.displayNode() is None
    assert not lone.isDisplayFlagSet()


@pytest.mark.parametrize("had_one", [True, False])
def test_the_display_flag_goes_back_when_the_render_fails(
    scene: Scene, home: Path, had_one: bool
) -> None:
    geo = scene.node("/obj/boxgeo")
    if not had_one:
        geo.displayNode().flags.discard("Display")
    other = geo.createNode("box", "box2")
    scene.capture.fail_at_frame = 1.0
    error = refused(scene, home, source="node", path="/obj/boxgeo/box2")
    assert error.code == "CAPTURE_FAILED"
    shown = geo.displayNode()
    assert (shown.path() if shown else None) == ("/obj/boxgeo/box1" if had_one else None)
    assert not other.isDisplayFlagSet()


# Section: clean up that does not go as it should


def test_a_render_node_that_will_not_go_is_cleanup_failed(scene: Scene, home: Path) -> None:
    scene.capture.undestroyable = {"flipbook"}
    error = refused(scene, home, frames=[1, 2, 1])
    assert error.code == "CLEANUP_FAILED"
    [step] = error.details["cleanup"]
    assert step["step"] == "take away /out/nscr_capture"
    assert step["error"].startswith("OperationFailed")
    assert error.details["frames"] == [1.0, 2.0]
    # Every other step still ran: the fitted camera is gone.
    assert names(scene, "/obj") == ["boxgeo"]
    # The frames are there, and the run record names them.
    [record] = (home.parent / ".agent" / "captures").rglob("*_run.json")
    files = json.loads(record.read_text(encoding="utf-8"))["paths"]["files"]
    assert len(files) == 2 and all(Path(item).is_file() for item in files)


def test_clean_up_that_fails_beside_a_failed_render_is_in_its_details(
    scene: Scene, home: Path
) -> None:
    scene.capture.undestroyable = {"flipbook"}
    scene.capture.fail_at_frame = 1.0
    error = refused(scene, home)
    assert error.code == "CAPTURE_FAILED"
    steps = [step["step"] for step in error.details["cleanup"]]
    expected = ["release viewport material bindings"] if capture.sys.platform == "win32" else []
    assert steps == [*expected, "take away /out/nscr_capture"]


def test_a_view_that_will_not_go_back_is_cleanup_failed_and_the_rest_goes_back(
    scene: Scene, home: Path
) -> None:
    viewer = viewer_scene(scene)
    viewer.viewport.refuse_default = True
    error = refused(scene, home, kind="gui", camera="top", display="wire")
    assert error.code == "CLEANUP_FAILED"
    assert [step["step"] for step in error.details["cleanup"]] == ["put the viewport's camera back"]
    assert viewer.viewport.type() == "Perspective"
    assert viewer.viewport.shading() == {"SceneObject": "Smooth", "DisplayModel": "Smooth"}


# Section: a viewport sequence, a few frames at a time


def test_a_viewport_sequence_goes_in_pieces_with_progress(scene: Scene, home: Path) -> None:
    viewer_scene(scene)
    run = Run(scene, home, "gui")
    said = capture.capture_image({"frames": [1, 20, 1], "resolution": [32, 18]}, run.context)
    ranges = [seen["settings"]["frameRange"] for seen in scene.capture.seen]
    assert ranges == [(1.0, 8.0), (9.0, 16.0), (17.0, 20.0)]
    assert [note["done"] for note in run.notes] == [8, 16, 20]
    assert len(said["views"][0]["files"]) == 20


def test_a_viewport_sequence_stops_between_pieces(scene: Scene, home: Path) -> None:
    viewer_scene(scene)
    run = Run(scene, home, "gui")

    def note(said: dict[str, Any]) -> None:
        run.notes.append(said)
        run.cancel.set()

    context = tools.ToolContext(**{**run.context.__dict__, "progress": note})
    said = capture.capture_image({"frames": [1, 20, 1], "resolution": [32, 18]}, context)
    [shot] = said["views"]
    assert len(scene.capture.seen) == 1
    assert len(shot["files"]) == 8
    assert said["stopped_early"] is True


def test_a_capture_stopped_before_its_first_frame_writes_nothing(scene: Scene, home: Path) -> None:
    viewer_scene(scene)
    run = Run(scene, home, "gui")
    run.cancel.set()
    said = capture.capture_image({"frames": [1, 4, 1]}, run.context)
    [shot] = said["views"]
    assert shot["files"] == [] and shot["stopped_early"] is True
    assert scene.capture.seen == []
    assert list((home.parent / ".agent" / "captures").rglob("*_run.json")) == []


# Section: what the job row knows while a sequence runs


def test_the_job_row_names_the_run_and_each_frame_as_it_goes(scene: Scene, home: Path) -> None:
    store_path = home / store_module.STORE_FILE_NAME
    module = scene.module()
    dispatcher = Dispatcher(
        default_registry(),
        lock=threading.Lock(),
        kind="hython",
        session_id="s-1",
        identity=Identity(session_id="s-1", kind="hython", alias="w1", hou=module),
        receipts=receipts.Receipts(lambda: store_module.Store(store_path), session_id="s-1"),
        hou=module,
        wait_s=5.0,
        timeout_s=10.0,
        home=home,
        open_store=lambda: store_module.Store(store_path),
    )
    rows: list[Any] = []

    def look(frame: float) -> None:
        with store_module.Store(store_path) as store:
            rows.append(store.get_job("job-seq-1").outputs)

    scene.capture.after_frame = look
    envelope = Envelope(tool="capture.image", arguments={"frames": [1, 3, 1]}, operation_id="seq-1")
    assert dispatcher.dispatch(envelope).payload["ok"] is True
    runs = [row["capture"]["runs"][0] for row in rows]
    assert all(run["run_id"].startswith("run-") and "$F4" in run["path"] for run in runs)
    expected = [0, 1, 2, 3] if capture.sys.platform == "win32" else [0, 1, 2]
    assert [len(run["files"]) for run in runs] == expected
    assert [row["capture"]["frames_done"] for row in rows] == expected
    assert all(not path.endswith(".cleanup.png") for run in runs for path in run["files"])
