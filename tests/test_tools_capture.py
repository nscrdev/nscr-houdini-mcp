"""`hou_capture` through the server, down to a bridge making the picture.

Every call goes the whole way: the server checks the arguments, the router
sends the call, a real dispatcher with real receipts and job rows runs the
bridge's `capture.image` against the stand in for `hou`, and the server reads
the files that came back with Pillow. The stand in writes real PNG files, so
the numbers, the crops, the sheet and the thumbnail are all read from disk.
What a real Houdini draws is in the integration test.
"""

from __future__ import annotations

import base64
import io
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp_types import ImageContent, TextContent
from PIL import Image

from fake_hou import Desktop, NetworkEditorTab, Pane, Rect, Scene, Window
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, receipts
from nscr_houdini_mcp.bridge.dispatch import Dispatcher
from nscr_houdini_mcp.bridge.envelope import Envelope
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.identity import Identity
from nscr_houdini_mcp.tools import capture as capture_tool
from test_server import talk
from test_tools_sessions import Bench

TYPES = ("geo", "null", "cam", "box", "flipbook", "copnet", "fractalnoise")


class Through:
    """Sends each call to a real dispatcher over the stand in, as a session would."""

    def __init__(self, module: Any, home: Path, kind: str) -> None:
        store_path = home / store_module.STORE_FILE_NAME
        self.identity = Identity(session_id="s-1", kind=kind, alias="w1", hou=module)
        self.dispatcher = Dispatcher(
            default_registry(),
            lock=threading.Lock(),
            kind=kind,
            session_id="s-1",
            identity=self.identity,
            receipts=receipts.Receipts(lambda: store_module.Store(store_path), session_id="s-1"),
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


def bench_for(tmp_path: Path, scene: Scene, kind: str = "hython") -> Bench:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    made = Bench(home)
    made.session("s-1", "w1", kind=kind)
    made.sent = Through(scene.module(), home, kind)  # type: ignore[assignment]
    return made


@pytest.fixture
def bench(tmp_path: Path, scene: Scene) -> Bench:
    return bench_for(tmp_path, scene)


def call(bench: Bench, *calls: tuple[str, dict[str, Any]]) -> list[Any]:
    _, results = talk(bench.serve(), *calls)
    return results


def shoot(bench: Bench, **arguments: Any) -> Any:
    [result] = call(bench, ("hou_capture", arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def refused(result: Any) -> dict[str, Any]:
    assert result.is_error is True
    return result.structured_content["error"]


def images(result: Any) -> list[ImageContent]:
    return [block for block in result.content if isinstance(block, ImageContent)]


def decoded(block: ImageContent) -> Image.Image:
    with Image.open(io.BytesIO(base64.b64decode(block.data))) as opened:
        return opened.copy()


def sent(bench: Bench) -> list[dict[str, Any]]:
    return [item for item in bench.sent.calls if item["tool"] == "capture.image"]


# Section: one picture


def test_a_capture_returns_the_file_its_numbers_and_a_thumbnail(bench: Bench) -> None:
    result = shoot(bench, resolution=[1280, 720])
    body = ok(result)
    path = Path(body["path"])
    assert path.is_file()
    assert path.parent.parent.name == "captures"
    assert (body["width"], body["height"]) == (1280, 720)
    assert body["route"] == "flipbook_rop"
    assert body["camera"]["kind"] == "fitted"
    assert body["frame"] == 1.0
    assert body["run_id"] in path.name
    stats = body["image_stats"]
    assert stats["channels"] == "RGBA"
    assert stats["non_empty"] is True
    assert stats["min"][3] == 0 and stats["max"][3] == 255
    assert len(stats["mean"]) == 4
    assert body["state"] == "done"
    assert body["job_id"] == f"job-{body['trace']['operation_id']}"
    # The text block mirrors the result, and the thumbnail rides beside it.
    [text, image] = result.content
    assert isinstance(text, TextContent)
    thumb = decoded(image)
    assert max(thumb.size) == 512
    assert body["thumb"] == {
        "kind": "thumb",
        "width": 512,
        "height": 288,
        "bytes": len(base64.b64decode(image.data)),
        "mime_type": "image/png",
    }
    assert image.mime_type == "image/png"


def test_return_image_none_and_full(bench: Bench) -> None:
    none, full = call(
        bench,
        ("hou_capture", {"return_image": "none", "resolution": [200, 100]}),
        ("hou_capture", {"return_image": "full", "resolution": [200, 100]}),
    )
    assert images(none) == []
    assert "thumb" not in ok(none)
    [whole] = images(full)
    assert decoded(whole).size == (200, 100)
    assert ok(full)["thumb"]["kind"] == "full"


def test_a_full_image_too_big_to_send_comes_back_as_a_thumbnail(
    bench: Bench, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture_tool, "FULL_MAX_BYTES", 10)
    result = shoot(bench, return_image="full", resolution=[900, 300])
    body = ok(result)
    assert body["thumb"]["kind"] == "thumb"
    assert decoded(images(result)[0]).size == (512, 171)
    assert any("full image" in item for item in body["warnings"])


def test_a_thumbnail_is_kept_under_its_byte_bound(tmp_path: Path) -> None:
    path = tmp_path / "noise.png"
    Image.frombytes("RGBA", (1024, 1024), os.urandom(1024 * 1024 * 4)).save(path)
    data, mime, size = capture_tool.thumbnail(str(path), max_bytes=40_000)
    assert mime == "image/jpeg"
    assert len(data) <= 40_000
    assert max(size) <= 512


def test_a_region_crops_once_even_when_the_same_call_comes_again(
    bench: Bench, scene: Scene
) -> None:
    arguments = {"resolution": [400, 200], "region": [0.25, 0.5, 0.75, 1.0]}
    first = ok(shoot(bench, operation_id="crop-1", **arguments))
    assert (first["width"], first["height"]) == (200, 100)
    assert first["region"] == [0.25, 0.5, 0.75, 1.0]
    again = ok(shoot(bench, operation_id="crop-1", **arguments))
    assert again["path"] == first["path"]
    assert (again["width"], again["height"]) == (200, 100)
    with Image.open(first["path"]) as image:
        assert image.size == (200, 100)
    # The session answered the second from its receipt: one render only.
    assert len(scene.capture.seen) == 1


def test_quad_stitches_a_contact_sheet(bench: Bench, scene: Scene) -> None:
    body = ok(shoot(bench, views="quad", resolution=[160, 90], return_image="none"))
    assert [view["view"] for view in body["views"]] == ["persp", "top", "front", "right"]
    assert (body["width"], body["height"]) == (320, 180)
    assert "_viewport_sheet_" in Path(body["path"]).name
    assert Path(body["path"]).is_file()
    for view in body["views"]:
        assert Path(view["path"]).is_file()
        assert view["image_stats"]["non_empty"] is True
    assert body["image_stats"]["non_empty"] is True
    assert len(scene.capture.seen) == 4


def test_turntable4_is_four_orbits_and_a_sheet(bench: Bench) -> None:
    body = ok(shoot(bench, views="turntable4", resolution=[64, 64], return_image="none"))
    assert [view["camera"]["orbit"] for view in body["views"]] == [0.0, 90.0, 180.0, 270.0]
    assert (body["width"], body["height"]) == (128, 128)


def test_an_empty_picture_is_capture_empty(bench: Bench, scene: Scene) -> None:
    scene.capture.blank = True
    error = refused(shoot(bench))
    assert error["code"] == "CAPTURE_EMPTY"
    assert error["details"]["views"][0]["image_stats"]["non_empty"] is False
    assert "check the camera" in error["hint"]


def test_a_render_that_fails_is_capture_failed_with_its_error(bench: Bench, scene: Scene) -> None:
    scene.capture.fail_at_frame = 1.0
    result = shoot(bench)
    error = refused(result)
    assert error["code"] == "CAPTURE_FAILED"
    assert error["details"]["error"].startswith("OperationFailed")
    assert result.content[0].text.startswith("CAPTURE_FAILED")
    assert "camera" in error["hint"]


def test_the_network_in_hython_is_ui_unavailable(bench: Bench) -> None:
    result = shoot(bench, source="network")
    error = refused(result)
    assert error["code"] == "UI_UNAVAILABLE"
    assert "hou_inspect" in error["hint"]
    assert result.content[0].text.startswith("UI_UNAVAILABLE")


@pytest.mark.parametrize(
    ("arguments", "argument"),
    [
        ({"region": [0.5, 0.5, 0.4, 1.0]}, "region"),
        ({"region": [0, 0, 1]}, "region"),
        ({"resolution": [640]}, "resolution"),
        ({"resolution": [640, 9000]}, "resolution"),
        ({"frames": [1, 10]}, "frames"),
        ({"frames": [5, 1, 1]}, "frames"),
        ({"frames": [1, 5000, 1]}, "frames"),
        ({"frames": [1, 5, 1], "frame": 3}, "frames"),
        ({"operation_id": "no spaces"}, "operation_id"),
        ({"wait_s": 90}, "wait_s"),
        ({"camera": {"orbit": "north"}}, "camera.orbit"),
        ({"name": ""}, "name"),
    ],
)
def test_what_the_server_refuses_before_asking_the_session(
    bench: Bench, arguments: dict[str, Any], argument: str
) -> None:
    error = refused(shoot(bench, **arguments))
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == argument
    assert sent(bench) == []


def test_what_the_session_is_sent(bench: Bench) -> None:
    ok(
        shoot(
            bench,
            camera={"orbit": 30, "elevation": 10},
            display="wire",
            name="look",
            return_image="none",
            resolution=[320, 240],
        )
    )
    [first] = sent(bench)
    assert first["arguments"] == {
        "camera": {"orbit": 30, "elevation": 10},
        "display": "wire",
        "name": "look",
        "resolution": [320, 240],
    }
    assert first["operation_id"]


# Section: sequences


def test_a_short_sequence_answers_with_every_frame(bench: Bench) -> None:
    result = shoot(bench, frames=[1, 3, 1], resolution=[64, 36])
    body = ok(result)
    assert body["frames"] == [1.0, 2.0, 3.0]
    assert len(body["paths"]) == 3
    assert body["path"] == body["paths"][0]
    assert all(Path(item).is_file() for item in body["paths"])
    assert body["paths"][1].endswith(".0002.png")
    assert len(images(result)) == 1


def test_a_long_sequence_is_a_job_to_follow(bench: Bench, scene: Scene) -> None:
    scene.capture.delay_s = 0.3
    [started] = call(
        bench, ("hou_capture", {"frames": [1, 4, 1], "resolution": [64, 36], "timeout_s": 0.2})
    )
    handle = ok(started)
    assert handle["kind"] == "capture"
    assert handle["state"] == "running"
    assert images(started) == []
    job_id = handle["job_id"]
    body: dict[str, Any] = {}
    for _ in range(10):
        [held] = call(bench, ("hou_jobs", {"job_id": job_id, "wait_s": 5}))
        body = ok(held)
        if body["state"] not in ("queued", "running"):
            break
    assert body["state"] == "done"
    outputs = body["outputs"]
    assert outputs["frames"] == [1.0, 2.0, 3.0, 4.0]
    assert len(outputs["paths"]) == 4
    assert outputs["image_stats"]["non_empty"] is True
    assert outputs["operation_id"] == handle["operation_id"]
    assert body["progress"]["total"] == 4


def test_a_single_capture_past_its_timeout_names_its_job(bench: Bench, scene: Scene) -> None:
    scene.capture.delay_s = 0.5
    error = refused(shoot(bench, timeout_s=0.1))
    assert error["code"] == "TIMEOUT"
    assert error["details"]["job_id"].startswith("job-")


# Section: a session with a user interface


def test_the_network_editor_in_a_gui_session(tmp_path: Path, scene: Scene) -> None:
    editor = NetworkEditorTab(scene)
    editor.window = Window(0, 0, 600, 400, 2.0)
    editor.geometry = Rect(100, 100, 200, 100)
    editor.window.painted.append((editor.geometry, (200, 30, 30, 255)))
    # A node drawn in the editor, so the picture is not one flat colour.
    editor.window.painted.append((Rect(150, 120, 50, 20), (30, 200, 30, 255)))
    pane = Pane()
    pane.add(editor)
    scene.ui.desktop = Desktop()
    scene.ui.desktop.tabs = [editor]
    bench = bench_for(tmp_path, scene, kind="gui")
    body = ok(shoot(bench, source="network", return_image="none"))
    assert body["route"] == "network_grab"
    assert (body["width"], body["height"]) == (400, 200)
    assert body["image_stats"]["non_empty"] is True
    assert body["image_stats"]["max"][:3] == [200, 200, 30]


def test_the_render_node_in_a_gui_session_says_its_framing_is_unverified(
    tmp_path: Path, scene: Scene
) -> None:
    scene.ui.desktop = Desktop()
    bench = bench_for(tmp_path, scene, kind="gui")
    result = shoot(bench, return_image="none")
    body = ok(result)
    assert body["route"] == "flipbook_rop"
    assert body["framing_unverified"] is True
    assert [item["route"] for item in body["tried"]] == [
        "viewport_flipbook",
        "viewport_flipbook_tab",
    ]


# Section: the listing


def test_hou_capture_is_listed_after_hou_jobs_with_its_inputs(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    names = [tool.name for tool in listed.tools]
    assert names.index("hou_capture") > names.index("hou_jobs")
    [tool] = [tool for tool in listed.tools if tool.name == "hou_capture"]
    assert set(tool.input_schema["properties"]) == {
        "session",
        "source",
        "path",
        "camera",
        "frame_target",
        "display",
        "guides",
        "resolution",
        "frame",
        "frames",
        "region",
        "views",
        "name",
        "return_image",
        "operation_id",
        "wait_s",
        "timeout_s",
    }
    assert tool.input_schema["additionalProperties"] is False


def test_a_long_result_is_a_summary_line_in_text(bench: Bench) -> None:
    result = shoot(bench, views="quad", resolution=[64, 36], return_image="none")
    text = result.content[0].text
    assert text.startswith("hou_capture: viewport via flipbook_rop, 128x72")
    assert "structuredContent" in text


# Section: what counts as empty, and images of more than 8 bits


def test_what_is_empty_and_what_is_only_flat(tmp_path: Path) -> None:
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (8, 8), (40, 90, 200, 255)).save(opaque)
    stats, size = capture_tool.look(str(opaque))
    assert stats["non_empty"] is True and stats["flat"] is True
    assert size == (8, 8)
    clear = tmp_path / "clear.png"
    Image.new("RGBA", (8, 8), (0, 0, 0, 0)).save(clear)
    assert capture_tool.look(str(clear))[0]["non_empty"] is False
    # A clear COP or pane is still a picture: only a camera sees nothing.
    assert capture_tool.look(str(clear), rendered=False)[0]["non_empty"] is True
    nothing = tmp_path / "nothing.png"
    nothing.write_bytes(b"")
    assert capture_tool.look(str(nothing), rendered=False)[0]["non_empty"] is False


def test_a_16_bit_grey_image_is_read_at_its_depth(tmp_path: Path) -> None:
    path = tmp_path / "deep.png"
    Image.new("I;16", (6, 4), 40000).save(path)
    stats, _ = capture_tool.look(str(path), rendered=False)
    assert stats["max"] == [40000] and stats["min"] == [40000]
    assert stats["mean"] == [40000.0]
    assert stats["depth"] == 16 and stats["stats_depth"] == 16
    assert stats["flat"] is True and stats["non_empty"] is True
    data, mime, size = capture_tool.thumbnail(str(path))
    assert mime == "image/png" and size == (6, 4)
    assert (
        decoded(
            ImageContent(type="image", data=base64.b64encode(data).decode(), mime_type=mime)
        ).getpixel((0, 0))
        == 156
    )


def test_a_crop_is_made_once_whatever_the_session_said(tmp_path: Path) -> None:
    path = tmp_path / "cut.png"
    Image.new("RGBA", (100, 50), (1, 2, 3, 255)).save(path)
    capture_tool.crop(str(path), [0.0, 0.0, 0.5, 0.5])
    capture_tool.crop(str(path), [0.0, 0.0, 0.5, 0.5])
    with Image.open(path) as image:
        assert image.size == (50, 25)
        assert image.info[capture_tool.CROP_KEY] == "0,0,0.5,0.5"


def test_a_flat_capture_is_a_picture_with_a_note(bench: Bench, scene: Scene) -> None:
    scene.capture.flat = (70, 70, 70, 255)
    body = ok(shoot(bench, return_image="none"))
    assert body["image_stats"]["flat"] is True
    assert body["image_stats"]["non_empty"] is True
    assert any("flat colour" in item for item in body["warnings"])


# Section: finishing a capture once


def finishing_call(bench: Bench) -> Any:
    from nscr_houdini_mcp.tools.base import Call

    return Call(capture_tool.HOU_CAPTURE, {}, bench.router(bench.config), config=bench.config)


def test_a_capture_is_finished_once_however_many_ask(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "shot.png"
    Image.new("RGBA", (40, 20), (5, 5, 5, 255)).save(path)
    data = {
        "source": "viewport",
        "views": [{"view": "single", "files": [str(path)], "frames": [1.0], "run_id": "run-a"}],
        "region": [0.0, 0.0, 0.5, 0.5],
    }
    finished: list[int] = []
    real = capture_tool.finish
    gate = threading.Event()

    def slow(given: Any, **rest: Any) -> dict[str, Any]:
        finished.append(1)
        gate.wait(2.0)
        return real(given, **rest)

    monkeypatch.setattr(capture_tool, "finish", slow)
    answers: list[dict[str, Any]] = []

    def ask() -> None:
        answers.append(capture_tool.finalised(finishing_call(bench), "op-once", data))

    threads = [threading.Thread(target=ask) for _ in range(3)]
    for thread in threads:
        thread.start()
    gate.set()
    for thread in threads:
        thread.join(10.0)
    assert len(finished) == 1
    assert len(answers) == 3
    assert all(answer == answers[0] for answer in answers)
    assert (answers[0]["width"], answers[0]["height"]) == (20, 10)
    assert list(tmp_path.glob("*.partial")) == [] and list(tmp_path.glob("*.part")) == []


def test_temporary_names_are_this_writer_s_own(tmp_path: Path) -> None:
    from nscr_houdini_mcp import outputs

    target = str(tmp_path / "a.png")
    first, second = outputs.temporary_beside(target), outputs.temporary_beside(target)
    assert first != second
    assert Path(first).parent == Path(second).parent == tmp_path
    assert Path(first).is_file() and Path(second).is_file()


def test_no_thumbnail_is_sent_when_none_fits(
    bench: Bench, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "noise.png"
    Image.frombytes("RGB", (256, 256), os.urandom(256 * 256 * 3)).save(path)
    assert capture_tool.thumbnail(str(path), max_bytes=10) is None
    monkeypatch.setattr(capture_tool, "THUMB_MAX_BYTES", 10)
    result = shoot(bench)
    body = ok(result)
    assert images(result) == []
    assert "thumb" not in body
    assert any("no thumbnail" in item for item in body["warnings"])


def test_a_capture_stopped_before_any_frame_is_an_answer_not_an_error() -> None:
    said = capture_tool.finish(
        {
            "source": "viewport",
            "views": [{"view": "single", "files": [], "frames": [], "stopped_early": True}],
            "stopped_early": True,
        }
    )
    assert said["path"] is None and said["stopped_early"] is True
