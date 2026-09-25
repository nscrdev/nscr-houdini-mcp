"""`hou_capture` through the real server, against a real worker.

A client starts the server over stdio the way any client does, the server
starts a hython worker through the pool, and the worker draws a box through
the flipbook render node. Nothing is stood in for. What a user interface
does with the viewport and pane routes is not here: a worker has none.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap in
this file's own config is one), a state folder of this file's own, the pool's
port range, and every worker stopped again whatever happened, with a check
that nothing is left. The scene file and every capture are inside the test's
own temporary folder.
"""

from __future__ import annotations

import asyncio
import base64
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
from mcp_types import ImageContent
from PIL import Image

import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import registry


def hython_available() -> bool:
    try:
        pool.hython_path()
    except pool.HythonNotFound:
        return False
    return True


pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

PORT_RANGE = support.POOL_PORTS

SERVER_CODE = "from nscr_houdini_mcp.cli import main; raise SystemExit(main([]))"

# A worker start is a cold Houdini, so the client waits longer than it would
# for an ordinary call.
READ_TIMEOUT_S = 300.0

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# A box beside a ball the display flag is not on, in a scene saved where the
# test can find its captures.
BUILD = """
geo = hou.node('/obj').createNode('geo', 'boxgeo')
box = geo.createNode('box', 'box1')
ball = geo.createNode('sphere', 'ball')
ball.parmTuple('t').set((3, 0, 0))
box.setDisplayFlag(True)
box.setRenderFlag(True)
hou.hipFile.save({hip!r})
result = [box.path(), ball.path()]
"""

# What the scene holds that a capture could have left behind.
LOOK = """
result = {
    'out': sorted(node.name() for node in hou.node('/out').children()),
    'obj': sorted(node.name() for node in hou.node('/obj').children()),
    'display': hou.node('/obj/boxgeo').displayNode().path(),
    'render': hou.node('/obj/boxgeo').renderNode().path(),
    'undo': len(hou.undos.undoLabels()),
}
"""


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("capture")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    scenes = root / "scenes"
    scenes.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    made: dict[str, Any] = {"home": home, "scratch": scratch, "scenes": scenes}
    with support.persistent_client(server_params(made), timeout_s=READ_TIMEOUT_S) as send:
        made["send"] = send
        try:
            [started] = run(made, ("hou_sessions", {"action": "start"}))
            made["worker"] = ok(started)["session"]
            hip = scenes / "capture_scene.hip"
            [built] = run(made, ("hou_python", {"code": BUILD.format(hip=str(hip))}))
            made["box"], made["ball"] = ok(built)["result"]
            made["hip"] = hip
            yield made
        finally:
            left = support.stop_everything(home, pool.PoolConfig(home=home))
            assert left == [], f"workers were left running: {left}"
            with pool.open_store(home) as store:
                for worker in store.list_workers(active_only=False):
                    assert worker.pid is None or not pool.worker_is_alive(worker), worker.alias
            assert registry.live_entries(home) == []


def server_params(place: dict[str, Any]) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER_CODE],
        env={
            "NSCR_MCP_HOME": str(place["home"]),
            "HOUDINI_TEMP_DIR": str(place["scratch"]),
            "PYTHONIOENCODING": "utf-8",
        },
    )


async def _run(place: dict[str, Any], calls: list[tuple[str, dict]]) -> list[Any]:
    async with Client(
        server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
    ) as connected:
        results = [await connected.call_tool(name, arguments) for name, arguments in calls]
        await support.stop_server_bound_workers(connected, results)
        return results


def run(place: dict[str, Any], *calls: tuple[str, dict]) -> list[Any]:
    if "send" in place:
        return place["send"](*calls)
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def look(place: dict[str, Any]) -> dict[str, Any]:
    [said] = run(place, ("hou_python", {"code": LOOK}))
    return ok(said)["result"]


def is_png(path: str) -> bool:
    data = Path(path).read_bytes()
    return len(data) > len(PNG_SIGNATURE) and data.startswith(PNG_SIGNATURE)


def plausible(stats: dict[str, Any]) -> None:
    """A box on a clear background: some pixels covered, most not, none blown out."""
    assert stats["non_empty"] is True, stats
    assert stats["channels"] == "RGBA"
    assert stats["min"][3] == 0 and stats["max"][3] == 255, stats
    assert 2.0 < stats["mean"][3] < 200.0, stats
    assert max(stats["max"][:3]) > 16, stats


def framed(path: str) -> None:
    """The drawn part sits in the middle of the frame and fills a fair share of it."""
    with Image.open(path) as image:
        width, height = image.size
        left, top, right, bottom = image.getchannel("A").getbbox()
    middle = ((left + right) / 2 / width, (top + bottom) / 2 / height)
    assert abs(middle[0] - 0.5) < 0.1 and abs(middle[1] - 0.5) < 0.1, (middle, image.size)
    assert (right - left) / width > 0.25 or (bottom - top) / height > 0.25
    assert left > 0 and top > 0 and right < width and bottom < height


@pytest.mark.parametrize("update_mode", ["AutoUpdate", "Manual"])
def test_a_changing_scene_writes_different_sequence_frames(
    tmp_path: Path, update_mode: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    scratch = tmp_path / "houdini-temp"
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    place = {"home": home, "scratch": scratch}
    hip = tmp_path / "moving.hip"

    async def capture() -> dict[str, Any]:
        async with Client(
            server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
        ) as connected:
            worker = ok(await connected.call_tool("hou_sessions", {"action": "start"}))["session"]
            try:
                built = ok(
                    await connected.call_tool(
                        "hou_python",
                        {
                            "code": (
                                "geo = hou.node('/obj').createNode('geo', 'moving')\n"
                                "box = geo.createNode('box')\n"
                                "box.parmTuple('size').set((1, 2, 3))\n"
                                "turn = geo.createNode('xform')\n"
                                "turn.setInput(0, box)\n"
                                "turn.parm('ry').setExpression('$F*3')\n"
                                "turn.setDisplayFlag(True)\n"
                                "turn.setRenderFlag(True)\n"
                                "hou.setFrame(7)\n"
                                f"hou.hipFile.save({str(hip)!r})\n"
                                "result = [list(turn.geometryAtFrame(f).points()[0].position()) "
                                "for f in (1, 12, 23)]\n"
                                f"hou.setUpdateMode(hou.updateMode.{update_mode})"
                            )
                        },
                    )
                )
                assert len({tuple(point) for point in built["result"]}) == 3
                result = ok(
                    await connected.call_tool(
                        "hou_capture",
                        {
                            "source": "viewport",
                            "camera": "front",
                            "frames": [1, 23, 11],
                            "resolution": [320, 180],
                            "return_image": "none",
                        },
                    )
                )
                frame = ok(
                    await connected.call_tool("hou_python", {"code": "result = hou.frame()"})
                )
                assert frame["result"] == 7
                return result
            finally:
                await connected.call_tool(
                    "hou_sessions", {"action": "stop", "session": worker["session_id"]}
                )

    try:
        body = asyncio.run(capture())
    finally:
        assert support.stop_everything(home, pool.PoolConfig(home=home)) == []
    assert len(body["paths"]) == 3
    pixels = []
    for path in body["paths"]:
        with Image.open(path) as image:
            pixels.append(image.convert("RGBA").tobytes())
    differences = [
        sum(a != b for a, b in zip(pixels[0], other, strict=True)) for other in pixels[1:]
    ]
    print({"paths": body["paths"], "different_channel_values": differences})
    assert all(count > 100 for count in differences), differences


def test_repeated_viewport_captures_leave_the_worker_alive(tmp_path: Path) -> None:
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    home.mkdir()
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    place = {"home": home, "scratch": scratch}
    capture = (
        "hou_capture",
        {
            "source": "viewport",
            "camera": "front",
            "resolution": [320, 180],
            "return_image": "none",
        },
    )
    try:
        started, built, *captures, after = run(
            place,
            ("hou_sessions", {"action": "start"}),
            ("hou_python", {"code": BUILD.format(hip=str(tmp_path / "repeated.hip"))}),
            *([capture] * 5),
            ("hou_python", {"code": LOOK}),
        )
        ok(started)
        ok(built)
        for result in captures:
            assert is_png(ok(result)["path"])
            assert ok(result)["route"] == "flipbook_rop"
        assert len({ok(result)["path"] for result in captures}) == 5
        assert ok(after)["result"]["out"] == []
        assert ok(after)["result"]["obj"] == ["boxgeo"]
    finally:
        assert support.stop_everything(home, pool.PoolConfig(home=home)) == []


def test_material_bearing_geometry_survives_repeated_viewport_captures(tmp_path: Path) -> None:
    home, scratch = tmp_path / "home", tmp_path / "scratch"
    home.mkdir()
    scratch.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    place = {"home": home, "scratch": scratch}

    async def capture() -> None:
        async with Client(
            server_params(place), mode="auto", read_timeout_seconds=READ_TIMEOUT_S
        ) as connected:
            worker = ok(
                await connected.call_tool("hou_sessions", {"action": "start", "weight": "light"})
            )["session"]
            try:
                built = ok(
                    await connected.call_tool(
                        "hou_python",
                        {
                            "code": (
                                "import os\n"
                                "g = hou.node('/obj').createNode('geo', 'retest')\n"
                                "t = g.createNode('testgeometry_rubbertoy', 'toy')\n"
                                "x = g.createNode('xform', 'spin')\n"
                                "x.setInput(0, t)\n"
                                "x.parm('ry').setExpression('$F*10')\n"
                                "x.setDisplayFlag(True)\n"
                                "geo = x.geometry()\n"
                                "result = {'points': len(geo.points()), "
                                "'materials': list(set("
                                "geo.primStringAttribValues('shop_materialpath'))), "
                                "'threading': os.environ.get("
                                "'HOUDINI_VULKAN_VIEWER_MULTITHREADING')}"
                            )
                        },
                    )
                )
                assert built["result"]["points"] > 0
                assert any(built["result"]["materials"]), built
                print(built["result"])
                paths = []
                for frame in (1, 6, 11):
                    result = await connected.call_tool(
                        "hou_capture",
                        {
                            "session": worker["session_id"],
                            "source": "viewport",
                            "frame_target": "/obj/retest/spin",
                            "resolution": [320, 240],
                            "frame": frame,
                            "return_image": "none",
                            "wait_s": 30,
                        },
                    )
                    body = ok(result)
                    assert body["image_stats"]["non_empty"] is True
                    assert is_png(body["path"])
                    paths.append(body["path"])
                    print({"frame": frame, "path": body["path"]})
                assert len(set(paths)) == 3
                info = ok(
                    await connected.call_tool(
                        "hou_sessions", {"action": "info", "session": worker["session_id"]}
                    )
                )
                assert info["session"]["state"] == "live"
            finally:
                await connected.call_tool(
                    "hou_sessions", {"action": "stop", "session": worker["session_id"]}
                )

    try:
        asyncio.run(capture())
    finally:
        assert support.stop_everything(home, pool.PoolConfig(home=home)) == []


def test_a_fresh_worker_s_first_capture_is_a_framed_sequence(place: dict[str, Any]) -> None:
    """Run first: a worker that has captured nothing yet, asked for frames."""
    [platform, sequence] = run(
        place,
        ("hou_python", {"code": "import os\nresult = os.environ.get('QT_QPA_PLATFORM')"}),
        ("hou_capture", {"frames": [1, 2, 1], "resolution": [320, 180], "return_image": "none"}),
    )
    assert ok(platform)["result"] == "offscreen"
    body = ok(sequence)
    assert len(body["paths"]) == 2
    for item in body["paths"]:
        assert is_png(item)
        framed(item)
    captures = place["scenes"] / ".agent" / "captures"
    assert list(captures.rglob("*.probe.png")) == []


# A camera an artist animated and locked a parameter on, looking at the box.
SHOT_CAMERA = """
cam = hou.node('/obj').createNode('cam', 'shotcam')
cam.parmTuple('t').set((0, 0, 6))
for frame in (1, 10):
    cam.parm('ty').setKeyframe(hou.Keyframe(0.0, hou.frameToTime(frame)))
cam.parm('winx').lock(True)
result = cam.path()
"""

CAMERA_STATE = """
cam = hou.node('/obj/shotcam')
result = {
    'keys': {parm.name(): len(parm.keyframes()) for parm in cam.parms() if parm.keyframes()},
    'locked': sorted(parm.name() for parm in cam.parms() if parm.isLocked()),
    'values': {parm.name(): parm.eval() for parm in cam.parms()
               if isinstance(parm.eval(), float)},
    'outputs': [node.path() for node in cam.outputs()],
    'obj': sorted(node.name() for node in hou.node('/obj').children()),
}
"""


def test_a_named_camera_is_looked_through_and_left_as_it_was(place: dict[str, Any]) -> None:
    [made] = run(place, ("hou_python", {"code": SHOT_CAMERA}))
    camera = ok(made)["result"]
    try:
        [before] = run(place, ("hou_python", {"code": CAMERA_STATE}))
        [result] = run(
            place,
            ("hou_capture", {"camera": camera, "resolution": [320, 180], "return_image": "none"}),
        )
        [after] = run(place, ("hou_python", {"code": CAMERA_STATE}))
    finally:
        run(place, ("hou_python", {"code": f"hou.node({camera!r}).destroy()"}))
    body = ok(result)
    assert body["camera"] == {"kind": "node", "path": camera}
    framed(body["path"])
    assert ok(after)["result"] == ok(before)["result"]
    assert ok(before)["result"]["keys"]["ty"] == 2
    assert ok(before)["result"]["locked"] == ["winx"]


def test_a_fitted_persp_capture_of_a_box(place: dict[str, Any]) -> None:
    before = look(place)
    [result] = run(place, ("hou_capture", {"camera": "persp", "resolution": [640, 360]}))
    body = ok(result)
    after = look(place)
    assert body["route"] == "flipbook_rop"
    assert body["camera"]["kind"] == "fitted"
    assert (body["width"], body["height"]) == (640, 360)
    path = Path(body["path"])
    assert path.resolve().is_relative_to((place["scenes"] / ".agent" / "captures").resolve())
    assert is_png(str(path))
    plausible(body["image_stats"])
    framed(str(path))
    [image] = [block for block in result.content if isinstance(block, ImageContent)]
    assert base64.b64decode(image.data).startswith(PNG_SIGNATURE)
    assert body["thumb"]["width"] == 512
    # No render node, no camera, no flag moved and nothing on the undo stack.
    assert after == before
    assert after["out"] == []
    assert after["obj"] == ["boxgeo"]


def test_a_node_alone_puts_the_display_flag_back(place: dict[str, Any]) -> None:
    before = look(place)
    assert before["display"] == place["box"]
    [result] = run(
        place,
        ("hou_capture", {"source": "node", "path": place["ball"], "resolution": [320, 180]}),
    )
    body = ok(result)
    assert body["route"] == "flipbook_rop"
    assert body["camera"]["target"] == place["ball"]
    assert is_png(body["path"])
    plausible(body["image_stats"])
    framed(body["path"])
    assert look(place) == before


def test_quad_and_a_two_frame_sequence(place: dict[str, Any]) -> None:
    before = look(place)
    quad, sequence = run(
        place,
        ("hou_capture", {"views": "quad", "resolution": [320, 180], "return_image": "none"}),
        ("hou_capture", {"frames": [1, 2, 1], "resolution": [320, 180], "return_image": "none"}),
    )
    sheet = ok(quad)
    assert [view["view"] for view in sheet["views"]] == ["persp", "top", "front", "right"]
    assert (sheet["width"], sheet["height"]) == (640, 360)
    assert is_png(sheet["path"])
    for view in sheet["views"]:
        assert is_png(view["path"])
        assert view["image_stats"]["non_empty"] is True, view
        framed(view["path"])
    assert [view["camera"]["projection"] for view in sheet["views"]] == [
        "perspective",
        "ortho",
        "ortho",
        "ortho",
    ]
    frames = ok(sequence)
    assert frames["frames"] == [1.0, 2.0]
    assert [Path(item).name[-8:] for item in frames["paths"]] == ["0001.png", "0002.png"]
    assert all(is_png(item) for item in frames["paths"])
    plausible(frames["image_stats"])
    assert look(place) == before


# A shown camera, light and null away from the box, with the null's cross in
# front of the box where a capture would draw it.
AROUND = """
obj = hou.node('/obj')
cam = obj.createNode('cam', 'sidecam')
cam.parmTuple('t').set((6, 3, 0))
light = obj.createNode('hlight::2.0', 'sidelight')
light.parmTuple('t').set((-5, 4, 2))
marker = obj.createNode('null', 'marker')
marker.parmTuple('t').set((0.3, 0.3, 0.9))
result = [cam.path(), light.path(), marker.path()]
"""


def green(path: str) -> int:
    """How many drawn pixels are the green a null's cross is drawn in."""
    with Image.open(path) as image:
        data = image.convert("RGBA").tobytes()
    pixels = (data[index : index + 4] for index in range(0, len(data), 4))
    return sum(1 for r, g, b, a in pixels if a and g > r + 40 and g > b + 40)


def test_framing_all_leaves_out_cameras_lights_and_nulls(place: dict[str, Any]) -> None:
    before = look(place)
    [made] = run(place, ("hou_python", {"code": AROUND}))
    extras = ok(made)["result"]
    try:
        plain, guided = run(
            place,
            ("hou_capture", {"frame_target": "all", "resolution": [320, 180]}),
            ("hou_capture", {"frame_target": "all", "guides": True, "resolution": [320, 180]}),
        )
    finally:
        gone = "\n".join(f"hou.node({path!r}).destroy()" for path in extras)
        run(place, ("hou_python", {"code": gone}))
    body = ok(plain)
    assert body["camera"]["target"] == "all"
    assert not any("nothing to frame" in item for item in body.get("warnings", []))
    framed(body["path"])
    assert green(body["path"]) == 0
    # With guides the null's cross is drawn, and the framing is the same.
    framed(ok(guided)["path"])
    assert green(ok(guided)["path"]) > 20
    after = look(place)
    assert after["obj"] == before["obj"]
    assert after["out"] == []


def around(place: dict[str, Any], code: str, *calls: tuple[str, dict]) -> list[Any]:
    """Captures in a scene with more in it, which is taken away again after.

    `code` returns the paths it made; the box's object is shown again after,
    whatever the code did with it.
    """
    before = look(place)
    [made] = run(place, ("hou_python", {"code": code}))
    extras = ok(made)["result"]
    try:
        return run(place, *calls)
    finally:
        gone = "\n".join(f"hou.node({path!r}).destroy()" for path in extras)
        gone += "\nhou.node('/obj/boxgeo').setDisplayFlag(True)"
        run(place, ("hou_python", {"code": gone}))
        after = look(place)
        assert after["obj"] == before["obj"]
        assert after["out"] == []


def drawn_box(path: str) -> tuple[float, float]:
    """The width and height of what is drawn, as shares of the frame."""
    with Image.open(path) as image:
        width, height = image.size
        left, top, right, bottom = image.getchannel("A").getbbox()
    return (right - left) / width, (bottom - top) / height


# A simulation network shown, as every shelf simulation makes one.
SIMULATION = """
dop = hou.node('/obj').createNode('dopnet', 'sim')
dop.setDisplayFlag(True)
result = [dop.path()]
"""


def test_a_shown_simulation_does_not_stop_a_capture_of_all(place: dict[str, Any]) -> None:
    [result] = around(
        place,
        SIMULATION,
        ("hou_capture", {"frame_target": "all", "resolution": [320, 180]}),
    )
    body = ok(result)
    assert body["route"] == "flipbook_rop"
    framed(body["path"])


# An instance object copying a ball onto its point, beside the box. The ball's
# own object is hidden; the copy is drawn anyway.
INSTANCES = """
obj = hou.node('/obj')
source = obj.createNode('geo', 'ballsource')
source.createNode('sphere').setDisplayFlag(True)
source.setDisplayFlag(False)
placed = obj.createNode('instance', 'copies')
placed.parm('instancepath').set(source.path())
placed.parm('ptinstance').set('on')
placed.parmTuple('t').set((2.5, 0, 0))
result = [placed.path(), source.path()]
"""


def test_an_instance_object_is_drawn_and_framed(place: dict[str, Any]) -> None:
    [result] = around(
        place,
        INSTANCES,
        ("hou_capture", {"camera": "front", "frame_target": "all", "resolution": [320, 180]}),
    )
    body = ok(result)
    framed(body["path"])
    # The box and the ball side by side: four units across, two up.
    across, up = drawn_box(body["path"])
    assert across * 320 / (up * 180) > 1.6, (across, up)


# A ball above the box, shown only at frame 10, while the scene is at frame 1.
LATE = """
obj = hou.node('/obj')
late = obj.createNode('geo', 'late')
late.createNode('sphere').setDisplayFlag(True)
late.parmTuple('t').set((0, 3, 0))
late.parm('tdisplay').set(True)
late.parm('display').setExpression('$F == 10')
hou.setFrame(1)
result = [late.path()]
"""


def test_an_object_shown_only_at_the_frame_captured_is_drawn_and_framed(
    place: dict[str, Any],
) -> None:
    [result] = around(
        place,
        LATE,
        (
            "hou_capture",
            {"camera": "front", "frame_target": "all", "frame": 10, "resolution": [320, 180]},
        ),
    )
    body = ok(result)
    framed(body["path"])
    # The box and the ball above it: two units across, four and a half up.
    across, up = drawn_box(body["path"])
    assert up * 180 / (across * 320) > 1.6, (across, up)


# Nothing but a null: the box's object is hidden for the call.
ONLY_A_NULL = """
hou.node('/obj/boxgeo').setDisplayFlag(False)
marker = hou.node('/obj').createNode('null', 'marker')
result = [marker.path()]
"""


def test_a_scene_with_only_a_null_draws_no_cross_without_guides(place: dict[str, Any]) -> None:
    plain, guided = around(
        place,
        ONLY_A_NULL,
        ("hou_capture", {"frame_target": "all", "resolution": [320, 180]}),
        ("hou_capture", {"frame_target": "all", "guides": True, "resolution": [320, 180]}),
    )
    # Without guides nothing is drawn, so there is no picture to give.
    assert plain.is_error
    assert "CAPTURE_EMPTY" in plain.content[0].text
    body = ok(guided)
    assert any("frames the origin" in item for item in body["warnings"])
    assert green(body["path"]) > 20


# Fifteen hundred boxes, then the time framing all of them takes in the worker,
# cooked once first, as a capture would find them.
MANY = """
import time
from nscr_houdini_mcp.bridge import capture
obj = hou.node('/obj')
made = []
for index in range(1500):
    geo = obj.createNode('geo', f'many{index}')
    geo.createNode('box').setDisplayFlag(True)
    geo.parmTuple('t').set((index % 50, index // 50, 0))
    made.append(geo)
nodes, capped = capture.drawn_objects(hou, (1.0,))
capture.world_bounds(hou, nodes, (1.0,))
started = time.perf_counter()
bounds = capture.world_bounds(hou, nodes, (1.0,))
took = time.perf_counter() - started
for geo in made:
    geo.destroy()
result = {'took': took, 'count': len(nodes), 'capped': capped, 'bounds': bounds}
"""


def test_framing_many_objects_is_quick(place: dict[str, Any]) -> None:
    [said] = run(place, ("hou_python", {"code": MANY}))
    result = ok(said)["result"]
    assert result["count"] == 1501 and result["capped"] is False
    assert result["bounds"][1][:2] == [49.5, 29.5]
    # About 10 microseconds an object, 17 ms in all; reading the vectors with
    # list() took 0.3 s here. The bound leaves room for a busy machine.
    assert result["took"] < 0.15, result["took"]
