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
        return [await connected.call_tool(name, arguments) for name, arguments in calls]


def run(place: dict[str, Any], *calls: tuple[str, dict]) -> list[Any]:
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
