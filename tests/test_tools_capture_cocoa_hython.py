"""`hou_capture` in a worker whose shell names Qt's own macOS screen plugin.

A pool worker is put on the offscreen plugin unless its shell names one, and
a shell that names `cocoa` keeps it. On a dense display that plugin makes a
render node draw at twice the size asked, so the capture has to read its
drawing scale and make up for it, or say the framing is unverified. Either
is an answer; a corner of the frame given as the whole is not.

macOS only, and skipped when there is no Houdini. House rules as in the other
checks that start a Houdini: one worker, a state folder of this file's own,
the pool's port range, and every worker stopped again at the end.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

import support
from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge import registry
from test_tools_capture_hython import BUILD, SERVER_CODE, framed, hython_available, is_png, ok

pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
    pytest.mark.skipif(sys.platform != "darwin", reason="the cocoa screen plugin is macOS only"),
]

PORT_RANGE = support.POOL_PORTS
READ_TIMEOUT_S = 300.0


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    root = tmp_path_factory.mktemp("capture_cocoa")
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
        ok(started)
        hip = scenes / "cocoa_scene.hip"
        [built] = run(made, ("hou_python", {"code": BUILD.format(hip=str(hip))}))
        ok(built)
        yield made
    finally:
        left = support.stop_everything(home, pool.PoolConfig(home=home))
        assert left == [], f"workers were left running: {left}"
        assert registry.live_entries(home) == []


def run(place: dict[str, Any], *calls: tuple[str, dict]) -> list[Any]:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-c", SERVER_CODE],
        env={
            "NSCR_MCP_HOME": str(place["home"]),
            "HOUDINI_TEMP_DIR": str(place["scratch"]),
            "PYTHONIOENCODING": "utf-8",
            # Carried on to the worker, which keeps a plugin its shell names.
            "QT_QPA_PLATFORM": "cocoa",
        },
    )

    async def go() -> list[Any]:
        async with Client(params, mode="auto", read_timeout_seconds=READ_TIMEOUT_S) as client:
            return [await client.call_tool(name, arguments) for name, arguments in calls]

    return asyncio.run(go())


def test_a_worker_on_cocoa_is_calibrated_or_says_so(place: dict[str, Any]) -> None:
    platform, still, frames = run(
        place,
        ("hou_python", {"code": "import os\nresult = os.environ.get('QT_QPA_PLATFORM')"}),
        ("hou_capture", {"camera": "persp", "resolution": [320, 180], "return_image": "none"}),
        ("hou_capture", {"frames": [1, 2, 1], "resolution": [320, 180], "return_image": "none"}),
    )
    assert ok(platform)["result"] == "cocoa"
    for body in (ok(still), ok(frames)):
        for item in body.get("paths") or [body["path"]]:
            assert is_png(item)
            if not body.get("framing_unverified"):
                framed(item)
    captures = place["scenes"] / ".agent" / "captures"
    assert list(captures.rglob("*.probe.png")) == []
