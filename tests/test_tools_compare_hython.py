"""`hou_compare` with a scene linear file, through the real server and a real worker.

The unit tests read an EXR through a stand in `hou`. This file checks the same
route against Houdini itself: the COP `file` node, `addaovs`, the output
names, the layer, its buffer as float32 and the row order, and the display
transform from the session's OpenColorIO configuration. The EXR is made from
a PNG with the `iconvert` beside hython, so the test carries no binary file.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap
in this file's own config is one), a state folder of this file's own, the
pool's port range, and every worker stopped again whatever happened.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters
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

READ_TIMEOUT_S = 300.0
CONVERT_TIMEOUT_S = 120.0


def iconvert() -> Path:
    name = "iconvert.exe" if sys.platform == "win32" else "iconvert"
    found = pool.hython_path().parent / name
    if not found.is_file():
        pytest.skip("no iconvert beside hython")
    return found


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("compare")
    home = root / "home"
    home.mkdir()
    scratch = root / "houdini-temp"
    scratch.mkdir()
    images = root / "images"
    images.mkdir()
    (home / "config.toml").write_text(
        f"pool_cap = 1\nworker_ports = [{PORT_RANGE[0]}, {PORT_RANGE[1]}]\n", encoding="utf-8"
    )
    made: dict[str, Any] = {"home": home, "scratch": scratch, "images": images}
    with support.persistent_client(server_params(made), timeout_s=READ_TIMEOUT_S) as send:
        made["send"] = send
        try:
            [started] = run(made, ("hou_sessions", {"action": "start"}))
            made["worker"] = ok(started)["session"]
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
    if place.get("send") is not None:
        return place["send"](*calls)
    return asyncio.run(_run(place, list(calls)))


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def halves(path: Path) -> Path:
    """Light on top, dark below, so the row order shows whatever the curve does."""
    pixels = np.zeros((48, 64, 3), dtype=np.uint8)
    pixels[:24] = 230
    pixels[24:] = 40
    Image.fromarray(pixels).save(path)
    return path


def test_an_exr_is_read_by_the_session_through_its_display_transform(
    place: dict[str, Any],
) -> None:
    png = halves(place["images"] / "halves.png")
    exr = place["images"] / "halves.exr"
    subprocess.run([str(iconvert()), str(png), str(exr)], check=True, timeout=CONVERT_TIMEOUT_S)
    assert exr.is_file()

    registered, compared, leftovers = run(
        place,
        ("hou_compare", {"action": "set_reference", "reference": str(png), "name": "halves"}),
        (
            "hou_compare",
            {"candidate": {"source": "file", "path": str(exr)}, "reference": "halves"},
        ),
        (
            "hou_python",
            {
                "code": (
                    "result = [p for p in ('/img/nscr_compare_read', '/obj/nscr_compare_read')"
                    " if hou.node(p) is not None]"
                )
            },
        ),
    )
    assert ok(registered)["ref_id"].startswith("ref-")
    data = ok(compared)
    colour = data["colour"]["candidate"]
    assert colour["kind"] == "view_transform"
    assert colour["transform"] in ("ocio_display_view", "srgb_curve")
    if colour["transform"] == "ocio_display_view":
        assert colour["display"] and colour["view"]
    assert colour["channel"]
    assert data["steps"]["grid_px"] == [64, 48]
    read = data["steps"]["session_read"]["candidate"]
    assert read["route"] in ("OpenImageIO", "cop_file_node")
    assert read["scene_marked_changed"] is (read["route"] == "cop_file_node")
    if colour["transform"] == "srgb_curve":
        assert data["transfer_mismatch_possible"]["flag"] is False

    # Rows come from the top: the light half is on top in the aligned copy.
    candidate = np.asarray(Image.open(data["files"]["candidate"]).convert("L"), dtype=float)
    assert candidate[:20].mean() > candidate[28:].mean() + 60

    # The network the read was made in is gone again.
    assert ok(leftovers)["result"] == []
