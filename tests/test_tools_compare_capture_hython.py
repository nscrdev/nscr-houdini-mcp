"""`hou_compare` with a node candidate, captured by a real worker for the compare.

A reference is registered from a first capture of a box, seen through a
camera of the scene's own. A `node` candidate of the same box, framed by the
reference's camera, is then compared against it, and again once the box has
moved. A `render` candidate reads what a node and a job wrote. Nothing is
stood in for.

Skipped, not failed, when there is no Houdini on this machine. House rules as
in the other checks that start a Houdini: one worker at a time (the pool cap
in this file's own config is one), a state folder of this file's own, the
pool's port range, and every worker stopped again whatever happened. The
scene file, every capture and every compare are inside the test's own
temporary folder.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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

SIZE = [320, 180]

# A box at the origin and a camera looking at it down the Z axis, in a scene
# saved where the test can find its files.
BUILD = """
geo = hou.node('/obj').createNode('geo', 'boxgeo')
box = geo.createNode('box', 'box1')
box.setDisplayFlag(True)
box.setRenderFlag(True)
cam = hou.node('/obj').createNode('cam', 'cam1')
cam.parmTuple('t').set((0, 0, 6))
cam.parm('resx').set(320)
cam.parm('resy').set(180)
hou.hipFile.save({hip!r})
result = [box.path(), cam.path()]
"""

MOVE = "hou.node({box!r}).parmTuple('t').set(({x}, 0, 0))\nresult = True"


@pytest.fixture(scope="module")
def place(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A state folder with a config of its own and one worker, stopped at the end."""
    root = tmp_path_factory.mktemp("compare_capture")
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
        hip = scenes / "compare_scene.hip"
        [built] = run(made, ("hou_python", {"code": BUILD.format(hip=str(hip))}))
        made["box"], made["camera"] = ok(built)["result"]
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


def compared(place: dict[str, Any], candidate: dict[str, Any], **rest: Any) -> dict[str, Any]:
    arguments = {"candidate": candidate, "reference": "box", "return_image": "none", **rest}
    [result] = run(place, ("hou_compare", arguments))
    return ok(result)


def test_a_node_candidate_against_a_reference_from_a_first_capture(
    place: dict[str, Any],
) -> None:
    box, camera = place["box"], place["camera"]
    [first] = run(
        place,
        (
            "hou_capture",
            {
                "source": "node",
                "path": box,
                "camera": camera,
                "resolution": SIZE,
                "return_image": "none",
            },
        ),
    )
    reference = ok(first)
    assert reference["image_stats"]["non_empty"] is True
    [registered] = run(
        place,
        (
            "hou_compare",
            {
                "action": "set_reference",
                "reference": reference["path"],
                "name": "box",
                "camera": camera,
            },
        ),
    )
    assert ok(registered)["size_px"] == SIZE

    # The same box, framed by the reference's camera at its size: next to no difference.
    same = compared(place, {"source": "node", "path": box}, mode="regression")
    made = same["sources"]["candidate"]
    assert made["framed_by"] == "reference"
    assert made["camera"] == {"kind": "node", "path": camera}
    assert made["route"] == "flipbook_rop"
    assert made["job_id"] == f"job-{same['trace']['operation_id']}:capture"
    assert made["frame"] == 1.0
    assert made["size_px"] == SIZE
    path = Path(made["path"])
    assert path.is_file() and path != Path(reference["path"])
    assert path.resolve().is_relative_to((place["scenes"] / ".agent" / "captures").resolve())
    assert made["run_id"] in path.name
    with Image.open(path) as image:
        assert image.size == tuple(SIZE)
    kept = json.loads(Path(same["files"]["result"]).read_text(encoding="utf-8"))
    assert kept["sources"]["candidate"]["run_id"] == made["run_id"]
    assert (Path(same["folder"]) / kept["sources"]["candidate"]["path"]).resolve() == (
        path.resolve()
    )
    metrics = same["metrics"]
    assert metrics["mae"]["overall"] < 0.002, metrics
    assert metrics["diff_area_pct"] < 0.5, metrics
    for key in ("candidate", "reference", "diff", "overview"):
        assert Path(same["files"][key]).is_file(), key

    # The box moved half a unit to the right: the compare reports the shift.
    [moved] = run(place, ("hou_python", {"code": MOVE.format(box=box, x=0.5)}))
    try:
        assert ok(moved)["result"] is True
        before = compared(place, {"source": "node", "path": box}, mode="regression")
        shifted = compared(
            place, {"source": "node", "path": box}, mode="regression", auto_shift=True
        )
    finally:
        run(place, ("hou_python", {"code": MOVE.format(box=box, x=0)}))
    assert before["metrics"]["diff_area_pct"] > 2.0, before["metrics"]
    assert before["largest_region"] is not None
    assert before["series"]["id"] == same["series"]["id"]
    assert before["series"]["runs_before"] == 1
    shift = shifted["steps"]["shift_px"]
    assert shift["applied"] is True, shift
    # The candidate sits to the right, so it is moved left to line up.
    assert shift["dx"] < -10, shift
    assert abs(shift["dy"]) <= 1, shift
    assert shifted["metrics"]["mae"]["overall"] < before["metrics"]["mae"]["overall"] / 3

    # A render candidate reads the newest file the node wrote, and a job's own file.
    by_node = compared(place, {"source": "render", "path": box})
    newest = by_node["sources"]["candidate"]
    assert newest["run_id"] == shifted["sources"]["candidate"]["run_id"]
    assert newest["node"] == box
    by_job = compared(place, {"source": "render", "job_id": reference["job_id"]})
    assert by_job["sources"]["candidate"]["path"] == reference["path"]
    assert by_job["metrics"]["mae"]["overall"] == 0.0


# A Python job that writes a small PNG, by hand, to a path the job was handed.
PYTHON_RENDER = """
import struct, zlib
path = mcp.output_path('render', 'flat', 'png').replace('$F4', '0001')
width, height = 64, 32
rows = b''.join(b'\\x00' + bytes((128, 128, 128, 255)) * width for _ in range(height))
def chunk(tag, body):
    return (struct.pack('>I', len(body)) + tag + body
            + struct.pack('>I', zlib.crc32(tag + body) & 0xffffffff))
header = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
data = (b'\\x89PNG\\r\\n\\x1a\\n' + chunk(b'IHDR', header)
        + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))
with open(path, 'wb') as handle:
    handle.write(data)
result = path
"""


def test_a_render_candidate_by_the_job_id_of_a_python_job(place: dict[str, Any]) -> None:
    [written] = run(place, ("hou_python", {"code": PYTHON_RENDER}))
    job = ok(written)
    path = job["result"]
    assert Path(path).is_file()
    [registered] = run(
        place,
        ("hou_compare", {"action": "set_reference", "reference": path, "name": "flat"}),
    )
    ok(registered)
    [result] = run(
        place,
        (
            "hou_compare",
            {
                "candidate": {"source": "render", "job_id": job["job_id"]},
                "reference": "flat",
                "return_image": "none",
            },
        ),
    )
    body = ok(result)
    made = body["sources"]["candidate"]
    assert made["path"] == path
    assert made["job_kind"] == "python"
    assert body["metrics"]["mae"]["overall"] == 0.0
