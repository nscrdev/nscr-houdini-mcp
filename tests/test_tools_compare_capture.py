"""`hou_compare` with a candidate the session draws, or one a job or a node wrote.

Every call goes the whole way, as in the capture checks: the server checks
the arguments, a real dispatcher with real receipts and job rows runs the
bridge's `scene.info` and `capture.image` against the stand in for `hou`,
whose flipbook writes real PNG files, and the compare reads them from disk.
What a real Houdini draws is in the integration file.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp.tools import compare as compare_tool
from test_server import talk
from test_tools_capture import TYPES, Through
from test_tools_sessions import Bench

BOX = "/obj/boxgeo/box1"
CAMERA = "/obj/cam1"
LOOK = {"reference": "look", "return_image": "none"}


@pytest.fixture
def scene(tmp_path: Path) -> Iterator[Scene]:
    made = Scene(types=TYPES)
    made.hipFile.setName(str(tmp_path / "shot.hip"))
    geo = made.node("/obj").createNode("geo", "boxgeo")
    geo.createNode("box", "box1").setDisplayFlag(True)
    made.node("/obj").createNode("cam", "cam1")
    made.undos.labels.clear()
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def bench(tmp_path: Path, scene: Scene, monkeypatch: pytest.MonkeyPatch) -> Bench:
    monkeypatch.delenv("HOUDINI_TEMP_DIR", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1", kind="hython")
    made.sent = Through(scene.module(), home, "hython")  # type: ignore[assignment]
    return made


def call(bench: Bench, *calls: tuple[str, dict[str, Any]]) -> list[Any]:
    _, results = talk(bench.serve(), *calls)
    return results


def one(bench: Bench, tool: str, **arguments: Any) -> Any:
    [result] = call(bench, (tool, arguments))
    return result


def ok(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def refused(result: Any) -> dict[str, Any]:
    assert result.is_error is True
    return result.structured_content["error"]


def capture(bench: Bench, **arguments: Any) -> dict[str, Any]:
    return ok(one(bench, "hou_capture", return_image="none", **arguments))


def register(bench: Bench, image: str, **rest: Any) -> dict[str, Any]:
    return ok(
        one(bench, "hou_compare", action="set_reference", reference=image, name="look", **rest)
    )


def compare(bench: Bench, candidate: dict[str, Any], **rest: Any) -> Any:
    return one(
        bench, "hou_compare", candidate=candidate, reference="look", return_image="none", **rest
    )


def saved(result: dict[str, Any]) -> dict[str, Any]:
    return json.loads(Path(result["files"]["result"]).read_text(encoding="utf-8"))


# Section: a candidate captured for the compare


def test_a_viewport_candidate_is_captured_and_compared(
    bench: Bench, scene: Scene, tmp_path: Path
) -> None:
    first = capture(bench, resolution=[320, 180])
    register(bench, first["path"])
    result = ok(compare(bench, {"source": "viewport", "resolution": [320, 180]}))
    assert result["metrics"]["mae"]["overall"] == 0.0
    made = result["sources"]["candidate"]
    path = Path(made["path"])
    assert path.is_file() and path != Path(first["path"])
    assert path.parent.parent.name == "captures"
    assert made["run_id"] in path.name
    assert made["source"] == "viewport"
    assert made["route"] == "flipbook_rop"
    assert made["framed_by"] == "capture"
    assert made["camera"]["kind"] == "fitted"
    assert made["size_px"] == [320, 180]
    # The capture goes under an id derived from the compare's own.
    assert made["job_id"] == f"job-{result['trace']['operation_id']}:capture"
    # The file on record is the capture's, named relative to the result.
    kept = saved(result)["sources"]["candidate"]
    assert kept["run_id"] == made["run_id"]
    assert (Path(result["folder"]) / kept["path"]).resolve() == path.resolve()
    assert str(tmp_path) not in json.dumps(kept)
    # The capture is a job and a run like any other.
    status = ok(one(bench, "hou_jobs", job_id=made["job_id"]))
    assert status["state"] == "done" and status["kind"] == "capture"
    assert len(scene.capture.seen) == 2


def test_a_node_candidate_is_captured_with_the_capture_arguments(
    bench: Bench, scene: Scene
) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[200, 100])
    register(bench, first["path"])
    result = ok(
        compare(
            bench,
            {"source": "node", "path": BOX, "resolution": [200, 100], "display": "wire"},
        )
    )
    assert result["metrics"]["mae"]["overall"] == 0.0
    made = result["sources"]["candidate"]
    assert made["source"] == "node"
    assert made["camera"]["target"] == BOX
    seen = scene.capture.seen[-1]
    assert seen["size"] == (200, 100)
    assert seen["shadingmode"] != scene.capture.seen[0]["shadingmode"]


def test_a_candidate_region_crops_the_capture(bench: Bench) -> None:
    first = capture(bench, resolution=[400, 200])
    register(bench, first["path"])
    result = ok(
        compare(bench, {"source": "viewport", "resolution": [400, 200], "region": [0, 0, 0.5, 1]})
    )
    assert result["sources"]["candidate"]["size_px"] == [200, 200]


def test_an_empty_capture_is_capture_empty_and_no_compare_is_made(
    bench: Bench, scene: Scene
) -> None:
    first = capture(bench, resolution=[64, 32])
    register(bench, first["path"])
    scene.capture.blank = True
    error = refused(compare(bench, {"source": "viewport"}))
    assert error["code"] == "CAPTURE_EMPTY"
    assert not (Path(first["path"]).parents[3] / "compare").exists()


def test_a_node_the_session_does_not_have_is_node_not_found(bench: Bench) -> None:
    first = capture(bench, resolution=[64, 32])
    register(bench, first["path"])
    error = refused(compare(bench, {"source": "node", "path": "/obj/nothing"}))
    assert error["code"] == "NODE_NOT_FOUND"


# Section: the reference's camera


def test_the_reference_camera_frames_the_capture_at_the_reference_aspect(
    bench: Bench, scene: Scene
) -> None:
    first = capture(bench, source="node", path=BOX, camera=CAMERA, resolution=[300, 150])
    register(bench, first["path"], camera=CAMERA)
    result = ok(compare(bench, {"source": "node", "path": BOX}))
    made = result["sources"]["candidate"]
    assert made["framed_by"] == "reference"
    assert made["camera"] == {"kind": "node", "path": CAMERA}
    assert made["size_px"] == [300, 150]
    seen = scene.capture.seen[-1]
    assert seen["follows"] == CAMERA
    assert seen["size"] == (300, 150)
    assert result["metrics"]["mae"]["overall"] == 0.0


def test_a_candidate_camera_or_size_wins_over_the_reference(bench: Bench, scene: Scene) -> None:
    first = capture(bench, source="node", path=BOX, camera=CAMERA, resolution=[300, 150])
    register(bench, first["path"], camera=CAMERA)
    own = ok(compare(bench, {"source": "node", "path": BOX, "camera": "front"}))
    assert own["sources"]["candidate"]["framed_by"] == "candidate"
    assert own["sources"]["candidate"]["camera"]["kind"] == "fitted"
    assert scene.capture.seen[-1]["follows"] is None
    sized = ok(compare(bench, {"source": "node", "path": BOX, "resolution": [120, 120]}))
    assert sized["sources"]["candidate"]["framed_by"] == "reference"
    assert scene.capture.seen[-1]["size"] == (120, 120)
    assert scene.capture.seen[-1]["follows"] == CAMERA
    # A different framing is a different series.
    assert own["series"]["id"] != sized["series"]["id"]


def test_a_large_reference_is_captured_at_its_aspect_within_the_edge() -> None:
    assert compare_tool.reference_size(300, 150) == [300, 150]
    assert compare_tool.reference_size(4096, 1716) == [2048, 858]
    assert compare_tool.reference_size(1000, 5000) == [410, 2048]


# Section: a render: the newest file of a job or a node


def test_a_render_by_job_id_is_that_capture_s_file(bench: Bench) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[160, 90])
    register(bench, first["path"])
    later = capture(bench, source="node", path=BOX, resolution=[160, 90])
    result = ok(compare(bench, {"source": "render", "job_id": first["job_id"]}))
    made = result["sources"]["candidate"]
    assert made["source"] == "render"
    assert made["job_id"] == first["job_id"]
    assert made["run_id"] == first["run_id"] != later["run_id"]
    assert made["path"] == first["path"]
    assert made["node"] == BOX
    assert made["job_kind"] == "capture"
    assert result["metrics"]["mae"]["overall"] == 0.0


def test_a_render_by_node_is_the_newest_file_that_node_wrote(bench: Bench) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[160, 90])
    register(bench, first["path"])
    later = capture(bench, source="node", path=BOX, resolution=[160, 90])
    # A viewport capture is no picture of the node.
    capture(bench, resolution=[160, 90])
    result = ok(compare(bench, {"source": "render", "path": BOX}))
    made = result["sources"]["candidate"]
    assert made["run_id"] == later["run_id"]
    assert made["path"] == later["path"]
    assert made["job_id"] == later["job_id"]
    assert made["node"] == BOX


def test_a_sequence_s_newest_frame_is_the_render(bench: Bench) -> None:
    first = capture(bench, source="node", path=BOX, frames=[1, 3, 1], resolution=[64, 36])
    register(bench, first["paths"][0])
    result = ok(compare(bench, {"source": "render", "job_id": first["job_id"]}))
    assert result["sources"]["candidate"]["path"] == first["paths"][-1]


def test_a_node_whose_file_is_gone_falls_back_to_an_earlier_run(bench: Bench) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[160, 90])
    register(bench, first["path"])
    later = capture(bench, source="node", path=BOX, resolution=[160, 90])
    Path(later["path"]).unlink()
    result = ok(compare(bench, {"source": "render", "path": BOX}))
    assert result["sources"]["candidate"]["run_id"] == first["run_id"]


def test_a_job_still_running_is_job_running(bench: Bench, scene: Scene) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[64, 36])
    register(bench, first["path"])
    scene.capture.delay_s = 0.4
    handle = ok(
        one(
            bench,
            "hou_capture",
            source="node",
            path=BOX,
            frames=[1, 4, 1],
            resolution=[64, 36],
            timeout_s=0.2,
        )
    )
    assert handle["state"] == "running"
    by_job, by_node = call(
        bench,
        ("hou_compare", {"candidate": {"source": "render", "job_id": handle["job_id"]}, **LOOK}),
        ("hou_compare", {"candidate": {"source": "render", "path": BOX}, **LOOK}),
    )
    for result in (by_job, by_node):
        error = refused(result)
        assert error["code"] == "JOB_RUNNING"
        assert error["details"]["job_id"] == handle["job_id"]
        assert "hou_jobs" in error["hint"]
    body: dict[str, Any] = {}
    for _ in range(10):
        body = ok(one(bench, "hou_jobs", job_id=handle["job_id"], wait_s=5))
        if body["state"] not in ("queued", "running"):
            break
    assert body["state"] == "done"
    finished = ok(compare(bench, {"source": "render", "job_id": handle["job_id"]}))
    assert finished["sources"]["candidate"]["path"] == body["outputs"]["paths"][-1]


def test_a_node_nothing_wrote_is_no_output(bench: Bench) -> None:
    first = capture(bench, resolution=[64, 32])
    register(bench, first["path"])
    error = refused(compare(bench, {"source": "render", "path": BOX}))
    assert error["code"] == "NO_OUTPUT"
    assert error["details"] == {"node": BOX}


def test_a_job_that_wrote_no_image_is_no_output(bench: Bench, scene: Scene) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[64, 32])
    register(bench, first["path"])
    scene.capture.blank = True
    empty = refused(one(bench, "hou_capture", source="node", path=BOX, resolution=[64, 32]))
    assert empty["code"] == "CAPTURE_EMPTY"
    [job] = [
        row["job_id"]
        for row in ok(one(bench, "hou_jobs", action="list"))["jobs"]
        if row["job_id"] != first["job_id"]
    ]
    error = refused(compare(bench, {"source": "render", "job_id": job}))
    assert error["code"] == "NO_OUTPUT"
    assert error["details"]["error"] == "CAPTURE_EMPTY"
    # The empty capture is no picture of the node either: the earlier one is.
    by_node = ok(compare(bench, {"source": "render", "path": BOX}))
    assert by_node["sources"]["candidate"]["run_id"] == first["run_id"]


def test_a_job_whose_file_is_gone_is_no_output(bench: Bench) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[64, 32])
    register(bench, first["path"])
    later = capture(bench, source="node", path=BOX, resolution=[64, 32])
    Path(later["path"]).unlink()
    error = refused(compare(bench, {"source": "render", "job_id": later["job_id"]}))
    assert error["code"] == "NO_OUTPUT"
    assert error["details"]["state"] == "done"


def test_an_unknown_job_is_job_unknown(bench: Bench) -> None:
    first = capture(bench, resolution=[64, 32])
    register(bench, first["path"])
    error = refused(compare(bench, {"source": "render", "job_id": "job-nothing"}))
    assert error["code"] == "JOB_UNKNOWN"


# Section: what a capture's run records


def test_a_node_capture_s_runs_name_the_node_and_the_job(bench: Bench) -> None:
    made = capture(bench, source="node", path=BOX + "/", views="quad", resolution=[32, 32])
    with bench.store() as store:
        by_job = store.runs_made_by(job_id=made["job_id"])
        by_node = store.runs_made_by(source_node=BOX)
    # Four views and the sheet, newest first.
    assert len(by_job) == 5
    assert {run.run_id for run in by_job} == {run.run_id for run in by_node}
    assert by_node[0].run_id == made["run_id"]
    viewport = capture(bench, resolution=[32, 32])
    with bench.store() as store:
        [own] = store.runs_made_by(job_id=viewport["job_id"])
    assert own.source_node is None


# Section: what the capture says, and where its errors belong


def test_the_capture_s_warnings_frame_and_unsaved_scene_reach_the_compare(
    bench: Bench, scene: Scene
) -> None:
    first = capture(bench, source="node", path=BOX, camera=CAMERA, resolution=[160, 90])
    register(bench, first["path"])
    result = ok(
        compare(
            bench,
            {"source": "node", "path": BOX, "camera": CAMERA, "frame_target": BOX, "frame": 3},
        )
    )
    assert any("frame_target was not applied" in item for item in result["warnings"])
    made = result["sources"]["candidate"]
    assert made["frame"] == 3.0
    assert "unsaved_hip" not in made
    assert saved(result)["sources"]["candidate"]["frame"] == 3.0


def test_a_capture_of_an_unsaved_scene_says_so(tmp_path: Path, scene: Scene) -> None:
    scene.hipFile.setName("untitled.hip")
    home = tmp_path / "unsaved_home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1", kind="hython")
    made.sent = Through(scene.module(), home, "hython")  # type: ignore[assignment]
    first = capture(made, resolution=[64, 32])
    register(made, first["path"])
    result = ok(compare(made, {"source": "viewport", "resolution": [64, 32]}))
    assert result["sources"]["candidate"]["unsaved_hip"] is True
    assert any("has not been saved" in item for item in result["warnings"])


def test_a_reference_camera_that_is_gone_is_the_reference_s_error(
    bench: Bench, scene: Scene
) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[160, 90])
    register(bench, first["path"], camera="/obj/cam2")
    seen = len(scene.capture.seen)
    error = refused(compare(bench, {"source": "node", "path": BOX}))
    assert error["code"] == "NODE_NOT_FOUND"
    assert error["details"]["argument"] == "reference"
    assert error["details"]["camera"] == "/obj/cam2"
    assert error["hint"] == "register the reference again, or pass candidate.camera"
    assert len(scene.capture.seen) == seen
    # A camera the reference names that is not a camera is the reference's too.
    register(bench, first["path"], camera="/obj/boxgeo")
    error = refused(compare(bench, {"source": "node", "path": BOX}))
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == "reference"
    # The candidate's own camera is still the candidate's.
    error = refused(compare(bench, {"source": "node", "path": BOX, "camera": "/obj/cam9"}))
    assert error["details"]["argument"] == "candidate.camera"


@pytest.mark.parametrize("camera", ["cam1", "/obj/my cam", ""])
def test_set_reference_takes_a_camera_only_as_a_node_path(bench: Bench, camera: str) -> None:
    first = capture(bench, resolution=[64, 32])
    result = one(
        bench,
        "hou_compare",
        action="set_reference",
        reference=first["path"],
        name="look",
        camera=camera,
    )
    if camera == "":
        # An empty camera is no camera, as before.
        assert ok(result)["camera"] is None
        return
    error = refused(result)
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == "camera"


def test_a_capture_past_its_time_names_the_job_to_compare_later(bench: Bench, scene: Scene) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[64, 36])
    register(bench, first["path"])
    scene.capture.delay_s = 0.5
    error = refused(
        compare(bench, {"source": "node", "path": BOX, "resolution": [64, 36], "timeout_s": 0.1})
    )
    assert error["code"] == "TIMEOUT"
    job_id = error["details"]["job_id"]
    assert job_id.endswith(":capture")
    assert error["hint"] == "wait with hou_jobs, then compare with source render and this job_id"
    body: dict[str, Any] = {}
    for _ in range(10):
        body = ok(one(bench, "hou_jobs", job_id=job_id, wait_s=5))
        if body["state"] not in ("queued", "running"):
            break
    assert body["state"] == "done"
    later = ok(compare(bench, {"source": "render", "job_id": job_id}))
    assert later["sources"]["candidate"]["job_id"] == job_id


def test_a_bad_candidate_timeout_is_refused_before_the_session(bench: Bench) -> None:
    error = refused(compare(bench, {"source": "node", "path": BOX, "timeout_s": -1}))
    assert error["code"] == "BAD_ARGUMENTS"
    assert error["details"]["argument"] == "candidate.timeout_s"


# Section: series and aspect


def test_a_capture_made_another_way_is_another_series(bench: Bench) -> None:
    first = capture(bench, source="node", path=BOX, resolution=[320, 180])
    register(bench, first["path"])
    same = ok(compare(bench, {"source": "node", "path": BOX, "resolution": [320, 180]}))
    again = ok(compare(bench, {"source": "node", "path": BOX, "resolution": [320, 180]}))
    square = ok(compare(bench, {"source": "node", "path": BOX, "resolution": [320, 320]}))
    wired = ok(
        compare(bench, {"source": "node", "path": BOX, "resolution": [320, 180], "display": "wire"})
    )
    assert again["series"]["id"] == same["series"]["id"]
    assert square["series"]["id"] != same["series"]["id"]
    assert wired["series"]["id"] != same["series"]["id"]
    assert "capture" in square["series"]["changed"]
    # Off the reference's aspect by more than a pixel: said so.
    assert any("not at the aspect" in item for item in square["warnings"])
    assert not any("not at the aspect" in item for item in (same["warnings"] or []))


def test_what_is_off_the_reference_aspect() -> None:
    assert compare_tool.off_aspect((320, 320), (320, 180)) is True
    assert compare_tool.off_aspect((320, 181), (320, 180)) is False
    assert compare_tool.off_aspect((641, 360), (320, 180)) is False
    assert compare_tool.off_aspect((1280, 720), (1920, 1080)) is False


# Section: the reference and the mask are read before anything is captured


def test_an_unreadable_reference_costs_no_capture(
    bench: Bench, scene: Scene, tmp_path: Path
) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a picture")
    result = one(
        bench,
        "hou_compare",
        candidate={"source": "node", "path": BOX},
        reference=str(broken),
        return_image="none",
    )
    assert refused(result)["code"] == "IMAGE_UNREADABLE"
    assert scene.capture.seen == []


def test_a_mask_that_will_not_do_costs_no_capture(bench: Bench, scene: Scene) -> None:
    first = capture(bench, resolution=[64, 32])
    register(bench, first["path"])
    seen = len(scene.capture.seen)
    error = refused(compare(bench, {"source": "viewport"}, mask="/no/such/mask.png"))
    assert error["code"] == "FILE_NOT_FOUND"
    assert error["details"]["argument"] == "mask"
    assert len(scene.capture.seen) == seen
    compare_folder = Path(first["path"]).parents[2] / "compare"
    assert not compare_folder.exists() or not any(compare_folder.rglob("result.json"))


# Section: a render a Python job wrote


PYTHON_RENDER = """
from PIL import Image
path = mcp.output_path('render', 'beauty', 'png').replace('$F4', '0001')
Image.new('RGBA', (64, 32), (128, 128, 128, 255)).save(path)
result = path
"""


def test_a_render_by_job_id_reads_what_a_python_job_wrote(bench: Bench) -> None:
    written = ok(one(bench, "hou_python", code=PYTHON_RENDER))
    path = written["result"]
    register(bench, path)
    result = ok(compare(bench, {"source": "render", "job_id": written["job_id"]}))
    made = result["sources"]["candidate"]
    assert made["path"] == path
    assert made["job_kind"] == "python"
    assert result["metrics"]["mae"]["overall"] == 0.0
    with bench.store() as store:
        [run] = store.runs_made_by(job_id=written["job_id"])
    assert run.kind == "render"


# Section: partial alpha


def test_a_node_drawn_small_on_a_clear_background_is_no_alpha_warning(
    bench: Bench, tmp_path: Path
) -> None:
    import numpy as np
    from PIL import Image

    small = np.zeros((100, 100, 4), dtype=np.uint8)
    small[45:55, 45:55] = (200, 200, 200, 255)
    reference = tmp_path / "small.png"
    Image.fromarray(small).save(reference)
    register(bench, str(reference))
    result = ok(compare(bench, {"source": "file", "path": str(reference)}))
    assert result["steps"]["alpha"]["candidate"]["coverage_pct"] == 1.0
    assert not any("partial alpha" in item for item in result["warnings"] or [])
    half = small.copy()
    half[:, :50, 3] = 255
    cut = tmp_path / "half.png"
    Image.fromarray(half).save(cut)
    warned = ok(compare(bench, {"source": "file", "path": str(cut)}))
    assert any("partial alpha" in item for item in warned["warnings"])


# Section: run records by family


def test_a_node_s_runs_are_narrowed_to_the_family_before_the_limit(bench: Bench) -> None:
    with bench.store() as store:
        store.create_run("run-here", kind="capture", paths={}, hip_family="shot", source_node=BOX)
        for index in range(60):
            store.create_run(
                f"run-there-{index}",
                kind="capture",
                paths={},
                hip_family="other",
                source_node=BOX,
            )
        [here] = store.runs_made_by(source_node=BOX, hip_family="shot")
        assert here.run_id == "run-here"
        assert len(store.runs_made_by(source_node=BOX)) == 50
        indexes = {
            row[0]
            for row in store._read_all("SELECT name FROM sqlite_master WHERE type = 'index'", [])
        }
    assert {"runs_by_node", "runs_by_job"} <= indexes
