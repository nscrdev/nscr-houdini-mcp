"""`hou_compare` through the server, with the session's answers stood in for.

The pictures are real: the brief's reference drawings and images made here
with NumPy and Pillow. The store and the output folders are real too, inside
the test's own folder. What the session answers is chosen by the test; for a
scene linear file the stand in runs the bridge's own reader against a stand
in `hou`. What a real Houdini does with an EXR is in the integration file.
"""

from __future__ import annotations

import base64
import io
import json
import struct
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageCms

from nscr_houdini_mcp import imaging, references
from nscr_houdini_mcp.bridge import images
from nscr_houdini_mcp.bridge.handlers import default_registry
from nscr_houdini_mcp.bridge.tools import ToolContext
from test_router import Sent
from test_server import talk
from test_tools_scene import info, reply
from test_tools_sessions import Bench

BRIEF = Path(__file__).resolve().parent / "fixtures" / "brief" / "references"
FRONT = BRIEF / "front_shaded.png"
FRONT_MASK = BRIEF / "front_mask.png"
THREE_QUARTER = BRIEF / "three_quarter_shaded.png"
EIGHT_KEYS = BRIEF / "variants" / "count_8_three_quarter_shaded.png"
BACKGROUND = (236, 236, 233)


class Acting(Sent):
    """A stand in session whose replies may be worked out from the call."""

    def __call__(self, session: Any, tool: str, **rest: Any) -> Any:
        if self.replies and callable(self.replies[0]):
            act = self.replies.pop(0)
            self.replies.insert(0, act(tool, rest.get("arguments") or {}))
        return super().__call__(session, tool, **rest)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    folder = tmp_path / "project"
    folder.mkdir()
    return folder


@pytest.fixture
def hip(project: Path) -> Path:
    path = project / "keys_v001.hip"
    path.write_bytes(b"scene")
    return path


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bench:
    monkeypatch.delenv("HOUDINI_TEMP_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "PyOpenColorIO", None)
    home = tmp_path / "home"
    home.mkdir()
    made = Bench(home)
    made.session("s-1", "w1")
    return made


def run(bench: Bench, *replies: Any, **arguments: Any) -> Any:
    bench.sent = Acting(*replies)
    _, [result] = talk(bench.serve(), ("hou_compare", arguments))
    return result


def body(result: Any) -> dict[str, Any]:
    assert not result.is_error, result.content[0].text
    return result.structured_content


def code(result: Any) -> str:
    assert result.is_error is True
    return result.structured_content["error"]["code"]


def compare(bench: Bench, hip: Path, candidate: Path, reference: Any, **rest: Any) -> Any:
    return run(
        bench,
        info(hip),
        candidate={"source": "file", "path": str(candidate)},
        reference=str(reference),
        **rest,
    )


def register(bench: Bench, hip: Path, source: Path, **rest: Any) -> dict[str, Any]:
    return body(run(bench, info(hip), action="set_reference", reference=str(source), **rest))


def save(path: Path, pixels: np.ndarray, **options: Any) -> Path:
    data = np.clip(pixels, 0, 255).astype(np.uint8)
    Image.fromarray(data).save(path, **options)
    return path


def load(path: Path | str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)


def shifted(pixels: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """The picture moved right by dx and down by dy, the gap filled with the background."""
    out = np.empty_like(pixels)
    out[...] = np.array(BACKGROUND, dtype=pixels.dtype)
    height, width = pixels.shape[:2]
    out[dy:, dx:] = pixels[: height - dy, : width - dx]
    return out


# Section: the tool as listed


def test_the_tool_is_listed_last_and_within_its_token_budget(bench: Bench) -> None:
    listed, _ = talk(bench.serve())
    assert listed.tools[-1].name == "hou_compare"
    tool = listed.tools[-1]
    payload = tool.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert len(json.dumps(payload, separators=(",", ":"))) <= 1200
    assert "No pass or fail" in tool.description
    assert tool.input_schema["additionalProperties"] is False


# Section: numbers


def test_two_identical_images_give_zero_difference(bench: Bench, hip: Path) -> None:
    result = compare(bench, hip, FRONT, FRONT)
    data = body(result)
    metrics = data["metrics"]
    assert metrics["mae"]["overall"] == 0.0
    assert metrics["rmse"] == {"r": 0.0, "g": 0.0, "b": 0.0, "overall": 0.0}
    assert metrics["diff_area_pct"] == 0.0
    assert metrics["psnr_db"] is None
    assert "psnr_db" in data["missing"]
    assert data["largest_region"] is None
    assert metrics["role"] == "secondary"
    assert "match" not in json.dumps(metrics)
    # Every file is where the result says, under the scene's own folder.
    folder = Path(data["folder"])
    assert folder.is_relative_to(hip.parent / ".agent" / "compare")
    for key in ("candidate", "reference", "diff", "overview", "result"):
        assert Path(data["files"][key]).is_file(), key
    saved = json.loads(Path(data["files"]["result"]).read_text(encoding="utf-8"))
    assert saved["metrics"] == metrics
    assert saved["scene_stamp"] == {
        "hip_name": hip.name,
        "hip_path": str(hip),
        "scene_epoch": 0,
        "frame": 1.0,
    }
    # The overview comes back as a picture for the person as well.
    text, picture = result.content
    assert text.type == "text"
    assert picture.type == "image"
    assert picture.mime_type == "image/jpeg"
    assert "user" in picture.annotations.audience
    sheet = Image.open(io.BytesIO(base64.b64decode(picture.data)))
    assert sheet.size == Image.open(data["files"]["overview"]).size
    assert 3 * 256 <= sheet.width <= 3 * 256 + 40


def test_the_text_block_says_the_numbers_when_the_result_is_long(bench: Bench, hip: Path) -> None:
    result = compare(bench, hip, EIGHT_KEYS, THREE_QUARTER, match_exposure=True)
    text = result.content[0].text
    assert text.startswith("hou_compare: mae ")
    assert "(secondary)" in text


def test_a_known_shift_is_found_within_a_pixel(bench: Bench, hip: Path, tmp_path: Path) -> None:
    moved = save(tmp_path / "moved.png", shifted(load(FRONT), 5, 3))
    before = body(compare(bench, hip, moved, FRONT, name="before"))
    after = body(compare(bench, hip, moved, FRONT, auto_shift=True, name="after"))
    shift = after["steps"]["shift_px"]
    assert shift["applied"] is True
    assert abs(shift["estimate"][0] + 5) <= 1
    assert abs(shift["estimate"][1] + 3) <= 1
    assert (shift["dx"], shift["dy"]) == (-5, -3)
    assert before["steps"]["shift_px"] is None
    assert after["metrics"]["mae"]["overall"] < before["metrics"]["mae"]["overall"] / 5


def test_a_different_image_is_counted_and_boxed(bench: Bench, hip: Path) -> None:
    data = body(compare(bench, hip, EIGHT_KEYS, THREE_QUARTER, mode="regression"))
    metrics = data["metrics"]
    assert metrics["role"] == "primary"
    assert 0 < metrics["mae"]["overall"] < 0.2
    assert metrics["psnr_db"] > 0
    assert metrics["diff_area_pct"] > 0
    box = data["largest_region"]["box"]
    assert 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1
    # With no named regions, the automatic crop is cut around it.
    assert "largest_difference" in data["crops"]
    assert Path(data["files"]["crops"]["largest_difference"]).is_file()


def test_match_exposure_reports_the_gain_and_flags_the_luminance_gap(
    bench: Bench, hip: Path, tmp_path: Path
) -> None:
    dark = save(tmp_path / "dark.png", load(FRONT) * 0.5)
    data = body(compare(bench, hip, dark, FRONT, match_exposure=True))
    matched = data["metrics"]["exposure_matched"]
    assert 1.9 < matched["gain"] < 2.1
    assert matched["mae"]["overall"] < data["metrics"]["mae"]["overall"] / 10
    flagged = data["transfer_mismatch_possible"]
    assert flagged["flag"] is True
    assert "luminance" in flagged["why"][0]


# Section: framing and crops


def test_letterbox_and_fill_place_a_square_candidate_differently(
    bench: Bench, hip: Path, tmp_path: Path
) -> None:
    wide = save(tmp_path / "wide.png", np.full((540, 960, 3), 128))
    square = save(tmp_path / "square.png", np.full((600, 600, 3), 128))
    fit = body(compare(bench, hip, square, wide))["steps"]
    fill = body(compare(bench, hip, square, wide, align="fill"))["steps"]
    bars = fit["letterboxed"]
    assert bars["left"] > 0 and bars["right"] > 0
    assert bars["top"] == 0 and bars["bottom"] == 0
    assert abs(bars["left"] - bars["right"]) <= 1
    # The square is taller than the frame, so the grid grew to keep its pixels.
    assert fit["reference_upscaled"] > 1
    assert fill["letterboxed"] is None
    assert fill["grid_px"] == [960, 540]
    assert fill["placed_px"]["width"] == 960 and fill["placed_px"]["height"] == 960
    assert fill["placed_px"]["y"] == -210


def test_the_uncovered_bars_are_not_counted(bench: Bench, hip: Path, tmp_path: Path) -> None:
    wide = save(tmp_path / "wide.png", np.full((540, 960, 3), 128))
    square = save(tmp_path / "square.png", np.full((500, 500, 3), 128))
    data = body(compare(bench, hip, square, wide))
    assert data["metrics"]["mae"]["overall"] < 0.002
    assert data["metrics"]["evaluated_pct"] < 60


def test_a_detail_crop_keeps_native_resolution(bench: Bench, hip: Path, tmp_path: Path) -> None:
    rect = [0.25, 0.3, 0.75, 0.6]
    register(bench, hip, FRONT, name="front", regions={"keys": rect})
    big = tmp_path / "big.png"
    Image.open(FRONT).convert("RGB").resize((1920, 1080), Image.Resampling.LANCZOS).save(big)
    data = body(compare(bench, hip, big, "front", detail_crops=["keys"]))
    assert data["steps"]["grid_px"] == [1920, 1080]
    crop = data["crops"]["keys"]
    assert crop["crop_px"] == [480, 324, 1440, 648]
    assert crop["size_px"] == [960, 324]
    assert crop["metrics"]["pixels"] == 960 * 324
    # The overview numbers are counted at the working size, the crop's are not.
    assert data["steps"]["resized_to"] == [1024, 576]
    pair = Image.open(data["files"]["crops"]["keys"])
    assert pair.width >= 2 * 960 and pair.height >= 324
    assert set(data["crops"]) == {"keys"}
    # Left to choose, the crops are the regions the reference was registered with.
    chosen = body(compare(bench, hip, big, "front"))
    assert set(chosen["crops"]) == {"keys"}
    assert chosen["crops"]["keys"]["size_px"] == [960, 324]
    none = body(compare(bench, hip, big, "front", detail_crops="none"))
    assert none["crops"] is None


def test_region_cuts_both_sides_before_the_overview(bench: Bench, hip: Path) -> None:
    data = body(compare(bench, hip, FRONT, FRONT, region=[0.5, 0.5, 1.0, 1.0]))
    assert data["steps"]["crop_px"] == [480, 270, 960, 540]
    assert Image.open(data["files"]["candidate"]).size == (480, 270)
    assert Image.open(data["files"]["reference"]).size == (480, 270)
    assert data["steps"]["resized_to"] == [1024, 576]


def test_an_unknown_crop_name_is_refused_with_the_names_there_are(bench: Bench, hip: Path) -> None:
    register(bench, hip, FRONT, name="front", regions={"keys": [0.2, 0.2, 0.8, 0.8]})
    result = compare(bench, hip, FRONT, "front", detail_crops=["kyes"])
    assert code(result) == "BAD_ARGUMENTS"
    assert result.structured_content["error"]["details"]["did_you_mean"] == ["keys"]


# Section: masks


def half_alpha(tmp_path: Path) -> tuple[Path, Path]:
    """A reference whose alpha keeps the left half, and a candidate that differs on the right."""
    rgb = np.full((200, 400, 3), 100)
    alpha = np.zeros((200, 400, 1))
    alpha[:, :200] = 255
    reference = save(tmp_path / "ref_alpha.png", np.concatenate([rgb, alpha], axis=2))
    other = rgb.copy()
    other[:, 200:] = 250
    candidate = save(tmp_path / "cand.png", other)
    return reference, candidate


def test_mask_from_the_reference_alpha(bench: Bench, hip: Path, tmp_path: Path) -> None:
    reference, candidate = half_alpha(tmp_path)
    masked = body(compare(bench, hip, candidate, reference, mask="reference"))
    whole = body(compare(bench, hip, candidate, reference))
    # Only the resampling at the mask's edge leaves anything to count.
    assert masked["metrics"]["mae"]["overall"] < 0.002
    assert 49 < masked["metrics"]["evaluated_pct"] < 51
    assert masked["mask"]["source"] == "reference_alpha"
    assert whole["metrics"]["mae"]["overall"] > 0.2


def test_candidate_alpha_alone_never_decides_the_area(
    bench: Bench, hip: Path, tmp_path: Path
) -> None:
    rgb = np.full((200, 400, 3), 100)
    alpha = np.zeros((200, 400, 1))
    candidate = save(tmp_path / "cand_alpha.png", np.concatenate([rgb, alpha], axis=2))
    reference = save(tmp_path / "ref.png", rgb)
    data = body(compare(bench, hip, candidate, reference))
    assert data["metrics"]["evaluated_pct"] == 100.0
    assert data["mask"] is None


def test_a_registered_mask_is_the_reference_mask(bench: Bench, hip: Path) -> None:
    register(bench, hip, FRONT, name="front", mask=str(FRONT_MASK))
    data = body(compare(bench, hip, THREE_QUARTER, "front", mask="reference"))
    assert data["mask"]["source"] == "reference_record"
    silhouette = np.asarray(Image.open(FRONT_MASK).convert("L")) >= 128
    assert abs(data["metrics"]["evaluated_pct"] - silhouette.mean() * 100) < 1.0


def test_mask_reference_with_nothing_to_take_it_from_is_refused(bench: Bench, hip: Path) -> None:
    result = compare(bench, hip, FRONT, FRONT, mask="reference")
    assert code(result) == "BAD_ARGUMENTS"
    assert "set_reference" in result.content[0].text


def test_numbers_that_cannot_be_counted_still_leave_the_aligned_pair(
    bench: Bench, hip: Path, tmp_path: Path
) -> None:
    empty = save(tmp_path / "empty_mask.png", np.zeros((540, 960, 3)))
    data = body(compare(bench, hip, THREE_QUARTER, FRONT, mask=str(empty)))
    assert data["metrics"] is None
    assert "no pixels" in data["missing"]["all"]
    assert Path(data["files"]["candidate"]).is_file()
    assert Path(data["files"]["reference"]).is_file()
    assert Path(data["files"]["overview"]).is_file()


# Section: colour


def linear_profile(description: str = "linear test") -> bytes:
    """A matrix profile with sRGB primaries and straight line curves, made by hand."""

    def fixed(value: float) -> bytes:
        return struct.pack(">i", round(value * 65536))

    def xyz(x: float, y: float, z: float) -> bytes:
        return b"XYZ " + bytes(4) + fixed(x) + fixed(y) + fixed(z)

    text = description.encode("ascii") + b"\0"
    desc = b"desc" + bytes(4) + struct.pack(">I", len(text)) + text + bytes(4 + 4 + 2 + 1 + 67)
    curve = b"curv" + bytes(4) + struct.pack(">IH", 1, 0x0100) + bytes(2)
    tags = [
        (b"desc", desc),
        (b"wtpt", xyz(0.9642, 1.0, 0.8249)),
        (b"rXYZ", xyz(0.4360747, 0.2225045, 0.0139322)),
        (b"gXYZ", xyz(0.3850649, 0.7168786, 0.0971045)),
        (b"bXYZ", xyz(0.1430804, 0.0606169, 0.7141733)),
        (b"rTRC", curve),
        (b"gTRC", curve),
        (b"bTRC", curve),
    ]
    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    table = struct.pack(">I", len(tags))
    data = b""
    for signature, content in tags:
        padded = content + bytes(-len(content) % 4)
        table += signature + struct.pack(">II", offset + len(data), len(content))
        data += padded
    size = 128 + table_size + len(data)
    header = (
        struct.pack(">I", size)
        + bytes(4)
        + struct.pack(">I", 0x02100000)
        + b"mntrRGB XYZ "
        + struct.pack(">6H", 2026, 1, 1, 0, 0, 0)
        + b"acsp"
        + bytes(16)
        + bytes(8)
        + struct.pack(">I", 0)
        + fixed(0.9642)
        + fixed(1.0)
        + fixed(0.8249)
        + bytes(4)
        + bytes(16)
        + bytes(28)
    )
    assert len(header) == 128
    return header + table + data


def test_an_embedded_profile_is_converted_to_srgb(bench: Bench, hip: Path, tmp_path: Path) -> None:
    grey = np.full((64, 64, 3), 128)
    tagged = save(tmp_path / "linear.png", grey, icc_profile=linear_profile())
    plain = save(tmp_path / "plain.png", grey)
    data = body(compare(bench, hip, tagged, plain))
    colour = data["colour"]["candidate"]
    assert colour["profile"] == "embedded"
    assert colour["description"] == "linear test"
    assert colour["converted_to"] == "sRGB"
    assert data["colour"]["reference"]["profile"] == "assumed_srgb"
    assert data["steps"]["profile"] == {"candidate": "embedded", "reference": "assumed_srgb"}
    # Half way in linear light is well above half way in sRGB.
    converted = load(data["files"]["candidate"])
    assert 180 <= int(converted[32, 32, 0]) <= 195


def test_an_srgb_profile_changes_nothing_and_no_profile_is_an_assumption(
    tmp_path: Path,
) -> None:
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    pixels = np.random.default_rng(7).integers(0, 255, (32, 32, 3))
    tagged = imaging.read_display_file(save(tmp_path / "srgb.png", pixels, icc_profile=srgb))
    plain = imaging.read_display_file(save(tmp_path / "plain.png", pixels))
    assert tagged.colour["profile"] == "embedded"
    assert plain.colour["profile"] == "assumed_srgb"
    assert np.abs(tagged.rgb - plain.rgb).max() <= 1.5 / 255


def test_a_file_that_is_not_an_image_is_refused(bench: Bench, hip: Path, tmp_path: Path) -> None:
    notes = tmp_path / "notes.png"
    notes.write_text("not a picture", encoding="utf-8")
    result = compare(bench, hip, notes, FRONT)
    assert code(result) == "IMAGE_UNREADABLE"
    assert result.structured_content["error"]["details"]["argument"] == "candidate"


# Section: references


def test_set_reference_writes_an_immutable_record_and_a_new_id_on_replace(
    bench: Bench, hip: Path
) -> None:
    first = register(
        bench, hip, FRONT, name="front", camera="/obj/cam_front", regions={"keys": [0, 0, 1, 1]}
    )
    record_path = Path(first["record"])
    kept = record_path.read_bytes()
    assert first["ref_id"].startswith("ref-")
    assert first["sha256"] == references.file_hash(FRONT)
    assert first["colour"]["profile"] == "assumed_srgb"
    assert first["size_px"] == [960, 540]
    assert Path(first["image"]).parent == hip.parent / ".agent" / "reference"
    assert first["replaces"] is None

    second = register(bench, hip, THREE_QUARTER, name="front")
    assert second["ref_id"] != first["ref_id"]
    assert second["replaces"] == first["ref_id"]
    assert record_path.read_bytes() == kept
    with pytest.raises(FileExistsError):
        references.write_once(record_path, {"ref_id": first["ref_id"]})

    listed = body(run(bench, info(hip), action="list_references"))["references"]
    [row] = listed
    assert row["name"] == "front"
    assert row["ref_id"] == second["ref_id"]
    assert row["replaced"] == [first["ref_id"]]


def test_a_reference_is_found_by_name_and_an_unknown_name_says_so(bench: Bench, hip: Path) -> None:
    register(bench, hip, FRONT, name="front")
    data = body(compare(bench, hip, FRONT, "front"))
    assert data["sources"]["reference_id"].startswith("ref-")
    assert data["metrics"]["mae"]["overall"] == 0.0
    result = compare(bench, hip, FRONT, "frnt")
    assert code(result) == "REFERENCE_UNKNOWN"
    assert result.structured_content["error"]["details"]["did_you_mean"] == ["front"]


def test_full_scene_info_lists_the_registered_references(bench: Bench, hip: Path) -> None:
    register(bench, hip, FRONT, name="front")
    register(bench, hip, THREE_QUARTER, name="three_quarter")
    bench.sent = Acting(info(hip))
    _, [result] = talk(bench.serve(), ("hou_scene", {"detail": "full"}))
    assert result.structured_content["references"] == ["front", "three_quarter"]
    bench.sent = Acting(info(hip))
    _, [summary] = talk(bench.serve(), ("hou_scene", {}))
    assert "references" not in summary.structured_content


def test_bad_regions_are_refused(bench: Bench, hip: Path) -> None:
    result = run(
        bench,
        info(hip),
        action="set_reference",
        reference=str(FRONT),
        regions={"keys": [0.5, 0.2, 0.4, 0.8]},
    )
    assert code(result) == "BAD_ARGUMENTS"


# Section: series


def test_a_series_carries_its_trend_and_a_change_of_setup_starts_another(
    bench: Bench, hip: Path
) -> None:
    register(bench, hip, THREE_QUARTER, name="tq")
    first = body(compare(bench, hip, EIGHT_KEYS, "tq"))
    second = body(compare(bench, hip, THREE_QUARTER, "tq"))
    assert first["series"]["new"] is True
    assert first["trend"] is None
    assert second["series"] == {"id": first["series"]["id"], "new": False, "runs_before": 1}
    trend = second["trend"]
    assert trend["runs"] == 1
    assert trend["earlier"][0]["run_id"] == first["run_id"]
    assert trend["change_since_last"]["mae"] < 0

    other = body(compare(bench, hip, THREE_QUARTER, "tq", tolerance=0.1))
    assert other["series"]["new"] is True
    assert other["series"]["id"] != first["series"]["id"]
    assert other["series"]["changed"] == ["settings"]
    assert other["series"]["previous_series"] == first["series"]["id"]


def test_a_replaced_reference_starts_a_new_series(bench: Bench, hip: Path) -> None:
    register(bench, hip, THREE_QUARTER, name="tq")
    first = body(compare(bench, hip, EIGHT_KEYS, "tq"))
    register(bench, hip, THREE_QUARTER, name="tq")
    again = body(compare(bench, hip, EIGHT_KEYS, "tq"))
    assert again["series"]["new"] is True
    assert again["series"]["changed"] == ["reference"]
    assert again["series"]["id"] != first["series"]["id"]


# Section: sources this build does not have yet


@pytest.mark.parametrize("source", ["viewport", "node", "render"])
def test_capture_sources_are_not_yet_available(bench: Bench, source: str) -> None:
    result = run(bench, candidate={"source": source}, reference="front")
    assert code(result) == "NOT_YET_AVAILABLE"
    error = result.structured_content["error"]
    assert error["details"]["tool"] == "hou_capture"
    assert "hou_capture" in error["hint"]
    assert bench.sent.calls == []


# Section: scene linear files, through a stand in Houdini


class Layer:
    def __init__(self, pixels: np.ndarray, *, half: bool = False) -> None:
        self.pixels = pixels
        self.half = half

    def bufferResolution(self) -> tuple[int, int]:  # noqa: N802 - Houdini's own name
        return self.pixels.shape[1], self.pixels.shape[0]

    def channelCount(self) -> int:  # noqa: N802
        return self.pixels.shape[2]

    def allBufferElements(self) -> bytes:  # noqa: N802
        kind = np.float16 if self.half else np.float32
        return np.ascontiguousarray(self.pixels[::-1], dtype=kind).tobytes()


class Parm:
    def __init__(self, reader: Reader, name: str) -> None:
        self.reader, self.name = reader, name

    def set(self, value: Any) -> None:
        self.reader.parms[self.name] = value

    def pressButton(self) -> None:  # noqa: N802
        self.reader.pressed.append(self.name)


class Reader:
    def __init__(self, stand: StandIn) -> None:
        self.stand = stand
        self.parms: dict[str, Any] = {}
        self.pressed: list[str] = []

    def parm(self, name: str) -> Parm:
        return Parm(self, name)

    def outputNames(self) -> tuple[str, ...]:  # noqa: N802
        return tuple(self.stand.outputs)

    def layerAtFrame(self, frame: float, index: int) -> Layer:  # noqa: N802
        self.stand.read.append((frame, index))
        return self.stand.layers[index]


class Holder:
    def __init__(self, stand: StandIn) -> None:
        self.stand = stand
        self.destroyed = False

    def createNode(self, kind: str) -> Reader:  # noqa: N802
        assert kind == "file"
        reader = Reader(self.stand)
        self.stand.readers.append(reader)
        return reader

    def destroy(self) -> None:
        self.destroyed = True


class StandIn:
    """Just enough of `hou` for the reader: a COP network, a file node, its layers."""

    def __init__(self, layers: list[Layer], outputs: list[str]) -> None:
        self.layers = layers
        self.outputs = outputs
        self.holders: list[Holder] = []
        self.readers: list[Reader] = []
        self.read: list[tuple[float, int]] = []
        self.undo_disabled = 0
        stand = self

        class Undos:
            @staticmethod
            @contextmanager
            def disabler() -> Any:
                stand.undo_disabled += 1
                yield

        class Parent:
            def createNode(self, kind: str, name: str) -> Holder:  # noqa: N802
                assert kind == "copnet"
                holder = Holder(stand)
                stand.holders.append(holder)
                return holder

        self.undos = Undos()
        self._parent = Parent()

    def node(self, path: str) -> Any:
        return self._parent if path == "/img" else None

    def frame(self) -> float:
        return 12.0


def linear_ramp() -> np.ndarray:
    """Scene linear values whose sRGB display values are known, with alpha."""
    pixels = np.zeros((4, 6, 4), dtype=np.float32)
    pixels[..., :3] = 0.2140  # about half way on the sRGB curve
    pixels[0, :, :3] = 1.0  # the top row is white
    pixels[..., 3] = 1.0
    return pixels


def test_the_session_reader_writes_display_values_rows_from_the_top(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "PyOpenColorIO", None)
    exr = tmp_path / "beauty.exr"
    exr.write_bytes(b"stand in")
    stand = StandIn([Layer(np.zeros((1, 1, 1))), Layer(linear_ramp())], ["depth", "C"])
    out = tmp_path / "read.f32"
    data = images.read_exr(
        {"path": str(exr), "out_path": str(out)}, ToolContext(hou=stand, kind="hython")
    )
    assert (data["width"], data["height"], data["channels"]) == (6, 4, 4)
    assert data["colour"]["channel"] == "C"
    assert data["colour"]["transform"] == "srgb_curve"
    assert data["colour"]["alpha"] == "kept apart"
    [reader] = stand.readers
    assert reader.parms == {"filename": str(exr)}
    assert reader.pressed == ["addaovs"]
    assert stand.read == [(12.0, 1)]
    assert stand.holders[0].destroyed is True
    assert stand.undo_disabled == 1
    values = np.fromfile(out, dtype=np.float32).reshape(4, 6, 4)
    assert np.allclose(values[0, :, :3], 1.0)
    assert np.allclose(values[1:, :, :3], 0.5, atol=0.01)
    # Half floats are read too.
    half = StandIn([Layer(linear_ramp(), half=True)], ["C"])
    images.read_exr(
        {"path": str(exr), "out_path": str(tmp_path / "half.f32")}, ToolContext(hou=half)
    )
    assert np.allclose(np.fromfile(tmp_path / "half.f32", dtype=np.float32)[:3], 1.0)


def test_the_session_reader_refuses_what_it_should_not_write(tmp_path: Path) -> None:
    exr = tmp_path / "beauty.exr"
    exr.write_bytes(b"stand in")
    stand = StandIn([Layer(linear_ramp())], ["C"])
    taken = tmp_path / "taken.f32"
    taken.write_bytes(b"")
    for arguments, expected in (
        (
            {"path": str(tmp_path / "gone.exr"), "out_path": str(tmp_path / "a.f32")},
            "FILE_NOT_FOUND",
        ),
        ({"path": str(exr), "out_path": str(tmp_path / "a.png")}, "BAD_ARGUMENTS"),
        ({"path": "beauty.exr", "out_path": str(tmp_path / "a.f32")}, "BAD_ARGUMENTS"),
    ):
        with pytest.raises(images.BridgeError) as raised:
            images.read_exr(arguments, ToolContext(hou=stand))
        assert raised.value.code == expected
    with pytest.raises(FileExistsError):
        images.read_exr({"path": str(exr), "out_path": str(taken)}, ToolContext(hou=stand))


def test_the_session_reader_uses_the_opencolorio_view_when_there_is_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Processor:
        def applyRGB(self, pixels: np.ndarray) -> None:  # noqa: N802
            pixels *= 0.5

    class Config:
        @staticmethod
        def CreateFromFile(path: str) -> Config:  # noqa: N802
            made = Config()
            made.path = path
            return made

        def getDefaultDisplay(self) -> str:  # noqa: N802
            return "sRGB - Display"

        def getDefaultView(self, display: str) -> str:  # noqa: N802
            return "Standard"

        def getProcessor(self, transform: Any) -> Any:  # noqa: N802
            return types.SimpleNamespace(getDefaultCPUProcessor=Processor)

    fake = types.SimpleNamespace(
        Config=Config,
        GetCurrentConfig=Config,
        DisplayViewTransform=lambda **rest: rest,
        ROLE_SCENE_LINEAR="scene_linear",
    )
    monkeypatch.setitem(sys.modules, "PyOpenColorIO", fake)
    monkeypatch.setenv("OCIO", str(tmp_path / "config.ocio"))
    exr = tmp_path / "beauty.exr"
    exr.write_bytes(b"stand in")
    stand = StandIn([Layer(linear_ramp())], ["C"])
    data = images.read_exr(
        {"path": str(exr), "out_path": str(tmp_path / "v.f32")}, ToolContext(hou=stand)
    )
    colour = data["colour"]
    assert colour["transform"] == "ocio_display_view"
    assert colour["display"] == "sRGB - Display"
    assert colour["view"] == "Standard"
    assert colour["configuration"] == str(tmp_path / "config.ocio")
    assert colour["exposure"] == 0.0
    values = np.fromfile(tmp_path / "v.f32", dtype=np.float32).reshape(4, 6, 4)
    assert np.allclose(values[0, :, :3], 0.5)


def test_the_reader_is_registered_on_the_bridge_as_a_read() -> None:
    tool = default_registry().get("compare.read_exr")
    assert tool.mutating is False
    assert set(tool.required) == {"path", "out_path"}


def test_an_exr_candidate_goes_through_the_session_and_is_flagged_against_a_png(
    bench: Bench, hip: Path, tmp_path: Path
) -> None:
    exr = tmp_path / "beauty.exr"
    exr.write_bytes(b"stand in")
    grey = save(tmp_path / "grey.png", np.full((4, 6, 3), 128))
    stand = StandIn([Layer(linear_ramp())], ["C"])

    def session_reads(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        assert tool == "compare.read_exr"
        return reply(images.read_exr(arguments, ToolContext(hou=stand)))

    result = run(
        bench,
        info(hip),
        session_reads,
        candidate={"source": "file", "path": str(exr)},
        reference=str(grey),
    )
    data = body(result)
    assert [call["tool"] for call in bench.sent.calls] == ["scene.info", "compare.read_exr"]
    sent = bench.sent.calls[1]["arguments"]
    assert not Path(sent["out_path"]).exists()
    assert data["colour"]["candidate"]["kind"] == "view_transform"
    assert data["steps"]["view_transform"]["candidate"]["transform"] == "srgb_curve"
    assert data["steps"]["view_transform"]["reference"] is None
    flagged = data["transfer_mismatch_possible"]
    assert flagged["flag"] is True
    assert "different kinds of transform" in flagged["why"][0]
