"""`hou_compare`: a candidate image against a reference, as pictures and numbers.

Three actions.

- `compare`, the default. The candidate is a file; a `viewport` or `node`
  picture captured now, through the capture code in this process with the
  capture arguments the candidate carries; or a `render`, the newest file on
  disk that a finished job or a node wrote, from the run records. A
  registered reference that names a camera frames a capture unless the
  candidate names its own camera, at the reference's aspect unless the
  candidate names its own size, and the result says which camera framed it;
  a camera the reference names that will not do is the reference's error.
  The capture's run id and path, or the render's run and job, are the
  candidate's source in the result and in `result.json`. A job not yet
  ended is `JOB_RUNNING`; a job or node with no image on disk is
  `NO_OUTPUT`. The work follows
  the order in `imaging`: colour, alignment at native size, crops cut before
  anything is shrunk, then the overview. Every step is recorded in the
  result. The files go to a managed `compare` folder: the aligned pair, the
  difference map, the labelled overview sheet, one side by side file per
  detail crop and `result.json`, which names every file relative to itself
  and holds no other place on this machine than the scene's own path.
- `set_reference` registers an image under a name, with an optional camera
  path, named detail regions and a mask. The record is written once and never
  changed; registering the name again makes a new one with a new id.
- `list_references` lists the names registered for the scene.

The numbers are not a verdict. There is no match flag and no built in
threshold. In `likeness` mode they are labelled secondary, because lighting
and framing move them; in `regression` mode they are primary. Comparisons
that can be read against each other form a series, and a result in a series
that has earlier runs carries the trend.

A scene linear file (EXR, HDR) is read by the session, which applies its
display transform and writes the pixels into this compare's own folder; the
rest is done here, in the server process, with NumPy and Pillow.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sqlite3
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp_types import Annotations, ImageContent

from nscr_houdini_mcp import outputs as outputs_module
from nscr_houdini_mcp import references
from nscr_houdini_mcp import results as results_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools import capture as capture_tool
from nscr_houdini_mcp.tools import jobs as jobs_tool
from nscr_houdini_mcp.tools import outputs as outputs_tool
from nscr_houdini_mcp.tools.base import (
    OPERATION_ID_SEPARATOR,
    SESSION,
    Call,
    ToolSpec,
    inputs,
    outputs,
)

ACTIONS = ("compare", "set_reference", "list_references")
SOURCES = ("file", "viewport", "node", "render")
ALIGN = ("fit", "fill", "stretch", "none")
MODES = ("likeness", "regression")
RETURN_IMAGE = ("thumb", "none", "full")

# The sources the session draws for the compare, through the capture code.
CAPTURED = ("viewport", "node")

# What each source's candidate takes besides `source`.
CAPTURE_KEYS = ("path", "camera", "frame_target", "display", "resolution", "frame", "region")
CANDIDATE_KEYS = {
    "file": ("path",),
    "viewport": (*CAPTURE_KEYS, "timeout_s"),
    "node": (*CAPTURE_KEYS, "timeout_s"),
    "render": ("job_id", "path"),
}

# The longest edge of a capture made at a reference's aspect.
REFERENCE_EDGE = 2048

# What a capture made for a compare adds to the compare's operation id.
CAPTURE_SUFFIX = "capture"

# A camera node path as a reference records it: absolute, with no spaces.
CAMERA_PATH = re.compile(r"/[^\s]{0,1023}")

# The alpha coverage, in percent, between which a candidate's clear pixels
# are worth a warning. Outside it, the clear part or the covered part is a
# sliver, as for a node drawn small on a clear background.
MIN_COVERAGE_PCT = 5.0
MAX_COVERAGE_PCT = 95.0

# How many runs of one node are looked through for its newest file.
NODE_RUNS = 50

DEFAULT_TOLERANCE = 0.05

# Files the session reads, through its display transform.
LINEAR_SUFFIXES = (".exr", ".hdr")

# What a run may have written that a compare reads as an image.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", *LINEAR_SUFFIXES})

# The file a session writes a scene linear side into, inside the compare folder.
RAW_NAMES = {"candidate": "_candidate_read.f32", "reference": "_reference_read.f32"}

# How much room the automatic crop leaves around the largest difference.
AUTO_CROP_MARGIN = 0.1
AUTO_CROP_NAME = "largest_difference"

# The furthest `adjust` moves the candidate, as a share of the frame.
MAX_MOVE = 1.0

# The smallest `adjust.scale`. There is no fixed largest: the candidate it
# places may cover at most four times the frame's area, which `imaging`
# refuses before anything is enlarged.
MIN_SCALE = 0.05

MASK_NONE = "none"
MASK_REFERENCE = "reference"

# Views that show scene linear values on an sRGB display with no tone curve,
# so a file drawn for sRGB is a fair match for them.
PLAIN_VIEWS = frozenset({"un-tone-mapped", "untonemapped", "standard", "srgb", "plain srgb curve"})

LIKENESS_NOTE = "secondary: lighting, material and framing move these numbers; read the sheet first"
REGRESSION_NOTE = "primary: the same setup rendered again, so a change in the numbers is a change"

# Room kept for what is added to a result after its size is taken: the
# trace and the note about the picture.
RESULT_MARGIN = 2048

PARTIAL_ALPHA_WARNING = (
    "the candidate has partial alpha and no mask is set, so its transparent pixels count "
    "as the colour they store; pass mask to count only the reference's area"
)

# Where the conventions come from, named the way a person finds them.
CONVENTION_FILES = (
    f"{outputs_module.PROJECT_FILE_NAMES[0]} beside the scene",
    f"{outputs_module.USER_FILE_NAMES[0]} in the state folder",
)


def compare_tool(call: Call) -> Mapping[str, Any]:
    action = call.arguments.get("action") or "compare"
    return ACTION_HANDLERS[action](call)


# Section: the scene the call is about


@dataclass(frozen=True)
class Scene:
    """What a call needs to know about the scene to place and name files."""

    hip: str | None
    session_id: str
    home: Path
    scratch: Path | None
    place: Path
    stamp: dict[str, Any]


def scene_of(call: Call) -> Scene:
    reply = call.bridge("scene.info")
    data = dict(reply.get("data") or {})
    hip = None if data.get("untitled") else data.get("hip_path")
    home = Path(call.router.home)
    scratch = None if os.environ.get("HOUDINI_TEMP_DIR") else home / "temp"
    session_id = call.target().session_id
    place = guarded(
        "reference",
        lambda: references.folder(
            home=home, hip_path=hip, session_id=session_id, scratch_root=scratch
        ),
    )
    stamp = {
        "hip_name": data.get("hip_name"),
        "hip_path": hip,
        "scene_epoch": call.trace.get("scene_epoch"),
        "frame": data.get("frame"),
    }
    return Scene(hip, session_id, home, scratch, place, stamp)


def guarded(kind: str, work: Callable[[], Any]) -> Any:
    """Run output path work, turning its failures into coded errors."""
    try:
        return work()
    except outputs_module.AllocationFailed as error:
        raise CallError("OUTPUT_BUSY", str(error), details={"kind": kind}) from None
    except outputs_module.OutputError as error:
        raise CallError(
            "OUTPUT_REFUSED",
            str(error),
            details={
                "kind": kind,
                "exception": type(error).__name__,
                "conventions": list(CONVENTION_FILES),
            },
        ) from None
    except (store_module.StoreError, sqlite3.Error) as error:
        raise CallError(
            "STORE_UNAVAILABLE",
            "the coordination store could not be read",
            details={"exception": type(error).__name__},
        ) from None
    except OSError as error:
        raise CallError(
            "OUTPUT_UNWRITABLE",
            f"the folder for the {kind} files could not be written",
            details={"kind": kind, "exception": type(error).__name__},
        ) from None


# Section: arguments


def a_file(value: Any, argument: str) -> Path:
    """An absolute path to a file that is there."""
    if not isinstance(value, str) or not value.strip():
        raise CallError(
            "BAD_ARGUMENTS", f"{argument} needs a path to an image", details={"argument": argument}
        )
    path = Path(os.path.expanduser(value.strip()))
    if not path.is_absolute():
        raise CallError(
            "BAD_ARGUMENTS",
            f"{argument} has to be absolute, because the server's folder is not the scene's",
            details={"argument": argument},
        )
    if not path.is_file():
        raise CallError(
            "FILE_NOT_FOUND",
            f"there is no file at the path given as {argument}",
            details={"argument": argument, "folder_exists": path.parent.is_dir()},
        )
    return path


def check_regions(value: Any) -> dict[str, list[float]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CallError(
            "BAD_ARGUMENTS",
            "regions maps a name to [x0, y0, x1, y1] in shares of the frame",
            details={"argument": "regions"},
        )
    checked: dict[str, list[float]] = {}
    for name, rect in value.items():
        clean = outputs_module.sanitize_name(str(name))
        if clean != name:
            raise CallError(
                "BAD_ARGUMENTS",
                "a region name may hold only letters, digits, underscore and dash",
                details={"argument": f"regions.{name}", "did_you_mean": [clean]},
            )
        box = imaging().check_rect(rect)
        if box is None:
            raise CallError(
                "BAD_ARGUMENTS",
                f"regions.{name} must be [x0, y0, x1, y1] from 0 to 1, with x0 < x1 and y0 < y1",
                details={"argument": f"regions.{name}"},
            )
        checked[name] = box
    return checked


def check_region(value: Any) -> list[float] | None:
    if value is None:
        return None
    box = imaging().check_rect(value)
    if box is None:
        raise CallError(
            "BAD_ARGUMENTS",
            "region must be [x0, y0, x1, y1] from 0 to 1, with x0 < x1 and y0 < y1",
            details={"argument": "region"},
        )
    return box


def check_adjust(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {"dx", "dy", "scale"}:
        raise CallError(
            "BAD_ARGUMENTS",
            "adjust takes dx, dy and scale, in shares of the frame",
            details={"argument": "adjust"},
        )
    moved: dict[str, float] = {}
    for key, fallback in (("dx", 0.0), ("dy", 0.0), ("scale", 1.0)):
        number = value.get(key, fallback)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise CallError(
                "BAD_ARGUMENTS", f"adjust.{key} must be a number", details={"argument": key}
            )
        moved[key] = float(number)
    for key in ("dx", "dy"):
        if not -MAX_MOVE <= moved[key] <= MAX_MOVE:
            raise CallError(
                "BAD_ARGUMENTS",
                f"adjust.{key} is a share of the frame, from -1 to 1",
                details={"argument": f"adjust.{key}"},
            )
    if moved["scale"] < MIN_SCALE:
        raise CallError(
            "BAD_ARGUMENTS",
            f"adjust.scale must be at least {MIN_SCALE:g}; above 1 it is limited by area, "
            "since the candidate it places may cover at most four times the frame",
            details={"argument": "adjust.scale"},
        )
    return moved


def text_or_none(value: Any, argument: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise CallError("BAD_ARGUMENTS", f"{argument} must be text", details={"argument": argument})
    return value


def check_camera(value: str | None) -> str | None:
    """A reference's camera: a node path, which a later capture looks through."""
    if value is None:
        return None
    text = value.strip()
    if not CAMERA_PATH.fullmatch(text):
        raise CallError(
            "BAD_ARGUMENTS",
            "camera is the path of a camera node, such as /obj/cam1",
            details={"argument": "camera", "given": value[:200]},
        )
    return text.rstrip("/")


def check_tolerance(value: Any) -> float:
    if value is None:
        return DEFAULT_TOLERANCE
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise CallError(
            "BAD_ARGUMENTS",
            "tolerance is a per pixel difference from 0 to 1",
            details={"argument": "tolerance"},
        )
    return float(value)


def imaging() -> Any:
    """The image module, imported when a call first needs it."""
    from nscr_houdini_mcp import imaging as module

    return module


# Section: set_reference and list_references


def set_reference(call: Call) -> dict[str, Any]:
    source = a_file(call.arguments.get("reference"), "reference")
    regions = check_regions(call.arguments.get("regions"))
    mask_value = call.arguments.get("mask")
    mask = None
    if mask_value not in (None, MASK_NONE):
        if mask_value == MASK_REFERENCE:
            raise CallError(
                "BAD_ARGUMENTS",
                "a reference's mask is a file; pass its path",
                details={"argument": "mask"},
            )
        mask = a_file(mask_value, "mask")
        readable(lambda: imaging().read_mask(mask), "mask")
    camera = check_camera(text_or_none(call.arguments.get("camera"), "camera"))
    given = text_or_none(call.arguments.get("name"), "name")
    if given is not None and outputs_module.sanitize_name(given) != given:
        raise CallError(
            "BAD_ARGUMENTS",
            "a reference name may hold only letters, digits, underscore and dash",
            details={"argument": "name", "did_you_mean": [outputs_module.sanitize_name(given)]},
        )
    name = given or outputs_module.sanitize_name(source.stem)
    colour, size = colour_on_record(source)
    scene = scene_of(call)

    def register() -> dict[str, Any]:
        with call.router.store(create=True) as store:
            return references.register(
                store,
                home=scene.home,
                hip_path=scene.hip,
                session_id=scene.session_id,
                name=name,
                source=source,
                colour=colour,
                size=size,
                camera=camera,
                regions=regions,
                mask=mask,
                scratch_root=scene.scratch,
            )

    record = guarded("reference", register)
    return {
        "name": record["name"],
        "name_from_file": given is None,
        "ref_id": record["ref_id"],
        "record": str(scene.place / f"{record['ref_id']}.json"),
        "image": str(references.resolve(scene.place, record["image"])),
        "sha256": record["sha256"],
        "size_px": [record["width"], record["height"]] if record["width"] else None,
        "colour": record["colour"],
        "camera": record["camera"],
        "regions": record["regions"],
        "mask": record["mask"],
        "replaces": record["replaces"],
        "series": "compares against this id start a new series",
    }


def colour_on_record(source: Path) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """What a reference's record says about its colour, read from the file here.

    A scene linear file is read only as far as its header: its size and
    channels are recorded, and the display transform is applied at compare
    time, by the session.
    """
    module = imaging()
    if source.suffix.lower() in LINEAR_SUFFIXES:
        header = readable(lambda: module.read_linear_header(source), "reference")
        return {
            "kind": module.VIEW_KIND,
            "profile": "scene_linear",
            "format": header["format"],
            "channels": header["channels"],
            "note": "brought to display values by the session at compare time",
        }, (header["width"], header["height"])
    picture = readable(lambda: module.read_display_file(source), "reference")
    if picture.resized_on_read:
        width, height = picture.resized_on_read["from"]
        return picture.colour, (width, height)
    return picture.colour, picture.size


def list_references(call: Call) -> dict[str, Any]:
    scene = scene_of(call)
    return {"references": references.listing(scene.place), "folder": str(scene.place)}


def readable(read: Callable[[], Any], side: str) -> Any:
    """Read an image, turning a file that will not read into a coded error."""
    module = imaging()
    try:
        return read()
    except module.ImageError as error:
        raise CallError(
            error.code, error.message, details={"argument": side, **error.details}
        ) from None


# Section: compare


def compare(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    wanted = arguments.get("candidate")
    if not isinstance(wanted, Mapping):
        raise CallError(
            "BAD_ARGUMENTS",
            'compare needs candidate, such as {"source": "file", "path": "/abs/image.png"}',
            details={"argument": "candidate"},
        )
    source = wanted.get("source") or "file"
    check_candidate(wanted, source)
    if not arguments.get("reference"):
        raise CallError(
            "BAD_ARGUMENTS",
            "compare needs reference: a registered name or an image path",
            details={"argument": "reference"},
        )
    candidate_path = a_file(wanted.get("path"), "candidate.path") if source == "file" else None
    region = check_region(arguments.get("region"))
    adjust = check_adjust(arguments.get("adjust"))
    settings = {
        "align": arguments.get("align") or "fit",
        "adjust": adjust,
        "auto_shift": bool(arguments.get("auto_shift")),
        "mode": arguments.get("mode") or "likeness",
        "match_exposure": bool(arguments.get("match_exposure")),
        "tolerance": check_tolerance(arguments.get("tolerance")),
    }
    if source == "render":
        # Before the session is asked: one busy with the job would hold the call.
        render_ready(call, wanted)

    scene = scene_of(call)
    record, reference_path = which_reference(scene, str(arguments["reference"]))
    wanted_crops = crops_wanted(arguments.get("detail_crops", "auto"), record)
    settings["detail_crops"] = sorted(wanted_crops) if wanted_crops is not None else "auto"
    name = text_or_none(arguments.get("name"), "name") or (
        record["name"] if record else reference_path.stem
    )

    def allocate() -> outputs_module.OutputPlan:
        conventions = outputs_module.load_conventions(home=scene.home, hip_path=scene.hip)
        with call.router.store(create=True) as store:
            return outputs_module.allocate(
                store,
                "compare",
                name=str(name),
                hip_path=scene.hip,
                session_id=scene.session_id,
                conventions=conventions,
                scratch_root=scene.scratch,
            )

    plan = guarded("compare", allocate)
    folder = Path(plan.directory)
    # What the candidate's own run says about it: from the capture made now,
    # or from the run that wrote the render.
    made: dict[str, Any] = {}
    noted: list[str] = []
    try:
        # 1. Colour first: both sides to display values, with a record of how.
        # The reference and the mask are read before anything is captured, so
        # a reference or a mask that will not do costs no capture.
        reference = load_side(call, reference_path, "reference", plan)
        mask_map, mask_record = mask_for(arguments.get("mask"), record, reference, scene.place)
        if source in CAPTURED:
            candidate_path, made, noted = captured(
                call, wanted, source, record, str(name), reference.size
            )
        elif source == "render":
            candidate_path, made = rendered(call, wanted, scene)
        if candidate_path is None:
            raise CallError("NO_OUTPUT", "the candidate has no image", details={"source": source})
        candidate = load_side(call, candidate_path, "candidate", plan)
    except BaseException:
        give_back(call, plan)
        raise

    module = imaging()
    try:
        report = guarded(
            "compare",
            lambda: run(
                candidate,
                reference,
                folder=folder,
                settings=settings,
                region=region,
                crops=wanted_crops,
                mask_map=mask_map,
            ),
        )
    except module.AlignRefused as error:
        give_back(call, plan)
        raise CallError("BAD_ARGUMENTS", str(error), details={"argument": "adjust.scale"}) from None

    reference_hash = references.file_hash(reference_path)
    warnings = list(plan.warnings)
    warnings.extend(item for item in noted if item not in warnings)
    if record and record.get("sha256") != reference_hash:
        warnings.append("the registered reference image changed on disk since it was registered")
    if mask_record is None and partly_covered(candidate.alpha_note):
        warnings.append(PARTIAL_ALPHA_WARNING)
    key = {
        "reference": {"ref_id": record["ref_id"] if record else None, "sha256": reference_hash},
        "camera": made["camera"] if "camera" in made else (record or {}).get("camera"),
        "crop": region,
        "mask": mask_record and {"source": mask_record["source"], "sha256": mask_record["sha256"]},
        "colour": {"candidate": candidate.colour, "reference": reference.colour},
        "settings": settings,
    }
    if source in CAPTURED:
        # What the capture was asked for, so a capture made another way is
        # another series.
        key["capture"] = {
            "size_px": made.get("size_px"),
            "display": wanted.get("display"),
            "frame": made.get("frame"),
            "region": wanted.get("region"),
        }
    series, trend = follow_series(scene.place, key, str(plan.name), report["metrics_numbers"])

    saved = {
        "name": plan.name,
        "run_id": plan.run_id,
        "files": report["files"],
        "metrics": report["metrics"],
        "largest_region": report["largest_region"],
        "crops": report["crops"],
        "missing": report["missing"] or None,
        "transfer_mismatch_possible": report["transfer_mismatch_possible"],
        "steps": report["steps"],
        "colour": {"candidate": candidate.colour, "reference": reference.colour},
        "mask": mask_record,
        "sources": {
            "candidate": candidate_record(source, made, candidate_path, folder),
            "reference": {
                "file": reference_path.name,
                "ref_id": record["ref_id"] if record else None,
                "name": record["name"] if record else None,
                "sha256": reference_hash,
            },
        },
        "series": series,
        "trend": trend,
        "scene_stamp": scene.stamp,
        "warnings": warnings or None,
    }
    write_result(folder / "result.json", saved)
    references.log_run(
        scene.place,
        series["id"],
        {
            "run_id": plan.run_id,
            "name": plan.name,
            "when": time.time(),
            "when_iso": datetime.now().isoformat(timespec="seconds"),
            "key": key,
            "metrics": report["metrics_numbers"],
            "result": relative_to(folder / "result.json", scene.place),
        },
    )
    # What the caller gets is the saved record with places it can open here.
    returned = {
        **saved,
        "folder": str(folder),
        "files": absolute(folder, report["files"]),
        "sources": {
            "candidate": {**saved["sources"]["candidate"], "path": str(candidate_path)},
            "reference": {**saved["sources"]["reference"], "path": str(reference_path)},
        },
    }
    how = arguments.get("return_image") or "thumb"
    returned.update(attach_picture(call, how, report, beside=returned))
    return returned


def give_back(call: Call, plan: outputs_module.OutputPlan) -> None:
    """Hand back a compare place that never got its files."""
    try:
        with call.router.store(create=True) as store:
            outputs_module.release(store, plan)
    except Exception:  # noqa: BLE001 - the error that brought us here is the one to report
        pass


def relative_to(path: Path, base: Path) -> str | None:
    """A path written from another folder, or nothing when there is no way between them."""
    try:
        return Path(os.path.relpath(path, base)).as_posix()
    except ValueError:
        return None


def absolute(folder: Path, files: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in files.items():
        if isinstance(value, Mapping):
            out[key] = absolute(folder, value)
        else:
            out[key] = str(folder / value)
    return out


def which_reference(scene: Scene, handle: str) -> tuple[dict[str, Any] | None, Path]:
    """A registered reference by name or id first, then a file by its absolute path."""
    record = references.find(scene.place, handle)
    if record is not None:
        image = references.resolve(scene.place, record.get("image"))
        if image is None or not image.is_file():
            raise CallError(
                "FILE_NOT_FOUND",
                f"the image registered as {record.get('name')} is not in the reference folder",
                hint="register the reference again with set_reference",
                details={"argument": "reference", "ref_id": record.get("ref_id")},
            )
        return record, image
    if Path(os.path.expanduser(handle)).is_absolute():
        return None, a_file(handle, "reference")
    known = references.names(scene.place)
    raise CallError(
        "REFERENCE_UNKNOWN",
        f"no reference is registered as {handle}",
        hint="pass a name from list_references, or an absolute path to an image",
        details={"argument": "reference", "did_you_mean": did_you_mean(handle, known)},
    )


# Section: the candidate, when it is not a file


def check_candidate(wanted: Mapping[str, Any], source: str) -> None:
    """Refuse what the candidate's source does not take, before the session is asked."""
    known = CANDIDATE_KEYS[source]
    for key in wanted:
        # A file candidate has always let other keys by.
        if key != "source" and key not in known and source != "file":
            raise CallError(
                "BAD_ARGUMENTS",
                f"a {source} candidate takes no {key}",
                details={
                    "argument": f"candidate.{key}",
                    "did_you_mean": did_you_mean(str(key), list(known)),
                    "arguments": ["source", *known],
                },
            )
    if source in CAPTURED:
        if source == "node" and not wanted.get("path"):
            raise CallError(
                "BAD_ARGUMENTS",
                "a node candidate needs path: the node to capture",
                details={"argument": "candidate.path"},
            )
        as_candidate(lambda: capture_tool.checked({**wanted, "source": source}))
    if source == "render" and bool(wanted.get("job_id")) == bool(wanted.get("path")):
        raise CallError(
            "BAD_ARGUMENTS",
            "a render candidate takes job_id, or path: the node whose output to use",
            details={"argument": "candidate"},
        )


def as_candidate(work: Callable[[], Any]) -> Any:
    """Run capture code, naming a refused argument as part of the candidate."""
    try:
        return work()
    except CallError as error:
        where = error.details.get("argument")
        if where and not str(where).startswith("candidate"):
            error.details["argument"] = f"candidate.{where}"
        raise


def captured(
    call: Call,
    wanted: Mapping[str, Any],
    source: str,
    record: Mapping[str, Any] | None,
    name: str,
    reference_px: tuple[int, int],
) -> tuple[Path, dict[str, Any], list[str]]:
    """Capture the candidate now, through the capture code in this process.

    A registered reference that names a camera frames the capture unless the
    candidate names its own camera, and sets its size, at the reference's
    aspect, unless the candidate names its own size. The capture goes under
    an id derived from the compare's own. Its warnings come back beside it.
    """
    taken = (*CAPTURE_KEYS, "timeout_s")
    arguments = {key: wanted[key] for key in taken if wanted.get(key) is not None}
    arguments["source"] = source
    arguments["name"] = outputs_module.sanitize_name(f"{name}_candidate")[: capture_tool.MAX_NAME]
    framed_by = "candidate" if "camera" in arguments else "capture"
    if record and record.get("camera") and "camera" not in arguments:
        arguments["camera"] = record["camera"]
        framed_by = "reference"
        if "resolution" not in arguments and record.get("width") and record.get("height"):
            arguments["resolution"] = reference_size(int(record["width"]), int(record["height"]))
    derived = f"{call.operation_id()}{OPERATION_ID_SEPARATOR}{CAPTURE_SUFFIX}"
    try:
        said = as_candidate(lambda: capture_tool.take(call, arguments, operation_id=derived))
    except CallError as error:
        raise blamed(error, framed_by, record) from None
    if not said.get("path"):
        raise CallError(
            "NO_OUTPUT",
            "the capture stopped before it wrote an image",
            details={"source": source, "job_id": said.get("job_id")},
        )
    made: dict[str, Any] = {
        "run_id": said.get("run_id"),
        "job_id": said.get("job_id"),
        "route": said.get("route"),
        "camera": said.get("camera"),
        "framed_by": framed_by,
        "size_px": [said.get("width"), said.get("height")],
        "frame": said.get("frame"),
    }
    for flag in ("framing_unverified", "unsaved_hip"):
        if said.get(flag):
            made[flag] = True
    notes = [str(item) for item in said.get("warnings") or ()]
    width, height = said.get("width"), said.get("height")
    if width and height and off_aspect((int(width), int(height)), reference_px):
        notes.append(
            f"the capture is {width}x{height}, not at the aspect of the "
            f"{reference_px[0]}x{reference_px[1]} reference, so align decides how the two meet"
        )
    return Path(str(said["path"])), made, notes


def blamed(error: CallError, framed_by: str, record: Mapping[str, Any] | None) -> CallError:
    """A capture's error, put on what caused it.

    A camera taken from the reference is the reference's, not the caller's.
    A capture that ran past its time is still going: the render source reads
    it once it ends.
    """
    where = str(error.details.get("argument") or "")
    if framed_by == "reference" and where in ("camera", "candidate.camera"):
        error.details["argument"] = "reference"
        error.details["camera"] = (record or {}).get("camera")
        error.hint = "register the reference again, or pass candidate.camera"
    elif error.code == "TIMEOUT" and error.details.get("job_id"):
        error.hint = "wait with hou_jobs, then compare with source render and this job_id"
    return error


def off_aspect(size: tuple[int, int], reference_px: tuple[int, int]) -> bool:
    """Whether a size is off the reference's aspect by more than a pixel."""
    width, height = size
    rw, rh = reference_px
    if not (width and height and rw and rh):
        return False
    return abs(height - width * rh / rw) > 1.0 and abs(width - height * rw / rh) > 1.0


def partly_covered(note: Mapping[str, Any]) -> bool:
    """Whether a side's alpha leaves a real share of it clear, and a real share covered."""
    if not note.get("partial"):
        return False
    coverage = note.get("coverage_pct")
    return coverage is None or MIN_COVERAGE_PCT < float(coverage) < MAX_COVERAGE_PCT


def reference_size(width: int, height: int) -> list[int]:
    """The reference's own size, or its aspect at `REFERENCE_EDGE` when it is larger."""
    scale = min(1.0, REFERENCE_EDGE / max(width, height))
    return [max(1, round(width * scale)), max(1, round(height * scale))]


def rendered(call: Call, wanted: Mapping[str, Any], scene: Scene) -> tuple[Path, dict[str, Any]]:
    """The newest image a finished job, or a node, wrote, as its run records say."""
    job_id = text_or_none(wanted.get("job_id"), "candidate.job_id")
    if job_id is not None:
        return from_job(call, job_id)
    node = str(wanted["path"]).rstrip("/") or "/"
    return from_node(call, node, scene)


def render_ready(call: Call, wanted: Mapping[str, Any]) -> None:
    """`JOB_RUNNING` for a job, or this session's newest run of a node, not yet ended."""
    job_id = text_or_none(wanted.get("job_id"), "candidate.job_id")
    if job_id is not None:
        if not call.router.store_path.is_file():
            raise jobs_tool.unknown(job_id)
        ended_job(call, job_id)
        return
    session_id = call.target().session_id
    node = str(wanted["path"]).rstrip("/") or "/"
    for run in runs_of(call, source_node=node):
        if run.session_id == session_id and run.job_id:
            if job_kept(call, run.job_id):
                ended_job(call, run.job_id)
            return


def from_job(call: Call, job_id: str) -> tuple[Path, dict[str, Any]]:
    job = ended_job(call, job_id)
    details: dict[str, Any] = {"job_id": job_id, "state": job.state, "kind": job.kind}
    files: list[str] = []
    run_id = None
    runs = runs_of(call, job_id=job_id)
    if job.kind == "capture":
        # The answer `hou_jobs` gives, finished once, so a crop is made once
        # and a capture that came back empty is not taken for a picture.
        answer, why_not = capture_answer(call, job)
        if why_not is None:
            files = [str(item) for item in (answer.get("paths") or [answer.get("path")]) if item]
            run_id = answer.get("run_id")
        else:
            details["error"] = why_not
    else:
        for run in runs:
            files = run_files(run)
            if files:
                run_id = run.run_id
                break
    chosen = newest(files)
    if chosen is None:
        raise no_output(details)
    node = next((run.source_node for run in runs if run.run_id == run_id), None)
    return chosen, {"run_id": run_id, "job_id": job_id, "job_kind": job.kind, "node": node}


def from_node(call: Call, node: str, scene: Scene) -> tuple[Path, dict[str, Any]]:
    family = outputs_module.hip_family(scene.hip)
    folder = outputs_tool.scene_folder(scene.hip)
    for run in runs_of(call, source_node=node, hip_family=family):
        if scene.hip is None and run.session_id != scene.session_id:
            continue
        if scene.hip is not None and not outputs_tool.made_here(run, folder):
            continue
        job = ended_job(call, run.job_id) if run.job_id and job_kept(call, run.job_id) else None
        if job is not None and job.kind == "capture":
            # Finished once before it is read; an empty capture is no picture.
            if capture_answer(call, job)[1] is not None:
                continue
        chosen = newest(run_files(run))
        if chosen is not None:
            return chosen, {"run_id": run.run_id, "job_id": run.job_id, "node": node}
    raise no_output({"node": node})


def capture_answer(call: Call, job: Any) -> tuple[dict[str, Any], str | None]:
    """A capture job's answer, finished once, and the code saying why it is no picture."""
    answer = capture_tool.job_answer(call, job) or {}
    if isinstance(answer.get("error"), Mapping):
        return answer, str(answer["error"].get("code"))
    if answer.get("empty"):
        return answer, "CAPTURE_EMPTY"
    return answer, None


def ended_job(call: Call, job_id: str) -> Any:
    """A job's row brought up to date, or `JOB_RUNNING` while it has not ended."""
    with jobs_tool.capped(call, jobs_tool.BUSY_S) as store:
        job = jobs_tool.found(store, job_id)
    job = jobs_tool.settle_one(call, job, busy_s=jobs_tool.BUSY_S)
    if job.state not in jobs_tool.FINAL:
        raise CallError(
            "JOB_RUNNING",
            f"job {job_id} is {job.state}, so its output is not final",
            hint=f"wait with hou_jobs (job_id {job_id}, wait_s), then compare again",
            details={"job_id": job_id, "state": job.state, "kind": job.kind},
        )
    return job


def job_kept(call: Call, job_id: str) -> bool:
    try:
        with call.router.store() as store:
            return store is not None and store.get_job(job_id) is not None
    except (store_module.StoreError, sqlite3.Error):
        return False


def runs_of(call: Call, **which: str) -> list[store_module.RunRecord]:
    def read() -> list[store_module.RunRecord]:
        with call.router.store() as store:
            return [] if store is None else store.runs_made_by(**which, limit=NODE_RUNS)

    return guarded("render", read)


def run_files(run: store_module.RunRecord) -> list[str]:
    """The images a run wrote that are there and hold something."""
    paths = run.paths if isinstance(run.paths, Mapping) else {}
    if paths.get("is_directory"):
        return []
    listed = paths.get("files")
    if isinstance(listed, list) and listed:
        found = [str(item) for item in listed if outputs_module.files_on_disk(str(item))]
    else:
        found = outputs_module.files_on_disk(str(paths.get("path") or ""))
    return [item for item in found if Path(item).suffix.lower() in IMAGE_SUFFIXES]


def newest(files: list[str]) -> Path | None:
    """The file written last, and the later one in the list when two share a time."""
    there = [Path(item) for item in files if item and Path(item).is_file()]
    there = [path for path in there if path.stat().st_size > 0]
    if not there:
        return None
    return max(enumerate(there), key=lambda pair: (pair[1].stat().st_mtime, pair[0]))[1]


def no_output(details: dict[str, Any]) -> CallError:
    return CallError(
        "NO_OUTPUT",
        "no image written by that job or node is on disk",
        hint="capture it with a node candidate, or name the image as a file",
        details=details,
    )


def candidate_record(
    source: str, made: Mapping[str, Any], path: Path, folder: Path
) -> dict[str, Any]:
    """The candidate as `result.json` keeps it: its file, and the run that made it."""
    said: dict[str, Any] = {
        "source": source,
        "file": path.name,
        "sha256": references.file_hash(path),
    }
    if made:
        said.update(made)
        # Relative to the result, so the record names no other place on this machine.
        said["path"] = relative_to(path, folder)
    return said


def crops_wanted(value: Any, record: Mapping[str, Any] | None) -> dict[str, list[float]] | None:
    """Named crops to cut, or nothing for the automatic choice.

    `auto` is every region the reference was registered with, or, when it has
    none, one crop around the largest difference once that is known.
    """
    regions = dict((record or {}).get("regions") or {})
    if value in (None, "auto"):
        return regions or None
    if value == "none":
        return {}
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CallError(
            "BAD_ARGUMENTS",
            "detail_crops is auto, none, or a list of region names",
            details={"argument": "detail_crops"},
        )
    unknown = [item for item in value if item not in regions]
    if unknown:
        raise CallError(
            "BAD_ARGUMENTS",
            f"the reference has no region named {unknown[0]}",
            hint="name regions when the reference is registered with set_reference",
            details={
                "argument": "detail_crops",
                "did_you_mean": did_you_mean(unknown[0], list(regions)),
                "regions": sorted(regions),
            },
        )
    return {item: regions[item] for item in value}


def load_side(call: Call, path: Path, side: str, plan: outputs_module.OutputPlan) -> Any:
    """One side in display values. A scene linear file is read by the session.

    The session writes its pixels into this compare's own folder, at a path
    the output conventions check, and the file is removed once read. A
    session that was asked to stop removes it itself.
    """
    module = imaging()
    if path.suffix.lower() not in LINEAR_SUFFIXES:
        return readable(lambda: module.read_display_file(path), side)
    readable(lambda: module.read_linear_header(path), side)
    raw = Path(guarded("compare", lambda: outputs_module.inside(plan, RAW_NAMES[side])))
    try:
        reply = call.bridge("compare.read_exr", {"path": str(path), "out_path": str(raw)})
        data = dict(reply.get("data") or {})
        if data.get("cancelled"):
            raise CallError(
                "TOOL_FAILED",
                f"the session stopped reading the {side} before it was done",
                hint="call again when the session is free",
                details={"argument": side},
            )
        picture = readable(
            lambda: module.picture_from_raw(
                raw,
                width=int(data["width"]),
                height=int(data["height"]),
                channels=int(data["channels"]),
                colour=dict(data.get("colour") or {}),
            ),
            side,
        )
    finally:
        raw.unlink(missing_ok=True)
    picture.path = str(path)
    picture.alpha_note = dict(data.get("alpha") or {"present": picture.alpha is not None})
    picture.resized_on_read = data.get("resized_on_read")
    picture.read_notes = {
        key: data[key] for key in ("route", "scene_marked_changed") if key in data
    }
    return picture


def mask_for(
    value: Any, record: Mapping[str, Any] | None, reference: Any, place: Path
) -> tuple[Any, dict[str, Any] | None]:
    """The area that is counted, from the reference side or a file. Never candidate alpha."""
    if value in (None, MASK_NONE):
        return None, None
    if value == MASK_REFERENCE:
        registered = (record or {}).get("mask")
        if registered:
            path = references.resolve(place, registered.get("path"))
            if path is None or not path.is_file():
                raise CallError(
                    "FILE_NOT_FOUND",
                    "the mask registered with the reference is not in the reference folder",
                    hint="register the reference again with set_reference",
                    details={"argument": "mask"},
                )
            return readable(lambda: imaging().read_mask(path), "mask"), {
                "source": "reference_record",
                "file": path.name,
                "sha256": references.file_hash(path),
            }
        if reference.alpha is not None:
            return reference.alpha, {
                "source": "reference_alpha",
                "file": Path(reference.path).name,
                "sha256": references.file_hash(reference.path),
            }
        raise CallError(
            "BAD_ARGUMENTS",
            "the reference has no mask and no alpha to take one from",
            hint="register a mask with set_reference, or pass a mask file",
            details={"argument": "mask"},
        )
    path = a_file(value, "mask")
    return readable(lambda: imaging().read_mask(path), "mask"), {
        "source": "file",
        "file": path.name,
        "sha256": references.file_hash(path),
    }


def is_plain(colour: Mapping[str, Any]) -> bool:
    """Whether a side's values are what a plain sRGB display shows."""
    if colour.get("kind") != imaging().VIEW_KIND:
        return True
    if colour.get("transform") == "srgb_curve":
        return True
    display = str(colour.get("display") or "").lower()
    view = str(colour.get("view") or "").strip().lower()
    return "srgb" in display and view in PLAIN_VIEWS


def transfer_reasons(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> list[str]:
    """Why the two sides may not have gone through the same transfer, from their records."""
    view_kind = imaging().VIEW_KIND
    both = candidate.get("kind") == view_kind and reference.get("kind") == view_kind
    if both:
        views = [(side.get("display"), side.get("view")) for side in (candidate, reference)]
        if views[0] != views[1]:
            return ["the two sides went through different display views"]
        return []
    reasons = []
    for side, colour in (("candidate", candidate), ("reference", reference)):
        if not is_plain(colour):
            reasons.append(
                f"the {side} went through the {colour.get('view')} view on "
                f"{colour.get('display')}, not a plain sRGB display"
            )
    return reasons


def run(
    candidate: Any,
    reference: Any,
    *,
    folder: Path,
    settings: Mapping[str, Any],
    region: list[float] | None,
    crops: dict[str, list[float]] | None,
    mask_map: Any,
) -> dict[str, Any]:
    """Align, cut, shrink, count and draw, in that order, writing every file.

    File names in the report are relative to `folder`.
    """
    module = imaging()
    import numpy as np

    tolerance = float(settings["tolerance"])
    # 2. Align at native resolution.
    aligned = module.align(
        candidate,
        reference,
        mode=str(settings["align"]),
        adjust=settings["adjust"],
        auto_shift=bool(settings["auto_shift"]),
    )
    width, height = aligned.size
    counted = aligned.valid.copy()
    if mask_map is not None:
        counted &= module.resize(mask_map, width, height) >= 0.5
    sides = (("candidate", candidate), ("reference", reference))
    steps: dict[str, Any] = {
        "profile": {side: picture.colour.get("profile") for side, picture in sides},
        "view_transform": {
            side: picture.colour if picture.colour.get("kind") == module.VIEW_KIND else None
            for side, picture in sides
        },
        "alpha": {side: picture.alpha_note for side, picture in sides},
        "resized_on_read": {side: picture.resized_on_read for side, picture in sides},
        "session_read": {side: getattr(picture, "read_notes", None) for side, picture in sides},
        **aligned.steps,
    }
    files: dict[str, Any] = {}
    crop_rows: dict[str, Any] = {}
    missing: dict[str, str] = {}

    # 3. Crop before shrinking: named regions from the full aligned frame.
    for crop_name, rect in (crops or {}).items():
        box = module.box_px(rect, width, height)
        crop_rows[crop_name] = detail(
            module,
            crop_name,
            rect,
            box,
            aligned.candidate,
            aligned.reference,
            counted,
            tolerance,
            folder,
            files,
        )
    cand, ref = aligned.candidate, aligned.reference
    steps["crop_px"] = None
    if region is not None:
        box = module.box_px(region, width, height)
        cand, ref, counted = (module.cut(item, box) for item in (cand, ref, counted))
        steps["crop_px"] = list(box)

    module.to_image(cand).save(folder / "candidate.png")
    module.to_image(ref).save(folder / "reference.png")
    files["candidate"] = "candidate.png"
    files["reference"] = "reference.png"

    # 4. The overview, at the working size, never larger than the image.
    native_h, native_w = ref.shape[:2]
    size = module.working_size(native_w, native_h)
    small_cand = np.clip(module.resize(cand, *size), 0.0, 1.0)
    small_ref = np.clip(module.resize(ref, *size), 0.0, 1.0)
    small_counted = module.resize(counted.astype(np.float32), *size) >= 0.5
    steps["resized_to"] = list(size)

    numbers = None
    try:
        numbers = module.metrics(small_cand, small_ref, small_counted, tolerance)
    except module.MetricsUnavailable as error:
        missing["all"] = str(error)
    if numbers is not None and numbers["psnr_db"] is None:
        missing["psnr_db"] = "no difference where counted, so no finite value"
    over = (np.abs(small_cand - small_ref).max(axis=2) > tolerance) & small_counted
    largest = module.largest_region(over)
    if largest is not None:
        largest["of"] = "region" if region is not None else "frame"

    if crops is None and largest is not None:
        rect = padded(largest["box"])
        box = module.box_px(rect, native_w, native_h)
        crop_rows[AUTO_CROP_NAME] = detail(
            module, AUTO_CROP_NAME, rect, box, cand, ref, counted, tolerance, folder, files
        )

    matched = None
    if settings["match_exposure"]:
        try:
            matched = module.exposure_matched(small_cand, small_ref, small_counted, tolerance)
        except module.MetricsUnavailable as error:
            missing["exposure_matched"] = str(error)

    reasons = transfer_reasons(candidate.colour, reference.colour)
    gap = module.mean_luminance_gap(small_cand, small_ref, small_counted)
    if gap is not None and gap > module.LUMINANCE_MISMATCH:
        reasons.append(f"mean luminance differs by {round(gap * 100)} percent after alignment")

    heat = module.heat_map(small_cand, small_ref, small_counted)
    module.to_image(heat).save(folder / "diff.png")
    files["diff"] = "diff.png"
    labels = ["candidate", "reference", "difference"]
    sheet = module.side_by_side([small_cand, small_ref, heat], labels, edge=module.PANEL_EDGE)
    sheet.save(folder / "overview.jpg", quality=85)
    files["overview"] = "overview.jpg"
    files["result"] = "result.json"

    metrics = None
    if numbers is not None:
        likeness = settings["mode"] == "likeness"
        metrics = {
            "role": "secondary" if likeness else "primary",
            "note": LIKENESS_NOTE if likeness else REGRESSION_NOTE,
            "units": "display values from 0 to 1, counted at the working size",
            "tolerance": tolerance,
            **numbers,
        }
        if matched is not None:
            metrics["exposure_matched"] = matched
    return {
        "files": files,
        "metrics": metrics,
        "metrics_numbers": numbers,
        "largest_region": largest,
        "crops": crop_rows or None,
        "missing": missing,
        "transfer_mismatch_possible": {"flag": bool(reasons), "why": reasons},
        "steps": steps,
        "full_panels": (small_cand, small_ref, heat),
        "sheet": sheet,
    }


def detail(
    module: Any,
    name: str,
    rect: list[float],
    box: tuple[int, int, int, int],
    candidate: Any,
    reference: Any,
    counted: Any,
    tolerance: float,
    folder: Path,
    files: dict[str, Any],
) -> dict[str, Any]:
    """One detail crop at native size: its numbers and its side by side file."""
    cand, ref, mask = (module.cut(item, box) for item in (candidate, reference, counted))
    crops = folder / "crops"
    crops.mkdir(exist_ok=True)
    file_name = f"{outputs_module.sanitize_name(name)}.png"
    module.side_by_side([cand, ref], ["candidate", "reference"]).save(crops / file_name)
    files.setdefault("crops", {})[name] = f"crops/{file_name}"
    row: dict[str, Any] = {
        "box": [round(value, 4) for value in rect],
        "crop_px": list(box),
        "size_px": [box[2] - box[0], box[3] - box[1]],
    }
    try:
        row["metrics"] = module.metrics(cand, ref, mask, tolerance)
    except module.MetricsUnavailable as error:
        row["missing"] = str(error)
    return row


def padded(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = box
    grow_x, grow_y = (x1 - x0) * AUTO_CROP_MARGIN, (y1 - y0) * AUTO_CROP_MARGIN
    return [
        max(0.0, x0 - grow_x),
        max(0.0, y0 - grow_y),
        min(1.0, x1 + grow_x),
        min(1.0, y1 + grow_y),
    ]


def follow_series(
    place: Path, key: Mapping[str, Any], name: str, numbers: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Which series this run belongs to, and the trend when it has earlier runs."""
    identity = references.series_id(key)
    earlier = references.series_runs(place, identity)
    if earlier:
        trend = references.trend(earlier)
        if trend is not None:
            last = trend["earlier"][-1]
            trend["change_since_last"] = references.change_since(last, numbers)
        return {"id": identity, "new": False, "runs_before": len(earlier)}, trend
    series: dict[str, Any] = {"id": identity, "new": True, "runs_before": 0}
    prior = references.last_series_for(place, name, besides=identity)
    if prior is not None and isinstance(prior.get("key"), Mapping):
        series["previous_series"] = prior["series_id"]
        series["changed"] = references.changed_fields(prior["key"], key)
        series["why_new"] = "the reference, camera, crop, mask, colour or settings changed"
    else:
        series["why_new"] = "the first compare of this setup"
    return series, None


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    text = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    path.write_text(text + "\n", encoding="utf-8")


def attach_picture(
    call: Call, how: str, report: Mapping[str, Any], *, beside: Mapping[str, Any]
) -> dict[str, Any]:
    """The overview as image content, for the person as much as the agent.

    The picture has to fit in the reply beside the structured result and the
    text. A full sheet that would not is sent as the thumbnail instead, and
    one that still would not is left out; the result says which.
    """
    if how == "none":
        return {}
    module = imaging()
    room = (
        results_module.REPLY_BUDGET_BYTES
        - results_module.TEXT_BLOCK_CAP
        - results_module.structured_size(beside)
        - RESULT_MARGIN
    )
    said: dict[str, Any] = {}
    encoded = None
    if how == "full":
        full = module.side_by_side(
            list(report["full_panels"]), ["candidate", "reference", "difference"]
        )
        encoded = base64.b64encode(module.jpeg_bytes(full)).decode("ascii")
        if len(encoded) > room:
            said["image_downgraded"] = True
            said["image_note"] = (
                f"the full sheet is {len(encoded)} bytes encoded, over the {room} the reply"
                " has room for, so the thumbnail was sent"
            )
            encoded = None
    if encoded is None:
        encoded = base64.b64encode(module.jpeg_bytes(report["sheet"])).decode("ascii")
        if len(encoded) > room:
            said["image_omitted"] = True
            said["image_note"] = (
                "even the thumbnail is over the room the reply has, so none was sent"
            )
            return said
    call.attach(
        ImageContent(
            type="image",
            data=encoded,
            mime_type="image/jpeg",
            annotations=Annotations(audience=["user", "assistant"]),
        )
    )
    return said


def summary(result: Mapping[str, Any]) -> str:
    """One line for a result too long to repeat in the text block."""
    metrics = result.get("metrics") or {}
    series = result.get("series") or {}
    if "references" in result:
        return f"hou_compare: {len(result['references'])} references"
    if "ref_id" in result:
        return f"hou_compare: registered {result.get('name')} as {result.get('ref_id')}"
    if metrics:
        numbers = (
            f"mae {metrics['mae']['overall']}, rmse {metrics['rmse']['overall']}, "
            f"psnr_db {metrics['psnr_db']}, diff_area_pct {metrics['diff_area_pct']} "
            f"({metrics['role']})"
        )
    else:
        numbers = f"no numbers: {(result.get('missing') or {}).get('all')}"
    return (
        f"hou_compare: {numbers}; sheet {(result.get('files') or {}).get('overview')}; "
        f"series {series.get('id')}{' (new)' if series.get('new') else ''}"
    )


ACTION_HANDLERS: dict[str, Callable[[Call], dict[str, Any]]] = {
    "compare": compare,
    "set_reference": set_reference,
    "list_references": list_references,
}

_STRING = {"type": "string"}

HOU_COMPARE = ToolSpec(
    name="hou_compare",
    description=(
        "Compare a candidate image with a reference: "
        "aligned side by side sheet, difference map, numbers. No pass or fail; read both. "
        "Boxes are [x0,y0,x1,y1] in 0..1."
    ),
    input_schema=inputs(
        {
            "action": {"enum": list(ACTIONS)},
            "session": SESSION,
            "reference": _STRING,
            "candidate": {
                "properties": {
                    "source": {"enum": list(SOURCES)},
                    **{key: {} for key in (*CAPTURE_KEYS, "timeout_s", "job_id")},
                }
            },
            "align": {"enum": list(ALIGN)},
            "adjust": {"properties": {"dx": {}, "dy": {}, "scale": {}}},
            "auto_shift": {"type": "boolean"},
            "region": {"type": "array"},
            "mode": {"enum": list(MODES)},
            "mask": {},
            "detail_crops": {},
            "match_exposure": {"type": "boolean"},
            "tolerance": {"type": "number"},
            "name": {},
            "camera": {},
            "regions": {"type": "object"},
            "return_image": {"enum": list(RETURN_IMAGE)},
        }
    ),
    output_schema=outputs({}),
    handler=compare_tool,
    summary=summary,
    open_world=False,
)
