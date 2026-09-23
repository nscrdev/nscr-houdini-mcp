"""`hou_compare`: a candidate image against a reference, as pictures and numbers.

Three actions.

- `compare`, the default. The candidate is a file in this build; a viewport
  capture, a node's own image and a render are named in the schema and answer
  `NOT_YET_AVAILABLE` until the capture tool arrives. The work follows the
  order in `imaging`: colour, alignment at native size, crops cut before
  anything is shrunk, then the overview. Every step is recorded in the
  result. The files go to a managed `compare` folder: the aligned pair, the
  difference map, the labelled overview sheet, one side by side file per
  detail crop and `result.json` with everything the call returned.
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
display transform; the rest is done here, in the server process, with NumPy
and Pillow.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
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
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.errors import did_you_mean
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools.base import SESSION, Call, ToolSpec, inputs, outputs

ACTIONS = ("compare", "set_reference", "list_references")
SOURCES = ("file", "viewport", "node", "render")
ALIGN = ("fit", "fill", "stretch", "none")
MODES = ("likeness", "regression")
RETURN_IMAGE = ("thumb", "none", "full")

# The tool that will produce a candidate from the session itself.
CAPTURE_TOOL = "hou_capture"

DEFAULT_TOLERANCE = 0.05

# Files the session reads, through its display transform.
LINEAR_SUFFIXES = (".exr", ".hdr")

# How much room the automatic crop leaves around the largest difference.
AUTO_CROP_MARGIN = 0.1
AUTO_CROP_NAME = "largest_difference"

MASK_NONE = "none"
MASK_REFERENCE = "reference"

LIKENESS_NOTE = "secondary: lighting, material and framing move these numbers; read the sheet first"
REGRESSION_NOTE = "primary: the same setup rendered again, so a change in the numbers is a change"

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
    if not 0.05 <= moved["scale"] <= 20.0:
        raise CallError(
            "BAD_ARGUMENTS",
            "adjust.scale must be from 0.05 to 20",
            details={"argument": "adjust.scale"},
        )
    return moved


def text_or_none(value: Any, argument: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise CallError("BAD_ARGUMENTS", f"{argument} must be text", details={"argument": argument})
    return value


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
    camera = text_or_none(call.arguments.get("camera"), "camera")
    colour, size = colour_on_record(source)
    scene = scene_of(call)
    name = text_or_none(call.arguments.get("name"), "name") or source.stem

    def register() -> dict[str, Any]:
        with call.router.store(create=True) as store:
            return references.register(
                store,
                home=scene.home,
                hip_path=scene.hip,
                session_id=scene.session_id,
                name=str(name),
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
        "ref_id": record["ref_id"],
        "record": str(scene.place / f"{record['ref_id']}.json"),
        "image": record["image"],
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
    """What a reference's record says about its colour, read from the file here."""
    if source.suffix.lower() in LINEAR_SUFFIXES:
        return {
            "kind": imaging().VIEW_KIND,
            "profile": "scene_linear",
            "note": "brought to display values by the session at compare time",
        }, None
    picture = readable(lambda: imaging().read_display_file(source), "reference")
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
    if source != "file":
        raise CallError(
            "NOT_YET_AVAILABLE",
            f"a {source} candidate comes with the capture tool, which this build does not have",
            hint=f"capture with {CAPTURE_TOOL} when it arrives; for now pass the image as a file",
            details={"source": source, "tool": CAPTURE_TOOL, "available": ["file"]},
        )
    if not arguments.get("reference"):
        raise CallError(
            "BAD_ARGUMENTS",
            "compare needs reference: a registered name or an image path",
            details={"argument": "reference"},
        )
    candidate_path = a_file(wanted.get("path"), "candidate.path")
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

    scene = scene_of(call)
    record, reference_path = which_reference(scene, str(arguments["reference"]))
    wanted_crops = crops_wanted(arguments.get("detail_crops", "auto"), record)
    settings["detail_crops"] = sorted(wanted_crops) if wanted_crops is not None else "auto"

    # 1. Colour first: both sides to display values, with a record of how.
    candidate = load_side(call, candidate_path, "candidate", scene)
    reference = load_side(call, reference_path, "reference", scene)
    mask_map, mask_record = mask_for(arguments.get("mask"), record, reference)

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

    reference_hash = references.file_hash(reference_path)
    warnings = list(plan.warnings)
    if record and record.get("sha256") != reference_hash:
        warnings.append("the registered reference image changed on disk since it was registered")
    key = {
        "reference": {"ref_id": record["ref_id"] if record else None, "sha256": reference_hash},
        "camera": record.get("camera") if record else None,
        "crop": region,
        "mask": mask_record and {"source": mask_record["source"], "sha256": mask_record["sha256"]},
        "colour": {"candidate": candidate.colour, "reference": reference.colour},
        "settings": settings,
    }
    series, trend = follow_series(scene.place, key, str(plan.name), report["metrics_numbers"])

    result = {
        "name": plan.name,
        "run_id": plan.run_id,
        "folder": str(folder),
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
            "candidate": str(candidate_path),
            "reference": str(reference_path),
            "reference_id": record["ref_id"] if record else None,
            "reference_sha256": reference_hash,
        },
        "series": series,
        "trend": trend,
        "scene_stamp": scene.stamp,
        "warnings": warnings or None,
    }
    write_result(folder / "result.json", result)
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
            "result": str(folder / "result.json"),
        },
    )
    attach_picture(call, arguments.get("return_image") or "thumb", report)
    return result


def which_reference(scene: Scene, handle: str) -> tuple[dict[str, Any] | None, Path]:
    """A registered reference by name or id, or a file given by its path."""
    record = references.find(scene.place, handle)
    if record is not None:
        image = Path(str(record.get("image")))
        if not image.is_file():
            raise CallError(
                "FILE_NOT_FOUND",
                f"the image registered as {record.get('name')} is not in the reference folder",
                hint="register the reference again with set_reference",
                details={"argument": "reference", "ref_id": record.get("ref_id")},
            )
        return record, image
    looks_like_path = any(mark in handle for mark in ("/", "\\")) or Path(handle).suffix
    if looks_like_path:
        return None, a_file(handle, "reference")
    known = references.names(scene.place)
    raise CallError(
        "REFERENCE_UNKNOWN",
        f"no reference is registered as {handle}",
        details={"argument": "reference", "did_you_mean": did_you_mean(handle, known)},
    )


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


def load_side(call: Call, path: Path, side: str, scene: Scene) -> Any:
    """One side in display values. A scene linear file is read by the session."""
    module = imaging()
    if path.suffix.lower() not in LINEAR_SUFFIXES:
        return readable(lambda: module.read_display_file(path), side)
    temp = scene.home / "temp"
    guarded("compare", lambda: temp.mkdir(parents=True, exist_ok=True))
    raw = temp / f"compare-{secrets.token_hex(6)}.f32"
    try:
        reply = call.bridge("compare.read_exr", {"path": str(path), "out_path": str(raw)})
        data = dict(reply.get("data") or {})
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
    return picture


def mask_for(
    value: Any, record: Mapping[str, Any] | None, reference: Any
) -> tuple[Any, dict[str, Any] | None]:
    """The area that is counted, from the reference side or a file. Never candidate alpha."""
    if value in (None, MASK_NONE):
        return None, None
    if value == MASK_REFERENCE:
        registered = (record or {}).get("mask")
        if registered:
            path = a_file(registered.get("path"), "mask")
            return readable(lambda: imaging().read_mask(path), "mask"), {
                "source": "reference_record",
                "path": str(path),
                "sha256": references.file_hash(path),
            }
        if reference.alpha is not None:
            return reference.alpha, {
                "source": "reference_alpha",
                "path": reference.path,
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
        "path": str(path),
        "sha256": references.file_hash(path),
    }


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
    """Align, cut, shrink, count and draw, in that order, writing every file."""
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
    steps: dict[str, Any] = {
        "profile": {
            "candidate": candidate.colour.get("profile"),
            "reference": reference.colour.get("profile"),
        },
        "view_transform": {
            side: picture.colour if picture.colour.get("kind") == module.VIEW_KIND else None
            for side, picture in (("candidate", candidate), ("reference", reference))
        },
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
    files["candidate"] = str(folder / "candidate.png")
    files["reference"] = str(folder / "reference.png")

    # 4. The overview, at the working size.
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

    reasons = []
    kinds = {candidate.colour.get("kind"), reference.colour.get("kind")}
    if len(kinds) > 1:
        reasons.append("the two sides went through different kinds of transform")
    gap = module.mean_luminance_gap(small_cand, small_ref, small_counted)
    if gap is not None and gap > module.LUMINANCE_MISMATCH:
        reasons.append(f"mean luminance differs by {round(gap * 100)} percent after alignment")

    heat = module.heat_map(small_cand, small_ref, small_counted)
    module.to_image(heat).save(folder / "diff.png")
    files["diff"] = str(folder / "diff.png")
    labels = ["candidate", "reference", "difference"]
    sheet = module.side_by_side([small_cand, small_ref, heat], labels, edge=module.PANEL_EDGE)
    sheet.save(folder / "overview.jpg", quality=85)
    files["overview"] = str(folder / "overview.jpg")
    files["result"] = str(folder / "result.json")

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
    path = crops / f"{outputs_module.sanitize_name(name)}.png"
    module.side_by_side([cand, ref], ["candidate", "reference"]).save(path)
    files.setdefault("crops", {})[name] = str(path)
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


def attach_picture(call: Call, how: str, report: Mapping[str, Any]) -> None:
    """The overview as image content, for the person as much as the agent."""
    if how == "none":
        return
    module = imaging()
    if how == "full":
        picture = module.side_by_side(
            list(report["full_panels"]), ["candidate", "reference", "difference"]
        )
    else:
        picture = report["sheet"]
    call.attach(
        ImageContent(
            type="image",
            data=base64.b64encode(module.jpeg_bytes(picture)).decode("ascii"),
            mime_type="image/jpeg",
            annotations=Annotations(audience=["user", "assistant"]),
        )
    )


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
        "Compare a candidate image (file, viewport, node or render) with a reference: "
        "aligned side by side sheet, difference map, numbers. No pass or fail; read both. "
        "Boxes are [x0,y0,x1,y1] in 0..1."
    ),
    input_schema=inputs(
        {
            "action": {"enum": list(ACTIONS)},
            "session": SESSION,
            "reference": _STRING,
            "candidate": {"properties": {"source": {"enum": list(SOURCES)}, "path": {}}},
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
