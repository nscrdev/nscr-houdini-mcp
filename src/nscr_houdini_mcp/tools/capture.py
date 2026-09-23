"""`hou_capture`: save a picture of what a session shows, and look at it.

The session makes the picture through `capture.image`, which picks a route
and writes every file under the `capture` kind of the output table
(`$HIP/.agent/captures/<date>/<time>_<name>_<run_id>.png`). This side reads
what was written, with Pillow, and says what is in it:

- `image_stats`: the mean, the least and the most of each channel, the
  depth, `flat` for an image that is one value in every channel, and
  `non_empty`, which is false for an empty file and, for the viewport and
  node sources, for an alpha that is zero everywhere. A flat image is a
  picture, with a note. A capture whose every image is empty is
  `CAPTURE_EMPTY`, with the files left where they are.
- `region` crops each saved image in place, at its native size, to
  `[x0, y0, x1, y1]` as fractions of the width and height from the top left.
  A crop is made once: the cut image carries a note saying so, and one that
  carries it is left alone, so a reply sent again does not crop twice.
- `views: quad` writes persp, top, front and right, and `turntable4` four
  orbits a quarter turn apart; both add a contact sheet of the four, two by
  two, whose path is `path`. Each view is listed with its own route, camera
  and numbers.
- `return_image`: `thumb`, the default, sends a thumbnail no longer than 512
  pixels on its long edge and no bigger than a bounded number of bytes, as
  image content beside the text. `full` sends the whole image when it is
  small enough, and a thumbnail with a warning when it is not. `none` sends
  no image.

A sequence (`frames`) is a job. The call waits up to `inline_wait_s` from the
config, or `timeout_s` when that is shorter, and a sequence still rendering
then answers with the job to follow in `hou_jobs`, whose finished answer
carries the same paths and numbers without the thumbnail. A single capture
waits up to `timeout_s` and answers `TIMEOUT` with the `job_id` past it.

Every capture takes a receipt under its operation id, so the same id sent
again after a lost reply gets the same files back rather than a second
render.

This module never imports `hou`.
"""

from __future__ import annotations

import base64
import io
import math
import os
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mcp_types import ImageContent
from PIL import Image, ImageStat, UnidentifiedImageError
from PIL.PngImagePlugin import PngInfo

from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp import jobs as job_rules
from nscr_houdini_mcp import outputs as outputs_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_TIMEOUT_S
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools import python as python_tool
from nscr_houdini_mcp.tools.base import (
    OPERATION_ID_MAX,
    OPERATION_ID_SEPARATOR,
    Call,
    ToolSpec,
    inputs,
    outputs,
)

SOURCES = ("viewport", "node", "network", "cop", "pane")
DISPLAYS = ("shaded", "wire", "shaded_wire", "matcap")
VIEW_SETS = ("single", "quad", "turntable4")
RETURN_IMAGE = ("thumb", "none", "full")
# Sources a camera draws, where a clear frame means nothing was in view.
RENDERED = ("viewport", "node")

DEFAULT_RESOLUTION = [1280, 720]
MAX_RESOLUTION = 8192
MAX_FRAMES = 1000
MAX_NAME = 64
MAX_WAIT_S = 50.0
MAX_TIMEOUT_S = 3600.0

# The thumbnail's long edge, and the most it may weigh before it is sent as a
# JPEG, then smaller.
THUMB_EDGE = 512
THUMB_MAX_BYTES = 192_000
# The most a full image may weigh to be sent whole.
FULL_MAX_BYTES = 4_000_000
# What a thumbnail with transparency is laid on when it has to be a JPEG.
MATTE = (128, 128, 128)

_OPERATION_ID = re.compile(rf"[A-Za-z0-9_-]{{1,{OPERATION_ID_MAX}}}")

# The arguments the session is sent. The rest are this side's.
SENT = (
    "source",
    "path",
    "camera",
    "frame_target",
    "display",
    "guides",
    "frame",
    "frames",
    "views",
    "name",
)


def capture(call: Call) -> dict[str, Any]:
    arguments = call.arguments
    sent = checked(arguments)
    wanted = arguments.get("return_image") or "thumb"
    sequence = sent.get("frames") is not None
    asked = arguments.get("timeout_s")
    run_for = DEFAULT_TIMEOUT_S if asked is None else float(asked)
    if sequence:
        # A sequence is followed as a job once it runs past a short wait.
        inline = call.config.inline_wait_s if call.config else config_module.DEFAULT_INLINE_WAIT_S
        run_for = min(float(inline), run_for)
    call.arguments["timeout_s"] = run_for
    job_id = job_rules.job_id_for(call.operation_id())
    try:
        reply = call.bridge("capture.image", sent, mutating=True)
    except CallError as error:
        if error.code == "TIMEOUT" and error.details.get("still_running") and sequence:
            return handle(call, job_id)
        if error.code in python_tool.FOLLOWABLE:
            error.details["job_id"] = job_id
        raise
    said = finalised(call, call.operation_id(), dict(reply.get("data") or {}))
    if said.get("empty") and not said.get("stopped_early"):
        raise empty_error(said)
    said["job_id"] = job_id
    said["state"] = "done"
    attach(call, said, wanted)
    return said


# Section: arguments


def checked(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """What the session is sent, after the checks that need no scene.

    The session checks the rest against the scene: that a node is there, that
    a camera is a camera, and what each source takes.
    """
    operation_id = arguments.get("operation_id")
    if operation_id is not None and not _OPERATION_ID.fullmatch(str(operation_id)):
        raise bad("operation_id", "operation_id is 1 to 120 letters, digits, dash or underscore")
    for name, top in (("wait_s", MAX_WAIT_S), ("timeout_s", MAX_TIMEOUT_S)):
        value = arguments.get(name)
        if value is not None and not 0 <= _number(value, name) <= top:
            raise bad(name, f"{name} must be from 0 to {top:g}")
    camera = arguments.get("camera")
    if isinstance(camera, Mapping):
        unknown = sorted(set(camera) - {"orbit", "elevation"})
        if unknown or "orbit" not in camera:
            raise bad("camera", "an orbit camera is {orbit: degrees, elevation: degrees}")
        for key in ("orbit", "elevation"):
            if key in camera:
                _number(camera[key], f"camera.{key}")
    if arguments.get("frame") is not None:
        _number(arguments["frame"], "frame")
    frames = arguments.get("frames")
    if frames is not None:
        _frames(frames)
        if arguments.get("frame") is not None:
            raise bad("frames", "send frame or frames, not both")
    name = arguments.get("name")
    if name is not None and not 1 <= len(str(name)) <= MAX_NAME:
        raise bad("name", f"name is 1 to {MAX_NAME} characters")
    resolution = arguments.get("resolution", DEFAULT_RESOLUTION)
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or not all(_whole(item) and 1 <= item <= MAX_RESOLUTION for item in resolution)
    ):
        raise bad("resolution", f"resolution is [width, height], each 1 to {MAX_RESOLUTION}")
    region = region_of(arguments.get("region"))
    sent = {key: arguments[key] for key in SENT if arguments.get(key) is not None}
    sent["resolution"] = [int(item) for item in resolution]
    if region is not None:
        sent["region"] = region
    return sent


def region_of(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise bad("region", "region is [x0, y0, x1, y1], fractions from the top left")
    x0, y0, x1, y1 = (_number(item, "region") for item in value)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise bad("region", "region needs 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return [x0, y0, x1, y1]


def _frames(value: Any) -> None:
    if not isinstance(value, list) or len(value) != 3 or not all(_whole(item) for item in value):
        raise bad("frames", "frames is [start, end, step], whole numbers")
    start, end, step = value
    if step <= 0 or end < start:
        raise bad("frames", "frames needs step above zero and end at or after start")
    if (end - start) // step + 1 > MAX_FRAMES:
        raise bad("frames", f"a sequence holds at most {MAX_FRAMES} frames")


def _whole(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value) and value == int(value)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise bad(name, f"{name} must be a finite number")
    return float(value)


def bad(argument: str, message: str) -> CallError:
    return CallError("BAD_ARGUMENTS", message, details={"argument": argument})


# Section: what was written


def finish(data: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
    """The session's answer with the images read: crops, numbers and a sheet.

    It changes files, so it runs once per operation, through `finalised`.
    What it reports as `empty` the call turns into `CAPTURE_EMPTY`; a
    finished job read back later reports it as it is. `strict` raises here,
    for a caller that finishes without the claim.
    """
    views = list(data.get("views") or ())
    if views and data.get("stopped_early") and not any(view.get("files") for view in views):
        # Stopped before any frame: nothing to read, and nothing wrong.
        return {
            "source": data.get("source"),
            "path": None,
            "paths": [],
            "frames": [],
            "stopped_early": True,
            "warnings": list(data.get("warnings") or []),
        }
    region = data.get("region")
    sequence = bool(data.get("sequence"))
    rendered = data.get("source") in RENDERED
    notes: list[str] = []
    shaped: list[dict[str, Any]] = []
    sizes: list[tuple[int, int] | None] = []
    all_empty = True
    empty_frames: list[Any] = []
    for view in data.get("views") or ():
        files = [str(item) for item in view.get("files") or ()]
        frames = list(view.get("frames") or ())
        if region is not None:
            for item in files:
                crop(item, region)
        read = [look(item, rendered=rendered) for item in files]
        entry: dict[str, Any] = {
            "view": view.get("view"),
            "path": files[0] if files else None,
            "route": view.get("route"),
            "camera": view.get("camera"),
            "image_stats": read[0][0] if read else None,
            "run_id": view.get("run_id"),
        }
        sizes.append(read[0][1] if read else None)
        if sequence:
            entry["paths"] = files
            entry["frames"] = frames
        for key in ("framing_unverified", "tried", "stopped_early"):
            if view.get(key):
                entry[key] = view[key]
        for index, (stats, _) in enumerate(read):
            if stats["non_empty"]:
                all_empty = False
            elif sequence and index < len(frames):
                empty_frames.append(frames[index])
        if read and read[0][0].get("flat") and read[0][0]["non_empty"]:
            notes.append(f"the {entry['view']} image is one flat colour")
        shaped.append(entry)
    if not shaped:
        raise CallError("CAPTURE_EMPTY", "the session answered with no image")
    if strict and all_empty:
        raise empty_error({"views": shaped})
    first = shaped[0]
    said: dict[str, Any] = {"source": data.get("source")}
    sheet = data.get("sheet")
    if isinstance(sheet, Mapping) and sheet.get("path") and len(shaped) > 1:
        stitch(str(sheet["path"]), [str(item["path"]) for item in shaped])
        stats, size = look(str(sheet["path"]), rendered=False)
        said.update(path=sheet["path"], run_id=sheet.get("run_id"), image_stats=stats)
        said["width"], said["height"] = size or (None, None)
        said["views"] = shaped
        said["route"] = first["route"]
        said["camera"] = None
    else:
        said.update(
            path=first["path"],
            run_id=first["run_id"],
            image_stats=first["image_stats"],
            route=first["route"],
            camera=first["camera"],
        )
        said["width"], said["height"] = sizes[0] or (None, None)
        for key in ("tried",):
            if first.get(key):
                said[key] = first[key]
    if sequence:
        said["paths"] = first.get("paths", [])
        said["frames"] = first.get("frames", [])
        if empty_frames:
            said["empty_frames"] = empty_frames
    else:
        frames = (data.get("views") or [{}])[0].get("frames") or [None]
        said["frame"] = frames[0]
    if any(item.get("framing_unverified") for item in shaped):
        said["framing_unverified"] = True
    if region is not None:
        said["region"] = region
    for key in ("unsaved_hip", "stopped_early"):
        if data.get(key):
            said[key] = True
    if data.get("warnings") or notes:
        said["warnings"] = list(data.get("warnings") or []) + notes
    if all_empty:
        said["empty"] = True
    return said


def empty_error(said: Mapping[str, Any]) -> CallError:
    """`CAPTURE_EMPTY`, with the numbers of each view that came back empty."""
    views = said.get("views") or [
        {"view": "single", "route": said.get("route"), "image_stats": said.get("image_stats")}
    ]
    return CallError(
        "CAPTURE_EMPTY",
        "the capture wrote only empty images",
        details={
            "views": [
                {
                    "view": item.get("view"),
                    "route": item.get("route"),
                    "image_stats": item.get("image_stats"),
                }
                for item in views
            ]
        },
    )


# How long a second caller waits for the first to finish the same capture.
FINISH_WAIT_S = 30.0
FINISH_POLL_S = 0.1


def finalised(call: Call, operation_id: str | None, data: Mapping[str, Any]) -> dict[str, Any]:
    """The capture's answer with its files finished, made once per operation.

    Cropping and stitching write files, and a retry after a lost reply and a
    job status read can both arrive for the same capture. The first takes a
    claim in the store under the operation id and keeps the finished answer
    there; the others wait for it and read it, so no file is finished twice.
    Without a store to claim in, the files are finished here: each is cut
    once, whatever else reads it.
    """
    if not operation_id:
        return finish(data)
    key = f"{operation_id}{OPERATION_ID_SEPARATOR}finish"
    runs = [view.get("run_id") for view in data.get("views") or ()]
    sheet = data.get("sheet") if isinstance(data.get("sheet"), Mapping) else {}
    digest = store_module.digest_arguments({"runs": runs, "sheet": sheet.get("run_id")})
    try:
        with call.router.store(create=True) as store:
            claim = store.begin_operation(key, digest)
    except (store_module.StoreError, sqlite3.Error, store_module.OperationMismatch, CallError):
        return finish(data)
    if not claim.claimed:
        if not claim.outcome_unknown:
            outcome = claim.record.outcome if isinstance(claim.record.outcome, Mapping) else {}
            return dict(outcome.get("said") or {})
        return waited(call, key)
    try:
        said = finish(data)
    except BaseException:
        settle(call, lambda store: store.drop_operation(key))
        raise
    settle(call, lambda store: store.finish_operation(key, outcome={"said": said}))
    return said


def waited(call: Call, key: str) -> dict[str, Any]:
    """The answer another call is finishing, once it has."""
    deadline = time.monotonic() + FINISH_WAIT_S
    while time.monotonic() < deadline:
        try:
            with call.router.store() as store:
                record = None if store is None else store.get_operation(key)
        except (store_module.StoreError, sqlite3.Error, CallError):
            record = None
        if record is None:
            break
        if record.state != "running":
            outcome = record.outcome if isinstance(record.outcome, Mapping) else {}
            return dict(outcome.get("said") or {})
        time.sleep(FINISH_POLL_S)
    raise CallError(
        "OUTCOME_UNKNOWN",
        "another call is still finishing this capture",
        hint="ask again shortly with the same operation_id",
    )


def settle(call: Call, action: Any) -> None:
    """One write of the finishing claim. A store that will not take it costs the claim only."""
    try:
        with call.router.store() as store:
            if store is not None:
                action(store)
    except (store_module.StoreError, sqlite3.Error, CallError):
        pass


def look(path: str, *, rendered: bool = True) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """What is in one image: its numbers, and its size.

    Empty means a file with nothing in it, or for a rendered source an alpha
    that is zero everywhere: the camera saw nothing. A flat image, one value
    in every channel, is not empty, since a constant colour COP or a close up
    of one surface is a real picture; it says `flat`. A 16 bit grey image is
    read at its own depth. Pillow reads a 16 bit colour image at 8 bits, and
    `stats_depth` says so.
    """
    try:
        if os.path.getsize(path) == 0:
            return {"readable": False, "non_empty": False}, None
        with Image.open(path) as opened:
            image = opened.copy()
            depth = _file_depth(opened)
    except (OSError, UnidentifiedImageError):
        return {"readable": False, "non_empty": False}, None
    if image.mode.startswith("I;16") or image.mode in ("I", "F"):
        stats = _deep_grey(image)
    else:
        if image.mode not in ("RGB", "RGBA", "L", "LA"):
            image = image.convert("RGBA")
        stat = ImageStat.Stat(image)
        stats = {
            "channels": "".join(image.getbands()),
            "mean": [round(value, 2) for value in stat.mean],
            "min": [int(pair[0]) for pair in stat.extrema],
            "max": [int(pair[1]) for pair in stat.extrema],
            "stats_depth": 8,
        }
    stats["depth"] = depth or stats["stats_depth"]
    bands = stats["channels"]
    stats["flat"] = all(a == b for a, b in zip(stats["min"], stats["max"], strict=True))
    clear = rendered and "A" in bands and stats["max"][bands.index("A")] == 0
    stats["non_empty"] = not clear
    return stats, image.size


def _file_depth(opened: Image.Image) -> int | None:
    """The bits a channel holds in the file, where the file says."""
    if opened.format == "PNG":
        try:
            with open(opened.filename, "rb") as handle:
                header = handle.read(26)
        except (OSError, TypeError):
            return None
        return header[24] if len(header) == 26 else None
    return None


def _deep_grey(image: Image.Image) -> dict[str, Any]:
    """Numbers for a grey image of more than 8 bits, at its own depth."""
    wide = image.convert("F" if image.mode == "F" else "I")
    low, high = wide.getextrema()
    # The newer name where this Pillow has it, the older one otherwise.
    flattened = getattr(wide, "get_flattened_data", None) or wide.getdata
    values = list(flattened())
    mean = sum(values) / len(values) if values else 0.0
    as_number = float if image.mode == "F" else int
    return {
        "channels": "L",
        "mean": [round(mean, 2)],
        "min": [as_number(low)],
        "max": [as_number(high)],
        "stats_depth": 32 if image.mode in ("I", "F") else 16,
    }


# The note a crop leaves in the PNG it cut, so the same image is never cut twice.
CROP_KEY = "nscr_crop"


def crop(path: str, region: Sequence[float]) -> None:
    """Cut one saved image down to the region, once, in place.

    The region is written into the image as a text note. An image that
    carries one has been cut already, whatever size the session said it
    wrote, so a reply sent again, or a job read back twice, leaves it alone.
    """
    try:
        with Image.open(path) as opened:
            image = opened.copy()
            done = opened.info.get(CROP_KEY)
    except (OSError, UnidentifiedImageError):
        return
    if done is not None:
        return
    left, top, right, bottom = crop_box(region, image.size)
    note = PngInfo()
    note.add_text(CROP_KEY, ",".join(f"{value:g}" for value in region))
    save(image.crop((left, top, right, bottom)), path, note)


def crop_box(region: Sequence[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The region in pixels: at least one pixel, and never off the image."""
    x0, y0, x1, y1 = region
    width, height = size
    left = min(width - 1, int(math.floor(x0 * width)))
    top = min(height - 1, int(math.floor(y0 * height)))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    return left, top, right, bottom


def stitch(path: str, parts: Sequence[str]) -> None:
    """Two by two contact sheet of the views, each at its own size in an equal cell."""
    images = []
    for item in parts:
        with Image.open(item) as opened:
            images.append(opened.convert("RGBA"))
    cell_w = max(image.size[0] for image in images)
    cell_h = max(image.size[1] for image in images)
    columns = 2
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGBA", (cell_w * columns, cell_h * rows), (0, 0, 0, 0))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * cell_w, (index // columns) * cell_h))
    save(sheet, path)


def save(image: Image.Image, path: str, note: PngInfo | None = None) -> None:
    """Write an image whole, through a file of this writer's own beside it."""
    partial = outputs_module.temporary_beside(path)
    try:
        image.save(partial, format="PNG", pnginfo=note)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


# Section: the picture sent back


def attach(call: Call, said: dict[str, Any], wanted: str) -> None:
    """Put the thumbnail, or the whole image, beside the text."""
    if wanted == "none" or not said.get("path"):
        return
    path = str(said.get("paths", [None])[0] if said.get("paths") else said["path"])
    notes = said.setdefault("warnings", [])
    try:
        made: tuple[bytes, str, tuple[Any, Any]] | None
        kind = "thumb"
        if wanted == "full" and Path(path).stat().st_size <= FULL_MAX_BYTES:
            made = (Path(path).read_bytes(), "image/png", (said.get("width"), said.get("height")))
            kind = "full"
        else:
            if wanted == "full":
                notes.append(
                    f"the full image is over {FULL_MAX_BYTES} bytes, so a thumbnail came back"
                )
            made = thumbnail(path)
    except (OSError, UnidentifiedImageError):
        notes.append("the image could not be read for a thumbnail")
        return
    finally:
        if not notes:
            said.pop("warnings", None)
    if made is None:
        said.setdefault("warnings", []).append(
            f"no thumbnail of this image fits in {THUMB_MAX_BYTES} bytes, so none came back"
        )
        return
    data, mime, (width, height) = made
    call.attach(
        ImageContent(type="image", data=base64.b64encode(data).decode("ascii"), mime_type=mime)
    )
    said["thumb"] = {
        "kind": kind,
        "width": width,
        "height": height,
        "bytes": len(data),
        "mime_type": mime,
    }


def thumbnail(
    path: str, *, edge: int = THUMB_EDGE, max_bytes: int | None = None
) -> tuple[bytes, str, tuple[int, int]] | None:
    """A small copy of an image: at most `edge` on its long side and `max_bytes` whole.

    Nothing when no encoding of it fits in `max_bytes`, which is
    `THUMB_MAX_BYTES` unless given.
    """
    max_bytes = THUMB_MAX_BYTES if max_bytes is None else max_bytes
    with Image.open(path) as opened:
        image = opened.copy()
    if image.mode.startswith("I") or image.mode == "F":
        # Shown at 8 bits: the top byte of a 16 bit value.
        scale = 1.0 if image.mode == "F" else 1.0 / 256.0
        image = image.convert("I").convert("F").point(lambda value: value * scale).convert("L")
    image.thumbnail((edge, edge))
    data = encode(image, "PNG")
    if len(data) <= max_bytes:
        return data, "image/png", image.size
    flat = image
    if image.mode in ("RGBA", "LA", "P"):
        flat = Image.new("RGB", image.size, MATTE)
        flat.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
    flat = flat.convert("RGB")
    while True:
        for quality in (85, 70, 50):
            data = encode(flat, "JPEG", quality=quality)
            if len(data) <= max_bytes:
                return data, "image/jpeg", flat.size
        if max(flat.size) <= 16:
            # Nothing small enough fits: better no thumbnail than one over the bound.
            return None
        flat = flat.resize((max(1, flat.size[0] // 2), max(1, flat.size[1] // 2)))


def encode(image: Image.Image, kind: str, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=kind, **options)
    return buffer.getvalue()


# Section: a sequence followed as a job


def handle(call: Call, job_id: str) -> dict[str, Any]:
    """What a sequence still rendering answers: the job to follow."""
    record = python_tool.read_job(call, job_id)
    scene = record.scene if record is not None and isinstance(record.scene, dict) else {}
    return {
        "job_id": job_id,
        "state": record.state if record is not None else "running",
        "session": call.trace.get("session_id"),
        "kind": "capture",
        "started_at": (record.started_at or record.created_at) if record is not None else None,
        "operation_id": call.trace.get("operation_id"),
        "scene_epoch": scene.get("scene_epoch", call.trace.get("scene_epoch")),
    }


def job_answer(call: Call, record: Any) -> dict[str, Any] | None:
    """A finished capture job's answer, read the way the call would have read it.

    Finished under the same claim as the call's own answer, so a job read and
    a retry arriving together finish the files once.
    """
    kept = record.outputs if isinstance(record.outputs, Mapping) else {}
    answer = kept.get("answer")
    if not isinstance(answer, Mapping):
        return None
    try:
        return finalised(call, record.operation_id, answer)
    except CallError as error:
        return {"error": error.as_dict()}


def summary_line(data: Mapping[str, Any]) -> str:
    """What a client that reads only text is shown of a long result."""
    if "path" not in data and data.get("job_id"):
        return f"hou_capture: {data.get('state')} as job {data['job_id']}"
    size = f"{data.get('width')}x{data.get('height')}"
    line = f"hou_capture: {data.get('source')} via {data.get('route')}, {size}, {data.get('path')}"
    if data.get("paths"):
        line += f" ({len(data['paths'])} frames)"
    if data.get("framing_unverified"):
        line += "; framing unverified"
    return line


HOU_CAPTURE = ToolSpec(
    name="hou_capture",
    description=(
        "Save an image of the viewport, one node alone, the network editor, a COP or a "
        "pane. Returns the path, image stats and a thumbnail; a sequence may return a job_id."
    ),
    input_schema=inputs(
        {
            "session": {"type": "string"},
            "source": {"enum": list(SOURCES)},
            "path": {"type": "string"},
            "camera": {"type": ["string", "object"]},
            "frame_target": {"type": "string"},
            "display": {"enum": list(DISPLAYS)},
            "guides": {"type": "boolean"},
            "resolution": {"type": "array"},
            "frame": {"type": "number"},
            "frames": {"type": "array"},
            "region": {"type": "array"},
            "views": {"enum": list(VIEW_SETS)},
            "name": {"type": "string"},
            "return_image": {"enum": list(RETURN_IMAGE)},
            "operation_id": {"type": "string"},
            "wait_s": {"type": "number"},
            "timeout_s": {"type": "number"},
        }
    ),
    output_schema=outputs({}),
    handler=capture,
    open_world=False,
    summary=summary_line,
)
