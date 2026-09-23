"""`hou_capture`: save a picture of what a session shows, and look at it.

The session makes the picture through `capture.image`, which picks a route
and writes every file under the `capture` kind of the output table
(`$HIP/.agent/captures/<date>/<time>_<name>_<run_id>.png`). This side reads
what was written, with Pillow, and says what is in it:

- `image_stats`: the mean, the least and the most of each channel, and
  `non_empty`, which is false for an image with nothing in it: every channel
  one value, or an alpha channel that is zero everywhere. A capture whose
  every image is empty is `CAPTURE_EMPTY`, with the files left where they are.
- `region` crops each saved image in place, at its native size, to
  `[x0, y0, x1, y1]` as fractions of the width and height from the top left.
  A crop is made once: an image that is no longer the size the session wrote
  is taken as cropped already, so a reply sent again does not crop twice.
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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mcp_types import ImageContent
from PIL import Image, ImageStat, UnidentifiedImageError

from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp import jobs as job_rules
from nscr_houdini_mcp.bridge.dispatch import DEFAULT_TIMEOUT_S
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.tools import python as python_tool
from nscr_houdini_mcp.tools.base import OPERATION_ID_MAX, Call, ToolSpec, inputs, outputs

SOURCES = ("viewport", "node", "network", "cop", "pane")
DISPLAYS = ("shaded", "wire", "shaded_wire", "matcap")
VIEW_SETS = ("single", "quad", "turntable4")
RETURN_IMAGE = ("thumb", "none", "full")

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
    said = finish(dict(reply.get("data") or {}), strict=True)
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


def finish(data: Mapping[str, Any], *, strict: bool) -> dict[str, Any]:
    """The session's answer with the images read: crops, numbers and a sheet.

    `strict` makes a capture with nothing in it an error. A finished job read
    back later is not strict: it reports `empty` instead.
    """
    region = data.get("region")
    sequence = bool(data.get("sequence"))
    shaped: list[dict[str, Any]] = []
    sizes: list[tuple[int, int] | None] = []
    all_empty = True
    empty_frames: list[Any] = []
    for view in data.get("views") or ():
        files = [str(item) for item in view.get("files") or ()]
        frames = list(view.get("frames") or ())
        if region is not None:
            for item in files:
                crop(item, region, view.get("native"))
        read = [look(item) for item in files]
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
        shaped.append(entry)
    if not shaped:
        raise CallError("CAPTURE_EMPTY", "the session answered with no image")
    if strict and all_empty:
        raise CallError(
            "CAPTURE_EMPTY",
            "the capture wrote only empty images",
            details={
                "views": [
                    {
                        "view": item["view"],
                        "route": item["route"],
                        "image_stats": item["image_stats"],
                    }
                    for item in shaped
                ]
            },
        )
    first = shaped[0]
    said: dict[str, Any] = {"source": data.get("source")}
    sheet = data.get("sheet")
    if isinstance(sheet, Mapping) and sheet.get("path") and len(shaped) > 1:
        stitch(str(sheet["path"]), [str(item["path"]) for item in shaped])
        stats, size = look(str(sheet["path"]))
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
    if data.get("warnings"):
        said["warnings"] = list(data["warnings"])
    if not strict and all_empty:
        said["empty"] = True
    return said


def look(path: str) -> tuple[dict[str, Any], tuple[int, int] | None]:
    """What is in one image: its numbers, and its size."""
    try:
        with Image.open(path) as opened:
            image = opened.copy()
    except (OSError, UnidentifiedImageError):
        return {"readable": False, "non_empty": False}, None
    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert("RGBA")
    bands = image.getbands()
    stat = ImageStat.Stat(image)
    low = [int(pair[0]) for pair in stat.extrema]
    high = [int(pair[1]) for pair in stat.extrema]
    varies = any(a != b for a, b in zip(low, high, strict=True))
    if "A" in bands and high[bands.index("A")] == 0:
        varies = False
    stats = {
        "channels": "".join(bands),
        "mean": [round(value, 2) for value in stat.mean],
        "min": low,
        "max": high,
        "non_empty": varies,
    }
    return stats, image.size


def crop(path: str, region: Sequence[float], native: Any) -> None:
    """Cut one saved image down to the region, once, in place."""
    try:
        with Image.open(path) as opened:
            image = opened.copy()
    except (OSError, UnidentifiedImageError):
        return
    width, height = image.size
    if isinstance(native, (list, tuple)) and len(native) == 2 and None not in native:
        if (width, height) != (int(native[0]), int(native[1])):
            # Not the size the session wrote: this image was cropped already.
            return
    left, top, right, bottom = crop_box(region, (width, height))
    save(image.crop((left, top, right, bottom)), path)


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


def save(image: Image.Image, path: str) -> None:
    partial = f"{path}.part"
    image.save(partial, format="PNG")
    os.replace(partial, path)


# Section: the picture sent back


def attach(call: Call, said: dict[str, Any], wanted: str) -> None:
    """Put the thumbnail, or the whole image, beside the text."""
    if wanted == "none" or not said.get("path"):
        return
    path = str(said.get("paths", [None])[0] if said.get("paths") else said["path"])
    try:
        if wanted == "full":
            data = Path(path).read_bytes()
            if len(data) <= FULL_MAX_BYTES:
                mime = "image/png"
                width, height = said.get("width"), said.get("height")
                kind = "full"
            else:
                said.setdefault("warnings", []).append(
                    f"the full image is over {FULL_MAX_BYTES} bytes, so a thumbnail came back"
                )
                data, mime, (width, height) = thumbnail(path)
                kind = "thumb"
        else:
            data, mime, (width, height) = thumbnail(path)
            kind = "thumb"
    except (OSError, UnidentifiedImageError):
        said.setdefault("warnings", []).append("the image could not be read for a thumbnail")
        return
    call.attachments.append(
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
    path: str, *, edge: int = THUMB_EDGE, max_bytes: int = THUMB_MAX_BYTES
) -> tuple[bytes, str, tuple[int, int]]:
    """A small copy of an image: at most `edge` on its long side and `max_bytes` whole."""
    with Image.open(path) as opened:
        image = opened.copy()
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
            return data, "image/jpeg", flat.size
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


def job_answer(outputs_: Any) -> dict[str, Any] | None:
    """A finished capture job's answer, read the way the call would have read it."""
    kept = outputs_ if isinstance(outputs_, Mapping) else {}
    answer = kept.get("answer")
    if not isinstance(answer, Mapping):
        return None
    try:
        return finish(answer, strict=False)
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
