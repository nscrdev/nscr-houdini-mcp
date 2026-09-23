"""`capture.image`: a picture of what a session shows, written to a managed path.

Five sources. `viewport` is the 3D view, `node` one node's output on its own,
`network` the network editor, `cop` a COP's output image and `pane` any pane
tab by name. Every file goes to a path from the `capture` kind of the output
table, one per view, with the frame before the extension for a sequence, and
the reply says which route made it and what was tried before.

The routes for a view, in order, and the first that writes a file wins:

- `viewport_flipbook`: the Scene Viewer that is showing, flipbooked with
  settings of its own: no MPlay, the beauty pass only and the guide
  objects left out unless guides are asked for, the resolution asked for.
  A camera, a display mode or a target to frame is applied for the capture
  and put back in a `finally`: the view type, the camera looked through, the
  default camera with its translation, rotation, pivot and ortho width, and
  the shading. A torn off copy of a viewport is never made, because it draws
  nothing and writes no file without saying so.
- `viewport_flipbook_tab`: when no Scene Viewer is showing, an existing one
  is made the current tab of its pane, flipbooked the same way, and the tab
  that was current is put back.
- `flipbook_rop`: a flipbook render node made for the capture, with a camera
  it needs: the one named, or a camera made for the capture and fitted to
  the target's bounds. In a session with a user interface this route cannot
  know what the artist's view frames, so its result says
  `framing_unverified`. It is the only route a session without one has.
  `node` isolates the node: its object alone is drawn, the node carries the
  display flag for the capture, and the flag goes back to where it was.
  It always looks through a camera made for the capture: one fitted to the
  target, or one that follows a named camera as its child and reads its lens
  and window by reference, so a named camera is never written. A worker
  draws at one pixel to a point because the pool, and the bridge's own
  start, put it on Qt's offscreen screen plugin. In a session with a user
  interface, or a hython on another plugin whose screen is not one to one, a
  tiny render of a known box says whether the render node draws larger than
  asked, read again when the screen's pixel ratio changes, and the made
  camera's window makes up for it. A scale that cannot be read leaves the
  framing marked unverified.

Framing `all` goes around the geometry of the objects shown at the frames
captured, the first and last of a sequence together, whatever frame the scene
is on: not cameras, lights, what sits inside them, or an object of a guide
type (a null, a bone, a rivet, a path) that still draws its stock shape.
Only object networks are walked, never what sits inside a simulation, and
the walk is made once per view. A simulation network draws no geometry node
to read, so it is drawn but not framed; an instance object is framed around
its points grown by what it copies. With nothing to frame the view or camera
frames the origin and says so. The viewport is framed for the picture's
shape, not its own, so a picture of another shape is not cut. Without guides
the objects drawn are all of them less those guides, left out by name, so
Houdini still decides what else is shown at each frame. A Scene Viewer
inside a geometry network or showing a stage keeps Houdini's own frame all
and draws every object.

`cop` reads the COP's image layer and writes it as an 8 bit PNG, or saves an
older COP's image with its own writer. `network` and `pane` grab the pane's
own window through Qt, reached from that one pane tab, made the current tab
for the grab and put back after, with pending paints let through first, and
crop it to the pane. Nothing here walks the widget tree.

Everything made for a capture, the render node, a fitted camera and a moved
display flag, is made and taken away with undo turned off, so the artist's
undo history is as it was. Every step that puts something back is tried,
whatever the one before it did; a step that fails makes a capture that
worked `CLEANUP_FAILED`, and goes in the details of one that did not. A
viewport sequence is flipbooked a few frames at a time, so it can be stopped
between them and says how far it got. The job row names each run, where its
files go and which frames are written, from the moment a place is handed
out, so a session that dies part way leaves a record of what it wrote.

A route that finishes without writing a file, or
writes an empty one, counts as a route that did not work, and the next one is
tried, and whatever frames it wrote are taken away. When a route that
applies ran and Houdini stopped it with an error, the answer is
`CAPTURE_FAILED` with that error. A capture stopped on
request keeps the frames it wrote and lists them. Only when every route that
applies fails is the answer
`UI_UNAVAILABLE`, or `CAPTURE_EMPTY` when every one ran and wrote nothing.

This module reads `hou` only through the objects it is handed. Its numbers
and file writing are plain Python, so they are tested without Houdini.
"""

from __future__ import annotations

import math
import os
import struct
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nscr_houdini_mcp.bridge.errors import BridgeError, did_you_mean, hide_paths
from nscr_houdini_mcp.bridge.tools import (
    ToolContext,
    _houdini,
    _is_hou_error,
    _near,
    _quiet,
    output_plan,
)

SOURCES = ("viewport", "node", "network", "cop", "pane")
VIEW_SETS = ("single", "quad", "turntable4")
NAMED_VIEWS = ("persp", "top", "front", "right")
DISPLAYS = ("shaded", "wire", "shaded_wire", "matcap")
TARGETS = ("all", "selection")

# The routes, by the names a reply and the capability probe use.
VIEWPORT = "viewport_flipbook"
VIEWPORT_TAB = "viewport_flipbook_tab"
FLIPBOOK_ROP = "flipbook_rop"
NETWORK_GRAB = "network_grab"
PANE_GRAB = "pane_grab"
COP_LAYER = "cop_layer"
COP2_SAVE = "cop2_save"

# What is made for a capture, and where.
ROP_TYPE = "flipbook"
ROP_PARENT = "/out"
ROP_NAME = "nscr_capture"
CAMERA_TYPE = "cam"
CAMERA_PARENT = "/obj"
CAMERA_NAME = "nscr_capture_cam"

# The display modes, as the viewport and the render node name them.
GL_SHADING = {"shaded": "Smooth", "wire": "Wire", "shaded_wire": "SmoothWire", "matcap": "MatCap"}
ROP_SHADING = {"shaded": "smooth", "wire": "wire", "shaded_wire": "smoothwire", "matcap": "matcap"}
VIEWPORT_TYPES = {"persp": "Perspective", "top": "Top", "front": "Front", "right": "Right"}
# The display sets a display mode is applied to: the objects shown, and the
# one being worked on.
DISPLAY_SETS = ("SceneObject", "DisplayModel")

# A fitted camera for each named view: orbit and elevation in degrees, and
# whether it is orthographic. The orbit turns about +Y from a camera on +Z.
FITTED = {
    "persp": (45.0, 25.0, False),
    "top": (0.0, 90.0, True),
    "front": (0.0, 0.0, True),
    "right": (90.0, 0.0, True),
}
TURNTABLE = (0.0, 90.0, 180.0, 270.0)
DEFAULT_ELEVATION = 25.0
# How much room a fitted camera leaves around the target.
MARGIN = 1.15
# The size a target with no extent is framed at.
MIN_EXTENT = 0.5
# The guides a capture without guides leaves out, and "all" does not frame:
# cameras and lights, by Houdini's own kinds, with whatever sits inside them
# such as a camera rig's handles, and the object types Houdini counts as
# geometry that draw a stock shape, by the type of the shape node they are
# made with. One of those whose display flag is on anything else draws
# geometry of its own and counts as geometry.
GUIDE_KINDS = ("ObjCamera", "ObjLight")
GUIDE_SHAPES = {
    "null": "control",
    "rivet": "control",
    "sticky": "control",
    "bone": "bonelink",
    "fetch": "sphere",
    "blend": "sphere",
    "pathcv": "control",
    "path": "convert",
    "handle": "merge",
    "muscle": "muscle",
}
NOTHING_TO_FRAME = "nothing to frame was found, so the {} frames the origin"
# How many objects framing "all" reads. Each costs a read of its displayed
# geometry's box and transform, about 10 microseconds once cooked, on the
# main thread.
MAX_FRAMED = 10000
TOO_MANY_TO_FRAME = f"only the first {MAX_FRAMED} shown objects were framed"

DEFAULT_RESOLUTION = (1280, 720)
MAX_RESOLUTION = 8192
MAX_FRAMES = 1000
FRAME_TOKEN = "$F4"

NO_FILE = "the route finished and wrote no image"
NETWORK_HINT = "read the network with hou_inspect mode tree; a picture of it needs a user interface"
PANE_HINT = "a pane exists only in a session with a user interface"


class Unavailable(Exception):
    """This route cannot make the picture here. The next one is tried."""


class Stopped(Exception):
    """The call was asked to stop before any frame was written."""


# How many frames one viewport flipbook call writes, so a long sequence can be
# stopped, and reports how far it got, between calls.
FLIPBOOK_PIECE = 8


class Attempt:
    """One route's try: the clean up steps that failed, and a note of each file written.

    Every clean up step is tried, whatever the one before it did, and a step
    that fails is kept rather than swallowed, so a capture never says it went
    well while it left something of the scene or the view changed.
    """

    def __init__(self, wrote: Callable[[list[str]], Any] | None = None) -> None:
        self.failures: list[dict[str, str]] = []
        self._wrote = wrote

    def attempt(self, step: str, action: Callable[[], Any]) -> None:
        try:
            action()
        except Exception as error:  # noqa: BLE001 - every step is tried; each failure is kept
            self.failures.append({"step": step, "error": _reason(error)})

    def wrote(self, files: Sequence[str]) -> None:
        if self._wrote is not None:
            _quiet(lambda: self._wrote(list(files)))


# Section: what was asked for


@dataclass(frozen=True)
class Spec:
    """One capture as asked for, checked against the scene."""

    source: str
    path: str | None
    camera: Any
    frame_target: str | None
    display: str | None
    guides: bool
    resolution: tuple[int, int]
    frame: float | None
    frames: tuple[float, ...] | None
    views: str
    name: str

    @property
    def sequence(self) -> bool:
        return self.frames is not None


def read_spec(arguments: Mapping[str, Any], hou: Any) -> Spec:
    """The arguments as a spec, or `BAD_ARGUMENTS` / `NODE_NOT_FOUND` naming the one at fault."""
    source = str(arguments.get("source") or "viewport")
    if source not in SOURCES:
        raise _bad("source", f"source must be one of {', '.join(SOURCES)}")
    path = arguments.get("path")
    path = None if path in (None, "") else str(path)
    camera = _camera(arguments.get("camera"), hou)
    target = arguments.get("frame_target")
    target = None if target in (None, "") else str(target)
    display = arguments.get("display")
    if display is not None and display not in DISPLAYS:
        raise _bad("display", f"display must be one of {', '.join(DISPLAYS)}")
    views = str(arguments.get("views") or "single")
    if views not in VIEW_SETS:
        raise _bad("views", f"views must be one of {', '.join(VIEW_SETS)}")
    frame = arguments.get("frame")
    if frame is not None:
        frame = _finite(frame, "frame")
    frames = _frames(arguments.get("frames"))
    if frame is not None and frames is not None:
        raise _bad("frames", "send frame or frames, not both")
    spec = Spec(
        source=source,
        path=path,
        camera=camera,
        frame_target=target,
        display=display,
        guides=bool(arguments.get("guides")),
        resolution=_resolution(arguments.get("resolution")),
        frame=frame,
        frames=frames,
        views=views,
        name=str(arguments.get("name") or source),
    )
    _fits_source(spec, hou)
    return spec


def _fits_source(spec: Spec, hou: Any) -> None:
    """Refuse what a source cannot take, and a node that is not there."""
    three_d = spec.source in ("viewport", "node")
    if not three_d:
        for name, given in (
            ("camera", spec.camera),
            ("frame_target", spec.frame_target),
            ("display", spec.display),
            ("frames", spec.frames),
        ):
            if given is not None:
                raise _bad(name, f"{name} is for the viewport and node sources")
        if spec.views != "single":
            raise _bad("views", "views is for the viewport and node sources")
    if spec.sequence and spec.views != "single":
        raise _bad("views", "a sequence is captured from one view")
    if spec.source in ("node", "cop", "pane") and not spec.path:
        what = "the pane tab's name" if spec.source == "pane" else "the node's path"
        raise _bad("path", f"source {spec.source} needs path: {what}")
    if spec.source == "node":
        node = _node_at(hou, spec.path, "path")
        if _object_of(node) is None:
            raise _bad("path", "source node takes an object or a geometry node")
    if spec.source == "cop":
        node = _node_at(hou, spec.path, "path")
        if _category(node) not in ("Cop", "Cop2"):
            raise _bad("path", "source cop takes a COP node", category=_category(node))
    if spec.source == "network" and spec.path:
        _node_at(hou, spec.path, "path")
    if spec.frame_target not in (None, *TARGETS):
        _node_at(hou, spec.frame_target, "frame_target")


def _camera(value: Any, hou: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - {"orbit", "elevation"})
        if unknown or "orbit" not in value:
            raise _bad("camera", "an orbit camera is {orbit: degrees, elevation: degrees}")
        orbit = _finite(value["orbit"], "camera.orbit")
        elevation = _finite(value.get("elevation", DEFAULT_ELEVATION), "camera.elevation")
        if not -90.0 <= elevation <= 90.0:
            raise _bad("camera", "camera.elevation must be from -90 to 90")
        return {"orbit": orbit, "elevation": elevation}
    text = str(value)
    if text in NAMED_VIEWS:
        return text
    if not text.startswith("/"):
        raise _bad(
            "camera",
            f"camera is a camera node's path, one of {', '.join(NAMED_VIEWS)}, or an orbit",
            did_you_mean=did_you_mean(text, NAMED_VIEWS),
        )
    node = _node_at(hou, text, "camera")
    if not _is_camera(node):
        raise _bad("camera", "that node is not a camera", type=_quiet(lambda: node.type().name()))
    return text


def _is_camera(node: Any) -> bool:
    name = _quiet(lambda: node.type().name())
    if name == CAMERA_TYPE:
        return True
    return _quiet(lambda: node.parm("focal")) is not None and (
        _quiet(lambda: node.parm("aperture")) is not None
    )


def _frames(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise _bad("frames", "frames is [start, end, step]")
    start, end, step = (_finite(item, "frames") for item in value)
    if any(float(item) != int(item) for item in (start, end, step)):
        raise _bad("frames", "frames are whole numbers, because each one names a file")
    if step <= 0 or end < start:
        raise _bad("frames", "frames needs step above zero and end at or after start")
    count = int((end - start) // step) + 1
    if count > MAX_FRAMES:
        raise _bad("frames", f"a sequence holds at most {MAX_FRAMES} frames", count=count)
    return tuple(float(start + index * step) for index in range(count))


def _resolution(value: Any) -> tuple[int, int]:
    if value is None:
        return DEFAULT_RESOLUTION
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        or not all(1 <= item <= MAX_RESOLUTION for item in value)
    ):
        raise _bad("resolution", f"resolution is [width, height], each 1 to {MAX_RESOLUTION}")
    return int(value[0]), int(value[1])


def _region(value: Any) -> list[float] | None:
    """The server's crop, carried back as it came. Its shape is checked, not used."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise _bad("region", "region is [x0, y0, x1, y1]")
    return [_finite(item, "region") for item in value]


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _bad(name, f"{name} must be a finite number")
    return float(value)


def _bad(argument: str, message: str, **details: Any) -> BridgeError:
    return BridgeError(
        "BAD_ARGUMENTS",
        message,
        {"argument": argument, **details},
        hint="fix the argument named in the details and call again",
    )


def _node_at(hou: Any, path: Any, argument: str) -> Any:
    node = _quiet(lambda: hou.node(str(path)))
    if node is None:
        raise BridgeError(
            "NODE_NOT_FOUND",
            f"no node at {path}",
            {"argument": argument, "path": path, "did_you_mean": _near(hou, str(path))},
            hint="read the scene and use a path that is there",
        )
    return node


# Section: the tool


def capture_image(arguments: Mapping[str, Any], context: ToolContext) -> dict[str, Any]:
    """Write the picture, or pictures, and say how each was made."""
    hou = _houdini(context)
    spec = read_spec(arguments, hou)
    gui = is_gui(hou, context)
    frames = spec.frames or ((spec.frame if spec.frame is not None else _current_frame(hou)),)
    wanted = views_of(spec)
    shots: list[dict[str, Any]] = []
    warnings: list[str] = []
    unsaved = False
    stopped = False
    written = JobOutputs(context)
    made_by = {"node_path": _owner(spec, hou), "job_id": written.job_id}
    for label, camera in wanted:
        if shots and context.should_stop():
            stopped = True
            break
        name = spec.name if len(wanted) == 1 else f"{spec.name}_{label}"
        plan = output_plan(context, hou, "capture", name, "png", **made_by)
        unsaved = unsaved or bool(plan.unsaved_hip)
        warnings.extend(item for item in plan.warnings if item not in warnings)
        path = sequence_path(plan.path) if spec.sequence else plan.path
        index = written.planned(label, plan, path)
        try:
            shot = shoot(
                hou,
                context,
                spec,
                camera,
                path,
                frames,
                gui,
                lambda files, index=index: written.wrote(index, files),
            )
        except BridgeError as error:
            files = list(getattr(error, "files", ()))
            if files:
                # The picture is there; only the clean up went wrong.
                _record(context, plan, files)
            else:
                _release(context, plan)
            raise
        except BaseException:
            # Nothing was written under this place, so its claim and record go.
            _release(context, plan)
            raise
        if shot["files"]:
            _record(context, plan, shot["files"])
        else:
            _release(context, plan)
        written.wrote(index, shot["files"])
        warnings.extend(item for item in shot.pop("warnings", ()) if item not in warnings)
        if shot.get("stopped_early"):
            stopped = True
            kept = "the frames written before the stop are kept, and listed"
            if shot["files"] and kept not in warnings:
                warnings.append(kept)
        shots.append({"view": label, "run_id": plan.run_id, "template": plan.template, **shot})
    sheet = None
    if len(wanted) > 1 and len(shots) == len(wanted) and not stopped:
        plan = output_plan(context, hou, "capture", f"{spec.name}_sheet", "png", **made_by)
        sheet = {"path": plan.path, "run_id": plan.run_id, "template": plan.template}
    return {
        "source": spec.source,
        "views": shots,
        "view_set": spec.views,
        "sheet": sheet,
        "gui": gui,
        "sequence": spec.sequence,
        "stopped_early": stopped,
        "unsaved_hip": unsaved,
        "warnings": warnings,
        "region": _region(arguments.get("region")),
    }


def _owner(spec: Spec, hou: Any) -> str | None:
    """The node a capture of one node's output is a picture of, as its run records it."""
    if spec.source not in ("node", "cop") or spec.path is None:
        return None
    node = _quiet(lambda: hou.node(spec.path))
    return (_quiet(lambda: node.path()) if node is not None else None) or spec.path


class JobOutputs:
    """What the capture has written so far, kept on its job row as it goes.

    Written as soon as each place is handed out and again after every frame,
    so a job whose session dies part way still says which runs it had, where
    their files go and which frames are there, for whoever reports on it or
    clears it away. The row takes the capture's answer in place of this when
    the call ends. Best effort: a store that cannot be written costs the note,
    never the capture.
    """

    def __init__(self, context: ToolContext) -> None:
        from nscr_houdini_mcp import jobs as job_rules

        self._context = context
        self._job_id = (
            job_rules.job_id_for(context.operation_id)
            if context.operation_id and context.open_store is not None
            else None
        )
        self.runs: list[dict[str, Any]] = []

    @property
    def job_id(self) -> str | None:
        """The job this capture runs as, when it runs as one."""
        return self._job_id

    def planned(self, view: str, plan: Any, path: str) -> int:
        self.runs.append(
            {
                "view": view,
                "run_id": plan.run_id,
                "path": path,
                "sidecar": plan.sidecar,
                "files": [],
            }
        )
        self._write()
        return len(self.runs) - 1

    def wrote(self, index: int, files: Sequence[str]) -> None:
        self.runs[index]["files"] = list(files)
        self._write()

    def _write(self) -> None:
        if self._job_id is None:
            return
        outputs = {
            "capture": {
                "runs": self.runs,
                "frames_done": sum(len(run["files"]) for run in self.runs),
            }
        }

        def write() -> None:
            with self._context.open_store() as store:
                store.update_job(self._job_id, outputs=outputs)

        _quiet(write)


def _record(context: ToolContext, plan: Any, files: Sequence[str]) -> None:
    """Name the files this run wrote in its record and beside its output. Best effort."""
    from nscr_houdini_mcp import outputs

    def write() -> None:
        with context.open_store() as store:
            outputs.record_files(store, plan, files)

    _quiet(write)


def _release(context: ToolContext, plan: Any) -> None:
    from nscr_houdini_mcp import outputs

    def give_back() -> None:
        with context.open_store() as store:
            outputs.release(store, plan)

    _quiet(give_back)


def views_of(spec: Spec) -> list[tuple[str, Any]]:
    """Each view to capture, as a label and the camera it is seen through."""
    if spec.views == "quad":
        return [(view, view) for view in NAMED_VIEWS]
    if spec.views == "turntable4":
        elevation = (
            spec.camera["elevation"] if isinstance(spec.camera, Mapping) else DEFAULT_ELEVATION
        )
        return [
            (f"orbit{int(angle)}", {"orbit": angle, "elevation": elevation}) for angle in TURNTABLE
        ]
    return [("single", spec.camera)]


def sequence_path(path: str) -> str:
    from nscr_houdini_mcp import outputs

    return outputs.sequence_path(path, FRAME_TOKEN)


def frame_files(path: str, frames: Sequence[float], sequence: bool) -> list[str]:
    """The files a route writes for these frames."""
    if not sequence:
        return [path]
    return [path.replace(FRAME_TOKEN, f"{int(round(frame)):04d}") for frame in frames]


def is_gui(hou: Any, context: ToolContext) -> bool:
    return context.kind == "gui" and getattr(hou, "ui", None) is not None


def _current_frame(hou: Any) -> float:
    frame = _quiet(lambda: hou.frame())
    return float(frame) if frame is not None else 1.0


# Section: trying the routes


Route = Callable[..., dict[str, Any]]


def routes_for(source: str, gui: bool) -> list[tuple[str, Route]]:
    """The routes that apply to a source here, in the order they are tried."""
    if source == "viewport":
        found: list[tuple[str, Route]] = []
        if gui:
            found += [(VIEWPORT, viewport_route), (VIEWPORT_TAB, viewport_tab_route)]
        return [*found, (FLIPBOOK_ROP, rop_route)]
    if source == "node":
        return [(FLIPBOOK_ROP, rop_route)]
    if source == "cop":
        return [(COP_LAYER, cop_layer_route), (COP2_SAVE, cop2_route)]
    if source == "network":
        return [(NETWORK_GRAB, network_route)] if gui else []
    return [(PANE_GRAB, pane_route)] if gui else []


def shoot(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Sequence[float],
    gui: bool,
    wrote: Callable[[list[str]], Any] | None = None,
) -> dict[str, Any]:
    """One view, by the first route that writes it.

    A route that fails takes its frames with it. A clean up step that failed
    on any route is reported: as `CLEANUP_FAILED` when a route then wrote the
    picture, and in the details of the error when none did.
    """
    tried: list[dict[str, Any]] = []
    cleanup: list[dict[str, str]] = []
    for route, run in routes_for(spec.source, gui):
        attempt = Attempt(wrote)
        try:
            shot = run(hou, context, spec, camera, path, frames, gui, attempt)
        except Stopped:
            cleanup += attempt.failures
            shot = {"files": [], "frames": [], "camera": None, "native": None}
            shot.update(route=route, stopped_early=True)
            if cleanup:
                raise cleanup_failed(spec, route, cleanup, shot) from None
            return shot
        except Unavailable as reason:
            discard(frame_files(path, frames, spec.sequence))
            cleanup += attempt.failures
            tried.append({"route": route, "reason": str(reason)})
            continue
        except BaseException as error:
            # A route that failed part way leaves nothing behind it.
            discard(frame_files(path, frames, spec.sequence))
            cleanup += attempt.failures
            if isinstance(error, BridgeError):
                if cleanup:
                    error.details["cleanup"] = cleanup
                raise
            if not _is_hou_error(error):
                raise
            tried.append({"route": route, "reason": _reason(error), "failed": True})
            continue
        cleanup += attempt.failures
        written = [item for item in shot["files"] if _written(item)]
        if not written or len(written) < len(shot["files"]):
            discard(frame_files(path, frames, spec.sequence))
            tried.append({"route": route, "reason": NO_FILE})
            continue
        shot["route"] = route
        if tried:
            shot["tried"] = tried
        if cleanup:
            raise cleanup_failed(spec, route, cleanup, shot)
        return shot
    details: dict[str, Any] = {"source": spec.source, "tried": tried}
    if cleanup:
        details["cleanup"] = cleanup
    if tried and all(item["reason"] == NO_FILE for item in tried):
        raise BridgeError(
            "CAPTURE_EMPTY",
            "every route ran and none wrote an image",
            details,
            hint="check the camera and the node shown, then capture again",
        )
    failed = [item for item in tried if item.get("failed")]
    if failed:
        # A route that applies here ran and Houdini said why it stopped.
        raise BridgeError(
            "CAPTURE_FAILED",
            "the capture ran and Houdini stopped it with an error",
            {**details, "error": failed[-1]["reason"]},
            hint="check the camera and the node shown, and the error in the details",
        )
    hint = {"network": NETWORK_HINT, "pane": PANE_HINT}.get(spec.source)
    raise BridgeError(
        "UI_UNAVAILABLE",
        f"no route could capture the {spec.source} in this session",
        {**details, "gui": gui},
        hint=hint or "use a source this session can show, or a session with a user interface",
    )


def cleanup_failed(
    spec: Spec, route: str, cleanup: list[dict[str, str]], shot: Mapping[str, Any]
) -> BridgeError:
    """The picture was made and something the capture changed was not put back."""
    error = BridgeError(
        "CLEANUP_FAILED",
        "the capture was written, and part of what it changed could not be put back",
        {
            "source": spec.source,
            "route": route,
            "cleanup": cleanup,
            "frames": list(shot.get("frames") or ()),
        },
        hint="look at the scene and the view for the steps named in the details",
    )
    # The files written, for the run record; never sent in the error itself.
    error.files = list(shot.get("files") or ())  # type: ignore[attr-defined]
    return error


def routes(hou: Any) -> list[str]:
    """What the capability probe reports: the routes this session has."""
    found: list[str] = []
    ui = getattr(hou, "ui", None) is not None and bool(_quiet(lambda: hou.isUIAvailable()))
    if ui:
        found += [VIEWPORT, VIEWPORT_TAB]
    category = _quiet(lambda: hou.ropNodeTypeCategory())
    if category is not None and _quiet(lambda: hou.nodeType(category, ROP_TYPE)) is not None:
        found.append(FLIPBOOK_ROP)
    if hasattr(hou, "ImageLayer"):
        found.append(COP_LAYER)
    categories = _quiet(lambda: hou.nodeTypeCategories()) or {}
    if "Cop2" in categories:
        found.append(COP2_SAVE)
    if ui:
        found += [NETWORK_GRAB, PANE_GRAB]
    return found


def discard(files: Iterable[str]) -> None:
    """Take away the files of a route that did not finish. A missing one is fine."""
    for item in files:
        try:
            os.remove(item)
        except OSError:
            pass


def _written(path: str) -> bool:
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


def _reason(error: BaseException) -> str:
    text = hide_paths(str(error)).strip().splitlines()
    said = text[0][:200] if text else ""
    return f"{type(error).__name__}: {said}" if said else type(error).__name__


# Section: the viewport routes


def scene_viewers(hou: Any) -> list[Any]:
    kind = hou.paneTabType.SceneViewer
    return [
        tab
        for tab in (_quiet(lambda: hou.ui.paneTabs()) or ())
        if _quiet(lambda tab=tab: tab.type()) == kind
    ]


def viewport_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Any,
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """The Scene Viewer that is showing."""
    showing = [tab for tab in scene_viewers(hou) if _quiet(lambda tab=tab: tab.isCurrentTab())]
    if not showing:
        raise Unavailable("no Scene Viewer is showing")
    return flipbook_viewer(hou, context, spec, camera, showing[0], path, frames, attempt)


def viewport_tab_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Any,
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """A Scene Viewer made the current tab of its pane for the capture, then put back."""
    tabs = scene_viewers(hou)
    if not tabs:
        raise Unavailable("this desktop has no Scene Viewer")
    if any(_quiet(lambda tab=tab: tab.isCurrentTab()) for tab in tabs):
        raise Unavailable("a Scene Viewer is already showing")
    tab = tabs[0]
    pane = _quiet(lambda: tab.pane())
    previous = _quiet(lambda: pane.currentTab()) if pane is not None else None
    tab.setIsCurrentTab()
    try:
        return flipbook_viewer(hou, context, spec, camera, tab, path, frames, attempt)
    finally:
        if previous is not None and previous is not tab:
            attempt.attempt("make the tab that was showing current again", previous.setIsCurrentTab)


def flipbook_viewer(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    viewer: Any,
    path: str,
    frames: Sequence[float],
    attempt: Attempt,
) -> dict[str, Any]:
    """Flipbook one viewer's current viewport, with its view put back after.

    A sequence goes a few frames per flipbook, looking between them at
    whether the call should stop and saying how far it has got, since one
    flipbook cannot be stopped from here once it has begun.
    """
    viewport = viewer.curViewport()
    if viewport is None:
        raise Unavailable("the Scene Viewer has no viewport")
    saved = ViewState.save(hou, viewport)
    done: list[float] = []
    stopped = False
    objects = shows_objects(viewer)
    scene = scene_objects(hou) if objects else None
    try:
        described, warnings = apply_view(hou, spec, camera, viewport, frames, objects, scene)
        settings, unset = flipbook_settings(hou, viewer, spec, path, frames, objects, scene)
        if unset:
            warnings.append(
                "these flipbook settings could not be set and keep the artist's: "
                + ", ".join(unset)
            )
        for start in range(0, len(frames), FLIPBOOK_PIECE):
            if context.should_stop():
                if not done:
                    raise Stopped
                stopped = True
                break
            piece = list(frames[start : start + FLIPBOOK_PIECE])
            settings.frameRange((piece[0], piece[-1]))
            viewer.flipbook(viewport=viewport, settings=settings, open_dialog=False)
            done += piece
            attempt.wrote(frame_files(path, done, spec.sequence))
            if len(frames) > 1 and context.progress is not None:
                context.progress(
                    {"done": len(done), "total": len(frames), "message": f"frame {piece[-1]:g}"}
                )
    finally:
        saved.restore(hou, viewport, attempt)
    shot = {
        "files": frame_files(path, done, spec.sequence),
        "frames": done,
        "camera": described,
        "native": list(spec.resolution),
        "warnings": warnings,
    }
    if stopped:
        shot["stopped_early"] = True
    return shot


# A setting whose value this build does not name.
_NO_VALUE = object()


def flipbook_settings(
    hou: Any,
    viewer: Any,
    spec: Spec,
    path: str,
    frames: Sequence[float],
    objects: bool = True,
    scene: SceneObjects | None = None,
) -> tuple[Any, list[str]]:
    """A copy of the viewer's flipbook settings with every one this capture needs set.

    The copy starts from the artist's dialog, so nothing is left to it: an
    object filter, a contact sheet, motion blur, depth of field, a background
    image or a colour transform the artist left on would all come along.
    Without guides, on a viewer showing objects, the guide objects are left
    out by name, as the beauty pass alone still draws a null's cross. Returns
    the settings and the names that could not be set.
    """
    settings = viewer.flipbookSettings().stash()
    kinds = getattr(hou, "flipbookObjectType", None)
    smoothing = getattr(hou, "flipbookAntialias", None)
    wanted = (
        ("outputToMPlay", False),
        ("output", path),
        ("frameRange", (frames[0], frames[-1])),
        ("frameIncrement", _step(frames)),
        ("useResolution", True),
        ("resolution", spec.resolution),
        ("beautyPassOnly", not spec.guides),
        ("visibleObjects", guide_mask(hou, spec, scene) if objects else "*"),
        ("visibleTypes", getattr(kinds, "Visible", _NO_VALUE)),
        ("useSheetSize", False),
        ("useMotionBlur", False),
        ("useDepthOfField", False),
        ("leaveFrameAtEnd", False),
        ("appendFramesToCurrent", False),
        ("backgroundImage", ""),
        ("overrideGamma", False),
        ("overrideLUT", False),
        ("initializeSimulations", False),
        ("renderAllViewports", False),
        ("scopeChannelKeyframesOnly", False),
        ("audioFilename", ""),
        ("outputZoom", 100),
        ("cropOutMaskOverlay", True),
        ("antialias", getattr(smoothing, "UseViewportSetting", _NO_VALUE)),
        ("setUseFrameTimeLimit", False),
        ("setUseFrameProgressLimit", False),
    )
    unset: list[str] = []
    for name, value in wanted:
        setter = getattr(settings, name, None)
        if setter is None or value is _NO_VALUE:
            unset.append(name)
            continue
        try:
            setter(value)
        except Exception as error:  # noqa: BLE001 - a setting this build refuses is named
            if not _is_hou_error(error) and not isinstance(error, (TypeError, ValueError)):
                raise
            unset.append(name)
    return settings, unset


@dataclass
class ViewState:
    """What a capture may change about a viewport, to be put back exactly."""

    kind: Any
    camera: Any
    default: Any
    shading: dict[str, Any] = field(default_factory=dict)
    # Put back on its own: a perspective view does not take it back with the
    # rest of its camera.
    ortho_width: Any = None

    @classmethod
    def save(cls, hou: Any, viewport: Any) -> ViewState:
        shading: dict[str, Any] = {}
        settings = _quiet(lambda: viewport.settings())
        for name in DISPLAY_SETS:
            kind = getattr(hou.displaySetType, name, None)
            shown = _quiet(lambda kind=kind: settings.displaySet(kind)) if kind else None
            if shown is not None:
                shading[name] = _quiet(lambda shown=shown: shown.shadedMode())
        return cls(
            kind=_quiet(lambda: viewport.type()),
            camera=_quiet(lambda: viewport.camera()),
            default=viewport.defaultCamera().stash(),
            shading=shading,
            ortho_width=_quiet(lambda: viewport.defaultCamera().orthoWidth()),
        )

    def restore(self, hou: Any, viewport: Any, attempt: Attempt) -> None:
        """Put each thing back on its own, so one that fails leaves the rest done."""
        if self.kind is not None:
            attempt.attempt("put the view type back", lambda: viewport.changeType(self.kind))
        if self.camera is not None:
            attempt.attempt(
                "look through the camera again", lambda: viewport.setCamera(self.camera)
            )
        else:
            attempt.attempt("look through the viewport's own camera", viewport.useDefaultCamera)
        attempt.attempt(
            "put the viewport's camera back", lambda: viewport.setDefaultCamera(self.default)
        )
        if self.ortho_width is not None:
            attempt.attempt(
                "put the viewport's ortho width back",
                lambda: viewport.defaultCamera().setOrthoWidth(self.ortho_width),
            )
        for name, mode in self.shading.items():
            kind = getattr(hou.displaySetType, name, None)
            if kind is None or mode is None:
                continue
            attempt.attempt(
                f"put the {name} shading back",
                lambda kind=kind, mode=mode: (
                    viewport.settings().displaySet(kind).setShadedMode(mode)
                ),
            )


def shows_objects(viewer: Any) -> bool:
    """Whether a Scene Viewer is at object level, where "all" and the guide mask apply.

    One inside a geometry network, or showing a stage, keeps Houdini's own
    frame all and draws every object, as it did before either was ours.
    """
    network = _quiet(lambda: viewer.pwd())
    if network is None:
        return True
    return _quiet(lambda: network.childTypeCategory().name()) == "Object"


def apply_view(
    hou: Any,
    spec: Spec,
    camera: Any,
    viewport: Any,
    frames: Sequence[float] = (),
    objects: bool = True,
    scene: SceneObjects | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Look through the camera asked for, shade and frame, on a viewport that is put back."""
    warnings: list[str] = []
    described: dict[str, Any] = {"kind": "viewport"}
    through = None
    if isinstance(camera, str) and camera.startswith("/"):
        through = hou.node(camera)
        viewport.setCamera(through)
        described = {"kind": "node", "path": camera}
    elif isinstance(camera, str):
        if _quiet(lambda: viewport.camera()) is not None:
            viewport.useDefaultCamera()
        viewport.changeType(getattr(hou.geometryViewportType, VIEWPORT_TYPES[camera]))
        described = {"kind": "view", "view": camera}
    elif isinstance(camera, Mapping):
        if _quiet(lambda: viewport.camera()) is not None:
            viewport.useDefaultCamera()
        viewport.changeType(hou.geometryViewportType.Perspective)
        view = viewport.defaultCamera()
        rotation = hou.hmath.buildRotate((-camera["elevation"], camera["orbit"], 0.0))
        view.setRotation(rotation.extractRotationMatrix3())
        viewport.setDefaultCamera(view)
        described = {"kind": "orbit", **camera}
    if spec.display is not None:
        mode = getattr(hou.glShadingType, GL_SHADING[spec.display])
        settings = viewport.settings()
        for name in DISPLAY_SETS:
            settings.displaySet(getattr(hou.displaySetType, name)).setShadedMode(mode)
    target = spec.frame_target
    if target is None and camera is not None and through is None:
        # A view turned for the capture is framed on everything unless told.
        target = "all"
    if target is not None and camera is None and _quiet(lambda: viewport.camera()) is not None:
        # Framing while looking through a camera node, with the camera locked
        # to the view, would move that camera. The viewport's own camera is
        # framed instead, and the camera node is looked through again after.
        viewport.useDefaultCamera()
        described["left_camera"] = True
    if target is not None:
        if through is not None:
            warnings.append("a camera node's view is not moved, so frame_target was not applied")
        elif target == "all" and not objects:
            viewport.frameAll()
        elif target == "all":
            # Houdini's own frame all counts cameras, lights and nulls, so a
            # shown camera away from the subject pulls the view off it.
            bounds = bounds_of_all(hou, frames, warnings, "view", scene)
            frame_box(hou, viewport, bounds, spec.resolution)
        elif target == "selection":
            viewport.frameSelected()
        else:
            bounds = world_bounds(hou, [hou.node(target)], frames)
            if bounds is None:
                warnings.append("the target has no geometry to frame")
            else:
                frame_box(hou, viewport, bounds, spec.resolution)
        described["target"] = target
    return described, warnings


def frame_box(
    hou: Any,
    viewport: Any,
    bounds: tuple[Sequence[float], Sequence[float]],
    resolution: tuple[int, int],
) -> None:
    """Frame a box so it stays whole in a picture of another shape than the viewport.

    The viewport frames for its own shape, and a flipbook of another shape
    is cut from that, so the box is grown about its middle by how far apart
    the two shapes are, whichever is the wider.
    """
    size = _quiet(lambda: viewport.size())
    grow = 1.0
    if size is not None and len(size) == 4 and size[2] > 0 and size[3] > 0:
        shown = size[2] / size[3]
        asked = resolution[0] / resolution[1]
        grow = max(shown / asked, asked / shown)
    low, high = bounds
    middle = [(a + b) / 2.0 for a, b in zip(low, high, strict=True)]
    low = [m + (a - m) * grow for a, m in zip(low, middle, strict=True)]
    high = [m + (b - m) * grow for b, m in zip(high, middle, strict=True)]
    viewport.frameBoundingBox(hou.BoundingBox(*low, *high))


# Section: the flipbook render node


def rop_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Sequence[float],
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """Render through a flipbook render node made for the capture and taken away after."""
    parent = _quiet(lambda: hou.node(ROP_PARENT))
    category = _quiet(lambda: hou.ropNodeTypeCategory())
    if parent is None or category is None:
        raise Unavailable(f"this scene has no {ROP_PARENT} network")
    if _quiet(lambda: hou.nodeType(category, ROP_TYPE)) is None:
        raise Unavailable(f"this build has no {ROP_TYPE} render node")
    isolated = hou.node(spec.path) if spec.source == "node" else None
    made: list[Any] = []
    put_back: list[tuple[str, Callable[[], Any]]] = []
    done: list[float] = []
    stopped = False
    scale: float | None = 1.0
    with hou.undos.disabler():
        try:
            rop = parent.createNode(ROP_TYPE, _free_name(parent, ROP_NAME))
            made.append(rop)
            objects = None
            if isolated is not None:
                objects, put_back = isolate(isolated)
            targets = [isolated] if isolated is not None else _targets(hou, spec.frame_target)
            if gui or needs_scale(hou):
                # A pool worker draws offscreen at one pixel to a point. A
                # session with a user interface, or a hython started some
                # other way, draws at its screen's ratio.
                scale = drawing_scale(hou, parent, os.path.dirname(path), attempt)
            # Read before the capture's own camera is made, so it is not named.
            scene = scene_objects(hou) if objects is None else None
            mask = guide_mask(hou, spec, scene) if scene is not None else None
            camera_path, described, warnings = rop_camera(
                hou, spec, camera, targets, made, scale or 1.0, frames, scene
            )
            if scale is None:
                warnings.append("the render node's drawing scale could not be read")
            _set(rop, "camera", camera_path)
            _set(rop, "picture", path)
            _set(rop, "mkpath", 1)
            _set(rop, "tres", 1)
            _set(rop, "res", spec.resolution)
            _set(rop, "trange", "normal")
            _set(rop, "f", (frames[0], frames[-1], _step(frames)))
            _set(rop, "sopsource", "display")
            if spec.display is not None:
                _set(rop, "shadingmode", ROP_SHADING[spec.display])
            if objects is not None:
                _set(rop, "vobjects", " ".join(objects))
                _set(rop, "forceobjects", " ".join(objects))
            if mask is not None:
                _set(rop, "vobjects", mask)
            for index, frame in enumerate(frames):
                if context.should_stop():
                    if not done:
                        raise Stopped
                    stopped = True
                    break
                rop.render(frame_range=(frame, frame))
                done.append(frame)
                attempt.wrote(frame_files(path, done, spec.sequence))
                if len(frames) > 1 and context.progress is not None:
                    context.progress(
                        {"done": index + 1, "total": len(frames), "message": f"frame {frame:g}"}
                    )
        finally:
            for label, step in reversed(put_back):
                attempt.attempt(label, step)
            for node in reversed(made):
                attempt.attempt(f"take away {_quiet(lambda node=node: node.path())}", node.destroy)
    shot: dict[str, Any] = {
        "files": frame_files(path, done, spec.sequence),
        "frames": done,
        "camera": described,
        "native": list(spec.resolution),
        "warnings": warnings,
    }
    if gui or scale is None:
        # What the artist's own view frames is not known to this route, and
        # a drawing scale that could not be read leaves the framing open too.
        shot["framing_unverified"] = True
    if stopped:
        shot["stopped_early"] = True
    return shot


# The drawing scale, worked out for the `hou` and the screen ratio it was read at.
_found: list[Any] = []

# The calibration: a unit box seen by an orthographic camera two units wide,
# in a square picture. Drawn right, the box covers the middle half each way.
PROBE_SIZE = 64
PROBE_NAME = "nscr_capture_probe"
# How far the edges may disagree, as a share of the box, before the reading is
# thrown away.
PROBE_SLACK = 0.1


# Where Qt is told which screen plugin to use.
QT_PLATFORM_ENV_VAR = "QT_QPA_PLATFORM"
OFFSCREEN = "offscreen"


def screen_ratio(hou: Any) -> float | None:
    """The screen's device pixel ratio, or nothing when nothing will say.

    With a user interface, the main window's. Without one, Qt's primary
    screen, when this process has a Qt application to ask.
    """
    if getattr(hou, "ui", None) is not None:
        window = _quiet(lambda: hou.ui.mainQtWindow())
        ratio = _quiet(lambda: window.devicePixelRatioF()) if window is not None else None
        if ratio:
            return float(ratio)
    return primary_screen_ratio()


def primary_screen_ratio() -> float | None:
    for binding in ("PySide6", "PySide2"):
        try:
            gui = __import__(f"{binding}.QtGui", fromlist=["QGuiApplication"])
        except ImportError:
            continue
        application = _quiet(lambda gui=gui: gui.QGuiApplication.instance())
        screen = (
            _quiet(lambda application=application: application.primaryScreen())
            if application is not None
            else None
        )
        ratio = (
            _quiet(lambda screen=screen: screen.devicePixelRatio()) if screen is not None else None
        )
        return float(ratio) if ratio else None
    return None


def needs_scale(hou: Any) -> bool:
    """Whether a session without a user interface has to read its drawing scale.

    Not on the offscreen screen plugin, which draws one pixel to a point, and
    not when the primary screen says it is one to one. Anything else, a
    screen plugin named in the shell or a ratio nobody will say, is read.
    """
    if os.environ.get(QT_PLATFORM_ENV_VAR, "").strip().lower() == OFFSCREEN:
        return False
    return screen_ratio(hou) != 1.0


def drawing_scale(hou: Any, parent: Any, folder: str, attempt: Attempt) -> float | None:
    """How much larger than asked the render node draws, in a session with a user interface.

    On a screen whose pixels are denser than its points, Qt can make a render
    node draw the frame that many times larger and keep only the corner it was
    asked the size of. A small render of a known box says by how much. It is
    read again whenever the main window's pixel ratio has changed, which is
    what moving Houdini to another screen does. Nothing when the picture could
    not be read or its edges disagree, which leaves the framing unverified.
    """
    ratio = screen_ratio(hou)
    platform = os.environ.get(QT_PLATFORM_ENV_VAR)
    if _found and _found[0] is hou and _found[1:3] == [ratio, platform]:
        return _found[3]
    obj = hou.node(CAMERA_PARENT)
    # A fixed name: the capture's own path can hold a frame token.
    probe_file = os.path.join(folder, f"{PROBE_NAME}_{os.getpid()}.probe.png")
    made: list[Any] = []
    scale: float | None = None
    try:
        holder = obj.createNode("geo", _free_name(obj, PROBE_NAME))
        made.append(holder)
        holder.createNode("box").setDisplayFlag(True)
        camera = obj.createNode(CAMERA_TYPE, _free_name(obj, CAMERA_NAME))
        made.append(camera)
        _set_camera(camera, "t", (0.0, 0.0, 5.0))
        _set_camera(camera, "resx", PROBE_SIZE)
        _set_camera(camera, "resy", PROBE_SIZE)
        _set_camera(camera, "projection", "ortho")
        _set_camera(camera, "orthowidth", 2.0)
        rop = parent.createNode(ROP_TYPE, _free_name(parent, ROP_NAME))
        made.append(rop)
        _set(rop, "camera", camera.path())
        _set(rop, "picture", probe_file)
        _set(rop, "tres", 1)
        _set(rop, "res", (PROBE_SIZE, PROBE_SIZE))
        _set(rop, "trange", "normal")
        _set(rop, "f", (1.0, 1.0, 1.0))
        _set(rop, "sopsource", "display")
        _set(rop, "vobjects", holder.path())
        _set(rop, "forceobjects", holder.path())
        rop.render(frame_range=(1.0, 1.0))
        scale = scale_from_box(alpha_box(probe_file), PROBE_SIZE)
    except (Unavailable, OSError, ValueError, zlib.error):
        scale = None
    except Exception as error:  # noqa: BLE001 - a Houdini refusal leaves the scale unknown
        if not _is_hou_error(error):
            raise
        scale = None
    finally:
        for node in reversed(made):
            attempt.attempt(f"take away {_quiet(lambda node=node: node.path())}", node.destroy)
        discard([probe_file])
    if scale is not None:
        _found[:] = [hou, ratio, platform, scale]
    return scale


def scale_from_box(box: tuple[int, int, int, int] | None, size: int) -> float | None:
    """The drawing scale from where the calibration box landed, or nothing.

    Drawn at scale `s`, the box's left edge is at `s` quarters of the width
    and its bottom edge the same distance up from the bottom, since the corner
    kept is the bottom left. Its width is half the frame times `s` when its
    right edge is inside the picture. All three have to agree.
    """
    if box is None:
        return None
    left, _, right, bottom = box
    quarter = size / 4.0
    from_left = left / quarter
    from_bottom = (size - bottom) / quarter
    readings = [from_left, from_bottom]
    if right < size:
        readings.append((right - left) / (2.0 * quarter))
    if from_left <= 0 or max(readings) - min(readings) > PROBE_SLACK * 4:
        return None
    middle = sum(readings) / len(readings)
    # To the nearest quarter, halves rounding up rather than to even.
    return math.floor(middle * 4.0 + 0.5) / 4.0


def alpha_box(path: str) -> tuple[int, int, int, int] | None:
    """The box around every pixel with any alpha in a PNG, read without an image library.

    Only what a render node writes is read: 8 or 16 bit, grey or colour with
    alpha, not interlaced. Anything else is `ValueError`.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("not a PNG")
    at = 8
    header = b""
    packed = bytearray()
    while at < len(data):
        (length,) = struct.unpack(">I", data[at : at + 4])
        tag = data[at + 4 : at + 8]
        body = data[at + 8 : at + 8 + length]
        at += 12 + length
        if tag == b"IHDR":
            header = body
        elif tag == b"IDAT":
            packed += body
        elif tag == b"IEND":
            break
    width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", header)
    channels = {4: 2, 6: 4}.get(colour)
    if channels is None or depth not in (8, 16) or interlace:
        raise ValueError("a PNG layout this does not read")
    step = channels * depth // 8
    stride = width * step
    raw = zlib.decompress(bytes(packed))
    rows: list[bytearray] = []
    previous = bytearray(stride)
    for index in range(height):
        start = index * (stride + 1)
        kind = raw[start]
        row = bytearray(raw[start + 1 : start + 1 + stride])
        _unfilter(kind, row, previous, step)
        rows.append(row)
        previous = row
    left, top, right, bottom = width, height, -1, -1
    alpha = step - depth // 8
    for y, row in enumerate(rows):
        for x in range(width):
            if row[x * step + alpha]:
                left, right = min(left, x), max(right, x)
                top, bottom = min(top, y), max(bottom, y)
    if right < 0:
        return None
    return left, top, right + 1, bottom + 1


def _unfilter(kind: int, row: bytearray, above: bytearray, step: int) -> None:
    for index in range(len(row)):
        a = row[index - step] if index >= step else 0
        b = above[index]
        c = above[index - step] if index >= step else 0
        if kind == 1:
            add = a
        elif kind == 2:
            add = b
        elif kind == 3:
            add = (a + b) // 2
        elif kind == 4:
            guess = a + b - c
            pa, pb, pc = abs(guess - a), abs(guess - b), abs(guess - c)
            add = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
        elif kind == 0:
            add = 0
        else:
            raise ValueError("an unknown PNG row filter")
        row[index] = (row[index] + add) & 0xFF


def _step(frames: Sequence[float]) -> float:
    return frames[1] - frames[0] if len(frames) > 1 else 1.0


def _set(node: Any, name: str, value: Any) -> None:
    """Set one parameter the node type has, read off the node rather than assumed."""
    if isinstance(value, (tuple, list)):
        target = _quiet(lambda: node.parmTuple(name))
    else:
        target = _quiet(lambda: node.parm(name))
    if target is None:
        kind = _quiet(lambda: node.type().name())
        raise Unavailable(f"the {kind} node has no parameter {name}")
    target.set(value)


def _free_name(parent: Any, base: str) -> str:
    taken = {
        _quiet(lambda child=child: child.name())
        for child in (_quiet(lambda: parent.children()) or ())
    }
    if base not in taken:
        return base
    index = 1
    while f"{base}{index}" in taken:
        index += 1
    return f"{base}{index}"


def isolate(node: Any) -> tuple[list[str], list[tuple[str, Callable[[], Any]]]]:
    """Show one node alone: its object, and the node carrying the display flag.

    Returns the objects the render draws and the steps that put the flags
    back, which run after the capture whatever happened: the node that had
    the flag gets it again, and when none had it the node gives it up.
    """
    owner = _object_of(node)
    put_back: list[tuple[str, Callable[[], Any]]] = []
    if owner is not node:
        previous = _quiet(lambda: owner.displayNode())
        if previous is None or _quiet(lambda: previous.path()) != _quiet(lambda: node.path()):
            node.setDisplayFlag(True)
            if previous is not None:
                put_back.append(
                    (
                        f"give the display flag back to {_quiet(lambda: previous.path())}",
                        lambda: previous.setDisplayFlag(True),
                    )
                )
            else:
                put_back.append(
                    (
                        f"take the display flag off {_quiet(lambda: node.path())}",
                        lambda: node.setDisplayFlag(False),
                    )
                )
    return [owner.path()], put_back


def _targets(hou: Any, target: str | None) -> list[Any] | None:
    if target in (None, "all"):
        return None
    if target == "selection":
        return list(_quiet(lambda: hou.selectedNodes()) or ())
    return [hou.node(target)]


def rop_camera(
    hou: Any,
    spec: Spec,
    camera: Any,
    targets: list[Any] | None,
    made: list[Any],
    scale: float = 1.0,
    frames: Sequence[float] = (),
    scene: SceneObjects | None = None,
) -> tuple[str, dict[str, Any], list[str]]:
    """The camera the render node looks through, always one made for the capture.

    A named camera is followed rather than used: a camera made beside it takes
    it as its parent, with no move of its own, and reads its lens and window
    through channel references, so an animated camera is followed frame by
    frame and nothing on it is ever written. Otherwise a camera is fitted to
    the target. On a render node that draws larger than asked, the made
    camera's window is widened to make up for it.
    """
    warnings: list[str] = []
    if isinstance(camera, str) and camera.startswith("/"):
        if spec.frame_target is not None:
            warnings.append("a camera node's view is not moved, so frame_target was not applied")
        named = hou.node(camera)
        follower = follow_camera(named, made, scale)
        return follower.path(), {"kind": "node", "path": camera}, warnings
    if isinstance(camera, Mapping):
        orbit, elevation, ortho = camera["orbit"], camera["elevation"], False
        view = "orbit"
    else:
        view = camera or "persp"
        orbit, elevation, ortho = FITTED[view]
    if targets is None:
        bounds = bounds_of_all(hou, frames, warnings, "camera", scene)
    else:
        found = world_bounds(hou, targets, frames)
        if found is None:
            warnings.append(NOTHING_TO_FRAME.format("camera"))
        bounds = found or ((-MIN_EXTENT,) * 3, (MIN_EXTENT,) * 3)
    parent = hou.node(CAMERA_PARENT)
    fitted = parent.createNode(CAMERA_TYPE, _free_name(parent, CAMERA_NAME))
    made.append(fitted)
    width, height = spec.resolution
    focal = _quiet(lambda: float(fitted.parm("focal").eval())) or 50.0
    aperture = _quiet(lambda: float(fitted.parm("aperture").eval())) or 41.4214
    fit = fit_camera(
        bounds,
        orbit=orbit,
        elevation=elevation,
        ortho=ortho,
        aspect=width / height,
        focal=focal,
        aperture=aperture,
    )
    _set_camera(fitted, "t", fit["t"])
    _set_camera(fitted, "r", fit["r"])
    _set_camera(fitted, "resx", width)
    _set_camera(fitted, "resy", height)
    _set_camera(fitted, "near", fit["near"])
    _set_camera(fitted, "far", fit["far"])
    if ortho:
        _set_camera(fitted, "projection", "ortho")
        _set_camera(fitted, "orthowidth", fit["orthowidth"])
    if scale != 1.0:
        for axis in ("x", "y"):
            _set_camera(fitted, f"win{axis}", (scale - 1.0) / 2.0)
            _set_camera(fitted, f"winsize{axis}", scale)
    described = {
        "kind": "fitted",
        "view": view,
        "orbit": orbit,
        "elevation": elevation,
        "projection": "ortho" if ortho else "perspective",
        "target": spec.path if spec.source == "node" else (spec.frame_target or "all"),
    }
    if scale != 1.0:
        described["window_scaled"] = scale
    return fitted.path(), described, warnings


# What a camera made to follow a named one reads from it, by channel reference.
FOLLOWED = (
    "focal",
    "aperture",
    "aspect",
    "orthowidth",
    "near",
    "far",
    "resx",
    "resy",
    "winx",
    "winy",
    "winsizex",
    "winsizey",
)
# Menus, copied by their value: a reference would read the item's number.
COPIED = ("projection", "focalunits")


def follow_camera(named: Any, made: list[Any], scale: float) -> Any:
    """A camera beside a named one that sees what it sees and writes nothing on it."""
    network = named.parent()
    follower = network.createNode(CAMERA_TYPE, _free_name(network, CAMERA_NAME))
    made.append(follower)
    follower.setInput(0, named)
    source = f"../{named.name()}"
    for name in FOLLOWED:
        if _quiet(lambda name=name: named.parm(name)) is None:
            continue
        mine = _quiet(lambda name=name: follower.parm(name))
        if mine is None:
            continue
        expression = f'ch("{source}/{name}")'
        if scale != 1.0 and name in ("winx", "winy"):
            axis = name[-1]
            expression += f' + ch("{source}/winsize{axis}") * {(scale - 1.0) / 2.0!r}'
        elif scale != 1.0 and name in ("winsizex", "winsizey"):
            expression += f" * {scale!r}"
        mine.setExpression(expression)
    for name in COPIED:
        theirs = _quiet(lambda name=name: named.parm(name))
        mine = _quiet(lambda name=name: follower.parm(name))
        if theirs is not None and mine is not None:
            mine.set(theirs.evalAsString())
    return follower


def _set_camera(node: Any, name: str, value: Any) -> None:
    if isinstance(value, (tuple, list)):
        node.parmTuple(name).set(tuple(value))
    else:
        node.parm(name).set(value)


def fit_camera(
    bounds: tuple[Sequence[float], Sequence[float]],
    *,
    orbit: float,
    elevation: float,
    ortho: bool,
    aspect: float,
    focal: float,
    aperture: float,
    margin: float = MARGIN,
) -> dict[str, Any]:
    """Where a camera goes to frame a box from an orbit and an elevation.

    The camera looks down its own -Z, turned first about X by the elevation
    and then about Y by the orbit, which is Houdini's default rotation order.
    A perspective camera stands back until the box's extent across and up
    fits the lens from the box's nearest depth; an orthographic one is as
    wide as the box across, or as tall scaled by the frame's aspect.
    """
    low, high = bounds
    center = [(a + b) / 2.0 for a, b in zip(low, high, strict=True)]
    turn, tilt = math.radians(orbit), math.radians(elevation)
    back = (math.sin(turn) * math.cos(tilt), math.sin(tilt), math.cos(turn) * math.cos(tilt))
    across = (math.cos(turn), 0.0, -math.sin(turn))
    up = (-math.sin(tilt) * math.sin(turn), math.cos(tilt), -math.sin(tilt) * math.cos(turn))
    half_across = half_up = depth = 0.0
    for corner in _corners(low, high):
        offset = [value - middle for value, middle in zip(corner, center, strict=True)]
        half_across = max(half_across, abs(_dot(offset, across)))
        half_up = max(half_up, abs(_dot(offset, up)))
        depth = max(depth, abs(_dot(offset, back)))
    half_across = max(half_across, MIN_EXTENT / 2.0)
    half_up = max(half_up, MIN_EXTENT / 2.0)
    fit: dict[str, Any] = {"r": (-elevation, orbit, 0.0), "orthowidth": None}
    if ortho:
        width = max(2.0 * half_across, 2.0 * half_up * aspect) * margin
        distance = depth + width + 1.0
        fit["orthowidth"] = width
    else:
        tan_across = (aperture / 2.0) / focal
        tan_up = tan_across / aspect
        distance = max(half_across / tan_across, half_up / tan_up) * margin + depth
    fit["t"] = tuple(middle + axis * distance for middle, axis in zip(center, back, strict=True))
    fit["near"] = max(1e-3, (distance - depth) * 0.01)
    fit["far"] = max(10000.0, (distance + depth) * 4.0)
    fit["distance"] = distance
    return fit


def _corners(low: Sequence[float], high: Sequence[float]) -> Iterable[tuple[float, float, float]]:
    for x in (low[0], high[0]):
        for y in (low[1], high[1]):
            for z in (low[2], high[2]):
                yield (x, y, z)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass
class SceneObjects:
    """The objects of the scene, sorted once per view: the guides, and the rest
    that Houdini counts as geometry."""

    guides: list[Any]
    geometry: list[Any]


def scene_objects(hou: Any) -> SceneObjects:
    """Every object in an object network, sorted into guides and geometry.

    Houdini's own kinds sort them, one filtered glob each for geometry, cameras
    and lights, a few milliseconds in all on a scene of ten thousand nodes. A
    node found inside anything that is not an object network, such as the
    networks inside a simulation, is not an object of the scene and is left
    out.
    """
    root = _quiet(lambda: hou.node("/obj"))
    kinds = getattr(hou, "nodeTypeFilter", None)
    if root is None or kinds is None:
        return SceneObjects([], [])

    def found(kind: Any) -> list[Any]:
        matched = _quiet(lambda: root.recursiveGlob("*", kind)) if kind is not None else None
        return [node for node in matched or () if _in_object_networks(node, root)]

    guides: dict[str, Any] = {}
    for name in GUIDE_KINDS:
        for node in found(getattr(kinds, name, None)):
            guides[node.path()] = node
    inside = tuple(path + "/" for path in guides)
    geometry = []
    for node in found(getattr(kinds, "ObjGeometry", None)):
        path = node.path()
        if path in guides:
            continue
        if path.startswith(inside) or _stock_shape(node):
            guides[path] = node
        else:
            geometry.append(node)
    return SceneObjects(list(guides.values()), geometry)


def _in_object_networks(node: Any, root: Any) -> bool:
    """Whether every network between a node and /obj holds objects, so the walk
    never counts what sits inside a simulation or a geometry network."""
    top = _quiet(lambda: root.path())
    current = _quiet(lambda: node.parent())
    while current is not None and _quiet(lambda current=current: current.path()) != top:
        held = _quiet(lambda current=current: current.childTypeCategory().name())
        if held != "Object":
            return False
        current = _quiet(lambda current=current: current.parent())
    return current is not None


def _stock_shape(node: Any) -> bool:
    """Whether an object is of a guide type and still draws the shape it came with."""
    type_name = _quiet(lambda: node.type().nameComponents()[2])
    shape = GUIDE_SHAPES.get(type_name)
    if shape is None:
        return False
    shown = _quiet(lambda: node.displayNode())
    return shown is None or _quiet(lambda: shown.type().name()) == shape


def bounds_of_all(
    hou: Any,
    frames: Sequence[float],
    warnings: list[str],
    what: str,
    objects: SceneObjects | None = None,
) -> tuple[Sequence[float], Sequence[float]]:
    """The box "all" frames, or one at the origin, with a warning, when there is none."""
    nodes, capped = drawn_objects(hou, frames, objects)
    if capped:
        warnings.append(TOO_MANY_TO_FRAME)
    bounds = world_bounds(hou, nodes, frames)
    if bounds is None:
        warnings.append(NOTHING_TO_FRAME.format(what))
        return (-MIN_EXTENT,) * 3, (MIN_EXTENT,) * 3
    return bounds


def framing_frames(hou: Any, frames: Sequence[float]) -> tuple[float, ...]:
    """The frames bounds are read at: a sequence is framed on its first and last
    frames together, so what moves across it stays in, and a single frame on
    itself, whatever frame the scene is on. Framing a sequence so cooks its
    last frame before the first is drawn."""
    if not frames:
        return (_current_frame(hou),)
    if frames[0] == frames[-1]:
        return (float(frames[0]),)
    return (float(frames[0]), float(frames[-1]))


def world_bounds(
    hou: Any, nodes: Sequence[Any], frames: Sequence[float] = ()
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The world space box around what the nodes draw at the frames framed.

    None when there is no box to go around: no geometry at all, or only
    empty geometry. A simulation network draws its objects without a
    geometry node to read, so it is drawn but not framed.
    """
    low = [math.inf] * 3
    high = [-math.inf] * 3
    for frame in framing_frames(hou, frames):
        for node in nodes:
            if node is None:
                continue
            for point in _node_corners(hou, node, frame):
                for axis in range(3):
                    low[axis] = min(low[axis], point[axis])
                    high[axis] = max(high[axis], point[axis])
    if not all(math.isfinite(value) for value in (*low, *high)):
        return None
    return (low[0], low[1], low[2]), (high[0], high[1], high[2])


def guide_mask(hou: Any, spec: Spec, objects: SceneObjects | None = None) -> str:
    """The objects a flipbook draws: all of them, less the guides unless guides are asked for.

    Houdini still decides per frame what is shown, so an object shown only at
    the frame captured, an instance object or a simulation is drawn as it would
    be anyway.
    """
    if spec.guides:
        return "*"
    guides = (objects or scene_objects(hou)).guides
    return " ".join(["*", *(f"^{node.path()}" for node in guides)])


def drawn_objects(
    hou: Any, frames: Sequence[float] = (), objects: SceneObjects | None = None
) -> tuple[list[Any], bool]:
    """The objects "all" frames, inside subnets too, and whether there were more.

    Every object Houdini counts as geometry that is shown at a frame framed
    and is not a guide, up to `MAX_FRAMED`.
    """
    at = framing_frames(hou, frames)
    drawn: list[Any] = []
    for node in (objects or scene_objects(hou)).geometry:
        if not any(_shown(node, frame) for frame in at):
            continue
        if len(drawn) == MAX_FRAMED:
            return drawn, True
        drawn.append(node)
    return drawn, False


def _shown(node: Any, frame: float) -> bool:
    shown = _quiet(lambda: node.isObjectDisplayedAtFrame(frame))
    if shown is None:
        shown = _quiet(lambda: node.isObjectDisplayed())
    return bool(shown) and _quiet(lambda: node.displayNode()) is not None


def _node_corners(hou: Any, node: Any, frame: float) -> list[tuple[float, float, float]]:
    owner = _object_of(node)
    if owner is None:
        return []
    source = _quiet(lambda: owner.displayNode()) if owner is node else node
    box = _box_at(source, frame)
    if box is None:
        return []
    low, high = box
    if owner is node:
        # An instance object draws another object's geometry at each of its
        # points: the box of the points grown by that geometry's own box.
        # Each point's own turn and scale are not read.
        copied = _instanced_box(owner, frame)
        if copied is not None:
            low = [a + b for a, b in zip(low, copied[0], strict=True)]
            high = [a + b for a, b in zip(high, copied[1], strict=True)]
    matrix = _quiet(lambda: owner.worldTransformAtTime(hou.frameToTime(frame)).asTuple())
    corners = list(_corners(low, high))
    if not matrix or len(matrix) != 16:
        return corners
    return [_apply(matrix, corner) for corner in corners]


def _box_at(source: Any, frame: float) -> tuple[list[float], list[float]] | None:
    """A geometry node's box at a frame, or None for a node with no geometry to read."""
    if source is None:
        return None
    geometry = _quiet(lambda: source.geometryAtFrame(frame))
    box = _quiet(lambda: geometry.boundingBox()) if geometry is not None else None
    if box is None or _quiet(lambda: box.isValid()) is False:
        return None
    # Read by index: a Houdini vector has no iterator, so list() would read on
    # until the fourth item fails, which costs far more than the three reads.
    low, high = box.minvec(), box.maxvec()
    return [low[0], low[1], low[2]], [high[0], high[1], high[2]]


def _instanced_box(node: Any, frame: float) -> tuple[list[float], list[float]] | None:
    if _quiet(lambda: node.type().nameComponents()[2]) != "instance":
        return None
    copied = _quiet(lambda: node.parm("instancepath").evalAsNodeAtFrame(frame))
    if copied is None:
        copied = _quiet(lambda: node.parm("instancepath").evalAsNode())
    shown = _quiet(lambda: copied.displayNode()) if copied is not None else None
    return _box_at(shown, frame)


def _apply(m: Sequence[float], p: Sequence[float]) -> tuple[float, float, float]:
    """A point times a row major 4 by 4 matrix, as Houdini multiplies them."""
    x, y, z = p
    w = x * m[3] + y * m[7] + z * m[11] + m[15]
    w = w if w else 1.0
    return (
        (x * m[0] + y * m[4] + z * m[8] + m[12]) / w,
        (x * m[1] + y * m[5] + z * m[9] + m[13]) / w,
        (x * m[2] + y * m[6] + z * m[10] + m[14]) / w,
    )


def _object_of(node: Any) -> Any:
    """The object a node draws in: itself for an object, its owner for a geometry node."""
    current = node
    while current is not None:
        if _category(current) == "Object":
            return current
        current = _quiet(lambda current=current: current.parent())
    return None


def _category(node: Any) -> str | None:
    return _quiet(lambda: node.type().category().name())


# Section: COP images


def cop_layer_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Any,
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """A Copernicus node's image layer, written as an 8 bit PNG."""
    node = hou.node(spec.path)
    if _category(node) != "Cop":
        raise Unavailable("the node is not a Copernicus COP")
    layer = node.layerAtFrame(frames[0]) if spec.frame is not None else node.layer()
    width, height = (int(value) for value in layer.bufferResolution())
    data = layer.allBufferElements(hou.imageLayerStorageType.Fixed8, 4)
    write_png(path, width, height, bytes(data), bottom_up=True)
    return {
        "files": [path],
        "frames": [frames[0]],
        "camera": None,
        "native": [width, height],
    }


def cop2_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Any,
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """An older COP's image, saved by its own writer."""
    node = hou.node(spec.path)
    if _category(node) != "Cop2":
        raise Unavailable("the node is not an older COP")
    node.saveImage(path, (frames[0], frames[0]))
    native = [_quiet(lambda: node.xRes()), _quiet(lambda: node.yRes())]
    return {
        "files": [path],
        "frames": [frames[0]],
        "camera": None,
        "native": native if None not in native else None,
    }


def write_png(path: str, width: int, height: int, rgba: bytes, *, bottom_up: bool) -> None:
    """An 8 bit RGBA PNG, written whole or not at all.

    Houdini keeps an image's first row at the bottom, so `bottom_up` turns it
    the right way up for a file whose first row is the top.
    """
    stride = width * 4
    if len(rgba) != stride * height:
        raise Unavailable("the image layer did not hold the pixels its size says")
    rows = [rgba[index * stride : (index + 1) * stride] for index in range(height)]
    if bottom_up:
        rows.reverse()
    raw = b"".join(b"\x00" + row for row in rows)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    body = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )
    from nscr_houdini_mcp import outputs

    partial = outputs.temporary_beside(path)
    try:
        with open(partial, "wb") as handle:
            handle.write(body)
        os.replace(partial, path)
    except BaseException:
        discard([partial])
        raise


def _chunk(tag: bytes, body: bytes) -> bytes:
    check = zlib.crc32(tag + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", check)


# Section: panes, grabbed through Qt


def network_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Any,
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """The network editor's own window, cropped to the editor."""
    kind = hou.paneTabType.NetworkEditor
    editors = [
        tab
        for tab in (_quiet(lambda: hou.ui.paneTabs()) or ())
        if _quiet(lambda tab=tab: tab.type()) == kind
    ]
    if not editors:
        raise Unavailable("this desktop has no network editor")
    showing = [tab for tab in editors if _quiet(lambda tab=tab: tab.isCurrentTab())]
    editor = (showing or editors)[0]
    shot: dict[str, Any] = {"files": [path], "frames": [frames[0]], "camera": None}
    previous = _quiet(lambda: editor.pwd()) if spec.path else None
    try:
        if spec.path:
            editor.setPwd(hou.node(spec.path))
            # Whether the editor has laid the new network out by the time it
            # is grabbed is not something this route can see.
            shot["framing_unverified"] = True
        shot["native"] = grab_pane(editor, path, attempt)
    finally:
        if previous is not None:
            attempt.attempt("show the network the editor showed", lambda: editor.setPwd(previous))
    return shot


def pane_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Any,
    gui: bool,
    attempt: Attempt,
) -> dict[str, Any]:
    """One pane tab by name, grabbed from its own window."""
    tab = _quiet(lambda: hou.ui.findPaneTab(spec.path))
    if tab is None:
        names = [
            _quiet(lambda item=item: item.name())
            for item in (_quiet(lambda: hou.ui.paneTabs()) or ())
        ]
        names = [name for name in names if name]
        raise _bad(
            "path",
            f"no pane tab named {spec.path}",
            did_you_mean=did_you_mean(str(spec.path), names),
            panes=names[:50],
        )
    native = grab_pane(tab, path, attempt)
    return {"files": [path], "frames": [frames[0]], "camera": None, "native": native}


def process_events() -> None:
    """Let the interface paint what has changed, so a grab reads what is shown.

    User input is left queued, so nothing the artist does runs in the middle
    of a capture.
    """
    for binding in ("PySide6", "PySide2"):
        try:
            widgets = __import__(f"{binding}.QtWidgets", fromlist=["QApplication"])
            core = __import__(f"{binding}.QtCore", fromlist=["QEventLoop"])
        except ImportError:
            continue
        application = widgets.QApplication.instance()
        if application is not None:
            application.processEvents(core.QEventLoop.ExcludeUserInputEvents)
        return


def grab_pane(tab: Any, path: str, attempt: Attempt) -> list[int]:
    """Grab the window one pane tab lives in and keep the part that is the pane.

    The tab is made the current one of its pane for the grab, so it is the
    one drawn, and the tab that was current is put back after.
    """
    pane = _quiet(lambda: tab.pane())
    previous = _quiet(lambda: pane.currentTab()) if pane is not None else None
    if not _quiet(lambda: tab.isCurrentTab()):
        tab.setIsCurrentTab()
    try:
        window = _quiet(lambda: tab.qtParentWindow())
        geometry = _quiet(lambda: tab.qtScreenGeometry())
        if window is None or geometry is None:
            raise Unavailable("the pane has no window to grab")
        process_events()
        pixmap = window.grab()
        origin = window.mapToGlobal(window.rect().topLeft())
        box = crop_box(
            (geometry.x(), geometry.y(), geometry.width(), geometry.height()),
            (origin.x(), origin.y()),
            (pixmap.width(), pixmap.height()),
            float(_quiet(lambda: pixmap.devicePixelRatio()) or 1.0),
        )
        if box is None:
            raise Unavailable("the pane is not inside its window on screen")
        left, top, right, bottom = box
        piece = pixmap.copy(left, top, right - left, bottom - top)
        if not piece.save(path, "PNG"):
            raise Unavailable(NO_FILE)
    finally:
        if previous is not None and previous is not tab:
            attempt.attempt("make the tab that was showing current again", previous.setIsCurrentTab)
    return [right - left, bottom - top]


def crop_box(
    pane: tuple[float, float, float, float],
    origin: tuple[float, float],
    size: tuple[int, int],
    ratio: float,
) -> tuple[int, int, int, int] | None:
    """The pane's rectangle in the grabbed window's pixels, or nothing when it is off it.

    The pane and the window's origin are in screen points; the grab is in
    device pixels, which is points times the device pixel ratio on a high
    density display. A pixel the pane only partly covers is kept, and the box
    is clamped to the grab.
    """
    x, y, width, height = pane
    left = math.floor((x - origin[0]) * ratio)
    top = math.floor((y - origin[1]) * ratio)
    right = math.ceil((x - origin[0] + width) * ratio)
    bottom = math.ceil((y - origin[1] + height) * ratio)
    left, right = max(0, left), min(int(size[0]), right)
    top, bottom = max(0, top), min(int(size[1]), bottom)
    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right, bottom


def job_spec(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """What a capture's job row says it runs."""
    return {
        "source": arguments.get("source") or "viewport",
        "views": arguments.get("views") or "single",
        "frames": arguments.get("frames"),
    }
