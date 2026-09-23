"""`capture.image`: a picture of what a session shows, written to a managed path.

Five sources. `viewport` is the 3D view, `node` one node's output on its own,
`network` the network editor, `cop` a COP's output image and `pane` any pane
tab by name. Every file goes to a path from the `capture` kind of the output
table, one per view, with the frame before the extension for a sequence, and
the reply says which route made it and what was tried before.

The routes for a view, in order, and the first that writes a file wins:

- `viewport_flipbook`: the Scene Viewer that is showing, flipbooked with
  settings of its own: no MPlay, the beauty pass only unless guides are asked
  for, the resolution asked for. A camera, a display mode or a target to
  frame is applied for the capture and put back in a `finally`: the view
  type, the camera looked through, the default camera with its translation,
  rotation, pivot and ortho width, and the shading. A torn off copy of a
  viewport is never made, because it draws nothing and writes no file
  without saying so.
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
  On a display whose pixels are denser than its points, this node draws the
  frame larger than asked and keeps only its bottom left corner. A tiny
  render of a known box says by how much, once per process, and the camera's
  screen window is widened by that much for the capture and put back after.

`cop` reads the COP's image layer and writes it as an 8 bit PNG, or saves an
older COP's image with its own writer. `network` and `pane` grab the pane's
own window through Qt, reached from that one pane tab, and crop it to the
pane. Nothing here walks the widget tree.

Everything made for a capture, the render node, a fitted camera and a moved
display flag, is made and taken away with undo turned off, so the artist's
undo history is as it was. A route that finishes without writing a file, or
writes an empty one, counts as a route that did not work, and the next one is
tried. Only when every route that applies fails is the answer
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

DEFAULT_RESOLUTION = (1280, 720)
MAX_RESOLUTION = 8192
MAX_FRAMES = 1000
FRAME_TOKEN = "$F4"

NO_FILE = "the route finished and wrote no image"
NETWORK_HINT = "read the network with hou_inspect mode tree; a picture of it needs a user interface"
PANE_HINT = "a pane exists only in a session with a user interface"


class Unavailable(Exception):
    """This route cannot make the picture here. The next one is tried."""


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
    for label, camera in wanted:
        if shots and context.should_stop():
            stopped = True
            break
        name = spec.name if len(wanted) == 1 else f"{spec.name}_{label}"
        plan = output_plan(context, hou, "capture", name, "png")
        unsaved = unsaved or bool(plan.unsaved_hip)
        warnings.extend(item for item in plan.warnings if item not in warnings)
        path = sequence_path(plan.path) if spec.sequence else plan.path
        shot = shoot(hou, context, spec, camera, path, frames, gui)
        warnings.extend(item for item in shot.pop("warnings", ()) if item not in warnings)
        stopped = stopped or bool(shot.get("stopped_early"))
        shots.append({"view": label, "run_id": plan.run_id, "template": plan.template, **shot})
    sheet = None
    if len(wanted) > 1 and len(shots) == len(wanted):
        plan = output_plan(context, hou, "capture", f"{spec.name}_sheet", "png")
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
    frame = _quiet(hou.frame)
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
) -> dict[str, Any]:
    """One view, by the first route that writes it."""
    tried: list[dict[str, str]] = []
    for route, run in routes_for(spec.source, gui):
        try:
            shot = run(hou, context, spec, camera, path, frames, gui)
        except Unavailable as reason:
            tried.append({"route": route, "reason": str(reason)})
            continue
        except BridgeError:
            raise
        except Exception as error:  # noqa: BLE001 - a Houdini refusal moves on to the next route
            if not _is_hou_error(error):
                raise
            tried.append({"route": route, "reason": _reason(error)})
            continue
        written = [item for item in shot["files"] if _written(item)]
        if not written or len(written) < len(shot["files"]):
            tried.append({"route": route, "reason": NO_FILE})
            continue
        shot["route"] = route
        if tried:
            shot["tried"] = tried
        return shot
    if tried and all(item["reason"] == NO_FILE for item in tried):
        raise BridgeError(
            "CAPTURE_EMPTY",
            "every route ran and none wrote an image",
            {"source": spec.source, "tried": tried},
            hint="check the camera and the node shown, then capture again",
        )
    hint = {"network": NETWORK_HINT, "pane": PANE_HINT}.get(spec.source)
    raise BridgeError(
        "UI_UNAVAILABLE",
        f"no route could capture the {spec.source} in this session",
        {"source": spec.source, "gui": gui, "tried": tried},
        hint=hint or "use a source this session can show, or a session with a user interface",
    )


def routes(hou: Any) -> list[str]:
    """What the capability probe reports: the routes this session has."""
    found: list[str] = []
    ui = getattr(hou, "ui", None) is not None and bool(_quiet(hou.isUIAvailable))
    if ui:
        found += [VIEWPORT, VIEWPORT_TAB]
    category = _quiet(hou.ropNodeTypeCategory)
    if category is not None and _quiet(lambda: hou.nodeType(category, ROP_TYPE)) is not None:
        found.append(FLIPBOOK_ROP)
    if hasattr(hou, "ImageLayer"):
        found.append(COP_LAYER)
    categories = _quiet(hou.nodeTypeCategories) or {}
    if "Cop2" in categories:
        found.append(COP2_SAVE)
    if ui:
        found += [NETWORK_GRAB, PANE_GRAB]
    return found


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
    return [tab for tab in (_quiet(hou.ui.paneTabs) or ()) if _quiet(tab.type) == kind]


def viewport_route(
    hou: Any, context: ToolContext, spec: Spec, camera: Any, path: str, frames: Any, gui: bool
) -> dict[str, Any]:
    """The Scene Viewer that is showing."""
    showing = [tab for tab in scene_viewers(hou) if _quiet(tab.isCurrentTab)]
    if not showing:
        raise Unavailable("no Scene Viewer is showing")
    return flipbook_viewer(hou, spec, camera, showing[0], path, frames)


def viewport_tab_route(
    hou: Any, context: ToolContext, spec: Spec, camera: Any, path: str, frames: Any, gui: bool
) -> dict[str, Any]:
    """A Scene Viewer made the current tab of its pane for the capture, then put back."""
    tabs = scene_viewers(hou)
    if not tabs:
        raise Unavailable("this desktop has no Scene Viewer")
    if any(_quiet(tab.isCurrentTab) for tab in tabs):
        raise Unavailable("a Scene Viewer is already showing")
    tab = tabs[0]
    pane = _quiet(tab.pane)
    previous = _quiet(pane.currentTab) if pane is not None else None
    tab.setIsCurrentTab()
    try:
        return flipbook_viewer(hou, spec, camera, tab, path, frames)
    finally:
        if previous is not None and previous is not tab:
            _quiet(previous.setIsCurrentTab)


def flipbook_viewer(
    hou: Any, spec: Spec, camera: Any, viewer: Any, path: str, frames: Sequence[float]
) -> dict[str, Any]:
    """Flipbook one viewer's current viewport, with its view put back after."""
    viewport = viewer.curViewport()
    if viewport is None:
        raise Unavailable("the Scene Viewer has no viewport")
    saved = ViewState.save(hou, viewport)
    try:
        described, warnings = apply_view(hou, spec, camera, viewport)
        settings = viewer.flipbookSettings().stash()
        settings.outputToMPlay(False)
        settings.output(path)
        settings.frameRange((frames[0], frames[-1]))
        if len(frames) > 1:
            settings.frameIncrement(frames[1] - frames[0])
        settings.useResolution(True)
        settings.resolution(spec.resolution)
        settings.beautyPassOnly(not spec.guides)
        viewer.flipbook(viewport=viewport, settings=settings, open_dialog=False)
    finally:
        saved.restore(hou, viewport)
    return {
        "files": frame_files(path, frames, spec.sequence),
        "frames": list(frames),
        "camera": described,
        "native": list(spec.resolution),
        "warnings": warnings,
    }


@dataclass
class ViewState:
    """What a capture may change about a viewport, to be put back exactly."""

    kind: Any
    camera: Any
    default: Any
    shading: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def save(cls, hou: Any, viewport: Any) -> ViewState:
        shading: dict[str, Any] = {}
        settings = _quiet(viewport.settings)
        for name in DISPLAY_SETS:
            kind = getattr(hou.displaySetType, name, None)
            shown = _quiet(lambda kind=kind: settings.displaySet(kind)) if kind else None
            if shown is not None:
                shading[name] = _quiet(shown.shadedMode)
        return cls(
            kind=_quiet(viewport.type),
            camera=_quiet(viewport.camera),
            default=viewport.defaultCamera().stash(),
            shading=shading,
        )

    def restore(self, hou: Any, viewport: Any) -> None:
        """Put each thing back on its own, so one that fails leaves the rest done."""
        if self.kind is not None:
            _quiet(lambda: viewport.changeType(self.kind))
        if self.camera is not None:
            _quiet(lambda: viewport.setCamera(self.camera))
        else:
            _quiet(viewport.useDefaultCamera)
        _quiet(lambda: viewport.setDefaultCamera(self.default))
        settings = _quiet(viewport.settings)
        for name, mode in self.shading.items():
            kind = getattr(hou.displaySetType, name, None)
            if kind is None or mode is None:
                continue
            _quiet(lambda kind=kind, mode=mode: settings.displaySet(kind).setShadedMode(mode))


def apply_view(
    hou: Any, spec: Spec, camera: Any, viewport: Any
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
        if _quiet(viewport.camera) is not None:
            viewport.useDefaultCamera()
        viewport.changeType(getattr(hou.geometryViewportType, VIEWPORT_TYPES[camera]))
        described = {"kind": "view", "view": camera}
    elif isinstance(camera, Mapping):
        if _quiet(viewport.camera) is not None:
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
    if target is not None:
        if through is not None:
            warnings.append("a camera node's view is not moved, so frame_target was not applied")
        elif target == "all":
            viewport.frameAll()
        elif target == "selection":
            viewport.frameSelected()
        else:
            bounds = world_bounds(hou, [hou.node(target)])
            if bounds is None:
                warnings.append("the target has no geometry to frame")
            else:
                viewport.frameBoundingBox(hou.BoundingBox(*bounds[0], *bounds[1]))
        described["target"] = target
    return described, warnings


# Section: the flipbook render node


def rop_route(
    hou: Any,
    context: ToolContext,
    spec: Spec,
    camera: Any,
    path: str,
    frames: Sequence[float],
    gui: bool,
) -> dict[str, Any]:
    """Render through a flipbook render node made for the capture and taken away after."""
    parent = _quiet(lambda: hou.node(ROP_PARENT))
    category = _quiet(hou.ropNodeTypeCategory)
    if parent is None or category is None:
        raise Unavailable(f"this scene has no {ROP_PARENT} network")
    if _quiet(lambda: hou.nodeType(category, ROP_TYPE)) is None:
        raise Unavailable(f"this build has no {ROP_TYPE} render node")
    isolated = hou.node(spec.path) if spec.source == "node" else None
    made: list[Any] = []
    put_back: list[Callable[[], Any]] = []
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
            camera_path, described, warnings = rop_camera(hou, spec, camera, targets, made)
            scale = drawing_scale(hou, parent, path)
            if scale is None:
                warnings.append("the render node's drawing scale could not be read")
            elif scale != 1.0:
                put_back.extend(widen_window(hou.node(camera_path), scale))
                described["window_scaled"] = scale
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
            for index, frame in enumerate(frames):
                if done and context.should_stop():
                    stopped = True
                    break
                rop.render(frame_range=(frame, frame))
                done.append(frame)
                if len(frames) > 1 and context.progress is not None:
                    context.progress(
                        {"done": index + 1, "total": len(frames), "message": f"frame {frame:g}"}
                    )
        finally:
            for step in reversed(put_back):
                _quiet(step)
            for node in reversed(made):
                _quiet(node.destroy)
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


# The drawing scale, worked out once for the `hou` it was read in.
_found: list[Any] = []

# The calibration: a unit box seen by an orthographic camera two units wide,
# in a square picture. Drawn right, the box's left edge is a quarter of the
# way across.
PROBE_SIZE = 32
PROBE_NAME = "nscr_capture_probe"


def drawing_scale(hou: Any, parent: Any, path: str) -> float | None:
    """How much larger than asked the render node draws, as a camera window sees it.

    On a display whose pixels are denser than its points, a render node in a
    session without a user interface draws the frame that many times larger
    and keeps only the corner it was asked the size of, so the picture shows
    the bottom left of the view. A small render of a known box says by how
    much, once per process, and the camera's window makes up for it. Nothing
    when the picture could not be read, which leaves the framing unverified.
    """
    if _found and _found[0] is hou:
        return _found[1]
    obj = hou.node(CAMERA_PARENT)
    probe_file = f"{path}.probe.png"
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
        box = alpha_box(probe_file)
        if box is not None and box[0] > 0:
            scale = round(box[0] / (PROBE_SIZE / 4) * 4) / 4
    except (Unavailable, OSError, ValueError, zlib.error):
        scale = None
    except Exception as error:  # noqa: BLE001 - a Houdini refusal leaves the scale unknown
        if not _is_hou_error(error):
            raise
        scale = None
    finally:
        for node in reversed(made):
            _quiet(node.destroy)
        try:
            os.remove(probe_file)
        except OSError:
            pass
    if scale is not None:
        _found[:] = [hou, scale]
    return scale


def widen_window(camera: Any, scale: float) -> list[Callable[[], Any]]:
    """Grow a camera's screen window so the whole view lands in the corner that is kept.

    The window keeps its bottom left corner and grows by the scale, which
    moves its centre by half of what it grew. Returns the steps that put the
    camera's own window back.
    """
    names = ("winx", "winy", "winsizex", "winsizey")
    parms = {name: _quiet(lambda name=name: camera.parm(name)) for name in names}
    if any(parm is None for parm in parms.values()):
        raise Unavailable("the camera has no screen window to widen")
    kept = {name: parm.eval() for name, parm in parms.items()}
    for axis in ("x", "y"):
        size = float(kept[f"winsize{axis}"])
        parms[f"win{axis}"].set(float(kept[f"win{axis}"]) + size * (scale - 1.0) / 2.0)
        parms[f"winsize{axis}"].set(size * scale)
    return [lambda name=name, value=value: parms[name].set(value) for name, value in kept.items()]


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
    taken = {_quiet(child.name) for child in (_quiet(parent.children) or ())}
    if base not in taken:
        return base
    index = 1
    while f"{base}{index}" in taken:
        index += 1
    return f"{base}{index}"


def isolate(node: Any) -> tuple[list[str], list[Callable[[], Any]]]:
    """Show one node alone: its object, and the node carrying the display flag.

    Returns the objects the render draws and the steps that put the flags
    back, which run after the capture whatever happened.
    """
    owner = _object_of(node)
    put_back: list[Callable[[], Any]] = []
    if owner is not node:
        previous = _quiet(owner.displayNode)
        if previous is None or _quiet(previous.path) != _quiet(node.path):
            node.setDisplayFlag(True)
            if previous is not None:
                put_back.append(lambda: previous.setDisplayFlag(True))
    return [owner.path()], put_back


def _targets(hou: Any, target: str | None) -> list[Any] | None:
    if target in (None, "all"):
        return None
    if target == "selection":
        return list(_quiet(hou.selectedNodes) or ())
    return [hou.node(target)]


def rop_camera(
    hou: Any, spec: Spec, camera: Any, targets: list[Any] | None, made: list[Any]
) -> tuple[str, dict[str, Any], list[str]]:
    """The camera the render node looks through: the one named, or one fitted for it."""
    warnings: list[str] = []
    if isinstance(camera, str) and camera.startswith("/"):
        if spec.frame_target is not None:
            warnings.append("a camera node's view is not moved, so frame_target was not applied")
        return camera, {"kind": "node", "path": camera}, warnings
    if isinstance(camera, Mapping):
        orbit, elevation, ortho = camera["orbit"], camera["elevation"], False
        view = "orbit"
    else:
        view = camera or "persp"
        orbit, elevation, ortho = FITTED[view]
    bounds = world_bounds(hou, targets)
    if bounds is None:
        warnings.append("nothing to frame was found, so the camera frames the origin")
        bounds = ((-MIN_EXTENT,) * 3, (MIN_EXTENT,) * 3)
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
    described = {
        "kind": "fitted",
        "view": view,
        "orbit": orbit,
        "elevation": elevation,
        "projection": "ortho" if ortho else "perspective",
        "target": spec.path if spec.source == "node" else (spec.frame_target or "all"),
    }
    return fitted.path(), described, warnings


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


def world_bounds(
    hou: Any, nodes: list[Any] | None
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """The world space box around what the nodes draw, or around every shown object."""
    if nodes is None:
        scene = _quiet(lambda: hou.node("/obj").children()) or ()
        nodes = [node for node in scene if _shown(node)]
    low = [math.inf] * 3
    high = [-math.inf] * 3
    for node in nodes:
        if node is None:
            continue
        for point in _node_corners(node):
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    if not all(math.isfinite(value) for value in (*low, *high)):
        return None
    return (low[0], low[1], low[2]), (high[0], high[1], high[2])


def _shown(node: Any) -> bool:
    shown = _quiet(node.isObjectDisplayed)
    if shown is None:
        shown = _quiet(node.isDisplayFlagSet)
    return bool(shown) and _quiet(node.displayNode) is not None


def _node_corners(node: Any) -> list[tuple[float, float, float]]:
    owner = _object_of(node)
    if owner is None:
        return []
    source = _quiet(owner.displayNode) if owner is node else node
    geometry = _quiet(source.geometry) if source is not None else None
    box = _quiet(geometry.boundingBox) if geometry is not None else None
    if box is None or _quiet(box.isValid) is False:
        return []
    low, high = list(box.minvec()), list(box.maxvec())
    matrix = _quiet(lambda: owner.worldTransform().asTuple())
    corners = list(_corners(low, high))
    if not matrix or len(matrix) != 16:
        return corners
    return [_apply(matrix, corner) for corner in corners]


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
        current = _quiet(current.parent)
    return None


def _category(node: Any) -> str | None:
    return _quiet(lambda: node.type().category().name())


# Section: COP images


def cop_layer_route(
    hou: Any, context: ToolContext, spec: Spec, camera: Any, path: str, frames: Any, gui: bool
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
    hou: Any, context: ToolContext, spec: Spec, camera: Any, path: str, frames: Any, gui: bool
) -> dict[str, Any]:
    """An older COP's image, saved by its own writer."""
    node = hou.node(spec.path)
    if _category(node) != "Cop2":
        raise Unavailable("the node is not an older COP")
    node.saveImage(path, (frames[0], frames[0]))
    native = [_quiet(node.xRes), _quiet(node.yRes)]
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
    partial = f"{path}.part"
    with open(partial, "wb") as handle:
        handle.write(body)
    os.replace(partial, path)


def _chunk(tag: bytes, body: bytes) -> bytes:
    check = zlib.crc32(tag + body) & 0xFFFFFFFF
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", check)


# Section: panes, grabbed through Qt


def network_route(
    hou: Any, context: ToolContext, spec: Spec, camera: Any, path: str, frames: Any, gui: bool
) -> dict[str, Any]:
    """The network editor's own window, cropped to the editor."""
    kind = hou.paneTabType.NetworkEditor
    editors = [tab for tab in (_quiet(hou.ui.paneTabs) or ()) if _quiet(tab.type) == kind]
    if not editors:
        raise Unavailable("this desktop has no network editor")
    showing = [tab for tab in editors if _quiet(tab.isCurrentTab)]
    editor = (showing or editors)[0]
    shot: dict[str, Any] = {"files": [path], "frames": [frames[0]], "camera": None}
    if spec.path:
        previous = _quiet(editor.pwd)
        editor.setPwd(hou.node(spec.path))
        # The editor draws the new network on its next paint, which a grab
        # in the same breath may not see.
        shot["framing_unverified"] = True
        try:
            shot["native"] = grab_pane(editor, path)
        finally:
            if previous is not None:
                _quiet(lambda: editor.setPwd(previous))
    else:
        shot["native"] = grab_pane(editor, path)
    return shot


def pane_route(
    hou: Any, context: ToolContext, spec: Spec, camera: Any, path: str, frames: Any, gui: bool
) -> dict[str, Any]:
    """One pane tab by name, grabbed from its own window."""
    tab = _quiet(lambda: hou.ui.findPaneTab(spec.path))
    if tab is None:
        names = [_quiet(item.name) for item in (_quiet(hou.ui.paneTabs) or ())]
        names = [name for name in names if name]
        raise _bad(
            "path",
            f"no pane tab named {spec.path}",
            did_you_mean=did_you_mean(str(spec.path), names),
            panes=names[:50],
        )
    return {"files": [path], "frames": [frames[0]], "camera": None, "native": grab_pane(tab, path)}


def grab_pane(tab: Any, path: str) -> list[int]:
    """Grab the window one pane tab lives in and keep the part that is the pane."""
    window = _quiet(tab.qtParentWindow)
    geometry = _quiet(tab.qtScreenGeometry)
    if window is None or geometry is None:
        raise Unavailable("the pane has no window to grab")
    pixmap = window.grab()
    origin = window.mapToGlobal(window.rect().topLeft())
    box = crop_box(
        (geometry.x(), geometry.y(), geometry.width(), geometry.height()),
        (origin.x(), origin.y()),
        (pixmap.width(), pixmap.height()),
        float(_quiet(pixmap.devicePixelRatio) or 1.0),
    )
    if box is None:
        raise Unavailable("the pane is not inside its window on screen")
    left, top, right, bottom = box
    piece = pixmap.copy(left, top, right - left, bottom - top)
    if not piece.save(path, "PNG"):
        raise Unavailable(NO_FILE)
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
