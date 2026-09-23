"""Reading a scene linear image inside a session, for the server to compare.

`compare.read_exr` reads an EXR, or any file the session can read, applies the
session's display transform and writes the display values to a raw float32
file the server named inside its compare folder, rows from the top. The
server reads that file and deletes it. The pixels never travel in a reply.

Two routes, in this order:

1. OpenImageIO, which Houdini ships. The file is read without touching the
   scene. The data window is placed inside the display window.
2. A COP `file` node, made for the purpose in a network of its own and
   destroyed again, with undo disabled, whatever the read did. The node takes
   the file's outputs with `addaovs`; the one named `C`, `rgba` or `rgb` is
   read if there is one, else the first. Samples are decoded by the layer's
   storage type and the data window is placed inside the display window. In a
   session with a user interface, making and removing a node marks the scene
   changed, so the reply says `scene_marked_changed`. The bridge's own
   unsaved mark is left alone: this tool is a read.

Colour is premultiplied in a scene linear file, so it is divided by alpha
before the display transform and alpha is handed back apart. An image over the
pixel budget is shrunk by a whole factor, averaging in linear light, before
anything else is done to it, and the reply says so.

The display transform comes from the session's OpenColorIO configuration: its
default display and view, from scene linear, with no exposure change. The
reply names the configuration by file name and content hash, never by where it
is, so the same configuration on two machines is the same record. When the
OpenColorIO module is not there, the plain sRGB curve is applied and the
record says so.

A call asked to stop writes nothing, and one asked to stop while writing
deletes what it wrote.

NumPy is imported when the tool runs, never at import time.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge.errors import BridgeError

ARGUMENTS = ("path", "out_path")
RAW_SUFFIX = ".f32"

# The most pixels an image is handled at; larger ones are shrunk on read.
PIXEL_BUDGET = 64_000_000

# Output names read first, in this order, when the file has several.
COLOUR_OUTPUTS = ("C", "rgba", "RGBA", "rgb", "RGB", "Cd", "beauty")

# Channel layers read first, in this order: the unprefixed `R`, `G`, `B`,
# then the usual names of the beauty layer.
COLOUR_LAYERS = ("", "C", "rgba", "RGBA", "beauty")

HOLDER_NAME = "nscr_compare_read"
HOLDER_TYPE = "copnet"
FILE_TYPE = "file"
HOLDER_PARENTS = ("/img", "/obj")

# Alpha this close to zero leaves the colour as it is when dividing by it.
ALPHA_FLOOR = 1e-6


def read_exr(arguments: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Read one image, bring it to display values and write it as float32."""
    hou = context.hou
    if hou is None:
        raise BridgeError("TOOL_FAILED", "this tool needs a Houdini and this process has none")
    path = str(arguments.get("path") or "")
    out_path = str(arguments.get("out_path") or "")
    if not path or not os.path.isabs(path):
        raise BridgeError("BAD_ARGUMENTS", "path has to be an absolute path")
    if not os.path.isfile(path):
        raise BridgeError("FILE_NOT_FOUND", "there is no image at that path", {"argument": "path"})
    target = Path(out_path)
    if not out_path or not target.is_absolute() or target.suffix != RAW_SUFFIX:
        raise BridgeError("BAD_ARGUMENTS", f"out_path has to be an absolute {RAW_SUFFIX} path")
    if not target.parent.is_dir():
        raise BridgeError("BAD_ARGUMENTS", "the folder for out_path is not there")

    import numpy

    read = _read_oiio(path, numpy)
    marked = False
    if read is None:
        read = _read_cop(hou, path, numpy)
        marked = True
    pixels, names, file_alpha, route = read
    height, width = pixels.shape[:2]
    rgb_index, alpha_index = _colour_channels(names, pixels.shape[2], file_alpha)
    factor = _factor(width, height)
    if factor > 1:
        pixels = _shrink(pixels, factor, numpy)
    rgb = pixels[..., rgb_index]
    alpha = pixels[..., alpha_index : alpha_index + 1] if alpha_index is not None else None
    if alpha is not None:
        safe = numpy.maximum(alpha, ALPHA_FLOOR)
        rgb = numpy.where(alpha > ALPHA_FLOOR, rgb / safe, rgb)
    shown, view = display(hou, numpy.ascontiguousarray(rgb, dtype=numpy.float32), numpy)
    shown = numpy.clip(numpy.nan_to_num(shown, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    planes = [shown] if alpha is None else [shown, numpy.clip(alpha, 0.0, 1.0)]
    out = numpy.ascontiguousarray(numpy.concatenate(planes, axis=2), dtype=numpy.float32)
    if context.should_stop():
        return {"cancelled": True, "written": False}
    handle = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(out.tobytes())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    if context.should_stop():
        target.unlink(missing_ok=True)
        return {"cancelled": True, "written": False}
    reply: dict[str, Any] = {
        "width": int(out.shape[1]),
        "height": int(out.shape[0]),
        "channels": int(out.shape[2]),
        "layout": "float32 rows from the top",
        "route": route,
        "scene_marked_changed": marked,
        "colour": {
            **view,
            "channel": ",".join(names[index] for index in rgb_index) if names else None,
            "alpha": {
                "present": alpha is not None,
                "premultiplied": alpha is not None,
                "unpremultiplied": alpha is not None,
            },
        },
        "alpha": {
            "present": alpha is not None,
            "partial": bool(alpha is not None and (alpha < 1.0).any()),
            "coverage_pct": (
                None
                if alpha is None
                else round(float(numpy.clip(alpha, 0.0, 1.0).mean(dtype=numpy.float64)) * 100, 2)
            ),
            "premultiplied": alpha is not None,
            "unpremultiplied": alpha is not None,
        },
    }
    if factor > 1:
        reply["resized_on_read"] = {
            "from": [int(width), int(height)],
            "to": [int(out.shape[1]), int(out.shape[0])],
            "factor": factor,
        }
    return reply


# Section: OpenImageIO


def _read_oiio(path: str, numpy: Any) -> tuple[Any, list[str], int | None, str] | None:
    """The whole display window through OpenImageIO, or nothing when it is not there."""
    try:
        import OpenImageIO as oiio  # noqa: N813 - the module's own name
    except ImportError:
        return None
    source = oiio.ImageInput.open(path)
    if source is None:
        return None
    try:
        spec = source.spec()
        try:
            data = source.read_image(0, 0, 0, spec.nchannels, "float")
        except TypeError:
            data = source.read_image("float")
    finally:
        source.close()
    if data is None:
        return None
    count = int(spec.nchannels)
    pixels = numpy.asarray(data, dtype=numpy.float32).reshape(spec.height, spec.width, count)
    full = (spec.full_x, spec.full_y, spec.full_width, spec.full_height)
    if not full[2] or not full[3]:
        full = (spec.x, spec.y, spec.width, spec.height)
    canvas = numpy.zeros((full[3], full[2], count), dtype=numpy.float32)
    _place(canvas, pixels, spec.x - full[0], spec.y - full[1])
    names = [str(name) for name in spec.channelnames]
    alpha = int(spec.alpha_channel) if int(getattr(spec, "alpha_channel", -1)) >= 0 else None
    return canvas, names, alpha, "OpenImageIO"


# Section: the COP route


def _read_cop(hou: Any, path: str, numpy: Any) -> tuple[Any, list[str], int | None, str]:
    """The colour output of a COP `file` node, placed in its display window."""
    with hou.undos.disabler():
        holder = _holder(hou)
        try:
            reader = holder.createNode(FILE_TYPE)
            reader.parm("filename").set(path)
            button = reader.parm("addaovs")
            if button is not None:
                button.pressButton()
            outputs = [str(name) for name in reader.outputNames()]
            index = _colour_output(outputs)
            if hasattr(reader, "layerAtFrame"):
                layer = reader.layerAtFrame(hou.frame(), index)
            else:
                layer = reader.layer(index)
            width, height = (int(value) for value in layer.bufferResolution())
            count = int(layer.channelCount())
            storage = str(_quiet(layer.storageType) or "")
            data = layer.allBufferElements()
            data_window = _quiet(layer.dataWindow)
            display_window = _quiet(layer.displayWindow)
        finally:
            holder.destroy()
    values = _decode(data, storage, width * height * count, numpy)
    # Houdini keeps image rows from the bottom up, in its own pixel coordinates.
    pixels = values.reshape(height, width, count)
    inner = _box(data_window, width, height)
    outer = _box(display_window, None, None) or inner
    if inner is not None and outer is not None:
        canvas = numpy.zeros((outer[3] - outer[1], outer[2] - outer[0], count), numpy.float32)
        _place(canvas, pixels, inner[0] - outer[0], inner[1] - outer[1])
        pixels = canvas
    channel = outputs[index] if index < len(outputs) else str(index)
    names = [f"{channel}.{letter}" for letter in "RGBA"[:count]]
    alpha = 3 if count >= 4 else None
    return pixels[::-1], names, alpha, "cop_file_node"


def _decode(data: bytes, storage: str, count: int, numpy: Any) -> Any:
    """Samples by the layer's storage type, brought to floats."""
    kind = storage.lower().replace("_", "")
    table = (
        ("float32", numpy.float32, 1.0),
        ("float16", numpy.float16, 1.0),
        ("int16", numpy.uint16, 65535.0),
        ("int8", numpy.uint8, 255.0),
        ("int32", numpy.uint32, 4294967295.0),
    )
    for name, dtype, top in table:
        if name in kind:
            values = numpy.frombuffer(data, dtype=dtype)
            if values.size != count:
                break
            return values.astype(numpy.float32) / top
    raise BridgeError(
        "TOOL_FAILED",
        "the image layer's storage type and size do not describe its buffer",
        {"storage": storage or None, "bytes": len(data), "values": count},
    )


def _box(value: Any, width: int | None, height: int | None) -> tuple[int, int, int, int] | None:
    """A window as left, bottom, right, top, right and top exclusive, however it came."""
    if value is None:
        return None
    low = _quiet(lambda: value.min())
    high = _quiet(lambda: value.max())
    if low is not None and high is not None:
        return int(low[0]), int(low[1]), int(high[0]), int(high[1])
    try:
        x0, y0, x1, y1 = (int(round(float(item))) for item in value)
    except (TypeError, ValueError):
        return None
    if width is None or height is None:
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None
    if (x1 - x0, y1 - y0) == (width, height):
        return x0, y0, x1, y1
    if (x1 - x0 + 1, y1 - y0 + 1) == (width, height):
        return x0, y0, x1 + 1, y1 + 1
    if (x1, y1) == (width, height):
        return x0, y0, x0 + width, y0 + height
    return None


def _place(canvas: Any, pixels: Any, left: int, top: int) -> None:
    """Paste a window into a larger frame, leaving out whatever falls outside it."""
    height, width = canvas.shape[:2]
    rows, cols = pixels.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(width, left + cols), min(height, top + rows)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = pixels[y0 - top : y1 - top, x0 - left : x1 - left]


def _holder(hou: Any) -> Any:
    """A COP network of this tool's own, in the first context that takes one."""
    last: BaseException | None = None
    for parent_path in HOLDER_PARENTS:
        parent = hou.node(parent_path)
        if parent is None:
            continue
        try:
            return parent.createNode(HOLDER_TYPE, HOLDER_NAME)
        except Exception as error:  # noqa: BLE001 - the next context may take it
            last = error
    raise BridgeError(
        "TOOL_FAILED",
        "no context in this session would take a COP network to read the image in",
        {"exception": type(last).__name__ if last else None},
    )


def _colour_output(names: list[str]) -> int:
    for wanted in COLOUR_OUTPUTS:
        if wanted in names:
            return names.index(wanted)
    return 0


def _colour_channels(
    names: list[str], count: int, alpha: int | None
) -> tuple[list[int], int | None]:
    """Which channels are red, green, blue and alpha, all from one layer.

    Channels are keyed by layer and letter, and the first channel under a key
    is kept, so `diffuse.R` never stands in for `R`. The unprefixed layer is
    tried first, then the usual names of the beauty layer, then any layer
    that has all three colours, in the order the file lists them. Alpha comes
    from the same layer or not at all. A file whose channels have no such
    names gives its first three channels that are not its alpha.
    """
    layers: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        layer, _, letter = name.rpartition(".")
        layers.setdefault(layer, {}).setdefault(letter.upper(), index)
    order = [layer for layer in COLOUR_LAYERS if layer in layers]
    order += [layer for layer in layers if layer not in order]
    for layer in order:
        found = layers[layer]
        if all(letter in found for letter in "RGB"):
            return [found["R"], found["G"], found["B"]], found.get("A")
    colour = [index for index in range(count) if index != alpha]
    if len(colour) >= 3:
        return colour[:3], alpha
    return ([colour[0]] * 3 if colour else [0, 0, 0]), alpha


def _factor(width: int, height: int) -> int:
    pixels = width * height
    return 1 if pixels <= PIXEL_BUDGET else math.ceil(math.sqrt(pixels / PIXEL_BUDGET))


def _shrink(pixels: Any, factor: int, numpy: Any) -> Any:
    height = pixels.shape[0] // factor * factor
    width = pixels.shape[1] // factor * factor
    cut = pixels[:height, :width]
    shape = (height // factor, factor, width // factor, factor, pixels.shape[2])
    return cut.reshape(shape).mean(axis=(1, 3), dtype=numpy.float64).astype(numpy.float32)


# Section: the display transform


def display(hou: Any, rgb: Any, numpy: Any) -> tuple[Any, dict[str, Any]]:
    """Scene linear to display values through the session's OpenColorIO view."""
    try:
        import PyOpenColorIO as ocio  # noqa: N813 - the module's own name
    except ImportError:
        return _srgb(rgb, numpy), {
            "kind": "view_transform",
            "transform": "srgb_curve",
            "configuration": None,
            "display": "sRGB",
            "view": "plain sRGB curve",
            "exposure": 0.0,
            "reason": "OpenColorIO is not importable in this session",
        }
    config_path = _quiet(lambda: hou.Color.ocio_configPath()) or os.environ.get("OCIO")
    try:
        if config_path:
            config = ocio.Config.CreateFromFile(config_path)
        else:
            config = ocio.GetCurrentConfig()
        display_name = _quiet(lambda: hou.Color.ocio_defaultDisplay()) or config.getDefaultDisplay()
        view_name = _quiet(lambda: hou.Color.ocio_defaultView()) or config.getDefaultView(
            display_name
        )
        transform = ocio.DisplayViewTransform(
            src=ocio.ROLE_SCENE_LINEAR, display=display_name, view=view_name
        )
        processor = config.getProcessor(transform).getDefaultCPUProcessor()
        shown = numpy.ascontiguousarray(rgb, dtype=numpy.float32).copy()
        processor.applyRGB(shown)
    except Exception as error:  # noqa: BLE001 - OpenColorIO raises its own kinds
        raise BridgeError(
            "TOOL_FAILED",
            "the session's OpenColorIO configuration could not bring the image to display values",
            {
                "exception": type(error).__name__,
                "reason": str(error)[:300],
                "configuration": Path(config_path).name if config_path else "the current one",
            },
            hint=(
                "point OCIO at a configuration with a scene_linear role and a default display"
                " and view, or compare a PNG or TIFF exported from the image"
            ),
        ) from None
    return shown, {
        "kind": "view_transform",
        "transform": "ocio_display_view",
        "configuration": Path(config_path).name if config_path else "the current configuration",
        "configuration_sha256": _config_hash(config_path),
        "display": display_name,
        "view": view_name,
        "exposure": 0.0,
    }


def _config_hash(config_path: str | None) -> str | None:
    if not config_path:
        return None
    try:
        return hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    except OSError:
        return None


def _srgb(rgb: Any, numpy: Any) -> Any:
    clipped = numpy.clip(rgb, 0.0, None)
    low = clipped * 12.92
    high = 1.055 * numpy.power(clipped, 1.0 / 2.4) - 0.055
    return numpy.where(clipped <= 0.0031308, low, high).astype(numpy.float32)


def _quiet(read: Any) -> Any:
    try:
        return read()
    except Exception:  # noqa: BLE001 - a fact the session will not give is left out
        return None
