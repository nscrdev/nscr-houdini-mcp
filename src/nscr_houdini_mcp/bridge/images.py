"""Reading a scene linear image inside a session, for the server to compare.

`compare.read_exr` reads an EXR, or any file a COP `file` node reads, through
Houdini's own image layer, applies the session's display transform and writes
the display values to a raw float32 file the server named, rows from the top.
The server reads that file and deletes it. The pixels never travel in a reply.

The read goes through a COP `file` node made for the purpose in a network of
its own and destroyed again, with undo disabled, so the scene is left as it
was apart from Houdini's own note that something changed. The node takes the
file's outputs with `addaovs`; the one named `C`, `rgba` or `rgb` is read if
there is one, else the first.

The display transform comes from the session's OpenColorIO configuration: its
default display and view, from scene linear, with no exposure change. What was
used is written into the reply, so the server can record it and tell when the
two sides of a comparison went through different kinds of transform. When the
OpenColorIO module is not there, the plain sRGB curve is applied and the
record says so.

NumPy is imported when the tool runs, never at import time.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge.errors import BridgeError

ARGUMENTS = ("path", "out_path")
RAW_SUFFIX = ".f32"

# Output names read first, in this order, when the file has several.
COLOUR_OUTPUTS = ("C", "rgba", "RGBA", "rgb", "RGB", "Cd", "beauty")

HOLDER_NAME = "nscr_compare_read"
HOLDER_TYPE = "copnet"
FILE_TYPE = "file"
HOLDER_PARENTS = ("/img", "/obj")


def read_exr(arguments: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Read one image layer, bring it to display values and write it as float32."""
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

    pixels, channel, names = _read_layer(hou, path, numpy)
    height, width, count = pixels.shape
    rgb = pixels[..., :3] if count >= 3 else numpy.repeat(pixels[..., :1], 3, axis=2)
    alpha = pixels[..., 3:4] if count >= 4 else None
    shown, view = display(hou, numpy.ascontiguousarray(rgb, dtype=numpy.float32), numpy)
    shown = numpy.clip(numpy.nan_to_num(shown, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    planes = [shown] if alpha is None else [shown, numpy.clip(alpha, 0.0, 1.0)]
    out = numpy.ascontiguousarray(numpy.concatenate(planes, axis=2), dtype=numpy.float32)
    handle = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(handle, "wb") as stream:
        stream.write(out.tobytes())
    return {
        "width": int(width),
        "height": int(height),
        "channels": int(out.shape[2]),
        "layout": "float32 rows from the top",
        "colour": {
            **view,
            "channel": channel,
            "outputs": names,
            "alpha": "kept apart" if alpha is not None else "none in the file",
        },
    }


def _read_layer(hou: Any, path: str, numpy: Any) -> tuple[Any, str, list[str]]:
    """The pixels of the colour output, as float32 rows from the top."""
    with hou.undos.disabler():
        holder = _holder(hou)
        try:
            reader = holder.createNode(FILE_TYPE)
            reader.parm("filename").set(path)
            button = reader.parm("addaovs")
            if button is not None:
                button.pressButton()
            names = [str(name) for name in reader.outputNames()]
            index = _colour_output(names)
            if hasattr(reader, "layerAtFrame"):
                layer = reader.layerAtFrame(hou.frame(), index)
            else:
                layer = reader.layer(index)
            width, height = (int(value) for value in layer.bufferResolution())
            count = int(layer.channelCount())
            data = layer.allBufferElements()
        finally:
            holder.destroy()
    expected = width * height * count
    if len(data) == expected * 4:
        values = numpy.frombuffer(data, dtype=numpy.float32)
    elif len(data) == expected * 2:
        values = numpy.frombuffer(data, dtype=numpy.float16).astype(numpy.float32)
    else:
        raise BridgeError(
            "TOOL_FAILED",
            "the image layer held a different number of values than its size says",
            {"bytes": len(data), "width": width, "height": height, "channels": count},
        )
    # Houdini keeps image rows from the bottom up.
    pixels = values.reshape(height, width, count)[::-1]
    channel = names[index] if index < len(names) else str(index)
    return pixels, channel, names


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
    config = ocio.Config.CreateFromFile(config_path) if config_path else ocio.GetCurrentConfig()
    display_name = _quiet(lambda: hou.Color.ocio_defaultDisplay()) or config.getDefaultDisplay()
    view_name = _quiet(lambda: hou.Color.ocio_defaultView()) or config.getDefaultView(display_name)
    transform = ocio.DisplayViewTransform(
        src=ocio.ROLE_SCENE_LINEAR, display=display_name, view=view_name
    )
    processor = config.getProcessor(transform).getDefaultCPUProcessor()
    shown = numpy.ascontiguousarray(rgb, dtype=numpy.float32).copy()
    processor.applyRGB(shown)
    return shown, {
        "kind": "view_transform",
        "transform": "ocio_display_view",
        "configuration": config_path or "the current configuration",
        "display": display_name,
        "view": view_name,
        "exposure": 0.0,
    }


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
