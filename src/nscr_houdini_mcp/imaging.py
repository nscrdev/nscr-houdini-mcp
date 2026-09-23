"""Comparing two pictures, in the server process, with NumPy and Pillow.

The order is fixed and every step is written down, so a number can always be
traced back to what was done to the pixels before it was counted:

1. Colour first. A PNG, JPEG or TIFF is brought to sRGB through its embedded
   ICC profile; with none, sRGB is assumed and the record says so. A scene
   linear file arrives from a session already through a display transform,
   with its own record of which one.
2. Align at native resolution. The candidate is placed on the reference's
   grid by `align`, then moved by `adjust`, then, when asked, by a
   translation found with phase correlation. When the candidate has more
   pixels than the reference, the grid is scaled up rather than the candidate
   shrunk, so no detail is thrown away before a crop is cut.
3. Crop before shrinking. Named regions and `region` are cut from the aligned
   native images, and the numbers for a detail crop are counted at that size.
4. Then the overview: both sides resized to a working long edge for the whole
   image numbers, the difference map and the sheet.

Numbers are in display values from 0 to 1. None of them is a verdict: there
is no threshold here and no pass or fail.

This module never imports `hou`.
"""

from __future__ import annotations

import io
import math
import struct
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

# The long edge both sides are shrunk to, when longer, for the whole image
# numbers. A smaller image is counted at its own size.
WORKING_EDGE = 1024

# The most pixels a side is read at. An image over it is shrunk by a whole
# factor while it is read, and the result says so.
PIXEL_BUDGET = 64_000_000

# The most `adjust.scale` may enlarge the candidate to, as a share of the
# grid's area.
MAX_PLACED_SHARE = 4.0

# Rows counted at a time, so no whole frame copy is made to count it.
BLOCK_ROWS = 256

# The long edge of each panel on the overview sheet.
PANEL_EDGE = 256

# How far the grid may be scaled up to keep a larger candidate at its own
# size, and the longest edge it may reach doing so.
MAX_UPSCALE = 4.0
MAX_GRID_EDGE = 4096

# Phase correlation runs on a copy no longer than this, and the shift found is
# scaled back up.
SHIFT_EDGE = 2048

# A correlation peak lower than this is not a shift, it is noise.
MIN_SHIFT_PEAK = 0.05

# A shift longer than this share of the image is not applied.
MAX_SHIFT_SHARE = 0.25

# Mean luminance differing by more than this share after alignment means the
# two sides may not have gone through the same transfer.
LUMINANCE_MISMATCH = 0.25

# The coarse grid the largest difference area is found on.
COARSE_CELLS = 64

# The difference that reaches full colour on the heat map.
HEAT_FULL = 0.25

# How the two sides were brought to display values.
FILE_KIND = "display_file"
VIEW_KIND = "view_transform"

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

ALIGN_MODES = ("fit", "fill", "stretch", "none")

HEAT_STOPS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.33, 0.0, 0.45],
        [0.8, 0.12, 0.16],
        [1.0, 0.63, 0.0],
        [1.0, 1.0, 0.82],
    ],
    dtype=np.float32,
)
UNEVALUATED = np.array([0.12, 0.13, 0.16], dtype=np.float32)
SHEET_BACKGROUND = (24, 24, 24)
SHEET_TEXT = (230, 230, 230)
SHEET_GAP = 6
LABEL_HEIGHT = 16

_HIGH_BIT_GREY = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")


class ImageError(Exception):
    """A picture that cannot be used, with a code the tool passes on."""

    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = dict(details or {})


class AlignRefused(ValueError):
    """An alignment that would place the candidate somewhere unreasonable."""


class MetricsUnavailable(Exception):
    """Numbers that cannot be counted, and why."""


@dataclass
class Picture:
    """One side, in display values: RGB from 0 to 1, alpha kept apart."""

    rgb: np.ndarray
    alpha: np.ndarray | None
    colour: dict[str, Any]
    path: str
    alpha_note: dict[str, Any] = field(default_factory=lambda: {"present": False})
    resized_on_read: dict[str, Any] | None = None
    # What the session said about how it read a scene linear side.
    read_notes: dict[str, Any] | None = None

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])


# Section: reading


def _open(path: str | Path, what: str) -> Image.Image:
    """Open an image without decoding it, refusing one past the size limit."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(path)
    except FileNotFoundError:
        raise ImageError("FILE_NOT_FOUND", f"there is no {what} at that path") from None
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise _too_large(what, None) from None
    except (OSError, SyntaxError, ValueError) as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            f"the {what} file is not an image this can read",
            details={"exception": type(error).__name__},
        ) from None
    limit = Image.MAX_IMAGE_PIXELS
    if limit and image.width * image.height > limit:
        raise _too_large(what, image.size)
    return image


def _too_large(what: str, size: tuple[int, int] | None) -> ImageError:
    return ImageError(
        "IMAGE_TOO_LARGE",
        f"the {what} has more pixels than this reads",
        details={"size_px": list(size) if size else None, "limit_px": Image.MAX_IMAGE_PIXELS},
    )


def _load(image: Image.Image, what: str) -> None:
    try:
        image.load()
    except (OSError, SyntaxError, ValueError) as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            f"the {what} file could not be decoded",
            details={"exception": type(error).__name__},
        ) from None


def reduce_factor(width: int, height: int, budget: int | None = None) -> int:
    """The whole factor that brings an image within the pixel budget, or 1."""
    limit = PIXEL_BUDGET if budget is None else budget
    pixels = width * height
    if pixels <= limit:
        return 1
    return math.ceil(math.sqrt(pixels / limit))


def _reduce_array(array: np.ndarray, factor: int) -> np.ndarray:
    """The block mean over `factor` by `factor` pixels."""
    if factor <= 1:
        return array
    height = array.shape[0] // factor * factor
    width = array.shape[1] // factor * factor
    cut = array[:height, :width].astype(np.float32)
    shape = (height // factor, factor, width // factor, factor) + array.shape[2:]
    return cut.reshape(shape).mean(axis=(1, 3), dtype=np.float64).astype(np.float32)


def _source_bits(image: Image.Image, rawmode: str | None) -> int:
    """How many bits a sample holds in the file, from the file's own description."""
    if image.mode == "F":
        return 32
    if image.mode == "I":
        return 16 if image.format == "PNG" else 32
    if image.mode.startswith("I;16") or (rawmode and ";16" in rawmode):
        return 16
    samples = getattr(image, "tag_v2", {}).get(258) if image.format == "TIFF" else None
    if isinstance(samples, (tuple, list)) and samples:
        return int(max(samples))
    if isinstance(samples, int):
        return samples
    return 1 if image.mode == "1" else 8


def _rawmode(image: Image.Image) -> str | None:
    tile = getattr(image, "tile", None) or []
    if tile and isinstance(tile[0].args, str):
        return tile[0].args
    return None


def _png16(path: str | Path, rawmode: str) -> np.ndarray:
    """A 16 bit RGB or RGBA PNG at full depth.

    Pillow keeps only the high byte of each sample. The same stream decoded a
    second time as little endian hands over the low byte in its place, and the
    two together are the file's values.
    """
    planes = []
    for order in (";16B", ";16L"):
        with Image.open(path) as image:
            image.tile = [image.tile[0]._replace(args=rawmode.replace(";16B", order))]
            image.load()
            planes.append(np.asarray(image, dtype=np.uint16))
    return planes[0] * 256 + planes[1]


def read_display_file(path: str | Path, *, budget: int | None = None) -> Picture:
    """A PNG, JPEG, TIFF or any file Pillow reads, brought to sRGB.

    An embedded ICC profile is converted from; with none, sRGB is assumed and
    the record says so rather than leaving it to be guessed. The full scale of
    a sample comes from the file's mode and bit depth, never from its pixels:
    16 bit greys and 16 bit RGB or RGBA PNGs are read at 16 bits. Alpha is
    kept apart, unpremultiplied when the file stores it premultiplied, and
    never mixed into the colour. An image over the pixel budget is shrunk by
    a whole factor as it is read, and the picture says so.
    """
    image = _open(path, "image")
    rawmode = _rawmode(image)
    bits = _source_bits(image, rawmode)
    colour: dict[str, Any] = {
        "kind": FILE_KIND,
        "format": image.format,
        "mode": image.mode,
        "source_bits": bits,
        "bits_read": min(bits, 8),
    }
    factor = reduce_factor(image.width, image.height, budget)
    original = image.size
    _load(image, "image")
    icc = image.info.get("icc_profile")
    premultiplied = image.mode in ("RGBa", "La")
    alpha = _alpha_of(image)

    if image.mode in _HIGH_BIT_GREY:
        grey = np.asarray(image, dtype=np.float32)
        top = 1.0 if image.mode == "F" else 65535.0 if bits == 16 else 4294967295.0
        grey = _reduce_array(np.clip(grey / top, 0.0, 1.0), factor)
        rgb = np.repeat(grey[..., None], 3, axis=2)
        colour["bits_read"] = bits
        colour["profile"] = "embedded_not_applied" if icc else "assumed_srgb"
        if icc:
            colour["reason"] = "a profile is not applied to high bit grey"
        return _picture(rgb, alpha, colour, path, premultiplied, factor, original)

    if bits == 16 and image.format == "PNG" and rawmode in ("RGB;16B", "RGBA;16B") and not icc:
        try:
            samples = _png16(path, rawmode).astype(np.float32) / 65535.0
        except (OSError, ValueError, AttributeError) as error:
            colour["reason"] = f"read at 8 bits: the 16 bit read failed ({type(error).__name__})"
        else:
            colour["bits_read"] = 16
            colour["profile"] = "assumed_srgb"
            samples = _reduce_array(samples, factor)
            if samples.shape[2] == 4:
                alpha = samples[..., 3]
            return _picture(samples[..., :3], alpha, colour, path, False, factor, original)
    elif bits > 8:
        colour["reason"] = (
            "read at 8 bits: the profile is applied at 8 bits"
            if icc
            else f"read at 8 bits: a {bits} bit {image.format} {image.mode} is read at 8"
        )

    if factor > 1:
        image = image.reduce(factor)
    base = image
    if image.mode in ("RGBA", "LA", "PA", "RGBa", "La") or (
        image.mode == "P" and "transparency" in image.info
    ):
        base = image.convert("RGBA").convert("RGB")
    elif image.mode == "P":
        base = image.convert("RGB")

    if icc:
        converted = _from_profile(base, icc, colour)
        if converted is not None:
            rgb = np.asarray(converted, dtype=np.float32) / 255.0
            return _picture(rgb, alpha, colour, path, premultiplied, factor, original)
    else:
        colour["profile"] = "assumed_srgb"
    try:
        plain = base.convert("RGB")
    except ValueError as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            f"a {image.mode} image cannot be brought to RGB without a profile",
            details={"exception": type(error).__name__, "mode": image.mode},
        ) from None
    rgb = np.asarray(plain, dtype=np.float32) / 255.0
    return _picture(rgb, alpha, colour, path, premultiplied, factor, original)


def _picture(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    colour: dict[str, Any],
    path: str | Path,
    premultiplied: bool,
    factor: int,
    original: tuple[int, int],
) -> Picture:
    """A picture with its alpha record and any shrink on read written down."""
    if alpha is not None and alpha.shape != rgb.shape[:2]:
        alpha = _fit_alpha(alpha, rgb.shape[1], rgb.shape[0], factor)
    colour["alpha"] = {"present": alpha is not None, "premultiplied": premultiplied}
    picture = Picture(rgb.astype(np.float32), alpha, colour, str(path))
    picture.alpha_note = alpha_note(
        alpha, premultiplied=premultiplied, unpremultiplied=premultiplied
    )
    if factor > 1:
        picture.resized_on_read = {
            "from": list(original),
            "to": [int(rgb.shape[1]), int(rgb.shape[0])],
            "factor": factor,
        }
    return picture


def _fit_alpha(alpha: np.ndarray, width: int, height: int, factor: int) -> np.ndarray:
    reduced = _reduce_array(alpha, factor)
    if reduced.shape != (height, width):
        reduced = resize(reduced, width, height)
    return reduced


def alpha_note(
    alpha: np.ndarray | None, *, premultiplied: bool, unpremultiplied: bool
) -> dict[str, Any]:
    """What a side's alpha is: there or not, partial or not, and how it was stored."""
    if alpha is None:
        return {"present": False}
    return {
        "present": True,
        "partial": bool((alpha < 1.0).any()),
        "coverage_pct": round(float(alpha.mean(dtype=np.float64)) * 100.0, 2),
        "premultiplied": premultiplied,
        "unpremultiplied": unpremultiplied,
    }


def _from_profile(image: Image.Image, icc: bytes, colour: dict[str, Any]) -> Image.Image | None:
    try:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        described = ImageCms.getProfileDescription(source).strip()
        converted = ImageCms.profileToProfile(
            image, source, ImageCms.createProfile("sRGB"), outputMode="RGB"
        )
    except (ImageCms.PyCMSError, OSError, ValueError) as error:
        colour["profile"] = "assumed_srgb"
        colour["reason"] = f"the embedded profile could not be used ({type(error).__name__})"
        return None
    colour["profile"] = "embedded"
    colour["description"] = described
    colour["converted_to"] = "sRGB"
    return converted


def _alpha_of(image: Image.Image) -> np.ndarray | None:
    bands = image.getbands()
    for band in ("A", "a"):
        if band in bands:
            return np.asarray(image.getchannel(band), dtype=np.float32) / 255.0
    if image.mode == "P" and "transparency" in image.info:
        return np.asarray(image.convert("RGBA").getchannel("A"), dtype=np.float32) / 255.0
    return None


def picture_from_raw(
    path: str | Path, *, width: int, height: int, channels: int, colour: Mapping[str, Any]
) -> Picture:
    """Display values a session wrote as float32, rows from the top."""
    data = np.fromfile(str(path), dtype=np.float32)
    expected = width * height * channels
    if data.size != expected or channels < 1:
        raise ImageError(
            "IMAGE_UNREADABLE",
            "the session wrote a different number of values than it said",
            details={"values": int(data.size), "expected": int(expected)},
        )
    pixels = data.reshape(height, width, channels)
    if channels >= 3:
        rgb = pixels[..., :3]
    else:
        rgb = np.repeat(pixels[..., :1], 3, axis=2)
    alpha = pixels[..., 3].copy() if channels >= 4 else None
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return Picture(rgb.astype(np.float32), alpha, dict(colour), str(path))


def read_mask(path: str | Path, *, budget: int | None = None) -> np.ndarray:
    """A mask file as a float map from 0 to 1: its alpha if it has one, else its grey."""
    image = _open(path, "mask")
    factor = reduce_factor(image.width, image.height, budget)
    _load(image, "mask")
    if factor > 1:
        image = image.reduce(factor)
    alpha = _alpha_of(image)
    if alpha is not None and alpha.min(initial=1.0) < 1.0:
        return alpha
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


# Section: scene linear headers

EXR_MAGIC = b"\x76\x2f\x31\x01"
HEADER_LIMIT = 1 << 20


def read_linear_header(path: str | Path) -> dict[str, Any]:
    """The size and channels an EXR or Radiance HDR file says it has, from its header alone."""
    try:
        with open(path, "rb") as stream:
            head = stream.read(HEADER_LIMIT)
    except FileNotFoundError:
        raise ImageError("FILE_NOT_FOUND", "there is no image at that path") from None
    except OSError as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            "the file could not be read",
            details={"exception": type(error).__name__},
        ) from None
    try:
        if head.startswith(EXR_MAGIC):
            return _exr_header(head)
        if head.startswith((b"#?RADIANCE", b"#?RGBE")):
            return _hdr_header(head)
    except (struct.error, ValueError, IndexError, KeyError, UnicodeDecodeError) as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            "the file's header could not be read",
            details={"exception": type(error).__name__},
        ) from None
    raise ImageError("IMAGE_UNREADABLE", "the file is not an EXR or HDR image")


def _exr_header(head: bytes) -> dict[str, Any]:
    at = 8
    found: dict[str, Any] = {"format": "EXR"}
    while True:
        end = head.index(b"\0", at)
        name = head[at:end].decode("ascii")
        at = end + 1
        if not name:
            break
        end = head.index(b"\0", at)
        kind = head[at:end].decode("ascii")
        (size,) = struct.unpack_from("<i", head, end + 1)
        value = head[end + 5 : end + 5 + size]
        if size < 0 or len(value) != size:
            raise ValueError("the header is cut short")
        at = end + 5 + size
        if kind == "box2i" and name in ("dataWindow", "displayWindow"):
            found[name] = list(struct.unpack("<4i", value))
        elif kind == "chlist" and name == "channels":
            found["channels"] = _channel_names(value)
    window = found.get("displayWindow") or found.get("dataWindow")
    if window is None or not found.get("channels"):
        raise ValueError("no window or no channels")
    found["width"] = window[2] - window[0] + 1
    found["height"] = window[3] - window[1] + 1
    if found["width"] < 1 or found["height"] < 1:
        raise ValueError("an empty window")
    return found


def _channel_names(value: bytes) -> list[str]:
    names, at = [], 0
    while at < len(value) and value[at] != 0:
        end = value.index(b"\0", at)
        names.append(value[at:end].decode("ascii"))
        at = end + 1 + 16
    return names


def _hdr_header(head: bytes) -> dict[str, Any]:
    lines = head.split(b"\n")
    for index, line in enumerate(lines[1:], start=1):
        if not line.strip():
            tokens = lines[index + 1].decode("ascii").split()
            sizes = {tokens[0][1]: int(tokens[1]), tokens[2][1]: int(tokens[3])}
            return {
                "format": "HDR",
                "width": sizes["X"],
                "height": sizes["Y"],
                "channels": ["R", "G", "B"],
            }
    raise ValueError("no resolution line")


# Section: resampling


def resize(
    array: np.ndarray,
    width: int,
    height: int,
    box: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """A float image or map at a new size, channel by channel.

    `box` is the part of the source, in source pixels and fractions of them,
    that the new size covers: the result is exactly that part of the whole
    image resized, without the whole image ever being made at the new size.
    """
    width, height = max(1, int(width)), max(1, int(height))
    source = (0.0, 0.0, float(array.shape[1]), float(array.shape[0]))
    area = source if box is None else tuple(float(value) for value in box)
    if area == source and array.shape[1] == width and array.shape[0] == height:
        return array.astype(np.float32, copy=True)
    flat = array.ndim == 2
    planes = array[..., None] if flat else array
    out = np.empty((height, width, planes.shape[2]), dtype=np.float32)
    shrinking = width < area[2] - area[0] or height < area[3] - area[1]
    method = Image.Resampling.LANCZOS if shrinking else Image.Resampling.BICUBIC
    for index in range(planes.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(planes[..., index], dtype=np.float32))
        resized = plane.resize((width, height), method, box=area)
        out[..., index] = np.asarray(resized, dtype=np.float32)
    return out[..., 0] if flat else out


def working_size(width: int, height: int, edge: int = WORKING_EDGE) -> tuple[int, int]:
    """The size with the long edge at most `edge`, keeping the aspect. Never larger."""
    scale = min(1.0, edge / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


# Section: alignment


@dataclass
class Aligned:
    """Both sides on one grid, and where the candidate really covers it."""

    candidate: np.ndarray
    reference: np.ndarray
    valid: np.ndarray
    reference_alpha: np.ndarray | None
    steps: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.reference.shape[1]), int(self.reference.shape[0])


def align(
    candidate: Picture,
    reference: Picture,
    *,
    mode: str = "fit",
    adjust: Mapping[str, float] | None = None,
    auto_shift: bool = False,
) -> Aligned:
    """Place the candidate on the reference's grid: `align`, `adjust`, `auto_shift`.

    `fit` letterboxes the candidate inside the reference frame, `fill` crops it
    to cover the frame, `stretch` scales each axis on its own and `none`
    places it pixel for pixel. `adjust` scales about the centre and moves by a
    share of the frame, `dx` to the right and `dy` down.
    """
    if mode not in ALIGN_MODES:
        raise ValueError(f"align must be one of {', '.join(ALIGN_MODES)}")
    rw, rh = reference.size
    cw, ch = candidate.size
    if mode == "fit":
        sx = sy = min(rw / cw, rh / ch)
    elif mode == "fill":
        sx = sy = max(rw / cw, rh / ch)
    elif mode == "stretch":
        sx, sy = rw / cw, rh / ch
    else:
        sx = sy = 1.0

    # A candidate with more pixels than the reference keeps them: the grid is
    # scaled up instead, within limits, so a crop sees the candidate's detail.
    # With `none` the candidate keeps its own pixels one for one, so the grid
    # grows by how much larger it is than the reference.
    ratio = max(cw / rw, ch / rh) if mode == "none" else 1.0 / min(sx, sy)
    grow = min(MAX_UPSCALE, ratio)
    grow = max(1.0, min(grow, MAX_GRID_EDGE / max(rw, rh)))
    gw, gh = round(rw * grow), round(rh * grow)
    fx, fy = gw / rw, gh / rh
    reference_rgb = resize(reference.rgb, gw, gh) if grow > 1.0 else reference.rgb.copy()
    reference_alpha = None
    if reference.alpha is not None:
        reference_alpha = resize(reference.alpha, gw, gh) if grow > 1.0 else reference.alpha.copy()

    moved = dict(adjust or {})
    scale = float(moved.get("scale", 1.0) or 1.0)
    dx, dy = float(moved.get("dx", 0.0) or 0.0), float(moved.get("dy", 0.0) or 0.0)
    if mode == "none":
        sx = sy = 1.0
    else:
        sx, sy = sx * fx, sy * fy
    sx, sy = sx * scale, sy * scale
    nw, nh = max(1, round(cw * sx)), max(1, round(ch * sy))
    if scale > 1.0 and nw * nh > MAX_PLACED_SHARE * gw * gh:
        raise AlignRefused(
            f"adjust.scale {scale:g} would place the candidate at {nw}x{nh}, more than "
            f"{MAX_PLACED_SHARE:g} times the area of the {gw}x{gh} frame"
        )
    ox = round((gw - nw) / 2 + dx * gw)
    oy = round((gh - nh) / 2 + dy * gh)
    placed, px, py = _visible_part(candidate.rgb, nw, nh, ox, oy, gw, gh)
    canvas, valid = _paste(placed, gw, gh, px, py)

    steps: dict[str, Any] = {
        "align": mode,
        "grid_px": [gw, gh],
        "reference_upscaled": round(grow, 4) if grow > 1.0 else None,
        "candidate_scale": [round(sx, 6), round(sy, 6)],
        "placed_px": {"x": ox, "y": oy, "width": nw, "height": nh},
        "letterboxed": _bars(ox, oy, nw, nh, gw, gh),
        "adjust": {"dx": dx, "dy": dy, "scale": scale} if adjust else None,
        "shift_px": None,
    }
    aligned = Aligned(np.clip(canvas, 0.0, 1.0), reference_rgb, valid, reference_alpha, steps)
    if auto_shift:
        found = estimate_shift(aligned.candidate, aligned.reference, aligned.valid)
        steps["shift_px"] = found
        if found["applied"]:
            moved_rgb = translate(aligned.candidate, found["dx"], found["dy"])
            moved_valid = translate(aligned.valid, found["dx"], found["dy"])
            before = valid_error(aligned.candidate, aligned.reference, aligned.valid)
            after = valid_error(moved_rgb, aligned.reference, moved_valid)
            found["mae"] = {
                "unshifted": None if before is None else round(before, 6),
                "shifted": None if after is None else round(after, 6),
            }
            if before is not None and after is not None and after < before:
                aligned.candidate, aligned.valid = moved_rgb, moved_valid
            else:
                # A weak peak can point the wrong way: the shift is kept only
                # when it brings the two closer where the candidate covers.
                found.update(
                    applied=False,
                    dx=0,
                    dy=0,
                    reason="the shift found did not lower the difference, so it was not applied",
                )
    return aligned


# How many rows a block of `valid_error` reads at a time.
ERROR_ROWS = 256


def valid_error(candidate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> float | None:
    """The mean absolute difference where the candidate covers, a block of rows at a time."""
    total = 0.0
    count = 0
    for top in range(0, reference.shape[0], ERROR_ROWS):
        rows = slice(top, top + ERROR_ROWS)
        inside = valid[rows]
        if not inside.any():
            continue
        gap = np.abs(candidate[rows][inside] - reference[rows][inside])
        total += float(gap.sum(dtype=np.float64))
        count += int(gap.size)
    return total / count if count else None


def _bars(ox: int, oy: int, nw: int, nh: int, gw: int, gh: int) -> dict[str, int] | None:
    """Uncovered margins, when the candidate leaves any part of the frame bare."""
    bars = {
        "left": max(0, ox),
        "top": max(0, oy),
        "right": max(0, gw - (ox + nw)),
        "bottom": max(0, gh - (oy + nh)),
    }
    return bars if any(bars.values()) else None


def _visible_part(
    source: np.ndarray, nw: int, nh: int, ox: int, oy: int, gw: int, gh: int
) -> tuple[np.ndarray, int, int]:
    """Only the part of the source that lands on the grid, resized, and where it goes.

    Enlarging the whole candidate and then cutting it down would hold the
    whole enlargement in memory. Only the frame's worth is made: the visible
    part of the grid is mapped back to the source, and that box of the source
    is resized straight to it, at the same place and scale as the whole.
    """
    height, width = source.shape[:2]
    kx, ky = nw / width, nh / height
    x0, x1 = max(0, ox), min(gw, ox + nw)
    y0, y1 = max(0, oy), min(gh, oy + nh)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, source.shape[2]), dtype=np.float32), gw, gh
    box = ((x0 - ox) / kx, (y0 - oy) / ky, (x1 - ox) / kx, (y1 - oy) / ky)
    return resize(source, x1 - x0, y1 - y0, box=box), x0, y0


def _paste(
    image: np.ndarray, width: int, height: int, ox: int, oy: int
) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((height, width, image.shape[2]), dtype=np.float32)
    valid = np.zeros((height, width), dtype=bool)
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(width, ox + image.shape[1]), min(height, oy + image.shape[0])
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = image[y0 - oy : y1 - oy, x0 - ox : x1 - ox]
        valid[y0:y1, x0:x1] = True
    return canvas, valid


def translate(array: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Move by whole pixels, filling what comes in from outside with zero."""
    out = np.zeros_like(array)
    height, width = array.shape[:2]
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 + dy : sy1 + dy, sx0 + dx : sx1 + dx] = array[sy0:sy1, sx0:sx1]
    return out


def estimate_shift(candidate: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> dict:
    """The translation that lines the candidate up with the reference.

    Phase correlation on luminance with a Hann window, found on a copy no
    longer than `SHIFT_EDGE` and refined to a fraction of a pixel on the
    peak's neighbours. Whole pixels are applied; the fraction is reported.
    """
    height, width = reference.shape[:2]
    a = luminance(candidate)
    b = luminance(reference)
    fill = float(b[valid].mean()) if valid.any() else float(b.mean())
    a = np.where(valid, a, fill)
    factor = 1.0
    if max(width, height) > SHIFT_EDGE:
        factor = max(width, height) / SHIFT_EDGE
        small = (max(1, round(width / factor)), max(1, round(height / factor)))
        a, b = resize(a, *small), resize(b, *small)
    h, w = b.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    a = (a - a.mean()) * window
    b = (b - b.mean()) * window
    cross = np.fft.rfft2(b) * np.conj(np.fft.rfft2(a))
    cross /= np.abs(cross) + 1e-12
    surface = np.fft.irfft2(cross, s=(h, w))
    py, px = np.unravel_index(int(np.argmax(surface)), surface.shape)
    peak = float(surface[py, px])
    fy = _refine(surface[(py - 1) % h, px], peak, surface[(py + 1) % h, px])
    fx = _refine(surface[py, (px - 1) % w], peak, surface[py, (px + 1) % w])
    sy = (py if py <= h // 2 else py - h) + fy
    sx = (px if px <= w // 2 else px - w) + fx
    sx, sy = sx * factor, sy * factor
    found = {
        "dx": round(sx),
        "dy": round(sy),
        "estimate": [round(sx, 3), round(sy, 3)],
        "peak": round(peak, 4),
        "applied": True,
    }
    if peak < MIN_SHIFT_PEAK:
        found.update(applied=False, reason="no clear correlation peak, so no shift was applied")
    elif abs(sx) > width * MAX_SHIFT_SHARE or abs(sy) > height * MAX_SHIFT_SHARE:
        found.update(applied=False, reason="the shift found is too large to trust")
    if not found["applied"]:
        found["dx"] = found["dy"] = 0
    return found


def _refine(before: float, peak: float, after: float) -> float:
    """Where a parabola through three samples peaks, from the middle one."""
    bend = before - 2.0 * peak + after
    if abs(bend) < 1e-12:
        return 0.0
    return float(max(-0.5, min(0.5, 0.5 * (before - after) / bend)))


def luminance(rgb: np.ndarray) -> np.ndarray:
    return (rgb @ LUMA).astype(np.float32)


# Section: regions


def box_px(rect: Sequence[float], width: int, height: int) -> tuple[int, int, int, int]:
    """A normalised rectangle as whole pixels, never empty."""
    x0, y0, x1, y1 = (float(value) for value in rect)
    left = min(width - 1, max(0, math.floor(x0 * width)))
    top = min(height - 1, max(0, math.floor(y0 * height)))
    right = max(left + 1, min(width, math.ceil(x1 * width)))
    bottom = max(top + 1, min(height, math.ceil(y1 * height)))
    return left, top, right, bottom


def check_rect(rect: Any) -> list[float] | None:
    """A normalised `[x0, y0, x1, y1]`, or nothing when it is not one."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        values = [float(value) for value in rect]
    except (TypeError, ValueError):
        return None
    if any(isinstance(value, bool) for value in rect):
        return None
    x0, y0, x1, y1 = values
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return values


def cut(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return array[top:bottom, left:right]


# Section: numbers


def metrics(
    candidate: np.ndarray, reference: np.ndarray, evaluated: np.ndarray, tolerance: float
) -> dict[str, Any]:
    """Mean absolute error, RMSE and PSNR per channel and overall, and the area over tolerance.

    Counted only where `evaluated` is true. Raises `MetricsUnavailable` when
    that leaves nothing.
    """
    count = int(evaluated.sum())
    if count == 0:
        raise MetricsUnavailable("the mask and the candidate's coverage leave no pixels to count")
    # A block of rows at a time: differences and their squares in float32 for
    # the block only, the running sums in float64.
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    over = 0
    for top in range(0, evaluated.shape[0], BLOCK_ROWS):
        rows = slice(top, top + BLOCK_ROWS)
        keep = evaluated[rows]
        if not keep.any():
            continue
        delta = candidate[rows][keep] - reference[rows][keep]
        absolute = np.abs(delta)
        sums += absolute.sum(axis=0, dtype=np.float64)
        squares += np.square(delta).sum(axis=0, dtype=np.float64)
        over += int(np.count_nonzero(absolute.max(axis=1) > tolerance))
    mae = sums / count
    rmse = np.sqrt(squares / count)
    overall_rmse = float(math.sqrt(squares.sum() / (3 * count)))
    return {
        "mae": _channels(mae, float(sums.sum() / (3 * count))),
        "rmse": _channels(rmse, overall_rmse),
        "psnr_db": _psnr(overall_rmse),
        "diff_area_pct": round(over * 100.0 / count, 4),
        "evaluated_pct": round(count * 100.0 / evaluated.size, 4),
        "pixels": count,
    }


def _channels(values: np.ndarray, overall: float) -> dict[str, float]:
    return {
        "r": round(float(values[0]), 6),
        "g": round(float(values[1]), 6),
        "b": round(float(values[2]), 6),
        "overall": round(overall, 6),
    }


def _psnr(rmse: float) -> float | None:
    """Peak signal to noise in decibels, or nothing when there is no difference."""
    if rmse <= 0.0:
        return None
    return round(20.0 * math.log10(1.0 / rmse), 4)


def exposure_matched(
    candidate: np.ndarray, reference: np.ndarray, evaluated: np.ndarray, tolerance: float
) -> dict[str, Any]:
    """The same numbers after one gain brings the candidate's mean luminance to the reference's."""
    if not evaluated.any():
        raise MetricsUnavailable("there are no pixels to take a mean luminance from")
    mean_candidate = float(luminance(candidate)[evaluated].mean())
    mean_reference = float(luminance(reference)[evaluated].mean())
    if mean_candidate <= 1e-6:
        raise MetricsUnavailable("the candidate is black where it is counted, so no gain fits")
    gain = mean_reference / mean_candidate
    matched = np.clip(candidate * gain, 0.0, 1.0)
    numbers = metrics(matched, reference, evaluated, tolerance)
    return {"gain": round(gain, 6), **{key: numbers[key] for key in _MATCHED_KEYS}}


_MATCHED_KEYS = ("mae", "rmse", "psnr_db", "diff_area_pct")


def mean_luminance_gap(
    candidate: np.ndarray, reference: np.ndarray, evaluated: np.ndarray
) -> float | None:
    """How far the candidate's mean luminance is from the reference's, as a share of it."""
    if not evaluated.any():
        return None
    mean_candidate = float(luminance(candidate)[evaluated].mean())
    mean_reference = float(luminance(reference)[evaluated].mean())
    return abs(mean_candidate - mean_reference) / max(mean_reference, 1e-6)


def largest_region(over: np.ndarray) -> dict[str, Any] | None:
    """The bounding box of the largest connected area over tolerance.

    Found on a coarse grid, so it is cheap and approximate: a box a person can
    look at, not an outline.
    """
    height, width = over.shape
    total = int(over.sum())
    if total == 0:
        return None
    cell = max(1, math.ceil(max(width, height) / COARSE_CELLS))
    rows, cols = math.ceil(height / cell), math.ceil(width / cell)
    padded = np.zeros((rows * cell, cols * cell), dtype=np.int32)
    padded[:height, :width] = over
    counts = padded.reshape(rows, cell, cols, cell).sum(axis=(1, 3))
    labels = np.zeros((rows, cols), dtype=np.int32)
    best: tuple[int, list[tuple[int, int]]] = (0, [])
    label = 0
    for start in zip(*np.nonzero(counts), strict=True):
        if labels[start]:
            continue
        label += 1
        labels[start] = label
        stack, cells, weight = [start], [], 0
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            weight += int(counts[row, col])
            for nr, nc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if 0 <= nr < rows and 0 <= nc < cols and counts[nr, nc] and not labels[nr, nc]:
                    labels[nr, nc] = label
                    stack.append((nr, nc))
        if weight > best[0]:
            best = (weight, cells)
    weight, cells = best
    top = min(row for row, _ in cells) * cell
    left = min(col for _, col in cells) * cell
    bottom = min(height, (max(row for row, _ in cells) + 1) * cell)
    right = min(width, (max(col for _, col in cells) + 1) * cell)
    return {
        "box": [
            round(left / width, 4),
            round(top / height, 4),
            round(right / width, 4),
            round(bottom / height, 4),
        ],
        "share_of_difference_pct": round(weight * 100.0 / total, 2),
        "areas": label,
    }


# Section: pictures out


def to_image(array: np.ndarray) -> Image.Image:
    """Float display values as an 8 bit RGB picture, a block of rows at a time."""
    data = np.empty(array.shape, dtype=np.uint8)
    for top in range(0, array.shape[0], BLOCK_ROWS):
        rows = slice(top, top + BLOCK_ROWS)
        data[rows] = (np.clip(array[rows], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(data)


def heat_map(candidate: np.ndarray, reference: np.ndarray, evaluated: np.ndarray) -> np.ndarray:
    """The difference as colour: black where equal, bright where far apart."""
    difference = np.abs(candidate - reference).mean(axis=2)
    level = np.sqrt(np.clip(difference / HEAT_FULL, 0.0, 1.0))
    position = level * (len(HEAT_STOPS) - 1)
    low = np.floor(position).astype(np.int32).clip(0, len(HEAT_STOPS) - 2)
    blend = (position - low)[..., None]
    colour = HEAT_STOPS[low] * (1.0 - blend) + HEAT_STOPS[low + 1] * blend
    colour[~evaluated] = UNEVALUATED
    return colour.astype(np.float32)


def side_by_side(
    panels: Sequence[np.ndarray], labels: Sequence[str] | None = None, *, edge: int | None = None
) -> Image.Image:
    """Panels in a row on a dark sheet, each with its label above it."""
    pictures = [to_image(panel) for panel in panels]
    if edge is not None:
        pictures = [
            picture.resize(working_size(*picture.size, edge), Image.Resampling.LANCZOS)
            for picture in pictures
        ]
    top = LABEL_HEIGHT if labels else 0
    width = sum(picture.width for picture in pictures) + SHEET_GAP * (len(pictures) + 1)
    height = max(picture.height for picture in pictures) + top + SHEET_GAP * 2
    sheet = Image.new("RGB", (width, height), SHEET_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    x = SHEET_GAP
    for index, picture in enumerate(pictures):
        if labels:
            draw.text((x, SHEET_GAP), labels[index], fill=SHEET_TEXT, font=font)
        sheet.paste(picture, (x, SHEET_GAP + top))
        x += picture.width + SHEET_GAP
    return sheet


def jpeg_bytes(image: Image.Image, quality: int = 85) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
