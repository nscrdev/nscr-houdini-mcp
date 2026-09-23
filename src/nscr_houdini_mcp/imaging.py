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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageFont

# The long edge both sides are resized to for the whole image numbers.
WORKING_EDGE = 1024

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


class MetricsUnavailable(Exception):
    """Numbers that cannot be counted, and why."""


@dataclass
class Picture:
    """One side, in display values: RGB from 0 to 1, alpha kept apart."""

    rgb: np.ndarray
    alpha: np.ndarray | None
    colour: dict[str, Any]
    path: str

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])


# Section: reading


def read_display_file(path: str | Path) -> Picture:
    """A PNG, JPEG, TIFF or any file Pillow reads, brought to sRGB.

    An embedded ICC profile is converted from; with none, sRGB is assumed and
    the record says so rather than leaving it to be guessed. Alpha is kept
    apart and never mixed into the colour.
    """
    try:
        image = Image.open(path)
        image.load()
    except FileNotFoundError:
        raise ImageError("FILE_NOT_FOUND", "there is no image at that path") from None
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            "the file is not an image this can read",
            details={"exception": type(error).__name__},
        ) from None
    colour: dict[str, Any] = {"kind": FILE_KIND, "format": image.format, "mode": image.mode}
    icc = image.info.get("icc_profile")
    alpha = _alpha_of(image)

    if image.mode in _HIGH_BIT_GREY:
        grey = np.asarray(image, dtype=np.float32)
        top = 1.0 if image.mode == "F" else 65535.0 if grey.max(initial=0) > 255 else 255.0
        grey = np.clip(grey / top, 0.0, 1.0)
        rgb = np.repeat(grey[..., None], 3, axis=2)
        colour["profile"] = "embedded_not_applied" if icc else "assumed_srgb"
        if icc:
            colour["reason"] = "a profile is not applied to high bit grey"
        return Picture(rgb.astype(np.float32), alpha, colour, str(path))

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
            return Picture(rgb, alpha, colour, str(path))
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
    return Picture(rgb, alpha, colour, str(path))


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
    if "A" in bands:
        return np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
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
    alpha = pixels[..., 3] if channels >= 4 else None
    rgb = np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return Picture(rgb.astype(np.float32), alpha, dict(colour), str(path))


def read_mask(path: str | Path) -> np.ndarray:
    """A mask file as a float map from 0 to 1: its alpha if it has one, else its grey."""
    try:
        image = Image.open(path)
        image.load()
    except FileNotFoundError:
        raise ImageError("FILE_NOT_FOUND", "there is no mask file at that path") from None
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as error:
        raise ImageError(
            "IMAGE_UNREADABLE",
            "the mask file is not an image this can read",
            details={"exception": type(error).__name__},
        ) from None
    alpha = _alpha_of(image)
    if alpha is not None and alpha.min(initial=1.0) < 1.0:
        return alpha
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


# Section: resampling


def resize(array: np.ndarray, width: int, height: int) -> np.ndarray:
    """A float image or map at a new size, channel by channel."""
    width, height = max(1, int(width)), max(1, int(height))
    if array.shape[1] == width and array.shape[0] == height:
        return array.astype(np.float32, copy=True)
    flat = array.ndim == 2
    planes = array[..., None] if flat else array
    out = np.empty((height, width, planes.shape[2]), dtype=np.float32)
    shrinking = width < array.shape[1] or height < array.shape[0]
    method = Image.Resampling.LANCZOS if shrinking else Image.Resampling.BICUBIC
    for index in range(planes.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(planes[..., index], dtype=np.float32))
        out[..., index] = np.asarray(plane.resize((width, height), method), dtype=np.float32)
    return out[..., 0] if flat else out


def working_size(width: int, height: int, edge: int = WORKING_EDGE) -> tuple[int, int]:
    """The size with the long edge at `edge`, keeping the aspect."""
    scale = edge / max(width, height)
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
    grow = 1.0 if mode == "none" else min(MAX_UPSCALE, 1.0 / min(sx, sy))
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
    ox = round((gw - nw) / 2 + dx * gw)
    oy = round((gh - nh) / 2 + dy * gh)
    placed = resize(candidate.rgb, nw, nh)
    canvas, valid = _paste(placed, gw, gh, ox, oy)

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
            aligned.candidate = translate(aligned.candidate, found["dx"], found["dy"])
            aligned.valid = translate(aligned.valid, found["dx"], found["dy"])
    return aligned


def _bars(ox: int, oy: int, nw: int, nh: int, gw: int, gh: int) -> dict[str, int] | None:
    """Uncovered margins, when the candidate leaves any part of the frame bare."""
    bars = {
        "left": max(0, ox),
        "top": max(0, oy),
        "right": max(0, gw - (ox + nw)),
        "bottom": max(0, gh - (oy + nh)),
    }
    return bars if any(bars.values()) else None


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
    delta = candidate[evaluated] - reference[evaluated]
    absolute = np.abs(delta)
    squared = delta.astype(np.float64) ** 2
    mae = absolute.mean(axis=0)
    rmse = np.sqrt(squared.mean(axis=0))
    overall_rmse = float(math.sqrt(squared.mean()))
    over = np.abs(candidate - reference).max(axis=2) > tolerance
    over &= evaluated
    return {
        "mae": _channels(mae, float(absolute.mean())),
        "rmse": _channels(rmse, overall_rmse),
        "psnr_db": _psnr(overall_rmse),
        "diff_area_pct": round(float(over.sum()) * 100.0 / count, 4),
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
    """Float display values as an 8 bit RGB picture."""
    data = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
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
