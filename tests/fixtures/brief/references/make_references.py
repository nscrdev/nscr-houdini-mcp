#!/usr/bin/env python3
"""Draw the reference images for the fixed modeling brief (BRIEF.md).

Everything here is synthetic: the object is described by the numbers below,
projected through the brief's three camera presets and drawn with Pillow. No
Houdini, no outside images. Run it again after changing a number and commit
the images it writes; BRIEF.md states the same numbers and must change with it.

Usage:
    python make_references.py            # writes next to this file
    python make_references.py --print    # also prints cameras and the crop box
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent

WIDTH, HEIGHT = 960, 540
SUPERSAMPLE = 4  # drawn at 3840x2160, then averaged down
FOCAL = 50.0  # mm
APERTURE = 41.4214  # mm, horizontal, the Houdini camera default

LIGHT = (-0.35, 0.75, 0.55)  # direction toward the light, world space
BACKGROUND = (236, 236, 233)
HOUSING = (128, 132, 138)
KEY = (214, 142, 74)
INK = (20, 20, 20)
PAPER = (255, 255, 255)


@dataclass(frozen=True)
class Design:
    """The brief's controls, in scene units (metres). Defaults are the brief."""

    length: float = 0.32  # housing, along X
    depth: float = 0.09  # housing, along Z
    height: float = 0.035  # housing, along Y, sitting on Y = 0
    key_count: int = 6
    end_margin: float = 0.03  # from each end of the housing to the key row
    key_size: float = 0.028  # square keys, X and Z
    key_height: float = 0.012  # above the housing top
    chamfer: float = 0.002  # on the four top edges of every key

    @property
    def pitch(self) -> float:
        return (self.length - 2 * self.end_margin) / self.key_count

    def key_centers(self) -> list[float]:
        start = -self.length / 2 + self.end_margin
        return [start + self.pitch * (i + 0.5) for i in range(self.key_count)]


@dataclass(frozen=True)
class Camera:
    name: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]

    def rotation(self) -> tuple[float, float]:
        """Houdini rx, ry in degrees (rotate order Rx Ry Rz, rz = 0)."""
        f = _norm(_sub(self.look_at, self.position))
        flat = math.hypot(f[0], f[2])
        # Straight down has no heading: call it zero. Adding 0.0 clears a -0.0.
        ry = 0.0 if flat < 1e-9 else math.degrees(math.atan2(-f[0], -f[2])) + 0.0
        rx = math.degrees(math.atan2(f[1], flat))
        return rx, ry

    def basis(self) -> tuple[tuple[float, ...], ...]:
        rx, ry = (math.radians(a) for a in self.rotation())
        cx, sx, cy, sy = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry)
        right = (cy, 0.0, -sy)
        up = (sy * sx, cx, cy * sx)
        back = (sy * cx, -sx, cy * cx)
        return right, up, back


CAMERAS = (
    Camera("front", (0.0, 0.06, 0.70), (0.0, 0.03, 0.0)),
    Camera("top", (0.0, 0.75, 0.0), (0.0, 0.0, 0.0)),
    Camera("three_quarter", (0.32, 0.24, 0.40), (0.0, 0.02, 0.0)),
)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    length = math.sqrt(_dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length)


def _newell(loop):
    n = [0.0, 0.0, 0.0]
    for i, p in enumerate(loop):
        q = loop[(i + 1) % len(loop)]
        n[0] += (p[1] - q[1]) * (p[2] + q[2])
        n[1] += (p[2] - q[2]) * (p[0] + q[0])
        n[2] += (p[0] - q[0]) * (p[1] + q[1])
    return _norm(tuple(n))


def _solid(loops):
    """Faces of a convex solid, each with an outward normal."""
    points = [p for loop in loops for p in loop]
    center = tuple(sum(c) / len(points) for c in zip(*points, strict=True))
    faces = []
    for loop in loops:
        normal = _newell(loop)
        if _dot(normal, _sub(loop[0], center)) < 0:
            loop, normal = loop[::-1], tuple(-c for c in normal)
        faces.append((loop, normal))
    return center, faces


def _rect(x0, x1, z0, z1, y):
    return [(x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)]


def _prism(lo, hi):
    """A box as bottom and top rectangles plus four walls."""
    return [lo, hi] + [[lo[i], lo[(i + 1) % 4], hi[(i + 1) % 4], hi[i]] for i in range(4)]


def solids(d: Design):
    h = d.depth / 2
    housing = _solid(
        _prism(
            _rect(-d.length / 2, d.length / 2, -h, h, 0.0),
            _rect(-d.length / 2, d.length / 2, -h, h, d.height),
        )
    )
    out = [housing]
    k, c = d.key_size / 2, d.chamfer
    y0, y1 = d.height, d.height + d.key_height
    for x in d.key_centers():
        bottom = _rect(x - k, x + k, -k, k, y0)
        shoulder = _rect(x - k, x + k, -k, k, y1 - c)
        top = _rect(x - k + c, x + k - c, -k + c, k - c, y1)
        # Bottom and four walls up to the shoulder, then four sloped chamfer
        # faces from the shoulder in to the smaller top, then the top.
        bottom_face, _, *walls = _prism(bottom, shoulder)
        slopes = [[shoulder[i], shoulder[(i + 1) % 4], top[(i + 1) % 4], top[i]] for i in range(4)]
        loops = [bottom_face, *walls, *slopes, top]
        out.append(_solid(loops))
    return out


def project(cam: Camera, p, scale: int):
    right, up, back = cam.basis()
    v = _sub(p, cam.position)
    depth = -_dot(v, back)
    k = (FOCAL / APERTURE) * WIDTH * scale / depth
    return (WIDTH * scale / 2 + _dot(v, right) * k, HEIGHT * scale / 2 - _dot(v, up) * k)


def _shade(color, normal):
    lit = 0.38 + 0.62 * max(0.0, _dot(normal, _norm(LIGHT)))
    return tuple(min(255, round(ch * lit)) for ch in color)


def draw(d: Design, cam: Camera, style: str) -> Image.Image:
    """One view at SUPERSAMPLE times the output size. style: line, shaded, mask."""
    s = SUPERSAMPLE
    mode, bg = ("L", 0) if style == "mask" else ("RGB", PAPER if style == "line" else BACKGROUND)
    image = Image.new(mode, (WIDTH * s, HEIGHT * s), bg)
    pen = ImageDraw.Draw(image)
    parts = solids(d)
    housing, keys = parts[0], parts[1:]
    # Every camera sits above the housing top, so keys always cover the
    # housing, and a row of equal convex keys sorts correctly far to near.
    keys.sort(key=lambda part: -math.dist(part[0], cam.position))
    for index, (_, faces) in enumerate([housing, *keys]):
        base = HOUSING if index == 0 else KEY
        for loop, normal in faces:
            if _dot(normal, _sub(cam.position, loop[0])) <= 0:
                continue
            pts = [project(cam, p, s) for p in loop]
            if style == "mask":
                pen.polygon(pts, fill=255)
            elif style == "shaded":
                pen.polygon(pts, fill=_shade(base, normal))
            else:
                pen.polygon(pts, fill=PAPER)
                pen.line([*pts, pts[0]], fill=INK, width=2 * s, joint="curve")
    return image


def finish(image: Image.Image, style: str) -> Image.Image:
    small = image.resize((WIDTH, HEIGHT), Image.Resampling.BOX)
    if style == "mask":
        small = small.point(lambda v: 255 if v >= 128 else 0)
    return small


def detail_box(d: Design, cam: Camera) -> tuple[int, int, int, int]:
    """16:9 box at the supersampled size around the key nearest the camera."""
    s = SUPERSAMPLE
    near = min(solids(d)[1:], key=lambda part: math.dist(part[0], cam.position))
    pts = [project(cam, p, s) for loop, _ in near[1] for p in loop]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    w = max(max(xs) - min(xs), (max(ys) - min(ys)) * 16 / 9) * 1.5
    w = 16 * math.ceil(w / 16)
    h = w * 9 // 16
    x0, y0 = round(cx - w / 2), round(cy - h / 2)
    return x0, y0, x0 + w, y0 + h


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--print", action="store_true", help="print cameras and the crop box")
    args = parser.parse_args()

    design = Design()
    for cam in CAMERAS:
        for style in ("line", "shaded", "mask"):
            finish(draw(design, cam, style), style).save(
                OUT / f"{cam.name}_{style}.png", optimize=True
            )

    three_q = CAMERAS[2]
    box = detail_box(design, three_q)
    for style in ("line", "shaded"):
        draw(design, three_q, style).crop(box).save(
            OUT / f"detail_chamfer_{style}.png", optimize=True
        )

    variants = OUT / "variants"
    variants.mkdir(exist_ok=True)
    step1 = replace(design, key_count=5)
    step2 = replace(step1, end_margin=0.06)
    step3 = replace(step2, depth=0.12, height=0.025)
    for name, variant in (
        ("count_8", replace(design, key_count=8)),
        ("handoff_1_count", step1),
        ("handoff_2_spacing", step2),
        ("handoff_3_proportion", step3),
    ):
        image = finish(draw(variant, three_q, "shaded"), "shaded")
        image.save(variants / f"{name}_three_quarter_shaded.png", optimize=True)

    if args.print:
        for cam in CAMERAS:
            rx, ry = cam.rotation()
            print(f"{cam.name}: t={cam.position} look_at={cam.look_at} r=({rx:.3f}, {ry:.3f}, 0)")
        full = (WIDTH * SUPERSAMPLE, HEIGHT * SUPERSAMPLE)
        norm = [round(v / full[i % 2], 4) for i, v in enumerate(box)]
        print(f"detail box at {full[0]}x{full[1]}: {box} normalized {norm}")
        print(f"pitch {design.pitch:.5f} gap {design.pitch - design.key_size:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
