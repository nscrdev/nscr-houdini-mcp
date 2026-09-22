"""Turning what a tool hands back into something JSON can carry.

A tool may return Houdini objects, arrays and raw bytes. None of that is JSON,
and a serialiser that raises in the middle of a reply loses the work the call
already did. So the reply is converted first, by walking it once:

- a vector or a matrix becomes a flat list of numbers
- a node or a parameter becomes its path, which is what a caller can act on
- an array becomes a list, capped
- bytes become base64 text, capped
- anything else becomes `repr()`, cut short

Nothing here raises, and that is load bearing rather than tidy: this runs
while the reply is being built, so anything that got through would leave the
caller with no answer at all. A deleted node raises from its own `repr`, a
mapping can raise while it is being read, so every one of those is caught and
the value becomes the name of its type.

There is a cap on everything: how many items of a list, how many keys of a
mapping, how many bytes, and how much in total. When something was cut or
replaced, the conversion says so, and the reply carries `lossy: true` with the
places it happened, so a caller never has to guess whether it got the whole
answer.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# How many numbers of an array are carried. Big arrays belong in a file.
MAX_ITEMS = 1024

# How many keys of one mapping are carried, and how many values in the whole
# reply. A scene can hand back a dictionary with a key per node.
MAX_KEYS = 256
MAX_VALUES = 20000

# How many bytes are carried inline, before base64 grows them by a third.
MAX_BYTES = 64 * 1024

# How long a `repr()` of something unknown may be.
MAX_REPR = 200

# How deep the walk goes before it stops describing and starts summarising.
MAX_DEPTH = 24


@dataclass
class Converted:
    """One converted reply, and what had to be cut to fit."""

    value: Any
    lossy: bool = False
    cut: list[str] = field(default_factory=list)
    left: int = MAX_VALUES


@dataclass(frozen=True)
class Caps:
    """What one conversion is allowed to carry."""

    items: int = MAX_ITEMS
    keys: int = MAX_KEYS
    byte_count: int = MAX_BYTES
    depth: int = MAX_DEPTH


def convert(
    value: Any,
    *,
    max_items: int = MAX_ITEMS,
    max_keys: int = MAX_KEYS,
    max_bytes: int = MAX_BYTES,
    max_depth: int = MAX_DEPTH,
    max_values: int = MAX_VALUES,
) -> Converted:
    """Convert one reply value, reporting anything that was cut."""
    result = Converted(None, left=max_values)
    caps = Caps(items=max_items, keys=max_keys, byte_count=max_bytes, depth=max_depth)
    result.value = _walk(value, "data", result, caps, max_depth)
    return result


def _cut(where: str, result: Converted) -> None:
    result.lossy = True
    if where not in result.cut:
        result.cut.append(where)
    del result.cut[16:]


def _walk(value: Any, where: str, result: Converted, caps: Caps, depth: int) -> Any:
    if depth <= 0 or result.left <= 0:
        _cut(where, result)
        return _describe(value)
    result.left -= 1
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # Not every float is JSON. Infinities and nan become text rather than
        # an invalid document.
        return value if value == value and abs(value) != float("inf") else str(value)
    if isinstance(value, (bytes, bytearray)):
        return _bytes(bytes(value), where, result, caps.byte_count)
    if isinstance(value, Mapping):
        return _mapping(value, where, result, caps, depth)

    array = _as_array(value)
    if array is not None:
        return _sequence(array, where, result, caps, depth)
    if isinstance(value, Sequence):
        listed = _listed(value)
        if listed is not None:
            return _sequence(listed, where, result, caps, depth)
    if isinstance(value, (set, frozenset)):
        listed = _listed(value)
        if listed is not None:
            return _sequence(sorted(_describe(item) for item in listed), where, result, caps, depth)

    houdini = _houdini_value(value)
    if houdini is not _UNKNOWN:
        return _walk(houdini, where, result, caps, depth - 1)

    _cut(where, result)
    return _describe(value)


def _mapping(value: Mapping[Any, Any], where: str, result: Converted, caps: Caps, depth: int):
    """One mapping, capped in breadth, reading it guarded from end to end."""
    try:
        items = list(value.items())
    except Exception:  # noqa: BLE001 - a mapping we cannot read is described instead
        _cut(where, result)
        return _describe(value)
    kept = items[: caps.keys]
    if len(items) > len(kept):
        _cut(where, result)
    converted: dict[str, Any] = {}
    for key, item in kept:
        name = _describe(key) if not isinstance(key, str) else key
        converted[name] = _walk(item, f"{where}.{name}", result, caps, depth - 1)
    return converted


def _sequence(items: list[Any], where: str, result: Converted, caps: Caps, depth: int) -> list[Any]:
    kept = items[: caps.items]
    if len(items) > len(kept):
        _cut(where, result)
    return [
        _walk(item, f"{where}[{index}]", result, caps, depth - 1) for index, item in enumerate(kept)
    ]


def _listed(value: Any) -> list[Any] | None:
    """A sequence as a plain list, or nothing when reading it raised."""
    try:
        return list(value)
    except Exception:  # noqa: BLE001 - a value we cannot read is described instead
        return None


def _bytes(raw: bytes, where: str, result: Converted, max_bytes: int) -> dict[str, Any]:
    kept = raw[:max_bytes]
    if len(raw) > max_bytes:
        _cut(where, result)
    return {
        "encoding": "base64",
        "bytes": len(raw),
        "data": base64.b64encode(kept).decode("ascii"),
    }


def _as_array(value: Any) -> list[Any] | None:
    """A numeric array as a plain list, when this is one.

    Read by duck typing so nothing here imports a package Houdini may not have
    loaded. An array is anything with a shape and a `tolist`.
    """
    if not hasattr(value, "tolist") or not hasattr(value, "shape"):
        return None
    try:
        listed = value.tolist()
    except Exception:  # noqa: BLE001 - a value we cannot read is handled below
        return None
    return listed if isinstance(listed, list) else [listed]


_UNKNOWN = object()

# Houdini objects the bridge knows how to say. The reader is chosen by class
# name, because `hou` is not importable where this module is also used.
_BY_NAME = {
    "Node": "path",
    "OpNode": "path",
    "SopNode": "path",
    "ObjNode": "path",
    "DopNode": "path",
    "LopNode": "path",
    "RopNode": "path",
    "CopNode": "path",
    "ChopNode": "path",
    "ShopNode": "path",
    "VopNode": "path",
    "TopNode": "path",
    "Parm": "path",
    "ParmTuple": "path",
    "NodeType": "name",
}

_TUPLE_TYPES = (
    "Vector2",
    "Vector3",
    "Vector4",
    "Quaternion",
    "Color",
    "Matrix2",
    "Matrix3",
    "Matrix4",
    "BoundingBox",
)


def _houdini_value(value: Any) -> Any:
    """What one Houdini object becomes, or `_UNKNOWN` when it is not one."""
    kind = type(value)
    if kind.__module__.split(".")[0] != "hou":
        return _UNKNOWN
    name = kind.__name__
    reader = _BY_NAME.get(name)
    if reader is not None:
        try:
            return str(getattr(value, reader)())
        except Exception:  # noqa: BLE001 - a deleted node still has to answer
            return _UNKNOWN
    if name in _TUPLE_TYPES or hasattr(value, "asTuple"):
        try:
            return list(value.asTuple())
        except Exception:  # noqa: BLE001 - fall through to the repr
            return _UNKNOWN
    return _UNKNOWN


def _describe(value: Any) -> str:
    """Short text for one value, even when the value refuses to describe itself.

    A node that has been deleted raises from its own `repr`, and this runs
    while a reply is being built, so the name of the type is the answer there.
    """
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - the type name is always available
        return f"<{type(value).__name__}>"
    return text if len(text) <= MAX_REPR else text[: MAX_REPR - 3] + "..."
