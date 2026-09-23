"""Turning what a tool hands back into something JSON can carry.

A tool may return Houdini objects, arrays and raw bytes. None of that is JSON,
and a serialiser that raises in the middle of a reply loses the work the call
already did. So the reply is converted first, by walking it once:

- a vector or a matrix becomes a flat list of numbers
- a node or a parameter becomes its path, which is what a caller can act on
- an array becomes a list, capped, and a scalar with a shape its number
- a set becomes a list of its members, sorted where they compare
- bytes become base64 text, capped
- text becomes UTF-8 safe text, capped, with a lone surrogate escaped
- anything else becomes `repr()`, cut short

A collection is only read as far as its cap, so a range of a billion numbers
costs as much as one of a thousand.

Nothing here raises, and that is load bearing rather than tidy: this runs
while the reply is being built, so anything that got through would leave the
caller with no answer at all. A deleted node raises from its own `repr`, a
mapping can raise while it is being read, so every one of those is caught and
the value becomes the name of its type.

There is a cap on everything: how many items of a list, how many keys of a
mapping, how many bytes, how long a string, and how much in total. A float
that is not a number or is infinite becomes text and counts as a change. When something was cut or
replaced, the conversion says so, and the reply carries `lossy: true` with the
places it happened, so a caller never has to guess whether it got the whole
answer.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any

# How many numbers of an array are carried. Big arrays belong in a file.
MAX_ITEMS = 1024

# How many keys of one mapping are carried, and how many values in the whole
# reply. A scene can hand back a dictionary with a key per node.
MAX_KEYS = 256
MAX_VALUES = 20000

# How many bytes are carried inline, before base64 grows them by a third.
MAX_BYTES = 64 * 1024

# How long one string may be. Text longer than this belongs in a file.
MAX_CHARS = 1_000_000

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
    chars: int = MAX_CHARS


def convert(
    value: Any,
    *,
    max_items: int = MAX_ITEMS,
    max_keys: int = MAX_KEYS,
    max_bytes: int = MAX_BYTES,
    max_depth: int = MAX_DEPTH,
    max_values: int = MAX_VALUES,
    max_chars: int = MAX_CHARS,
    root: str = "data",
) -> Converted:
    """Convert one reply value, reporting anything that was cut.

    `root` is what the places in `cut` start with.
    """
    result = Converted(None, left=max_values)
    caps = Caps(
        items=max_items,
        keys=max_keys,
        byte_count=max_bytes,
        depth=max_depth,
        chars=max_chars,
    )
    result.value = _walk(value, root, result, caps, max_depth)
    return result


def clean_text(text: str) -> tuple[str, bool]:
    """Text that UTF-8 can carry, and whether anything had to change.

    A lone surrogate, which a file name read with `surrogateescape` or a
    stray escape in a string can hold, cannot be written as UTF-8. It is
    written as its escape instead, so the text still says what was there.
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8"), True
    return text, False


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
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _text(value, where, result, caps.chars)
    if isinstance(value, float):
        if value == value and abs(value) != float("inf"):
            return value
        # Not every float is JSON. Infinities and nan become text, and the
        # reply says a value was changed to get there.
        _cut(where, result)
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return _bytes(bytes(value), where, result, caps.byte_count)
    if isinstance(value, Mapping):
        return _mapping(value, where, result, caps, depth)

    array = _as_array(value, caps.items)
    if array is not _UNKNOWN:
        if isinstance(array, list):
            return _sequence(array, where, result, caps, depth)
        # A number with a shape of its own, such as a numpy scalar or an
        # array with no dimensions, is the number.
        return _walk(array, where, result, caps, depth - 1)
    if isinstance(value, Sequence):
        listed = _listed(value, caps.items)
        if listed is not None:
            return _sequence(listed, where, result, caps, depth)
    if isinstance(value, (set, frozenset)):
        listed = _listed(value, caps.items)
        if listed is not None:
            try:
                listed = sorted(listed)
            except Exception:  # noqa: BLE001 - members that do not compare keep their order
                pass
            return _sequence(listed, where, result, caps, depth)

    houdini = _houdini_value(value)
    if houdini is not _UNKNOWN:
        return _walk(houdini, where, result, caps, depth - 1)

    _cut(where, result)
    return _describe(value)


def _text(value: str, where: str, result: Converted, most: int) -> str:
    if len(value) > most:
        _cut(where, result)
        value = value[:most]
    text, changed = clean_text(value)
    if changed:
        _cut(where, result)
    return text


def _mapping(value: Mapping[Any, Any], where: str, result: Converted, caps: Caps, depth: int):
    """One mapping, capped in breadth, read no further than the cap."""
    try:
        items = list(islice(value.items(), caps.keys + 1))
    except Exception:  # noqa: BLE001 - a mapping we cannot read is described instead
        _cut(where, result)
        return _describe(value)
    kept = items[: caps.keys]
    if len(items) > len(kept):
        _cut(where, result)
    converted: dict[str, Any] = {}
    for key, item in kept:
        name = key if isinstance(key, str) else _describe(key)
        name, changed = clean_text(name)
        if changed:
            _cut(where, result)
        converted[name] = _walk(item, f"{where}.{name}", result, caps, depth - 1)
    return converted


def _sequence(items: list[Any], where: str, result: Converted, caps: Caps, depth: int) -> list[Any]:
    kept = items[: caps.items]
    if len(items) > len(kept):
        _cut(where, result)
    return [
        _walk(item, f"{where}[{index}]", result, caps, depth - 1) for index, item in enumerate(kept)
    ]


def _listed(value: Any, most: int) -> list[Any] | None:
    """The first items of a collection, one past the cap so a cut shows, or
    nothing when reading it raised. Nothing past that is ever asked for."""
    try:
        return list(islice(iter(value), most + 1))
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


def _as_array(value: Any, most: int) -> Any:
    """A numeric array as a plain list, a scalar with a shape as the scalar,
    or `_UNKNOWN` when this is neither.

    Read by duck typing so nothing here imports a package Houdini may not have
    loaded. An array is anything with a shape and a `tolist`. A long one is
    sliced before it is listed, so only what can be carried is copied.
    """
    if not hasattr(value, "tolist") or not hasattr(value, "shape"):
        return _UNKNOWN
    try:
        shape = tuple(value.shape)
    except Exception:  # noqa: BLE001 - a shape we cannot read is no shape
        return _UNKNOWN
    head = value
    sized = shape and all(isinstance(size, int) for size in shape)
    if sized and any(size > most for size in shape):
        # Every axis is cut, not only the first: a wide inner axis would
        # otherwise be copied whole before the cap could see it.
        try:
            head = value[tuple(slice(0, most + 1) for _ in shape)]
        except Exception:  # noqa: BLE001 - one that will not slice is listed whole
            head = value
    try:
        listed = head.tolist()
    except Exception:  # noqa: BLE001 - a value we cannot read is handled below
        return _UNKNOWN
    if not shape:
        return listed
    return listed if isinstance(listed, list) else [listed]


_UNKNOWN = object()

# Houdini values that are a row of numbers read with `tuple()`. A matrix is
# read with `asTuple()`, which the vectors do not have in Houdini 22.
_TUPLE_TYPES = ("Vector2", "Vector3", "Vector4", "Quaternion")


def _houdini_value(value: Any) -> Any:
    """What one Houdini object becomes, or `_UNKNOWN` when it is not one.

    Read by class name, because `hou` is not importable where this module is
    also used. Any node, a parameter, a network box or anything else with a
    path is its path; a parameter tuple, which has none, is its node's path
    and its name; a node type of any context is its name.
    """
    kind = type(value)
    if kind.__module__.split(".")[0] != "hou":
        return _UNKNOWN
    name = kind.__name__
    try:
        if name == "ParmTuple":
            return f"{value.node().path()}/{value.name()}"
        if name.endswith("NodeType"):
            return str(value.name())
        if callable(getattr(value, "path", None)):
            return str(value.path())
        if name == "Color":
            return list(value.rgb())
        if name == "BoundingBox":
            return {"min": list(value.minvec()), "max": list(value.maxvec())}
        if callable(getattr(value, "asTuple", None)):
            return list(value.asTuple())
        if name in _TUPLE_TYPES:
            return list(tuple(value))
    except Exception:  # noqa: BLE001 - a deleted node still has to answer
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
