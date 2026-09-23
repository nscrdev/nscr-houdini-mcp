"""One undo entry per mutating call, and a rollback when the call fails.

What this covers, stated plainly: the artist's undo history and the graph
edits Houdini's own undo covers, which is creating, deleting, renaming and
wiring nodes and setting parameters and flags. It is not a transaction. Files
written during the call stay written, caches stay built, processes that were
started keep running, and callbacks that fired during the edit have already
fired. A reply never claims more than that.

A failed call is rolled back only when the group actually recorded something.
A group whose body raises before it touches the graph adds no undo entry, so
an undo there would reverse the artist's last edit instead of this call's. The
reply says which happened.

In a graphical session this runs on the main thread. Off it, the group is
silently not a group: ten creates inside one recorded ten entries.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Outcome:
    """What happened inside one undo group."""

    value: Any = None
    error: BaseException | None = None
    recorded: bool = False
    rolled_back: bool = False
    # Whether the undo stack could be read on both sides, so `recorded` is a
    # finding and not a default.
    counted: bool = False


def run_in_undo_group(function: Callable[[], Any], *, label: str, hou: Any) -> Outcome:
    """Run one callable inside one undo group, rolling back a failure.

    The label is what the bridge asked for. Houdini's own history may show the
    name of the last operation instead, so a reply echoes the label and does
    not promise what the undo menu reads.
    """
    before = _entries(hou)
    error: BaseException | None = None
    value: Any = None
    try:
        with hou.undos.group(label):
            try:
                value = function()
            except BaseException as raised:  # noqa: BLE001 - the group closes either way
                error = raised
    except BaseException as raised:  # noqa: BLE001 - the group itself refused
        if error is None:
            error = raised

    after = _entries(hou)
    # Without the stack from both sides there is no way to tell whether this
    # call put anything on it, and an undo on a guess would reverse somebody
    # else's edit. So it counts as nothing recorded. The stack is compared
    # whole rather than by length, because a stack at its limit drops its
    # oldest entry as it takes a new one and keeps the same length. A full
    # stack whose entries all read the same is the one case this cannot see.
    counted = before is not None and after is not None
    recorded = counted and after != before
    if error is None:
        return Outcome(value=value, recorded=recorded, counted=counted)

    rolled_back = False
    if recorded:
        try:
            hou.undos.performUndo()
            rolled_back = True
        except Exception:  # noqa: BLE001 - a failed rollback is reported, not raised
            rolled_back = False
    return Outcome(error=error, recorded=recorded, rolled_back=rolled_back, counted=counted)


def _entries(hou: Any) -> tuple[str, ...] | None:
    """The undo entries there are, or nothing when this build will not say."""
    try:
        return tuple(str(label) for label in hou.undos.undoLabels())
    except Exception:  # noqa: BLE001 - without the stack there is nothing to compare
        return None
