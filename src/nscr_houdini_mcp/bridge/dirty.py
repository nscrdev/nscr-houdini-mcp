"""Whether the scene has changes that are not on disk, as the bridge saw it.

A session with a user interface answers this itself. A headless one says it
has unsaved changes whatever it has, straight after a save included, so the
bridge keeps its own mark instead, from what it can see happen:

- `clean` after a save the bridge made, a save any code made while no call
  of ours was running, a load, or a new scene.
- `dirty` after any call that changes the scene, and after a merge. A call
  that may change it and left the undo stack as it was, such as Python code
  that only read, leaves the mark as it was. Code that edits with undo turned
  off is not seen that way.
- `unknown` at start, after a call that failed part way, and after code that
  saved or replaced the scene itself: it may have changed things after that
  and nothing here can tell.

Edits made from outside a call, such as a script Houdini ran on its own, are
not seen. The mark says what the bridge knows and no more.

This module never imports `hou`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

CLEAN = "clean"
DIRTY = "dirty"
UNKNOWN = "unknown"

# The bridge's own tools that leave the scene as it is on disk when they work.
SAVE_TOOLS = frozenset({"scene.save", "scene.save_as"})
LOAD_TOOLS = frozenset({"scene.open"})

# Scene events, as the identity passes them on.
SAVED = "saved"
LOADED = "loaded"
CLEARED = "cleared"
MERGED = "merged"


class DirtyMarker:
    """The mark, moved by the calls the bridge runs and the scene's own events."""

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._state = UNKNOWN
        self._why = "nothing has been seen since the session started"
        self._at: float | None = None
        # What a call that is running has seen of the scene's own events.
        self._events: list[str] | None = None

    @property
    def state_name(self) -> str:
        with self._lock:
            return self._state

    @property
    def unsaved(self) -> bool | None:
        """True for dirty, False for clean, nothing when it is not known."""
        with self._lock:
            return {CLEAN: False, DIRTY: True}.get(self._state)

    def state(self) -> dict[str, Any]:
        with self._lock:
            return {"state": self._state, "why": self._why, "at": self._at}

    def began(self, tool: str) -> None:
        """A call that may change the scene is about to run."""
        with self._lock:
            self._events = []

    def ended(self, tool: str, *, ok: bool, changed: bool | None = None) -> None:
        """A call that may have changed the scene has finished.

        `changed` is whether the undo stack moved during the call, when that
        could be read: false leaves the mark alone.
        """
        with self._lock:
            seen = self._events or []
            self._events = None
            if tool in SAVE_TOOLS:
                if ok:
                    self._set(CLEAN, f"saved by {tool}")
                return
            if tool in LOAD_TOOLS:
                if ok:
                    self._set(CLEAN, "loaded by scene.open")
                else:
                    self._set(UNKNOWN, "a load that did not finish")
                return
            if seen:
                self._set(UNKNOWN, f"{tool} {seen[-1]} the scene and may have changed it after")
            elif changed is False:
                return
            elif ok:
                self._set(DIRTY, f"changed by {tool}")
            else:
                self._set(UNKNOWN, f"{tool} failed part way")

    def event(self, what: str) -> None:
        """One of the scene's own events: saved, loaded, cleared or merged."""
        with self._lock:
            if self._events is not None:
                if what != MERGED:
                    self._events.append(what)
                return
            if what == MERGED:
                self._set(DIRTY, "a file was merged in")
            elif what in (SAVED, LOADED, CLEARED):
                self._set(CLEAN, {SAVED: "saved", LOADED: "loaded", CLEARED: "new scene"}[what])

    def _set(self, state: str, why: str) -> None:
        self._state = state
        self._why = why
        self._at = self._clock()
