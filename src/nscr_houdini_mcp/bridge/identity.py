"""Who this session is, and which scene it is holding.

Two handles and one counter, and the rules that keep them honest.

- `session_id` is random, minted once when the bridge starts, never reused and
  never changed. A call that means to continue earlier work carries it.
- `alias` is the readable name a person addresses the session by. A session
  with a scene takes its name from the scene file, a worker takes a number.
  The alias is settled at start and never changes afterwards: renaming a
  session under a caller that is holding the old name is worse than a name
  that has gone out of date. When the scene file changes so the name no longer
  matches it, every reply says so instead.
- `scene_epoch` counts how many times this process has thrown its scene away.
  Opening a file, starting a new scene and loading the same file again all
  replace the scene, and every node path a caller was holding goes with it. So
  a call that carries an epoch older than this one is refused before it runs.

How the counter is kept. Houdini reports scene changes through the hip file
event callbacks, and in this build a load reports four of them: `BeforeLoad`,
then `BeforeClear` and `AfterClear` for the scene it is dropping, then
`AfterLoad`. Counting clears and loads separately would count one load twice,
so a load holds the clear inside it and the epoch moves once.

A merge is left alone on purpose. It adds to the scene rather than replacing
it, so every path a caller is holding still means what it meant.

The scene summary handed back with a refusal is taken when the epoch moves,
which is on the main thread and right after the new scene has settled. Nothing
on the refusal path reads the object model, so a stale epoch is answered while
the session is busy with something else.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nscr_houdini_mcp.bridge import host

# Alias patterns. A worker is addressed by number, a session with a scene by
# the scene name, and a scene with no name yet by a plain word.
WORKER_ALIAS = "w{n}"
UNTITLED_ALIAS = "scene-{n}"

ALIAS_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# The networks a scene summary counts, in the order a person reads them.
CONTEXTS = ("/obj", "/out", "/stage", "/mat")

# Why the epoch moved, as the reply says it.
LOADED = "loaded"
CLEARED = "cleared"


def hip_stem(hip_path: str | None) -> str:
    """The scene file name with nothing in it that an alias cannot hold."""
    if not hip_path:
        return ""
    return ALIAS_SAFE.sub("-", Path(str(hip_path)).stem).strip("-")


def alias_template(kind: str, hip_path: str | None) -> str:
    """The pattern this session takes its default name from.

    A worker is `w1`, `w2` and so on, whatever it has open: it is addressed by
    number and its scene is the caller's business. A session with a user
    interface is named after the scene file, so two Houdinis on the same file
    come out as `shot-1` and `shot-2` rather than fighting over one name.
    """
    if kind != host.GUI:
        return WORKER_ALIAS
    stem = hip_stem(hip_path)
    return f"{stem}-{{n}}" if stem else UNTITLED_ALIAS


class Identity:
    """The session handles, the scene counter and the scene summary.

    Every reader takes the lock, so a call answering on a handler thread and a
    scene event arriving on the main thread cannot see half an update.
    """

    def __init__(
        self,
        *,
        session_id: str,
        kind: str = host.HYTHON,
        alias: str | None = None,
        hip_path: str | None = None,
        scene_epoch: int = 0,
        tracks_hip: bool = False,
        hou: Any | None = None,
        on_change: Callable[[int, str | None], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.kind = kind
        self._lock = threading.Lock()
        self._alias = alias
        # The scene name the alias was built from. It is what a later name is
        # compared against, so the comparison does not depend on the alias
        # still looking like the pattern that made it.
        self._alias_stem = hip_stem(hip_path) if tracks_hip else None
        self._tracks_hip = tracks_hip
        self._hip_path = hip_path
        self._epoch = scene_epoch
        self._reason: str | None = None
        self._changed_at: float | None = None
        self._nodes: dict[str, int] = {}
        self._counted_at: float | None = None
        self._hou = hou if hou is not None else host.houdini()
        self._on_change = on_change
        self._log = log or (lambda text: None)
        self._loading = False
        self._remove_watch: Callable[[], None] | None = None

    # Section: handles

    @property
    def alias(self) -> str | None:
        with self._lock:
            return self._alias

    def settle_alias(self, alias: str) -> None:
        """Record the name the store handed out. Called once, at start."""
        with self._lock:
            self._alias = alias

    @property
    def scene_epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def hip_path(self) -> str | None:
        with self._lock:
            return self._hip_path

    def trace(self) -> dict[str, Any]:
        """What every reply carries about who answered and which scene it was."""
        with self._lock:
            trace: dict[str, Any] = {
                "session_id": self.session_id,
                "alias": self._alias,
                "scene_epoch": self._epoch,
            }
            drift = self._drift()
        if drift is not None:
            trace["warnings"] = [drift]
        return trace

    def drift(self) -> dict[str, Any] | None:
        """The warning when the alias no longer matches the scene file."""
        with self._lock:
            return self._drift()

    def _drift(self) -> dict[str, Any] | None:
        if not self._tracks_hip:
            return None
        current = hip_stem(self._hip_path)
        if current == (self._alias_stem or ""):
            return None
        return {
            "code": "ALIAS_DRIFT",
            "message": "this session is named after the scene file it started with",
            "alias": self._alias,
            "named_after": self._alias_stem or None,
            "hip_stem": current or None,
            "hint": "address this session by its id, or rename it",
        }

    # Section: the scene

    def scene(self) -> dict[str, Any]:
        """The summary a refused call is answered with. Reads nothing live."""
        with self._lock:
            return {
                "session_id": self.session_id,
                "alias": self._alias,
                "scene_epoch": self._epoch,
                "hip_path": self._hip_path,
                "nodes": dict(self._nodes),
                "changed": self._reason,
                "changed_at": self._changed_at,
                "counted_at": self._counted_at,
            }

    def refresh(self) -> dict[str, int]:
        """Count the top level networks again, from wherever this is called.

        It touches the object model, so it runs on the main thread: at start,
        and from the scene event once the new scene has settled.
        """
        counts = self._count()
        path = self._read_hip_path()
        with self._lock:
            self._nodes = counts
            self._counted_at = time.time()
            if path is not None:
                self._hip_path = path
        return counts

    def _count(self) -> dict[str, int]:
        hou = self._hou
        if hou is None:
            return {}
        counts: dict[str, int] = {}
        for path in CONTEXTS:
            try:
                node = hou.node(path)
                if node is not None:
                    counts[path] = len(node.children())
            except Exception:  # noqa: BLE001 - a count we cannot take is one we do not give
                continue
        return counts

    def _read_hip_path(self) -> str | None:
        hou = self._hou
        if hou is None:
            return None
        try:
            return str(hou.hipFile.path())
        except Exception:  # noqa: BLE001 - a fact we cannot read is a fact we do not have
            return None

    def bump(self, reason: str) -> int:
        """Count one scene replacement and take a fresh summary."""
        self.refresh()
        with self._lock:
            self._epoch += 1
            self._reason = reason
            self._changed_at = time.time()
            epoch = self._epoch
            path = self._hip_path
        if self._on_change is not None:
            try:
                self._on_change(epoch, path)
            except Exception as error:  # noqa: BLE001 - a missed note is not a lost scene
                self._log(f"could not record the new scene epoch: {type(error).__name__}: {error}")
        return epoch

    # Section: watching Houdini

    def watch(self) -> Callable[[], None] | None:
        """Follow this process's scene changes. Returns the way to stop.

        A load reports a clear of its own inside it, so the clear is swallowed
        while a load is in flight and the epoch moves once for one load.
        """
        hou = self._hou
        if hou is None:
            return None
        try:
            events = hou.hipFileEventType
            before_load = events.BeforeLoad
            after_load = events.AfterLoad
            after_clear = events.AfterClear
        except AttributeError:
            self._log("this build reports no hip file events, so the scene epoch never moves")
            return None

        def on_event(event_type: Any = None, *_rest: Any) -> None:
            try:
                if event_type == before_load:
                    self._loading = True
                elif event_type == after_load:
                    self._loading = False
                    self.bump(LOADED)
                elif event_type == after_clear and not self._loading:
                    self.bump(CLEARED)
            except Exception as error:  # noqa: BLE001 - never raise into Houdini's event loop
                self._log(f"scene event: {type(error).__name__}: {error}")

        try:
            hou.hipFile.addEventCallback(on_event)
        except Exception as error:  # noqa: BLE001 - no watch is better than no bridge
            self._log(f"could not watch the scene: {type(error).__name__}: {error}")
            return None

        def remove() -> None:
            try:
                hou.hipFile.removeEventCallback(on_event)
            except Exception:  # noqa: BLE001 - the session may already be tearing down
                pass

        self._remove_watch = remove
        return remove

    def unwatch(self) -> None:
        """Stop following scene changes. Safe when nothing was ever watched."""
        remove, self._remove_watch = self._remove_watch, None
        if remove is not None:
            remove()
