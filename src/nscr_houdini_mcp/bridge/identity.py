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
- One exception. A session with a user interface that came up before its scene
  did, as one started with Houdini does while the file named on the command
  line is still to load, is named after the untitled scene it found. Its name
  is provisional: the first time the scene is loaded or saved under a file
  name, the session takes that name. Only until a call has reached it, since
  from then on a caller may be holding the name it has. The store holds the
  old name for the session while it runs, so a caller that read it and never
  called still reaches this session through it, never another one. A reply
  made while the rename is under way waits a moment for the new name rather
  than answer with the old one.
- `scene_epoch` counts how many times this process has thrown its scene away.
  Opening a file, starting a new scene and loading the same file again all
  replace the scene, and every node path a caller was holding goes with it. So
  a call that carries an epoch older than this one is refused before it runs.

How the counter is kept. Houdini reports scene changes through the hip file
event callbacks, and in this build a load reports four of them: `BeforeLoad`,
then `BeforeClear` and `AfterClear` for the scene it is dropping, then
`AfterLoad`. Counting each of those would count one load twice, so the count
moves on the clear and the load that follows only says what the scene is now.

The count moves on the clear rather than on the load because that is the
moment the old scene stops existing. A load that fails after clearing never
reports `AfterLoad` at all, and a session left with an empty scene must not go
on telling callers that their paths are still good.

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

from nscr_houdini_mcp.bridge import dirty as dirty_module
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

# How long after a load starts its clear is still read as part of that load.
# A load reports its clear within moments of starting, so this is a wide
# margin, and it is a window rather than a flag because a load that fails
# reports nothing at all and must not leave the next clear uncounted.
LOADING_WINDOW_S = 120.0

# How long a reply waits for a rename that is under way, so it carries the new
# name. The rename is one store write; a reply that waits this long without
# seeing it end goes out with the old name, and no warning about it.
RENAME_WAIT_S = 2.0


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
        on_rename: Callable[[str], str] | None = None,
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
        # Takes the scene file a provisional name should follow, and hands
        # back the name the store gave the session for it.
        self._on_rename = on_rename
        self._log = log or (lambda text: None)
        self._load_began: float | None = None
        self._counted_the_clear = False
        # The scene callback while it is registered with Houdini, whether or
        # not it has been told to go.
        self._on_event: Callable[..., None] | None = None
        # Read by the scene callback on the main thread. Cleared by `unwatch`
        # from whichever thread is stopping the bridge, which calls no `hou`.
        self._watching = False
        # Whether the scene has changes that are not on disk, as far as this
        # session can see. The scene events below move it, and so do the calls.
        self.dirty = dirty_module.DirtyMarker()
        # Whether the name was taken from a scene with no file yet, and may
        # still follow the first file the scene gets.
        self._provisional = on_rename is not None and tracks_hip and self._scene_is_new()
        # Clear while a rename is under way, which replies wait on.
        self._named = threading.Event()
        self._named.set()

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
    def provisional(self) -> bool:
        """Whether the name may still move to the scene's, once it has a file."""
        with self._lock:
            return self._provisional

    def keep_name(self) -> None:
        """A call has reached this session, so its name stays what it is."""
        with self._lock:
            self._provisional = False

    def _scene_is_new(self) -> bool:
        """Whether the scene this session starts with has never had a file."""
        hou = self._hou
        if hou is None:
            return False
        try:
            return bool(hou.hipFile.isNewFile())
        except Exception:  # noqa: BLE001 - a fact we cannot read is a fact we do not have
            return False

    def _follow_the_scene(self) -> None:
        """Give a provisional name the scene's, now that the scene has a file.

        Runs from the scene event, on the main thread. The name moves once:
        whatever happens to the scene afterwards, it is settled.
        """
        if not self.provisional or self._scene_is_new():
            return
        path = self._read_hip_path()
        with self._lock:
            if path is not None:
                self._hip_path = path
            hip = self._hip_path or ""
            stem = hip_stem(hip)
            if not self._provisional or not stem:
                return
            self._provisional = False
            if stem == self._alias_stem or self._on_rename is None:
                return
            self._named.clear()
        try:
            alias = self._on_rename(hip)
        except Exception as error:  # noqa: BLE001 - an old name is not a lost scene
            self._log(
                f"could not name the session after its scene: {type(error).__name__}: {error}"
            )
            return
        else:
            with self._lock:
                self._alias = alias
                self._alias_stem = stem
        finally:
            self._named.set()

    @property
    def scene_epoch(self) -> int:
        with self._lock:
            return self._epoch

    @property
    def hip_path(self) -> str | None:
        with self._lock:
            return self._hip_path

    def trace(self) -> dict[str, Any]:
        """What every reply carries about who answered and which scene it was.

        While the session is taking its scene's name, this waits a moment for
        the new name, so a reply does not go out under the one it is leaving.
        """
        self._named.wait(RENAME_WAIT_S)
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
        if not self._tracks_hip or not self._named.is_set():
            # Mid rename the name is on its way to the scene's, not behind it.
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
            "hint": "address this session by its id or its name; the name stays as it is",
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
        return self._replaced(reason, count=True)

    def settled(self, reason: str) -> int:
        """Say what the scene is now, without counting it again.

        For the second half of a replacement that has already been counted: a
        load whose clear moved the epoch a moment ago, where what is left to
        do is read the new scene and say why it changed.
        """
        return self._replaced(reason, count=False)

    def _replaced(self, reason: str, *, count: bool) -> int:
        self.refresh()
        with self._lock:
            if count:
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

    def _mid_load(self) -> bool:
        """Whether a load started recently enough to still be the one running.

        A load that fails reports nothing to say it is over, so this is read
        as a window rather than kept as a flag somebody has to clear.
        """
        began = self._load_began
        return began is not None and (time.monotonic() - began) <= LOADING_WINDOW_S

    def watch(self) -> Callable[[], None] | None:
        """Follow this process's scene changes. Returns the way to stop.

        A load reports its own clear, and the clear is what moves the epoch,
        so the load that follows says what the scene is now without counting
        a second time. A load that never finishes has still been counted.

        Watching twice registers one callback. A watch that was told to stop
        but whose callback is still registered is taken back instead.
        """
        hou = self._hou
        if hou is None:
            return None
        if self._on_event is not None and (self._watching or host.keep(self._on_event)):
            self._watching = True
            return self.unwatch
        try:
            events = hou.hipFileEventType
            before_load = events.BeforeLoad
            after_load = events.AfterLoad
            after_clear = events.AfterClear
        except AttributeError:
            self._log("this build reports no hip file events, so the scene epoch never moves")
            return None
        # The events that only move the unsaved mark, where the build has them.
        after_save = getattr(events, "AfterSave", None)
        after_merge = getattr(events, "AfterMerge", None)

        def on_event(event_type: Any = None, *_rest: Any) -> None:
            if not self._watching:
                # Told to stop. Taking the callback off is a `hou` call, and
                # this is the main thread, which is the only place it is free.
                host.take_off_now(on_event)
                return
            try:
                if event_type == before_load:
                    self._load_began = time.monotonic()
                    self._counted_the_clear = False
                elif event_type == after_clear:
                    # The old scene is gone from here, load or no load.
                    self.bump(CLEARED)
                    self._counted_the_clear = self._mid_load()
                    if not self._counted_the_clear:
                        self.dirty.event(dirty_module.CLEARED)
                elif event_type == after_load:
                    counted = self._counted_the_clear and self._mid_load()
                    self._load_began = None
                    self._counted_the_clear = False
                    if counted:
                        self.settled(LOADED)
                    else:
                        self.bump(LOADED)
                    self.dirty.event(dirty_module.LOADED)
                    self._follow_the_scene()
                elif after_save is not None and event_type == after_save:
                    self.dirty.event(dirty_module.SAVED)
                    self._follow_the_scene()
                elif after_merge is not None and event_type == after_merge:
                    self.dirty.event(dirty_module.MERGED)
            except Exception as error:  # noqa: BLE001 - never raise into Houdini's event loop
                self._log(f"scene event: {type(error).__name__}: {error}")

        try:
            hou.hipFile.addEventCallback(on_event)
        except Exception as error:  # noqa: BLE001 - no watch is better than no bridge
            self._log(f"could not watch the scene: {type(error).__name__}: {error}")
            return None

        self._on_event = on_event
        self._watching = True
        return self.unwatch

    def unwatch(self) -> None:
        """Stop following scene changes, without calling into `hou` here.

        Taking the callback off waits on the object model lock from any thread
        but the main one, and the main thread holds that lock for the whole of
        a cook. So this only sets a flag and puts the callback on the list to
        come off, which the main thread empties on its next visit. The callback
        also takes itself off on its next scene event, whichever comes first.
        Until then it does nothing at all.
        """
        self._watching = False
        on_event, hou = self._on_event, self._hou
        if on_event is None or hou is None:
            return

        def take_off() -> None:
            hou.hipFile.removeEventCallback(on_event)

        host.leave(hou, on_event, take_off)
