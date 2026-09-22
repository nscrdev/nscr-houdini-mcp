"""Where a tool's work actually runs.

Anything that changes the scene runs on the process main thread, in every
session kind. It has to. In Houdini 22, with a user interface and without:

- With a user interface, ten node creates inside one undo group from a handler
  thread added ten undo entries instead of one, and took about 94 ms per node
  against 2.5 ms on the main thread. The grouping does not fail loudly, it
  just does not happen.
- Headless, it is worse: undo recording is off on any thread but the main one
  (`hou.undos.areEnabled()` is false there, and this build has no way to turn
  it on), so a group on a worker thread records nothing at all and there is
  nothing to roll a failed call back with.

The two kinds get there by different routes, because they have different main
threads to reach.

- With a user interface: `hou.ui.postEventCallback` with a `threading.Event`
  and a result slot. Not `hdefereval`: from a handler thread the callback
  costs about 17 ms idle and 23 ms during playback, against 53 ms and 973 ms
  for `hdefereval`, which also does not exist outside a graphical Houdini.
- Headless: the bridge's own main thread runs a small loop and takes work off
  a queue. Where nothing is running that loop, the work falls back to a thread
  of its own and the reply says no undo entry was recorded, rather than
  quietly losing the grouping.

Reads run on a thread of their own in both kinds, one at a time, held open by
the session lock.

Two rules this module keeps:

- The work never runs on the thread that has to answer the request. A call may
  give up waiting while the work carries on, and a thread that is answering
  cannot also be working.
- Nothing here walks the user interface. Enumerating top level widgets on the
  main thread froze a session for the best part of a minute, so no code on
  this path touches a widget tree.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from nscr_houdini_mcp.bridge import host

# How long the seam waits for the main thread to pick work up before it says
# the main thread is busy.
DEFAULT_PICKUP_S = 1.0


class MarshalTimeout(Exception):
    """The main thread did not pick the work up in time."""


class Work:
    """One piece of work, and where it got to.

    `started` is set when the work is picked up, `finished` when it is done.
    The two are separate because a caller waits on them with separate budgets:
    one for being picked up, one for running.
    """

    def __init__(self, run: Callable[[], Any]) -> None:
        self._run = run
        self._claim = threading.Lock()
        self._taken = False
        self._cancelled = False
        self.started = threading.Event()
        self.finished = threading.Event()
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.result: Any = None
        self.error: BaseException | None = None

    def cancel(self) -> bool:
        """Stop this work before it starts. False means it is already running."""
        with self._claim:
            if self._taken:
                return False
            self._cancelled = True
            return True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def take(self) -> bool:
        """Claim the work. False means it was cancelled and must not run."""
        with self._claim:
            if self._cancelled or self._taken:
                return False
            self._taken = True
            return True

    def run(self) -> None:
        """Run the work here, on whichever thread called this."""
        if not self.take():
            return
        self.started_at = time.monotonic()
        self.started.set()
        try:
            self.result = self._run()
        except BaseException as error:  # noqa: BLE001 - the caller decides what it means
            self.error = error
        finally:
            self.finished_at = time.monotonic()
            self.finished.set()


class ThreadRunner:
    """Run the work on a thread of its own.

    One at a time: the caller holds the session lock across the whole
    operation, so there is never a second of these touching Houdini.
    """

    kind = host.HYTHON

    def submit(self, work: Work) -> Work:
        threading.Thread(target=work.run, name="nscr-mcp-op", daemon=True).start()
        return work


class MainLoop:
    """The main thread of a session that has no event loop of its own.

    The thread that owns the process calls `run_until` and stays in it. Work
    posted from a handler thread is run there, in the order it arrived, and
    nothing else happens on that thread in between.
    """

    def __init__(self) -> None:
        self._work: queue.Queue[Work] = queue.Queue()
        self._running = threading.Event()

    @property
    def running(self) -> bool:
        """Whether a thread is in `run_until` right now."""
        return self._running.is_set()

    def submit(self, work: Work) -> Work:
        self._work.put(work)
        return work

    def run_until(self, stop: threading.Event, *, tick_s: float = 0.05) -> None:
        """Run posted work on this thread until told to stop."""
        self._running.set()
        try:
            while not stop.is_set():
                try:
                    work = self._work.get(timeout=tick_s)
                except queue.Empty:
                    continue
                work.run()
        finally:
            self._running.clear()
            self._drop_the_rest()

    def _drop_the_rest(self) -> None:
        """Cancel whatever is still queued, so nobody waits on a stopped loop."""
        while True:
            try:
                self._work.get_nowait().cancel()
            except queue.Empty:
                return


class PumpRunner:
    """Run the work on a main thread that is taking work off a queue."""

    kind = host.HYTHON

    def __init__(self, loop: MainLoop) -> None:
        self._loop = loop

    def submit(self, work: Work) -> Work:
        return self._loop.submit(work)


class MainThreadRunner:
    """Run the work on the main thread of a graphical Houdini."""

    kind = host.GUI

    def __init__(self, hou: Any) -> None:
        self._hou = hou

    def submit(self, work: Work) -> Work:
        post_to_main_thread(work, hou=self._hou)
        return work


def post_to_main_thread(work: Work, *, hou: Any | None = None) -> Work:
    """Ask the main thread to run this work at its next event.

    The callback takes itself off again before it runs anything, because the
    event it is registered for fires for every interaction and this work is
    meant to happen once.
    """
    module = hou if hou is not None else host.houdini()
    if module is None:
        raise MarshalTimeout("this process has no user interface to post to")

    def on_event(*_rest: Any) -> None:
        try:
            module.ui.removeEventCallback(on_event)
        except Exception:  # noqa: BLE001 - a callback that cannot be removed still runs once
            pass
        work.run()

    module.ui.postEventCallback(on_event)
    return work


def run_on_main_thread(
    function: Callable[[], Any],
    *,
    timeout_s: float,
    pickup_s: float = DEFAULT_PICKUP_S,
    hou: Any | None = None,
) -> Any:
    """Run one callable on the main thread and wait for its answer.

    Raises `MarshalTimeout` when the main thread does not pick the work up
    within `pickup_s`, or does not finish it within `timeout_s`. Work that was
    not picked up is cancelled, so it never runs late.
    """
    work = post_to_main_thread(Work(function), hou=hou)
    if not work.started.wait(pickup_s):
        if work.cancel():
            raise MarshalTimeout("the main thread did not pick the work up")
    if not work.finished.wait(timeout_s):
        raise MarshalTimeout("the main thread is still running the work")
    if work.error is not None:
        raise work.error
    return work.result


def choose_runner(
    kind: str,
    *,
    mutating: bool,
    hou: Any | None = None,
    main_loop: MainLoop | None = None,
) -> Any:
    """Where one tool call should run.

    A mutating tool goes to the main thread, for the undo group: through the
    event callback with a user interface, and through the loop without one.
    Everything else, reads included, runs on a thread of its own under the
    session lock. So does a mutating tool in a session where nothing is
    running the loop, which is the case the reply marks as recording no undo.
    """
    module = hou if hou is not None else host.houdini()
    if mutating and module is not None:
        if kind == host.GUI:
            return MainThreadRunner(module)
        if main_loop is not None and main_loop.running:
            return PumpRunner(main_loop)
    return ThreadRunner()
