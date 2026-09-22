"""Where a tool's work actually runs.

Every `hou` call from a thread that is not the main thread takes Houdini's own
object model lock. The main thread holds that lock for the whole of a cook and
for the whole of any callback it is running, so a `hou` call from anywhere else
waits for the cook to end. Posting to the main thread is itself such a call.

That gives this module its one rule for a session with a user interface: no
thread that has to answer a request may call into `hou`. The only threads that
touch `hou` there are the main thread and one helper thread whose whole job is
to be the thread that waits for the lock. Reads are marshalled like writes,
because a read off the main thread waits on the same lock, is slower than the
same read on the main thread, and reads ambient state such as the current frame
from the wrong place.

A session without a user interface has no event loop holding that lock, so it
keeps the arrangement it had: reads on a thread of their own under the session
lock, and mutations through the loop the owner runs.

The pieces here:

- `Work` is one piece of work and a token. `take` and `cancel` share one claim,
  so a call that gives up cannot have its work run late, and work that has
  started cannot be cancelled out from under the thread running it.
- `Pulse` is one float: when the main thread last ran our code. A request
  thread reads it to decide whether the main thread is taking work at all,
  which needs no lock and no `hou`.
- `MainThreadRunner` keeps a queue and a poster thread. Submitting is a queue
  put. The poster does the blocking post. The main thread drains the queue,
  from the posted callback and from the pulse tick, whichever comes first.

Why the posted callback rather than the deferred evaluation module Houdini
ships: that module's wait has no timeout, so a caller cannot give up; its queue
is a module global shared with everything else in the process; it takes its own
loop callback off whenever its queue empties and adds it again on the next
call, which is another blocking call into `hou`; and its pickup is slower.

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
from collections import deque
from collections.abc import Callable
from typing import Any

from nscr_houdini_mcp.bridge import host

# How long the seam waits for the main thread to pick work up before it says
# the main thread is busy.
DEFAULT_PICKUP_S = 1.0

# How long the main thread may go without running our code before a call that
# will not wait that long is refused straight away. It sits above the gap
# between loop callbacks in a session that is playing back, so a session that
# is still serving calls is not called busy, and far below the length of a cook
# worth refusing.
DEFAULT_STALE_S = 2.0

# The one thread other than the main thread that calls into `hou`.
POSTER_THREAD_NAME = "nscr-mcp-poster"


class MarshalTimeout(Exception):
    """The main thread did not pick the work up in time."""


class Rejected(Exception):
    """The session could not take the work."""


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
        # Which of the two ways to the main thread reached this work first.
        self.picked_by: str | None = None

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

    @property
    def taken(self) -> bool:
        return self._taken

    def take(self) -> bool:
        """Claim the work. False means it was cancelled and must not run."""
        with self._claim:
            if self._cancelled or self._taken:
                return False
            self._taken = True
            return True

    def reject(self, error: BaseException) -> bool:
        """Wake the caller with an answer when the work cannot be delivered.

        The same claim as `take` and `cancel`, so work that is already running
        or already given up on is left alone.
        """
        with self._claim:
            if self._taken or self._cancelled:
                return False
            self._taken = True
            self.error = error
            self.started_at = time.monotonic()
            self.finished_at = self.started_at
        self.started.set()
        self.finished.set()
        return True

    def run(self, *, picked_by: str | None = None) -> None:
        """Run the work here, on whichever thread called this."""
        if not self.take():
            return
        self.picked_by = picked_by
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


class Pulse:
    """When the main thread last ran our code.

    One float, stamped on the main thread and read from anywhere. Reading it
    takes no lock and calls nothing in `hou`, which is what makes it safe on a
    thread that has to answer a request while Houdini is busy.

    Two places stamp it, both on the main thread: an event loop callback
    registered once, and the start of every piece of our work the main thread
    picks up. The second matters while Houdini is playing back, where the loop
    callback is called far less often but posted callbacks still land: a
    session that is serving calls stays fresh through its own pickups.
    """

    def __init__(
        self,
        *,
        stale_s: float = DEFAULT_STALE_S,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.stale_s = stale_s
        self._clock = clock
        self._log = log or (lambda text: None)
        self._at: float | None = None
        self._installed_at: float | None = None
        self._hou: Any | None = None
        self._retired = False
        # The runner sets this to its drain, so a tick is also a pickup.
        self.on_tick: Callable[[], None] | None = None

    def mark(self) -> None:
        """Say the main thread is here. Called on the main thread only."""
        self._at = self._clock()

    def age_s(self) -> float | None:
        """How long the main thread has been away, or nothing when unwatched.

        A pulse installed while the main thread was already busy has no stamp
        of its own yet, so it counts from the install rather than claiming the
        main thread has never been seen.
        """
        if self._installed_at is None:
            return None
        last = self._at if self._at is not None else self._installed_at
        return max(0.0, self._clock() - last)

    def away(self, *, limit_s: float | None = None) -> bool:
        """Whether the main thread has been away longer than it may be."""
        age = self.age_s()
        if age is None:
            return False
        return age > (self.stale_s if limit_s is None else limit_s)

    @property
    def installed(self) -> bool:
        return self._installed_at is not None

    @property
    def registered(self) -> bool:
        """Whether the callback is still on the main thread's list."""
        return self._hou is not None

    def install(self, hou: Any) -> None:
        """Start watching the main thread.

        Registering the callback is a call into `hou`, so this belongs on the
        main thread, or on the thread that starts the bridge while the session
        is idle. Never on a thread that is answering a request. A failure
        leaves the pulse uninstalled, in which case it never says the main
        thread is away and the pickup budget alone bounds a call.

        A pulse that was told to stop but whose callback has not yet come off
        is taken back rather than registered twice.
        """
        if self._installed_at is not None:
            return
        self._retired = False
        if self._hou is None:
            hou.ui.addEventLoopCallback(self._tick)
            self._hou = hou
        self._installed_at = self._clock()

    def uninstall(self) -> None:
        """Stop watching, without calling into `hou` from this thread.

        Taking the callback off is itself a `hou` call, and any thread but the
        main one waits on the object model lock to make it, which the main
        thread holds for the whole of a cook. So stopping only sets a flag:
        from here the pulse reports nothing, and the next tick takes the
        callback off from the main thread, where that call costs nothing.

        A session that never ticks again leaves one callback that does nothing
        but take itself off, and the process is going anyway.
        """
        self._retired = True
        self._installed_at = None
        self._at = None

    def _tick(self, *_rest: Any) -> None:
        """One visit from the main thread."""
        if self._retired:
            self._retire()
            return
        self.mark()
        tick = self.on_tick
        if tick is not None:
            tick()

    def _retire(self) -> None:
        """Take the callback off, on the main thread, and never fire again."""
        hou, self._hou = self._hou, None
        self.on_tick = None
        if hou is None:
            return
        try:
            hou.ui.removeEventLoopCallback(self._tick)
        except Exception as error:  # noqa: BLE001 - the session may already be tearing down
            self._log(f"could not stop watching the main thread: {type(error).__name__}: {error}")

    def state(self) -> dict[str, Any]:
        age = self.age_s()
        return {
            "installed": self.installed,
            "pulse_age_s": None if age is None else round(age, 3),
            "away": self.away(),
        }


class MainThreadQueue:
    """Work waiting for the main thread, in the order it was submitted."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: deque[Work] = deque()

    def put(self, work: Work) -> None:
        with self._lock:
            self._items.append(work)

    def drain(self, *, picked_by: str) -> int:
        """Run everything queued, here. Cancelled work is dropped, not run."""
        ran = 0
        while True:
            with self._lock:
                if not self._items:
                    return ran
                work = self._items.popleft()
            work.run(picked_by=picked_by)
            ran += 1

    def cancel_all(self) -> None:
        for work in self._take_all():
            work.cancel()

    def reject_all(self, error: BaseException) -> None:
        for work in self._take_all():
            work.reject(error)

    def _take_all(self) -> list[Work]:
        with self._lock:
            waiting = list(self._items)
            self._items.clear()
        return waiting

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class MainThreadRunner:
    """Run the work on the main thread of a graphical Houdini.

    Submitting is a queue put and nothing else, so the thread that has to
    answer the request never calls into `hou`. One daemon thread posts a kick
    to the main thread on its behalf and takes the wait for the object model
    lock. The main thread drains the queue from that kick and from the pulse's
    loop callback, whichever reaches it first.
    """

    kind = host.GUI

    def __init__(
        self,
        hou: Any,
        *,
        pulse: Pulse | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._hou = hou
        self._pulse = pulse
        self._log = log or (lambda text: None)
        self._queue = MainThreadQueue()
        self._kicks: queue.Queue[Any] = queue.Queue()
        self._kick_in_flight = threading.Event()
        self._closed = threading.Event()
        self._poster: threading.Thread | None = None
        self._starting = threading.Lock()
        if pulse is not None:
            pulse.on_tick = self._drain_from_loop

    def start(self) -> None:
        """Start the poster thread. Safe to call more than once."""
        with self._starting:
            if self._poster is not None and self._poster.is_alive():
                return
            self._poster = threading.Thread(
                target=self._post_loop, name=POSTER_THREAD_NAME, daemon=True
            )
            self._poster.start()

    def stop(self) -> None:
        """Stop taking work, and do not wait for a poster stuck in `hou`.

        A poster inside a post call is waiting for the object model lock, which
        it gets when the cook ends; it then sees the session is closed and
        ends. Waiting for it here would hold the shutdown open for the length
        of the cook.
        """
        self._closed.set()
        self._queue.cancel_all()
        self._kicks.put(None)
        poster, self._poster = self._poster, None
        if poster is not None:
            poster.join(0.1)

    @property
    def pending(self) -> int:
        """How much work the main thread has not taken yet. Takes no `hou`."""
        return len(self._queue)

    def submit(self, work: Work) -> Work:
        if self._closed.is_set():
            work.reject(Rejected("the session is stopping"))
            return work
        self._queue.put(work)
        self._kicks.put(True)
        return work

    def _post_loop(self) -> None:
        """The poster thread: the only one here that calls into `hou`."""
        while True:
            item = self._kicks.get()
            if item is None or self._closed.is_set():
                return
            if self._kick_in_flight.is_set():
                # A kick is already on its way and will drain whatever is
                # queued by the time it lands, so posting another would only
                # pile callbacks up.
                continue
            self._kick_in_flight.set()
            try:
                post_to_main_thread(self._on_kick, hou=self._hou)
            except BaseException as error:  # noqa: BLE001 - a caller waiting for ever is worse
                self._kick_in_flight.clear()
                self._log(f"could not post to the main thread: {type(error).__name__}: {error}")
                self._queue.reject_all(Rejected("the interface would not take the work"))

    def _on_kick(self, *_rest: Any) -> None:
        """The posted callback, on the main thread.

        The flag is cleared before the drain, so work submitted while this is
        running gets a kick of its own rather than waiting for the next one.
        """
        self._kick_in_flight.clear()
        if self._pulse is not None:
            self._pulse.mark()
        self._queue.drain(picked_by="kick")

    def _drain_from_loop(self) -> None:
        """The pulse tick, on the main thread."""
        self._queue.drain(picked_by="loop")

    def state(self) -> dict[str, Any]:
        poster = self._poster
        return {
            "queued_for_main": len(self._queue),
            "kick_in_flight": self._kick_in_flight.is_set(),
            "poster_alive": poster is not None and poster.is_alive(),
        }


def post_to_main_thread(callback: Callable[..., None], *, hou: Any | None = None) -> None:
    """Ask the main thread to run this callable at its next event.

    The callback fires exactly once: that is the interface's own contract for
    a posted callback, and there is no call on this build to take one off
    again. Posting takes Houdini's object model lock, so this blocks for as
    long as the main thread holds it and belongs on the poster thread alone.
    """
    module = hou if hou is not None else host.houdini()
    if module is None:
        raise MarshalTimeout("this process has no user interface to post to")
    module.ui.postEventCallback(callback)


def run_on_main_thread(
    function: Callable[[], Any],
    *,
    timeout_s: float,
    pickup_s: float = DEFAULT_PICKUP_S,
    hou: Any | None = None,
) -> Any:
    """Run one callable on the main thread and wait for its answer.

    For callers outside the request path, such as start up steps: it builds a
    runner of its own and takes it down again. A request is dispatched through
    the runner the bridge owns instead.

    Raises `MarshalTimeout` when the main thread does not pick the work up
    within `pickup_s`, or does not finish it within `timeout_s`. Work that was
    not picked up is cancelled, so it never runs late.
    """
    module = hou if hou is not None else host.houdini()
    if module is None:
        raise MarshalTimeout("this process has no user interface to post to")
    runner = MainThreadRunner(module)
    runner.start()
    work = Work(function)
    try:
        runner.submit(work)
        if not work.started.wait(pickup_s) and work.cancel():
            raise MarshalTimeout("the main thread did not pick the work up")
        if not work.finished.wait(timeout_s):
            raise MarshalTimeout("the main thread is still running the work")
    finally:
        runner.stop()
    if work.error is not None:
        raise work.error
    return work.result


def choose_runner(
    kind: str,
    *,
    mutating: bool,
    hou: Any | None = None,
    main_loop: MainLoop | None = None,
    main_thread: MainThreadRunner | None = None,
) -> Any:
    """Where one tool call should run.

    In a session with a user interface everything goes to the main thread,
    reads included: a read on any other thread waits on the object model lock
    for as long as a cook lasts, and reads the wrong frame while it is at it.

    Without a user interface a mutating tool goes to the loop the owner runs,
    for the undo group, and everything else runs on a thread of its own under
    the session lock. So does a mutating tool in a session where nothing is
    running the loop, which is the case the reply marks as recording no undo.
    """
    if kind == host.GUI and main_thread is not None:
        return main_thread
    module = hou if hou is not None else host.houdini()
    if mutating and module is not None and kind != host.GUI:
        if main_loop is not None and main_loop.running:
            return PumpRunner(main_loop)
    return ThreadRunner()
