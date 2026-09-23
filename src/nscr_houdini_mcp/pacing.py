"""How hard this server may drive a Houdini that a person is working in.

In a session with a user interface every call runs on Houdini's main thread,
the same thread that draws the interface and answers the mouse. One call at a
time is already the bridge's rule, but calls sent back to back leave the main
thread no room between them, and an agent in a loop can send a great many.
Two agents on one session can each keep it that busy.

So a call on its way to such a session is paced here, before it is sent, by
two rules:

- A minimum pause. The next call from this server starts no sooner than
  `min_pause_s` after the last one ended, and no sooner than that after the
  last one started, so calls this server sends side by side are spaced too.
- A rate cap. No more than `max_per_s` calls from this server start in any
  one second.

A call past either rule waits for its turn. It never fails for it, and the
caller hears how long it waited. The pace is kept per server process, which is
one client, and per session: a second session is paced on its own, and a
second client has its own allowance. Workers are headless, so nobody's
interface is at stake there and they are not paced.

Nothing here is kept anywhere but in memory, and nothing here imports `hou`.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# The window the rate cap counts over.
WINDOW_S = 1.0

DEFAULT_MIN_PAUSE_MS = 50
DEFAULT_MAX_CALLS_PER_S = 10


@dataclass
class _Pace:
    """What one session has had from this server lately."""

    last_start: float | None = None
    last_end: float | None = None
    starts: deque[float] = field(default_factory=deque)


class Pacer:
    """Admits calls to a session no faster than its two rules allow.

    `clock` and `sleep` are for tests, which run the rules on a clock of their
    own. `admit` hands back how long the call waited, in seconds; `done` is
    called when the call has ended, however it ended.
    """

    def __init__(
        self,
        *,
        min_pause_s: float = DEFAULT_MIN_PAUSE_MS / 1000.0,
        max_per_s: int = DEFAULT_MAX_CALLS_PER_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_pause_s = max(0.0, float(min_pause_s))
        self.max_per_s = max(0, int(max_per_s))
        self._clock = clock
        self._sleep = sleep
        self._paces: dict[str, _Pace] = {}
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        """Whether either rule is on. Zero turns a rule off."""
        return self.min_pause_s > 0 or self.max_per_s > 0

    def admit(self, key: str) -> float:
        """Wait for this call's turn, and say how long that was.

        The turn is taken under the lock and waited for outside it, so calls
        sent side by side each get a turn of their own, in the order they
        asked, and none of them holds the others up while it sleeps.
        """
        if not self.active:
            return 0.0
        with self._lock:
            pace = self._paces.setdefault(key, _Pace())
            now = self._clock()
            start = now
            if pace.last_start is not None:
                # Turns are handed out in order, so a later call never starts
                # before an earlier one.
                start = max(start, pace.last_start)
            if self.min_pause_s > 0:
                for mark in (pace.last_start, pace.last_end):
                    if mark is not None:
                        start = max(start, mark + self.min_pause_s)
            if self.max_per_s > 0:
                while pace.starts and pace.starts[0] <= start - WINDOW_S:
                    pace.starts.popleft()
                if len(pace.starts) >= self.max_per_s:
                    start = max(start, pace.starts[-self.max_per_s] + WINDOW_S)
                pace.starts.append(start)
                while len(pace.starts) > self.max_per_s:
                    pace.starts.popleft()
            pace.last_start = start
        waited = start - now
        if waited > 0:
            self._sleep(waited)
        return max(0.0, waited)

    def done(self, key: str) -> None:
        """Mark the end of a call, which the minimum pause counts from."""
        if not self.active:
            return
        with self._lock:
            pace = self._paces.setdefault(key, _Pace())
            pace.last_end = self._clock()

    def forget(self, key: str) -> None:
        """Drop what is kept for a session that has gone."""
        with self._lock:
            self._paces.pop(key, None)


def throttled_ms(waited_s: float) -> int:
    """A wait as the whole milliseconds a reply reports, nothing under one."""
    return int(round(waited_s * 1000.0)) if waited_s >= 0.0005 else 0
