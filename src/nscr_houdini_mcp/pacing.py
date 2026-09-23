"""How hard this server may drive a Houdini that a person is working in.

In a session with a user interface every call runs on Houdini's main thread,
the same thread that draws the interface and answers the mouse. One call at a
time is already the bridge's rule, but calls sent back to back leave the main
thread no room between them, and an agent in a loop can send a great many.

So a call on its way to such a session waits here for its turn, under three
rules:

- One at a time. Only one paced call from this server is out at the session
  at once. The next one's turn comes when that one has ended.
- A minimum pause. A turn comes no sooner than `min_pause_s` after the last
  call ended, so the main thread gets that long to itself between our calls,
  however many callers are waiting.
- A rate cap. No more than `max_per_s` turns start in any one second.

The wait for a turn is part of the caller's own `wait_s`, the time it said it
would queue for the session, never added to it: what is left of it goes on to
the bridge. When the turn is plainly further off than that, or the call asked
to be skipped when the session is busy, the call is refused at once with the
time until the turn, rather than sent late to a caller that has given up.

The pace is kept per server process and per session: a second session is
paced on its own. Several agents sharing one server process share its one
allowance; two server processes each have their own. Workers are headless,
so nobody's interface is at stake there and they are not paced.

Nothing here is kept anywhere but in memory, and nothing here imports `hou`.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# The window the rate cap counts over.
WINDOW_S = 1.0

DEFAULT_MIN_PAUSE_MS = 50
DEFAULT_MAX_CALLS_PER_S = 10


class NoTurn(Exception):
    """The call's turn would come too late for it, or it asked not to wait.

    `waited_s` is how long it had already waited. `retry_after_s` is how long
    until a turn could come, as far as can be told now: exact when only the
    pause and the cap stand in the way, a floor when another call is still out.
    """

    def __init__(self, *, waited_s: float, retry_after_s: float, reason: str) -> None:
        super().__init__(reason)
        self.waited_s = waited_s
        self.retry_after_s = retry_after_s
        self.reason = reason


@dataclass(frozen=True)
class Turn:
    """A call let through: how long it waited, and when, on the wall clock."""

    waited_s: float
    at: float


@dataclass
class _Pace:
    """What one session has had from this server lately."""

    out: bool = False
    last_end: float | None = None
    starts: deque[float] = field(default_factory=deque)
    queue: deque[int] = field(default_factory=deque)


class Pacer:
    """Hands out turns at a session no faster than its rules allow.

    `clock`, `wall` and `wait` are for tests, which run the rules on a clock
    of their own: `wait(condition, seconds)` is how a waiting caller gives the
    lock up until it is woken or the time has passed.
    """

    def __init__(
        self,
        *,
        min_pause_s: float = DEFAULT_MIN_PAUSE_MS / 1000.0,
        max_per_s: int = DEFAULT_MAX_CALLS_PER_S,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        wait: Callable[[threading.Condition, float], object] | None = None,
    ) -> None:
        self.min_pause_s = max(0.0, float(min_pause_s))
        self.max_per_s = max(0, int(max_per_s))
        self._clock = clock
        self._wall = wall
        self._wait = wait or (lambda condition, seconds: condition.wait(seconds))
        self._paces: dict[str, _Pace] = {}
        self._ready = threading.Condition()
        self._tickets = itertools.count(1)

    @property
    def active(self) -> bool:
        """Whether any rule is on. Zero turns the pause or the cap off, and
        with both off calls go straight through, several at once."""
        return self.min_pause_s > 0 or self.max_per_s > 0

    def admit(self, key: str, *, budget_s: float, skip_if_busy: bool = False) -> Turn:
        """Wait for this call's turn, no longer than `budget_s`.

        Raises `NoTurn` at once when the turn is known to be further off than
        the budget, or when anything at all stands in the way of a call that
        asked to be skipped; and when the budget runs out behind a call that
        is still out. Turns go in the order they were asked for.
        """
        if not self.active:
            return Turn(0.0, self._wall())
        with self._ready:
            pace = self._paces.setdefault(key, _Pace())
            ticket = next(self._tickets)
            pace.queue.append(ticket)
            started = self._clock()
            deadline = started + max(0.0, budget_s)
            granted = False
            try:
                while True:
                    now = self._clock()
                    first = pace.queue[0] == ticket
                    turn = self._earliest(pace, now)
                    if first and not pace.out and turn <= now:
                        granted = True
                        break
                    if skip_if_busy:
                        raise self._refuse(pace, started, now, turn, "the caller asked not to wait")
                    if first and not pace.out and turn > deadline:
                        raise self._refuse(pace, started, now, turn, "the turn is past the wait")
                    if now >= deadline:
                        raise self._refuse(pace, started, now, turn, "the wait ran out")
                    until = deadline if (pace.out or not first) else min(turn, deadline)
                    self._wait(self._ready, max(0.0, until - now))
            finally:
                if not granted:
                    pace.queue.remove(ticket)
                    self._ready.notify_all()
            pace.queue.popleft()
            pace.out = True
            if self.max_per_s > 0:
                pace.starts.append(now)
                while len(pace.starts) > self.max_per_s:
                    pace.starts.popleft()
            return Turn(max(0.0, now - started), self._wall())

    def done(self, key: str) -> None:
        """Mark the end of a call that had a turn, and wake whoever is next."""
        if not self.active:
            return
        with self._ready:
            pace = self._paces.setdefault(key, _Pace())
            pace.out = False
            pace.last_end = self._clock()
            self._ready.notify_all()

    def forget(self, key: str) -> None:
        """Drop what is kept for a session that has gone, when nothing waits on it."""
        with self._ready:
            pace = self._paces.get(key)
            if pace is not None and not pace.out and not pace.queue:
                del self._paces[key]

    def _earliest(self, pace: _Pace, now: float) -> float:
        """The first moment the pause and the cap allow, the call out aside."""
        turn = now
        if self.min_pause_s > 0 and pace.last_end is not None:
            turn = max(turn, pace.last_end + self.min_pause_s)
        if self.max_per_s > 0 and len(pace.starts) >= self.max_per_s:
            turn = max(turn, pace.starts[-self.max_per_s] + WINDOW_S)
        return turn

    def _refuse(self, pace: _Pace, started: float, now: float, turn: float, reason: str) -> NoTurn:
        ahead = turn - now
        if pace.out:
            # The call that is out has to end first, and then the pause runs.
            ahead = max(ahead, self.min_pause_s)
        return NoTurn(
            waited_s=max(0.0, now - started),
            retry_after_s=round(max(ahead, 0.001), 3),
            reason=reason,
        )


def throttled_ms(waited_s: float) -> int:
    """A wait as the whole milliseconds a reply reports, nothing under one."""
    return int(round(waited_s * 1000.0)) if waited_s >= 0.0005 else 0
