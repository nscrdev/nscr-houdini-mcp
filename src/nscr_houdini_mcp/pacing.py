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
the bridge. A call is refused at once only when the pacer's own rules put its
turn past that wait: the pause after each call ahead of it and the rate cap
alone come too late, however quickly those calls run. Otherwise it queues,
however long the call out may still run, and is refused only if its wait runs
out first. A call that asked to be skipped when the session is busy is
refused at once when anything stands in its way. So a call may be refused up
front when the calls queued ahead of it, at the rate cap, already fill its
`wait_s`, and `retry_after_s` says when to come back.

A refusal says when to come back, in `retry_after_s`: an estimate from the
calls queued ahead, the median time of the session's last few calls, the
pause and the cap, never less than the pause and never more than
`RETRY_CAP_S`. A call out longer than that median is guessed to run as long
again as it has so far. A timeout a call names is a ceiling, not an estimate, so it
plays no part in this.

A call sent again under the operation id of the call that is out skips the
pace altogether: the bridge answers it from that call's receipt, with no need
of the main thread.

The queue is bounded: past `max_queued` calls waiting on one session, a new
one is refused at once and told how many wait ahead of it. A call whose
caller has gone, by a cancel or its own deadline, leaves the queue before it
can reach the bridge.

The pace is kept per server process and per session: a second session is
paced on its own. Several agents sharing one server process share its one
allowance; two server processes each have their own. Workers are headless,
so nobody's interface is at stake there and they are not paced.

Nothing here is kept anywhere but in memory, and nothing here imports `hou`.
"""

from __future__ import annotations

import itertools
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# The window the rate cap counts over.
WINDOW_S = 1.0

DEFAULT_MIN_PAUSE_MS = 50
DEFAULT_MAX_CALLS_PER_S = 10

# The most calls that may wait on one session's turn at once.
DEFAULT_MAX_QUEUED = 32

# How often a waiting call looks at whether its caller has gone.
CANCEL_POLL_S = 0.05

# How many of a session's last calls its time per call is the median of, what
# is assumed before any has ended, and the longest a refusal says to wait.
DURATIONS_KEPT = 8
RUN_SEED_S = 0.25
RETRY_CAP_S = 30.0


class NoTurn(Exception):
    """The call's turn would come too late for it, or it asked not to wait.

    `waited_s` is how long it had already waited. `ahead` is how many calls
    were queued ahead of it. `retry_after_s` is an estimate of how long until
    a turn could come: what the call out may still take, then a turn for each
    call ahead at the session's recent time per call and the pause, and never
    sooner than the pause and the cap allow. It is at least the pause and at
    most `RETRY_CAP_S`.
    """

    def __init__(
        self, *, waited_s: float, retry_after_s: float, reason: str, ahead: int = 0
    ) -> None:
        super().__init__(reason)
        self.waited_s = waited_s
        self.retry_after_s = retry_after_s
        self.reason = reason
        self.ahead = ahead


@dataclass(frozen=True)
class Turn:
    """A call let through: how long it waited, and when, on the wall clock."""

    waited_s: float
    at: float


@dataclass
class _Pace:
    """What one session has had from this server lately."""

    out: bool = False
    # The call that is out: its operation id, and when it was let through.
    out_id: str | None = None
    out_since: float | None = None
    last_end: float | None = None
    starts: deque[float] = field(default_factory=deque)
    queue: deque[int] = field(default_factory=deque)
    # How long the last few calls held their turns, for the estimates.
    durations: deque[float] = field(default_factory=lambda: deque(maxlen=DURATIONS_KEPT))
    # The session has gone: the last call to leave drops what is kept.
    forgotten: bool = False


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
        max_queued: int = DEFAULT_MAX_QUEUED,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        wait: Callable[[threading.Condition, float], object] | None = None,
    ) -> None:
        self.min_pause_s = max(0.0, float(min_pause_s))
        self.max_per_s = max(0, int(max_per_s))
        self.max_queued = max(1, int(max_queued))
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

    def admit(
        self,
        key: str,
        *,
        budget_s: float,
        skip_if_busy: bool = False,
        cancelled: Callable[[], bool] | None = None,
        operation_id: str | None = None,
    ) -> Turn:
        """Wait for this call's turn, no longer than `budget_s`.

        Raises `NoTurn` at once when the pause and the cap alone put the turn
        past the budget, when the queue is full, or when anything at all
        stands in the way of a call that asked to be skipped; and when the
        budget runs out, or `cancelled` says the caller has gone, while it
        waits. Turns go in the order they were asked for.
        """
        if not self.active:
            return Turn(0.0, self._wall())
        with self._ready:
            pace = self._paces.setdefault(key, _Pace())
            if pace.forgotten:
                now = self._clock()
                raise self._refuse(pace, now, now, len(pace.queue), "the session has gone")
            if len(pace.queue) >= self.max_queued:
                now = self._clock()
                raise self._refuse(pace, now, now, len(pace.queue), "too many calls are queued")
            ticket = next(self._tickets)
            pace.queue.append(ticket)
            started = self._clock()
            deadline = started + max(0.0, budget_s)
            granted = False
            waited = False
            try:
                while True:
                    now = self._clock()
                    ahead = pace.queue.index(ticket)
                    first = ahead == 0
                    turn = self._earliest(pace, now)
                    if cancelled is not None and cancelled():
                        raise self._refuse(pace, started, now, ahead, "the caller went away")
                    if pace.forgotten:
                        raise self._refuse(pace, started, now, ahead, "the session has gone")
                    # A caller woken late may find its deadline already gone:
                    # it has given up, so it is not let through. One that has
                    # not waited yet is still on its way in, however small
                    # its budget, and may go if nothing stands in its way.
                    if waited and now > deadline:
                        raise self._refuse(pace, started, now, ahead, "the wait ran out")
                    if first and not pace.out and turn <= now:
                        granted = True
                        break
                    if skip_if_busy:
                        raise self._refuse(
                            pace, started, now, ahead, "the caller asked not to wait"
                        )
                    # However quickly the calls ahead run, the pause after
                    # each and the cap still stand between this call and its
                    # turn. When those alone come past the wait, stop now.
                    if self._soonest(pace, now, ahead) > deadline:
                        reason = "the wait ran out" if waited else "the turn is past the wait"
                        raise self._refuse(pace, started, now, ahead, reason)
                    if now >= deadline:
                        raise self._refuse(pace, started, now, ahead, "the wait ran out")
                    until = deadline if (pace.out or not first) else min(turn, deadline)
                    if cancelled is not None:
                        until = min(until, now + CANCEL_POLL_S)
                    self._wait(self._ready, max(0.0, until - now))
                    waited = True
            finally:
                if not granted:
                    pace.queue.remove(ticket)
                    self._drop_if_forgotten(key, pace)
                    self._ready.notify_all()
            pace.queue.popleft()
            pace.out = True
            pace.out_id = operation_id
            pace.out_since = now
            if self.max_per_s > 0:
                # The start this pushes out is a window old already, or the
                # cap would not have let this one through, so it is not kept.
                pace.starts.append(now)
                while len(pace.starts) > self.max_per_s:
                    pace.starts.popleft()
            return Turn(max(0.0, now - started), self._wall())

    def done(self, key: str, *, ran: bool = True) -> None:
        """Mark the end of a call that had a turn, and wake whoever is next.

        `ran` is false for a turn given back unused. Nothing reached the
        session, so the turn neither counts against the cap nor starts a
        pause, and says nothing about how long the session's calls take.
        """
        if not self.active:
            return
        with self._ready:
            pace = self._paces.get(key)
            if pace is None:
                return
            now = self._clock()
            if pace.out and pace.out_since is not None:
                if ran:
                    pace.durations.append(max(0.0, now - pace.out_since))
                elif pace.starts and pace.starts[-1] == pace.out_since:
                    pace.starts.pop()
            if ran:
                pace.last_end = now
            pace.out = False
            pace.out_id = None
            pace.out_since = None
            self._drop_if_forgotten(key, pace)
            self._ready.notify_all()

    def holds(self, key: str, operation_id: str | None) -> bool:
        """Whether the call out at this session carries this operation id."""
        if not operation_id:
            return False
        with self._ready:
            pace = self._paces.get(key)
            return pace is not None and pace.out and pace.out_id == operation_id

    def forget(self, key: str) -> None:
        """Drop what is kept for a session that has gone, what its calls took
        with the rest. With a call out or queued it is dropped when the last
        of them leaves, and the queued ones are refused at once."""
        with self._ready:
            pace = self._paces.get(key)
            if pace is None:
                return
            pace.forgotten = True
            self._drop_if_forgotten(key, pace)
            self._ready.notify_all()

    def _drop_if_forgotten(self, key: str, pace: _Pace) -> None:
        if pace.forgotten and not pace.out and not pace.queue and self._paces.get(key) is pace:
            del self._paces[key]

    def _earliest(self, pace: _Pace, now: float) -> float:
        """The first moment the pause and the cap allow, the call out aside."""
        turn = now
        if self.min_pause_s > 0 and pace.last_end is not None:
            turn = max(turn, pace.last_end + self.min_pause_s)
        if self.max_per_s > 0 and len(pace.starts) >= self.max_per_s:
            turn = max(turn, pace.starts[-self.max_per_s] + WINDOW_S)
        return turn

    def _soonest(self, pace: _Pace, now: float, ahead: int) -> float:
        """The first moment a call with `ahead` calls queued before it could
        have its turn were every call to take no time at all. Only the pause
        and the cap count here, so no turn can come sooner."""
        starts = list(pace.starts)
        turn = now
        if pace.out:
            # The call out ends no sooner than now, and the pause follows.
            turn = now + self.min_pause_s
        elif self.min_pause_s > 0 and pace.last_end is not None:
            turn = max(turn, pace.last_end + self.min_pause_s)
        for index in range(ahead + 1):
            if index:
                # The call before ends no sooner than it starts.
                turn += self.min_pause_s
            if self.max_per_s > 0 and len(starts) >= self.max_per_s:
                turn = max(turn, starts[-self.max_per_s] + WINDOW_S)
            starts.append(turn)
        return turn

    def _estimate(self, pace: _Pace, now: float, ahead: int) -> float:
        """Seconds until a turn could come with `ahead` calls queued first."""
        took = statistics.median(pace.durations) if pace.durations else RUN_SEED_S
        left = 0.0
        if pace.out:
            since = now if pace.out_since is None else pace.out_since
            out_for = max(0.0, now - since)
            # The call out has to end first, and then the pause runs. Past
            # what calls usually take, it is guessed to run as long again.
            left = (took - out_for if out_for < took else out_for) + self.min_pause_s
        guess = max(
            left + ahead * (took + self.min_pause_s),
            self._soonest(pace, now, ahead) - now,
        )
        return min(max(guess, self.min_pause_s, 0.001), RETRY_CAP_S)

    def _refuse(self, pace: _Pace, started: float, now: float, ahead: int, reason: str) -> NoTurn:
        return NoTurn(
            waited_s=max(0.0, now - started),
            retry_after_s=round(self._estimate(pace, now, ahead), 3),
            reason=reason,
            ahead=ahead,
        )


def throttled_ms(waited_s: float) -> int:
    """A wait as the whole milliseconds a reply reports, nothing under one."""
    return int(round(waited_s * 1000.0)) if waited_s >= 0.0005 else 0
