"""Who gets the session next.

One call at a time per Houdini process is settled: two handler threads working
the object model at once wedged a headless session for good, at full CPU, with
no exception and no recovery. The lock that enforces that is in the app.

This adds the one thing a lock on its own does not give: order. A plain lock
hands the session to whichever waiter the operating system wakes, so a call
that has been waiting twenty seconds can lose to one that arrived last. Here
every waiter takes a ticket and the session goes to the oldest ticket that is
still waiting. A caller that gives up takes its ticket with it.
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque

# The smallest wait given to the process lock once a waiter is at the head of
# the queue. Without it a zero wait caller could lose a lock that is free.
LOCK_GRACE_S = 0.05


class Gate:
    """The session, handed out in arrival order."""

    def __init__(self, lock: threading.Lock) -> None:
        self._lock = lock
        self._ready = threading.Condition()
        self._queue: deque[int] = deque()
        self._tickets = itertools.count(1)
        self._held = False

    def enter(self, *, wait_s: float, skip_if_busy: bool = False) -> bool:
        """Take the session, waiting no longer than asked. False means busy."""
        with self._ready:
            if skip_if_busy and (self._held or self._queue):
                return False
            ticket = next(self._tickets)
            self._queue.append(ticket)
            deadline = time.monotonic() + max(0.0, wait_s)
            while self._held or self._queue[0] != ticket:
                left = deadline - time.monotonic()
                if left <= 0:
                    self._drop(ticket)
                    return False
                self._ready.wait(left)
            self._queue.popleft()
            self._held = True
            left = deadline - time.monotonic()

        # The process wide lock is taken as well, so anything else that holds
        # it still keeps this call out.
        if not self._lock.acquire(timeout=max(LOCK_GRACE_S, left)):
            with self._ready:
                self._held = False
                self._ready.notify_all()
            return False
        return True

    def leave(self) -> None:
        """Give the session back to whoever has waited longest."""
        with self._ready:
            if not self._held:
                return
            self._held = False
            self._lock.release()
            self._ready.notify_all()

    @property
    def held(self) -> bool:
        return self._held

    def waiting(self) -> int:
        """How many calls are queued behind the one that holds the session."""
        with self._ready:
            return len(self._queue)

    def _drop(self, ticket: int) -> None:
        """Take a ticket out of the queue and wake whoever is next."""
        try:
            self._queue.remove(ticket)
        except ValueError:
            pass
        self._ready.notify_all()
