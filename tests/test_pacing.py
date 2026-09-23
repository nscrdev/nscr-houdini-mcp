"""Pacing calls to a Houdini with a user interface.

The pacer's rules are checked on their own first, on a clock the test turns,
then once on the real clock with callers side by side, then through the router,
which decides which calls are paced and what the bridge is told, and last
through the server, whose reply says how long a call waited.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client

from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.config import Config, ConfigError, parse_config
from nscr_houdini_mcp.pacing import NoTurn, Pacer, throttled_ms
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.router import Router
from nscr_houdini_mcp.server import _router_for, build_server
from nscr_houdini_mcp.tools.registry import TOOLS
from test_router import FakeFiles, FakeStore, Sent, record


class Clock:
    """A clock that moves only when a caller waits on it, or the test says."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.waits: list[float] = []

    def __call__(self) -> float:
        return self.now

    def wait(self, condition: Any, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def pass_(self, seconds: float) -> None:
        self.now += seconds


def pacer(clock: Clock, *, pause_ms: int = 50, per_s: int = 10, max_queued: int = 32) -> Pacer:
    return Pacer(
        min_pause_s=pause_ms / 1000.0,
        max_per_s=per_s,
        max_queued=max_queued,
        clock=clock,
        wall=clock,
        wait=clock.wait,
    )


def one_call(
    paced: Pacer, clock: Clock, key: str = "gui", *, runs_s: float = 0.0, budget_s: float = 30.0
) -> float:
    turn = paced.admit(key, budget_s=budget_s)
    clock.pass_(runs_s)
    paced.done(key)
    return turn.waited_s


# Section: the rules


def test_the_first_call_goes_at_once() -> None:
    clock = Clock()
    assert one_call(pacer(clock), clock) == 0.0
    assert clock.waits == []


def test_the_pause_counts_from_the_end_of_the_last_call() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock, runs_s=0.2)
    assert one_call(paced, clock) == pytest.approx(0.05)
    assert clock.now == pytest.approx(100.25)


def test_a_call_after_the_pause_has_passed_does_not_wait() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock)
    clock.pass_(0.06)
    assert one_call(paced, clock) == 0.0


def test_no_more_than_the_cap_start_in_any_second() -> None:
    clock = Clock()
    paced = pacer(clock, pause_ms=50, per_s=10)
    starts = []
    for _ in range(35):
        paced.admit("gui", budget_s=30.0)
        starts.append(clock.now)
        paced.done("gui")
    for index, start in enumerate(starts):
        inside = [other for other in starts if start - 1.0 < other <= start]
        assert len(inside) <= 10, index
    # Ten go at the pause, the eleventh waits for the window to open.
    assert starts[9] == pytest.approx(100.45)
    assert starts[10] == pytest.approx(101.0)
    gaps = [later - earlier for earlier, later in zip(starts, starts[1:], strict=False)]
    assert min(gaps) >= 0.05 - 1e-9


def test_only_one_call_is_out_at_a_time() -> None:
    clock = Clock()
    paced = pacer(clock)
    paced.admit("gui", budget_s=1.0)
    # The first is still out: a second waits the whole of its budget for it,
    # then is refused, with a floor of the pause for when to come back.
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=0.3)
    assert refused.value.waited_s == pytest.approx(0.3)
    assert refused.value.retry_after_s == pytest.approx(0.05)
    paced.done("gui")
    assert paced.admit("gui", budget_s=1.0).waited_s == pytest.approx(0.05)


def test_a_turn_further_off_than_the_wait_is_refused_at_once() -> None:
    clock = Clock()
    paced = pacer(clock, pause_ms=0, per_s=2)
    one_call(paced, clock)
    one_call(paced, clock)
    before = list(clock.waits)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=0.5)
    # Nothing was slept on: the turn is a second off and the caller gave half.
    assert clock.waits == before
    assert refused.value.waited_s == 0.0
    assert refused.value.retry_after_s == pytest.approx(1.0)
    # Nothing is left behind for the next caller to queue after.
    assert paced.admit("gui", budget_s=2.0).waited_s == pytest.approx(1.0)


def test_a_call_that_asked_to_be_skipped_is_refused_rather_than_held() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=30.0, skip_if_busy=True)
    assert clock.waits == []
    assert refused.value.retry_after_s == pytest.approx(0.05)
    clock.pass_(0.05)
    assert paced.admit("gui", budget_s=0.0, skip_if_busy=True).waited_s == 0.0


def test_sessions_are_paced_apart() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock, "one")
    assert one_call(paced, clock, "two") == 0.0
    assert one_call(paced, clock, "one") == pytest.approx(0.05)


def test_zero_turns_both_rules_off_and_lets_calls_out_side_by_side() -> None:
    clock = Clock()
    paced = pacer(clock, pause_ms=0, per_s=0)
    assert not paced.active
    for _ in range(50):
        paced.admit("gui", budget_s=0.0)
    assert clock.waits == []


def test_forgetting_a_session_lets_its_next_call_go_at_once() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock)
    paced.forget("gui")
    assert one_call(paced, clock) == 0.0


def test_the_cap_alone_can_put_a_queued_turn_past_the_wait() -> None:
    clock = Clock()
    paced = pacer(clock, pause_ms=0, per_s=1)
    one_call(paced, clock)
    queue = paced._paces["gui"].queue
    # One call waits ahead: at one start a second, its turn is at 101 and
    # this call's at 102 at the soonest, however quickly either runs.
    queue.append(-1)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=1.5)
    assert refused.value.reason == "the turn is past the wait"
    assert clock.waits == []
    assert refused.value.waited_s == 0.0
    assert refused.value.ahead == 1
    assert refused.value.retry_after_s == pytest.approx(2.0)
    # With nobody ahead the same wait is long enough.
    queue.clear()
    assert paced.admit("gui", budget_s=1.5).waited_s == pytest.approx(1.0)
    paced.done("gui")
    # With the cap's turn inside the wait, the call queues rather than being
    # refused, and is refused only when the call ahead has not moved by the
    # end of it.
    queue.append(-1)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=2.5)
    assert refused.value.reason == "the wait ran out"
    assert refused.value.waited_s == pytest.approx(2.5)


def test_retry_after_comes_from_recent_durations_not_from_the_timeout() -> None:
    clock = Clock()
    paced = pacer(clock)
    for _ in range(3):
        one_call(paced, clock, runs_s=0.2)
    paced.admit("gui", budget_s=30.0)
    clock.pass_(0.05)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=30.0, skip_if_busy=True)
    # The rest of a usual call, then the pause.
    assert refused.value.retry_after_s == pytest.approx(0.2)


def test_a_full_queue_says_when_a_place_would_come_from_the_depth() -> None:
    clock = Clock()
    paced = pacer(clock, max_queued=4)
    for _ in range(3):
        one_call(paced, clock, runs_s=0.1)
    paced.admit("gui", budget_s=30.0)
    clock.pass_(0.02)
    paced._paces["gui"].queue.extend([-1, -2, -3, -4])
    before = list(clock.waits)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=30.0)
    assert refused.value.reason == "too many calls are queued"
    assert refused.value.waited_s == 0.0
    assert refused.value.ahead == 4
    assert clock.waits == before
    # What the call out has left and the pause, then four calls and pauses.
    assert refused.value.retry_after_s == pytest.approx(0.08 + 0.05 + 4 * 0.15)


def test_a_call_out_past_its_estimate_gives_the_pause_as_the_floor() -> None:
    clock = Clock()
    paced = pacer(clock)
    paced.admit("gui", budget_s=1.0)
    clock.pass_(5.0)
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=30.0, skip_if_busy=True)
    assert refused.value.retry_after_s == pytest.approx(0.05)


def test_retry_after_is_capped() -> None:
    clock = Clock()
    paced = pacer(clock)
    for _ in range(3):
        one_call(paced, clock, runs_s=10.0)
    paced.admit("gui", budget_s=30.0)
    paced._paces["gui"].queue.extend(range(-10, 0))
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=30.0, skip_if_busy=True)
    assert refused.value.ahead == 10
    assert refused.value.retry_after_s == pytest.approx(30.0)


def test_durations_are_learned_per_session_and_forgotten_with_it() -> None:
    clock = Clock()
    paced = pacer(clock)

    def retry(key: str) -> float:
        paced.admit(key, budget_s=30.0)
        try:
            with pytest.raises(NoTurn) as refused:
                paced.admit(key, budget_s=30.0, skip_if_busy=True)
        finally:
            paced.done(key, ran=False)
        return refused.value.retry_after_s

    for _ in range(3):
        one_call(paced, clock, "one", runs_s=1.0)
    assert retry("one") == pytest.approx(1.05)
    # Another session has none of its own yet, so it starts from the seed.
    assert retry("two") == pytest.approx(0.3)
    paced.forget("one")
    assert retry("one") == pytest.approx(0.3)


def test_a_wait_is_reported_in_whole_milliseconds() -> None:
    assert throttled_ms(0.0) == 0
    assert throttled_ms(0.0004) == 0
    assert throttled_ms(0.0506) == 51
    assert throttled_ms(1.25) == 1250


def test_callers_side_by_side_leave_the_session_the_whole_pause_between_calls() -> None:
    # Twenty callers at once, each call taking a tenth of a second on a stand
    # in bridge that runs one call at a time, as the real one does.
    pause = 0.05
    paced = Pacer(min_pause_s=pause, max_per_s=10)
    bridge = threading.Lock()
    spans: list[tuple[float, float]] = []
    kept = threading.Lock()
    go = threading.Barrier(20)

    def caller() -> None:
        go.wait()
        paced.admit("gui", budget_s=30.0)
        try:
            with bridge:
                began = time.monotonic()
                time.sleep(0.1)
                ended = time.monotonic()
            with kept:
                spans.append((began, ended))
        finally:
            paced.done("gui")

    threads = [threading.Thread(target=caller) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(spans) == 20
    spans.sort()
    gaps = [later[0] - earlier[1] for earlier, later in zip(spans, spans[1:], strict=False)]
    # Every gap is the pause at least, so the main thread gets its break.
    assert min(gaps) >= pause - 0.002, gaps


def test_the_queue_for_one_session_is_bounded() -> None:
    paced = Pacer(min_pause_s=0.05, max_per_s=10, max_queued=3)
    paced.admit("gui", budget_s=1.0)
    outcomes: list[str] = []
    kept = threading.Lock()

    def waiter() -> None:
        try:
            paced.admit("gui", budget_s=5.0)
            outcome = "admitted"
            paced.done("gui")
        except NoTurn as refused:
            outcome = refused.reason
        with kept:
            outcomes.append(outcome)

    threads = [threading.Thread(target=waiter) for _ in range(3)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 5.0
    while len(paced._paces["gui"].queue) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    # Three wait; a fourth is refused at once, with when to come back.
    with pytest.raises(NoTurn) as refused:
        paced.admit("gui", budget_s=5.0)
    assert refused.value.reason == "too many calls are queued"
    assert refused.value.waited_s == 0.0
    assert refused.value.ahead == 3
    # Nothing learned yet: the rest of a quarter second for the call out and
    # the pause, at least the pause however long it has been out, then three
    # calls of a quarter second and their pauses.
    assert 0.95 - 1e-9 <= refused.value.retry_after_s <= 1.2 + 1e-9
    paced.done("gui")
    for thread in threads:
        thread.join(10)
    assert outcomes == ["admitted"] * 3


def test_a_call_whose_caller_went_away_leaves_the_queue() -> None:
    paced = Pacer(min_pause_s=0.05, max_per_s=10)
    paced.admit("gui", budget_s=1.0)
    gone = threading.Event()
    seen: list[BaseException] = []

    def waiter() -> None:
        try:
            paced.admit("gui", budget_s=30.0, cancelled=gone.is_set)
        except NoTurn as refused:
            seen.append(refused)

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.1)
    started = time.monotonic()
    gone.set()
    thread.join(5)
    assert time.monotonic() - started < 0.5
    assert [refused.reason for refused in seen] == ["the caller went away"]
    assert len(paced._paces["gui"].queue) == 0


# Section: which calls are paced, and what the bridge is told


class CountingBridge:
    """A stand in bridge that takes a little while per call and counts them."""

    def __init__(
        self,
        *,
        takes_s: float = 0.02,
        hold: threading.Event | None = None,
        hold_first_only: bool = False,
    ) -> None:
        self.takes_s = takes_s
        self.hold = hold
        self.hold_first_only = hold_first_only
        self.tools: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, session: client.Session, tool: str, **rest: Any) -> client.Answer:
        with self.lock:
            self.tools.append(tool)
            first = len(self.tools) == 1
        if self.hold is not None and (first or not self.hold_first_only):
            self.hold.wait(10)
            time.sleep(self.takes_s)
            return client.Answer(200, {"ok": True, "data": {}}, {})
        if self.hold_first_only:
            # A resend meets the first call still running, as the bridge
            # answers it from the receipt: at once, with the job to follow.
            return client.Answer(200, {"ok": True, "data": {"state": "running"}}, {})
        time.sleep(self.takes_s)
        return client.Answer(200, {"ok": True, "data": {}}, {})


def real_router(rows: list, send: Any, *, max_queued: int = 32) -> Router:
    return Router(
        home=Path("."),
        open_store=lambda path: FakeStore(rows),
        open_session=FakeFiles([row.session_id for row in rows]).open,
        send=send,
        renew_lease=lambda store, session_id: None,
        pacer=Pacer(min_pause_s=0.05, max_per_s=10, max_queued=max_queued),
    )


def test_forty_parallel_python_calls_mostly_queue_and_run() -> None:
    bridge = CountingBridge(takes_s=0.05)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    answers: list[str] = []
    refusals: list[CallError] = []
    kept = threading.Lock()
    go = threading.Barrier(40)

    def caller(index: int) -> None:
        go.wait()
        try:
            routed.call(
                target, "python.run", wait_s=50.0, timeout_s=10.0, operation_id=f"op-{index}"
            )
            answer = "sent"
        except CallError as error:
            answer = error.code
            with kept:
                refusals.append(error)
        with kept:
            answers.append(answer)

    threads = [threading.Thread(target=caller, args=(index,)) for index in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(answers) == 40
    # One out and 32 queued is the most that can arrive at once; the rest
    # are told the queue is full, and nothing else is refused.
    assert answers.count("sent") >= 33
    assert len(bridge.tools) == answers.count("sent")
    for error in refusals:
        assert error.details["reason"] == "too many calls are queued"
        assert error.details["queued_ahead"] == 32
        assert 32 * 0.05 <= error.details["retry_after_s"] <= 30.0


def test_short_calls_queue_behind_a_call_that_named_a_timeout() -> None:
    release = threading.Event()
    bridge = CountingBridge(hold=release, hold_first_only=True)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    first = threading.Thread(
        target=lambda: routed.call(
            target, "python.run", operation_id="op-1", wait_s=50.0, timeout_s=10.0
        )
    )
    first.start()
    try:
        while not bridge.tools:
            time.sleep(0.005)
        threading.Timer(0.2, release.set).start()
        started = time.monotonic()
        second = routed.call(target, "python.run", operation_id="op-2", wait_s=50.0, timeout_s=10.0)
        assert time.monotonic() - started < 5.0
        assert second["throttled_ms"] >= 150
        assert bridge.tools == ["python.run", "python.run"]
    finally:
        release.set()
        first.join(10)


def test_a_full_queue_refusal_reports_a_depth_based_retry() -> None:
    release = threading.Event()
    bridge = CountingBridge(hold=release)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge, max_queued=4)
    target = routed.resolve(None)
    outcomes: list[str] = []
    kept = threading.Lock()

    def caller(index: int) -> None:
        try:
            routed.call(target, "python.run", wait_s=30.0, operation_id=f"op-{index}")
            outcome = "sent"
        except CallError as error:
            outcome = error.details.get("reason", error.code)
        with kept:
            outcomes.append(outcome)

    threads = [threading.Thread(target=caller, args=(index,)) for index in range(5)]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not (
            bridge.tools and len(routed.pacer._paces["s-1"].queue) == 4
        ):
            time.sleep(0.005)
        with pytest.raises(CallError) as refused:
            routed.call(target, "python.run", wait_s=30.0, operation_id="op-late")
        details = refused.value.details
        assert details["reason"] == "too many calls are queued"
        assert details["queued_ahead"] == 4
        # Nothing learned yet: what is left of a quarter second for the call
        # out and the pause, then four calls of a quarter second and a pause.
        assert 1.25 - 1e-9 <= details["retry_after_s"] <= 1.5 + 1e-9
        assert "about" in refused.value.message
    finally:
        release.set()
        for thread in threads:
            thread.join(10)
    assert outcomes == ["sent"] * 5


def test_forty_callers_with_a_short_wait_are_mostly_busy_and_only_the_admitted_are_sent() -> None:
    bridge = CountingBridge()
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    answers: list[str] = []
    kept = threading.Lock()
    go = threading.Barrier(40)

    def caller() -> None:
        go.wait()
        try:
            routed.call(target, "bridge.ping", wait_s=0.1)
            answer = "sent"
        except CallError as error:
            answer = error.code
        with kept:
            answers.append(answer)

    threads = [threading.Thread(target=caller) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert len(answers) == 40
    sent = answers.count("sent")
    assert answers.count("SESSION_BUSY") == 40 - sent
    assert sent <= 5
    assert answers.count("SESSION_BUSY") >= 35
    assert len(bridge.tools) == sent


def test_a_queued_call_whose_caller_cancels_never_reaches_the_bridge() -> None:
    release = threading.Event()
    bridge = CountingBridge(hold=release)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    first = threading.Thread(target=lambda: routed.call(target, "bridge.ping", wait_s=5.0))
    first.start()
    while not bridge.tools:
        time.sleep(0.005)
    gone = threading.Event()
    refused: list[CallError] = []

    def queued() -> None:
        try:
            routed.call(target, "python.run", wait_s=30.0, operation_id="op-q", cancelled=gone)
        except CallError as error:
            refused.append(error)

    second = threading.Thread(target=queued)
    second.start()
    time.sleep(0.1)
    gone.set()
    second.join(5)
    release.set()
    first.join(10)
    assert [error.code for error in refused] == ["SESSION_BUSY"]
    assert refused[0].details["reason"] == "the caller went away"
    assert bridge.tools == ["bridge.ping"]


def test_a_cancelled_call_tells_its_thread() -> None:
    import anyio

    from nscr_houdini_mcp.server import in_daemon_thread

    gone = threading.Event()
    finished = threading.Event()

    def work() -> None:
        gone.wait(5)
        finished.set()

    async def cancel_it() -> None:
        with anyio.move_on_after(0.1):
            await in_daemon_thread(work, cancelled=gone)

    anyio.run(cancel_it)
    assert gone.is_set()
    assert finished.wait(5)


def router(rows: list, clock: Clock, *, pace_workers: bool = False, send: Any = None) -> Router:
    return Router(
        home=Path("."),
        open_store=lambda path: FakeStore(rows),
        open_session=FakeFiles([row.session_id for row in rows]).open,
        send=send or Sent(),
        renew_lease=lambda store, session_id: None,
        pacer=pacer(clock),
        pace_workers=pace_workers,
    )


def test_calls_to_a_session_with_an_interface_are_paced_and_say_how_long() -> None:
    clock = Clock()
    routed = router([record("s-1", "scene", kind="gui")], clock)
    target = routed.resolve(None)
    first = routed.call(target, "bridge.ping")
    second = routed.call(target, "bridge.ping")
    assert "throttled_ms" not in first
    assert second["throttled_ms"] == 50
    assert first["admitted_at"] == pytest.approx(100.0)
    assert second["admitted_at"] == pytest.approx(100.05)


def test_the_wait_comes_out_of_the_callers_wait_s() -> None:
    clock = Clock()
    sent = Sent()
    routed = router([record("s-1", "scene", kind="gui")], clock, send=sent)
    target = routed.resolve(None)
    routed.call(target, "bridge.ping", wait_s=5.0)
    routed.call(target, "bridge.ping", wait_s=5.0)
    routed.call(target, "bridge.ping")
    assert sent.calls[0]["wait_s"] == 5.0
    assert sent.calls[1]["wait_s"] == pytest.approx(4.95)
    # No wait named is the bridge's own second, and the wait comes out of that.
    assert sent.calls[2]["wait_s"] == pytest.approx(0.95)


def test_a_turn_past_the_wait_is_busy_at_once_and_nothing_is_sent() -> None:
    clock = Clock()
    sent = Sent()
    routed = router([record("s-1", "scene", kind="gui")], clock, send=sent)
    target = routed.resolve(None)
    for _ in range(10):
        routed.call(target, "bridge.ping", wait_s=5.0)
    waits_before = list(clock.waits)
    with pytest.raises(CallError) as refused:
        routed.call(target, "python.run", wait_s=0.1, operation_id="op-late")
    error = refused.value
    assert error.code == "SESSION_BUSY"
    assert error.details["paced"] is True
    assert error.details["retry_after_s"] == pytest.approx(0.55)
    assert error.details["throttled_ms"] == 0
    assert "nothing was sent" in error.message
    assert clock.waits == waits_before
    assert [call["tool"] for call in sent.calls].count("python.run") == 0


def test_a_call_that_asked_to_be_skipped_is_busy_at_once_with_what_it_waited() -> None:
    clock = Clock()
    sent = Sent()
    routed = router([record("s-1", "scene", kind="gui")], clock, send=sent)
    target = routed.resolve(None)
    routed.call(target, "bridge.ping")
    with pytest.raises(CallError) as refused:
        routed.call(target, "bridge.ping", skip_if_busy=True)
    assert refused.value.code == "SESSION_BUSY"
    assert refused.value.details["retry_after_s"] == pytest.approx(0.05)
    assert len(sent.calls) == 1


def test_workers_are_not_paced_unless_told_to_be() -> None:
    clock = Clock()
    routed = router([record("s-1", "w1")], clock)
    target = routed.resolve(None)
    replies = [routed.call(target, "bridge.ping") for _ in range(20)]
    assert all("throttled_ms" not in reply and "admitted_at" not in reply for reply in replies)
    assert clock.waits == []

    paced = router([record("s-1", "w1")], clock, pace_workers=True)
    target = paced.resolve(None)
    paced.call(target, "bridge.ping")
    assert paced.call(target, "bridge.ping")["throttled_ms"] == 50


def test_a_cancel_is_never_paced() -> None:
    clock = Clock()
    routed = router([record("s-1", "scene", kind="gui")], clock)
    target = routed.resolve(None)
    routed.call(target, "bridge.ping")
    assert "throttled_ms" not in routed.call(target, "bridge.cancel", {"operation_id": "op-1"})


def test_a_call_that_fails_after_its_wait_still_says_how_long() -> None:
    clock = Clock()
    busy = {"ok": False, "error": {"code": "SESSION_BUSY", "message": "busy"}}
    routed = router(
        [record("s-1", "scene", kind="gui")], clock, send=Sent({"ok": True, "data": {}}, busy)
    )
    target = routed.resolve(None)
    routed.call(target, "bridge.ping")
    with pytest.raises(CallError) as refused:
        routed.call(target, "bridge.ping")
    assert refused.value.code == "SESSION_BUSY"
    assert refused.value.trace["throttled_ms"] == 50
    assert refused.value.trace["admitted_at"] == pytest.approx(100.05)


def test_a_lost_reply_still_ends_the_call_for_the_pause() -> None:
    clock = Clock()
    lost = client.BridgeUnreachable("gone quiet")
    rows = [record("s-1", "scene", kind="gui")]
    routed = router(rows, clock, send=Sent(lost))
    target = routed.resolve(None)
    with pytest.raises(CallError):
        routed.call(target, "bridge.ping")
    clock.pass_(0.2)
    target = routed.resolve(None)
    assert "throttled_ms" not in routed.call(target, "bridge.ping")


# Section: the settings


def test_the_pacing_keys_have_their_defaults() -> None:
    config = Config(path=Path("config.toml"))
    assert config.gui_min_pause_ms == 50
    assert config.gui_max_calls_per_s == 10
    assert "gui_min_pause_ms = 50" in config_module.TEMPLATE
    assert "gui_max_calls_per_s = 10" in config_module.TEMPLATE
    assert "0 turns that rule off" in config_module.TEMPLATE


def test_no_config_can_ask_for_workers_to_be_paced() -> None:
    # Pacing workers is for tests, through the constructor, never a key.
    assert not hasattr(Config(path=Path("config.toml")), "treat_workers_as_gui")
    with pytest.raises(ConfigError) as refused:
        parse_config({"treat_workers_as_gui": True}, path=Path("config.toml"))
    assert refused.value.key == "treat_workers_as_gui"
    assert "treat_workers_as_gui" not in config_module.TEMPLATE


def test_the_pacing_keys_are_read_and_checked() -> None:
    raw = {"gui_min_pause_ms": 0, "gui_max_calls_per_s": 25}
    config = parse_config(raw, path=Path("config.toml"))
    assert (config.gui_min_pause_ms, config.gui_max_calls_per_s) == (0, 25)
    for bad in (
        {"gui_min_pause_ms": -1},
        {"gui_max_calls_per_s": -1},
        {"gui_min_pause_ms": 1.5},
        {"gui_max_calls_per_s": 100_000},
    ):
        with pytest.raises(ConfigError) as refused:
            parse_config(bad, path=Path("config.toml"))
        assert refused.value.key == next(iter(bad))


def test_the_server_builds_its_router_with_the_configured_pace() -> None:
    config = Config(path=Path("config.toml"), gui_min_pause_ms=75, gui_max_calls_per_s=4)
    routed = _router_for(config)
    assert routed.pacer.min_pause_s == pytest.approx(0.075)
    assert routed.pacer.max_per_s == 4
    assert routed.pace_workers is False
    assert _router_for(config, pace_workers=True).pace_workers


def test_only_the_constructor_paces_workers() -> None:
    config = Config(path=Path("config.toml"))
    plain = build_server(TOOLS, config_loader=lambda: config)
    paced = build_server(TOOLS, config_loader=lambda: config, pace_workers=True)
    assert plain.runtime.settings()[1].pace_workers is False
    assert paced.runtime.settings()[1].pace_workers is True


# Section: through the server


def test_the_reply_carries_throttled_ms_and_admitted_at_in_its_trace() -> None:
    clock = Clock()
    rows = [record("s-1", "scene", kind="gui")]
    sent = Sent()

    def paced_router(config: Config) -> Router:
        return router(rows, clock, send=sent)

    built = build_server(
        TOOLS,
        config_loader=lambda: Config(path=Path("config.toml")),
        router_factory=paced_router,
    )

    async def talk() -> list[Any]:
        async with Client(built) as connected:
            return [
                await connected.call_tool("hou_python", {"code": "result = 1"}) for _ in range(2)
            ]

    first, second = asyncio.run(talk())
    assert "throttled_ms" not in first.structured_content["trace"]
    assert first.structured_content["trace"]["admitted_at"] == pytest.approx(100.0)
    assert second.structured_content["trace"]["throttled_ms"] == 50


def test_a_resend_of_the_call_that_is_out_goes_straight_to_the_bridge() -> None:
    release = threading.Event()
    bridge = CountingBridge(hold=release, hold_first_only=True)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    first = threading.Thread(
        target=lambda: routed.call(
            target, "python.run", operation_id="op-1", wait_s=5.0, timeout_s=10.0
        )
    )
    first.start()
    try:
        while not bridge.tools:
            time.sleep(0.005)
        started = time.monotonic()
        again = routed.call(target, "python.run", operation_id="op-1", wait_s=5.0)
        took = time.monotonic() - started
        assert took < 0.1
        assert again["data"] == {"state": "running"}
        # Never paced: no turn was taken, so no admission time either.
        assert "admitted_at" not in again and "throttled_ms" not in again
        assert len(routed.pacer._paces["s-1"].queue) == 0
        assert bridge.tools == ["python.run", "python.run"]
    finally:
        release.set()
        first.join(10)


def test_a_call_behind_one_out_past_its_wait_waits_its_wait_then_says_when() -> None:
    # A timeout the call out named is a ceiling, not an estimate, so the call
    # behind it is given its wait; when that runs out it is told when to
    # come back from what calls usually take.
    release = threading.Event()
    bridge = CountingBridge(hold=release)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    first = threading.Thread(
        target=lambda: routed.call(
            target, "python.run", operation_id="op-1", wait_s=0.0, timeout_s=3.0
        )
    )
    first.start()
    try:
        while not bridge.tools:
            time.sleep(0.005)
        started = time.monotonic()
        with pytest.raises(CallError) as refused:
            routed.call(target, "bridge.ping", wait_s=1.0)
        took = time.monotonic() - started
        error = refused.value
        # The whole of its wait and no more, with room for a coarse clock.
        assert 0.9 <= took < 3.0
        assert error.code == "SESSION_BUSY"
        assert error.details["reason"] == "the wait ran out"
        assert error.details["queued_ahead"] == 0
        # Out past what calls usually take, so the pause is all there is to say.
        assert error.details["retry_after_s"] == pytest.approx(0.05)
        assert error.details["throttled_ms"] >= 900
        assert bridge.tools == ["python.run"]
    finally:
        release.set()
        first.join(10)


def test_a_call_behind_one_with_no_timeout_of_its_own_still_waits_its_turn() -> None:
    # A default timeout is a ceiling, not an estimate: a short call behind it
    # is given its wait, and is let through once the first ends.
    release = threading.Event()
    bridge = CountingBridge(hold=release, hold_first_only=True)
    routed = real_router([record("s-1", "scene", kind="gui")], bridge)
    target = routed.resolve(None)
    first = threading.Thread(target=lambda: routed.call(target, "bridge.ping", wait_s=5.0))
    first.start()
    try:
        while not bridge.tools:
            time.sleep(0.005)
        threading.Timer(0.2, release.set).start()
        second = routed.call(target, "bridge.ping", wait_s=2.0)
        assert second["throttled_ms"] >= 200
    finally:
        release.set()
        first.join(10)
