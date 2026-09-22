"""Getting work to the main thread of a graphical session, and giving up on it.

The fake Houdini here holds an object model lock for the whole of every
callback its main thread runs, and posting takes that lock, which is the one
behaviour the whole busy path turns on: a post during a cook blocks the thread
that posts. What these check is that no thread answering a request is ever that
thread, that work a caller gave up on never runs afterwards, and that the main
thread is reached whether or not a posted callback lands.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from fake_hou import Scene
from nscr_houdini_mcp.bridge import marshal


@pytest.fixture
def scene() -> Iterator[Scene]:
    made = Scene()
    try:
        yield made
    finally:
        made.ui.stop()


@pytest.fixture
def runners() -> Iterator[Any]:
    """Runners that are stopped again, whatever the test did to them."""
    made: list[marshal.MainThreadRunner] = []

    def build(
        module: Any, *, pulse: marshal.Pulse | None = None, start: bool = True
    ) -> marshal.MainThreadRunner:
        runner = marshal.MainThreadRunner(module, pulse=pulse)
        made.append(runner)
        if start:
            runner.start()
        return runner

    try:
        yield build
    finally:
        for runner in made:
            runner.stop()


# Section: the marshal and the token


def test_a_post_that_blocks_on_the_main_thread_does_not_block_submit(
    scene: Scene, runners: Any
) -> None:
    scene.ui.start()
    runner = runners(scene.module())
    ran: list[int] = []

    scene.ui.cook(1.0)
    _until(lambda: scene.ui.ran_on != [])

    work = marshal.Work(lambda: ran.append(1))
    began = time.monotonic()
    runner.submit(work)
    assert time.monotonic() - began < 0.05

    assert work.started.wait(0.3) is False
    assert work.cancel() is True

    _until(lambda: len(scene.ui.ran_on) > 1, timeout_s=5.0)
    time.sleep(0.3)
    assert work.taken is False
    assert ran == []


def _race_once() -> None:
    """One request thread giving up while the main thread picks the work up."""
    ran: list[int] = []
    cancelled: list[bool] = []
    work = marshal.Work(lambda: ran.append(1))
    both = threading.Barrier(2)

    def give_up() -> None:
        both.wait(5.0)
        cancelled.append(work.cancel())

    def pick_up() -> None:
        both.wait(5.0)
        work.run(picked_by="kick")

    threads = [threading.Thread(target=give_up), threading.Thread(target=pick_up)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)

    if cancelled == [True]:
        assert ran == []
        assert work.taken is False
    else:
        assert cancelled == [False]
        assert ran == [1]


def test_a_cancelled_work_is_never_executed() -> None:
    """The claim is one decision: it runs, or it is cancelled, never both."""
    for _ in range(200):
        _race_once()


def test_work_cancelled_before_the_kick_lands_is_dropped_by_the_drain(
    scene: Scene, runners: Any
) -> None:
    scene.ui.start()
    runner = runners(scene.module())
    ran: list[str] = []

    scene.ui.cook(0.6)
    _until(lambda: scene.ui.ran_on != [])

    works = [
        marshal.Work(lambda name=name: ran.append(name)) for name in ("first", "second", "third")
    ]
    for work in works:
        runner.submit(work)
    assert works[1].cancel() is True

    assert works[2].finished.wait(5.0)
    assert ran == ["first", "third"]


def test_one_kick_serves_every_queued_work(scene: Scene, runners: Any) -> None:
    scene.ui.start()
    runner = runners(scene.module())
    ran: list[int] = []

    scene.ui.cook(0.4)
    _until(lambda: scene.ui.ran_on != [])

    works = [marshal.Work(lambda index=index: ran.append(index)) for index in range(5)]
    for work in works:
        runner.submit(work)

    assert works[-1].finished.wait(5.0)
    assert ran == [0, 1, 2, 3, 4]
    assert [work.picked_by for work in works] == ["kick"] * 5
    # The cook, then one kick for all five. No pile of posted callbacks.
    assert len(scene.ui.ran_on) == 2


def test_the_loop_callback_drains_when_no_kick_has_landed(scene: Scene, runners: Any) -> None:
    """Nothing posts here at all, and the work still reaches the main thread."""
    module = scene.module()
    pulse = marshal.Pulse()
    runner = runners(module, pulse=pulse, start=False)
    pulse.install(module)
    scene.ui.start()
    try:
        ran: list[int] = []
        work = marshal.Work(lambda: ran.append(1))
        runner.submit(work)
        assert work.finished.wait(5.0)
        assert work.picked_by == "loop"
        assert ran == [1]
    finally:
        pulse.uninstall()


def test_a_post_the_interface_refuses_rejects_every_waiting_work(
    scene: Scene, runners: Any
) -> None:
    module = scene.module()
    module.ui = _RefusingInterface()
    runner = runners(module)

    work = marshal.Work(lambda: "never")
    runner.submit(work)
    assert work.finished.wait(0.5)
    assert isinstance(work.error, marshal.Rejected)


def test_the_posted_callback_needs_no_removal_call(scene: Scene, runners: Any) -> None:
    """There is no call to take a posted callback off, and none is needed."""
    assert not hasattr(scene.ui, "removeEventCallback")
    runner = runners(scene.module())
    ran: list[int] = []

    work = marshal.Work(lambda: ran.append(1))
    runner.submit(work)
    kick = scene.ui.posted.get(timeout=5.0)
    kick()
    kick()
    assert ran == [1]
    assert work.picked_by == "kick"


def test_stop_does_not_wait_for_a_blocked_poster(scene: Scene, runners: Any) -> None:
    scene.ui.start()
    runner = runners(scene.module())

    scene.ui.cook(1.0)
    _until(lambda: scene.ui.ran_on != [])

    work = marshal.Work(lambda: "never")
    runner.submit(work)
    poster = runner._poster

    began = time.monotonic()
    runner.stop()
    assert time.monotonic() - began < 0.3
    assert work.cancelled is True

    # The poster is still inside the post call, and ends when the cook does.
    assert poster is not None
    poster.join(5.0)
    assert poster.is_alive() is False


# Section: the pulse


def test_the_pulse_ages_when_the_main_thread_stops_ticking(scene: Scene) -> None:
    module = scene.module()
    pulse = marshal.Pulse(stale_s=0.3)
    pulse.install(module)
    scene.ui.start()
    try:
        _until(lambda: scene.ui.ticks > 1)
        assert pulse.age_s() < 0.1
        assert pulse.away() is False

        scene.ui.cook(1.0)
        _until(lambda: scene.ui.ran_on != [])
        _until(lambda: pulse.away(limit_s=0.5) is True, timeout_s=5.0)
        assert pulse.age_s() > 0.5
    finally:
        pulse.uninstall()


def test_the_pulse_stays_fresh_from_pickups_while_the_loop_is_starved(
    scene: Scene, runners: Any
) -> None:
    module = scene.module()
    pulse = marshal.Pulse(stale_s=0.3)
    runner = runners(module, pulse=pulse)
    pulse.install(module)
    scene.ui.starve_loop = True
    scene.ui.start()
    try:
        ages: list[float] = []
        for _ in range(3):
            work = marshal.Work(lambda: None)
            runner.submit(work)
            assert work.finished.wait(5.0)
            assert work.picked_by == "kick"
            ages.append(pulse.age_s())
            time.sleep(0.1)
        assert scene.ui.ticks == 0
        assert max(ages) < 0.3
        assert pulse.away() is False
    finally:
        pulse.uninstall()


def test_a_pulse_installed_while_the_main_thread_is_away_counts_from_the_install(
    scene: Scene,
) -> None:
    """A pulse with no stamp of its own yet ages from when it was installed.

    The main thread here never runs anything, which is what a session in the
    middle of a cook looks like from the outside.
    """
    now = [100.0]
    module = scene.module()
    pulse = marshal.Pulse(stale_s=2.0, clock=lambda: now[0])
    pulse.install(module)
    try:
        assert pulse.age_s() == 0.0
        assert pulse.away() is False
        now[0] = 103.0
        assert pulse.age_s() == 3.0
        assert pulse.away() is True
    finally:
        pulse.uninstall()


def test_a_pulse_that_could_not_install_never_says_away() -> None:
    pulse = marshal.Pulse(stale_s=0.01)
    assert pulse.installed is False
    assert pulse.age_s() is None
    assert pulse.away() is False
    assert pulse.state() == {"installed": False, "pulse_age_s": None, "away": False}
    with pytest.raises(AttributeError):
        pulse.install(object())
    assert pulse.installed is False


class _RefusingInterface:
    """An interface that will not take a posted callback."""

    def postEventCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        raise RuntimeError("the interface is going away")

    def addEventLoopCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        return None

    def removeEventLoopCallback(self, callback: Any) -> None:  # noqa: N802 - the name is Houdini's
        return None


def _until(ready: Any, timeout_s: float = 10.0) -> None:
    """Wait for something another thread is about to do."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.005)
    raise AssertionError("waited too long")
