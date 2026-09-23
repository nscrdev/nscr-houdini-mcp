"""Pacing calls to a Houdini with a user interface, on a clock the test turns.

The pacer's rules are checked on their own first, then through the router,
which decides which calls are paced, and last through the server, whose reply
says how long a call waited. Nothing here starts a Houdini or sleeps.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest
from mcp.client.client import Client

from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp import pacing
from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.config import Config, ConfigError, parse_config
from nscr_houdini_mcp.pacing import Pacer, throttled_ms
from nscr_houdini_mcp.results import CallError
from nscr_houdini_mcp.router import Router
from nscr_houdini_mcp.server import _router_for, build_server
from nscr_houdini_mcp.tools.registry import TOOLS
from test_router import FakeFiles, FakeStore, Sent, record


class Clock:
    """A clock that moves only when something sleeps on it, or the test says."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def pass_(self, seconds: float) -> None:
        self.now += seconds


def pacer(clock: Clock, *, pause_ms: int = 50, per_s: int = 10) -> Pacer:
    return Pacer(min_pause_s=pause_ms / 1000.0, max_per_s=per_s, clock=clock, sleep=clock.sleep)


def one_call(paced: Pacer, clock: Clock, key: str = "gui", *, runs_s: float = 0.0) -> float:
    waited = paced.admit(key)
    clock.pass_(runs_s)
    paced.done(key)
    return waited


# Section: the rules


def test_the_first_call_goes_at_once() -> None:
    clock = Clock()
    assert one_call(pacer(clock), clock) == 0.0
    assert clock.slept == []


def test_a_call_straight_after_another_waits_out_the_pause() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock, runs_s=0.2)
    assert one_call(paced, clock) == pytest.approx(0.05)
    # The pause counts from the end of the last call, not from its start.
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
        paced.admit("gui")
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


def test_a_long_call_leaves_room_for_the_next_without_a_wait_from_the_cap() -> None:
    clock = Clock()
    paced = pacer(clock, pause_ms=0, per_s=2)
    assert one_call(paced, clock, runs_s=0.6) == 0.0
    assert one_call(paced, clock, runs_s=0.6) == 0.0
    assert one_call(paced, clock) == 0.0


def test_sessions_are_paced_apart() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock, "one")
    assert one_call(paced, clock, "two") == 0.0
    assert one_call(paced, clock, "one") == pytest.approx(0.05)


def test_zero_turns_both_rules_off() -> None:
    clock = Clock()
    paced = pacer(clock, pause_ms=0, per_s=0)
    assert not paced.active
    for _ in range(50):
        assert one_call(paced, clock) == 0.0
    assert clock.slept == []


def test_calls_sent_side_by_side_each_get_a_turn_of_their_own() -> None:
    # Turns are handed out under the lock and slept outside it, so a second
    # call asking while the first is still running is spaced from its start.
    clock = Clock()
    paced = Pacer(min_pause_s=0.05, max_per_s=10, clock=clock, sleep=lambda seconds: None)
    first = paced.admit("gui")
    second = paced.admit("gui")
    third = paced.admit("gui")
    assert (first, second, third) == (0.0, pytest.approx(0.05), pytest.approx(0.10))


def test_forgetting_a_session_lets_its_next_call_go_at_once() -> None:
    clock = Clock()
    paced = pacer(clock)
    one_call(paced, clock)
    paced.forget("gui")
    assert one_call(paced, clock) == 0.0


def test_a_wait_is_reported_in_whole_milliseconds() -> None:
    assert throttled_ms(0.0) == 0
    assert throttled_ms(0.0004) == 0
    assert throttled_ms(0.0506) == 51
    assert throttled_ms(1.25) == 1250


# Section: which calls are paced


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


def test_workers_are_not_paced_unless_told_to_be() -> None:
    clock = Clock()
    routed = router([record("s-1", "w1")], clock)
    target = routed.resolve(None)
    replies = [routed.call(target, "bridge.ping") for _ in range(20)]
    assert all("throttled_ms" not in reply for reply in replies)
    assert clock.slept == []

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
    assert config.treat_workers_as_gui is False
    assert "gui_min_pause_ms = 50" in config_module.TEMPLATE
    assert "gui_max_calls_per_s = 10" in config_module.TEMPLATE
    # For tests only, so a person never meets it in the template.
    assert "treat_workers_as_gui" not in config_module.TEMPLATE


def test_the_pacing_keys_are_read_and_checked() -> None:
    raw = {"gui_min_pause_ms": 0, "gui_max_calls_per_s": 25, "treat_workers_as_gui": True}
    config = parse_config(raw, path=Path("config.toml"))
    assert (config.gui_min_pause_ms, config.gui_max_calls_per_s) == (0, 25)
    assert config.treat_workers_as_gui is True
    for bad in (
        {"gui_min_pause_ms": -1},
        {"gui_min_pause_ms": 1.5},
        {"gui_max_calls_per_s": 100_000},
        {"treat_workers_as_gui": "yes"},
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
    assert _router_for(Config(path=Path("c.toml"), treat_workers_as_gui=True)).pace_workers


# Section: through the server


def test_the_reply_carries_throttled_ms_in_its_trace_when_it_waited() -> None:
    clock = Clock()
    rows = [record("s-1", "scene", kind="gui")]
    sent = Sent()
    lock = threading.Lock()

    def paced_router(config: Config) -> Router:
        with lock:
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
    assert second.structured_content["trace"]["throttled_ms"] == 50
    assert pacing.DEFAULT_MIN_PAUSE_MS == 50
