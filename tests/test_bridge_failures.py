"""What a session does when the other end of a call goes away.

Three failures are produced on purpose here, because each one is a case the
design claims to survive and none of them can be waited for:

- The caller is ended in the middle of its own call. The work has to finish,
  the receipt has to be settled, and the session has to be there afterwards.
- The session is ended outright. Anything still addressing it has to be
  refused before a single byte is sent.
- The answer is thrown away after the work has already run. The second send of
  the same operation id has to answer from the receipt rather than run again.

Skipped, not failed, when there is no Houdini on this machine. One session at a
time, started here and stopped here, and never the one a person is working in.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import support
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client
from nscr_houdini_mcp.bridge.launcher import HythonBridge, hython_available

pytestmark = [
    pytest.mark.houdini,
    pytest.mark.skipif(not hython_available(), reason="no hython on this machine"),
]

PORT_RANGE = support.FAILURE_PORTS

# How long the session holds an answer a call asked it to lose, and how long
# the caller waits before it decides the answer is not coming. The caller has
# to give up first, or nothing is ever lost.
HOLD_S = 8.0
GIVE_UP_S = 2.0

# Long enough that a caller can be ended while the work is still running.
LONG_CALL_S = 12.0


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


@pytest.fixture
def bridge(home: Path) -> Iterator[HythonBridge]:
    with support.hython_session(
        home,
        port_range=PORT_RANGE,
        alias="failures",
        extra_args=["--drop-reply-s", str(HOLD_S)],
    ) as started:
        yield started


def nodes(bridge: HythonBridge) -> int:
    answer = bridge.call("scene.info")
    assert answer.payload["ok"] is True, answer.payload
    return int(answer.payload["data"]["nodes"]["/obj"])


# Section: the caller goes away in the middle of its own call


def test_a_caller_that_dies_mid_call_leaves_the_work_finished_and_the_session_up(
    bridge: HythonBridge, home: Path, tmp_path: Path
) -> None:
    """The call is running, the process that sent it is ended, nothing else is.

    What the session owes the caller is gone with the caller. What the session
    owes itself is not: the edit finishes, the receipt says so, and the next
    caller finds a session that is free rather than one stuck holding a call.
    """
    before = nodes(bridge)
    marker = tmp_path / "sent.txt"

    def running() -> bool:
        return bridge.health().payload["data"]["busy"] is True

    with support.client_that_dies_mid_call(
        home,
        bridge.session_id,
        tool="bridge.selfcheck",
        arguments={"creates": 1, "sleep_s": LONG_CALL_S},
        marker=marker,
        ready=running,
    ):
        operation_id = marker.read_text(encoding="utf-8").strip()
        # The work was never told to stop, so it runs to the end by itself.
        support.wait_until(lambda: bridge.health().payload["data"]["busy"] is False)

    assert nodes(bridge) == before + 1

    # The receipt is settled, so a fresh caller sending the same id is answered
    # from it rather than making a second node.
    with store_module.Store(home / store_module.STORE_FILE_NAME) as store:
        record = store.get_operation(operation_id)
    assert record is not None
    assert record.state == "done"

    again = bridge.call(
        "bridge.selfcheck",
        arguments={"creates": 1, "sleep_s": LONG_CALL_S},
        operation_id=operation_id,
        wait_s=10.0,
    )
    assert again.payload["replayed"] is True
    assert nodes(bridge) == before + 1
    assert bridge.health().payload["data"]["queued"] == 0


# Section: the session goes away


def test_a_call_to_a_session_that_was_killed_is_refused_before_anything_is_sent(
    bridge: HythonBridge, home: Path
) -> None:
    session_id = bridge.session_id
    assert support.kill_bridge(bridge) is not None

    with pytest.raises(client.SessionDead) as refused:
        client.Session.open(home, session_id)
    details = refused.value.details()
    assert details["code"] == "SESSION_DEAD"
    assert details["alias"] == "failures"
    assert details["live_session_id"] is None


# Section: the answer is thrown away


def test_an_answer_thrown_away_after_the_work_ran_is_replayed_from_the_receipt(
    bridge: HythonBridge,
) -> None:
    before = nodes(bridge)
    operation_id = client.new_operation_id()
    arguments = {"creates": 1, "drop_reply": True}

    answer = bridge.call(
        "bridge.selfcheck",
        arguments=arguments,
        operation_id=operation_id,
        http_timeout_s=GIVE_UP_S,
    )
    assert answer.payload["ok"] is True, answer.payload
    assert answer.payload["replayed"] is True
    assert nodes(bridge) == before + 1

    wrong = bridge.call(
        "bridge.selfcheck", arguments={"creates": 2}, operation_id=operation_id, wait_s=10.0
    )
    assert wrong.payload["error"]["code"] == "OPERATION_MISMATCH"
    assert nodes(bridge) == before + 1
