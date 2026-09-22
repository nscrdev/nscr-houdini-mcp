"""Several processes hitting one store at the same moment.

Children are started with the spawn method, which is the only one available on
every supported system, so a child imports this module by name and calls the
function it was given at module level. The runner that starts them is shared
with the other files that need a crowd, in `support`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import support
from nscr_houdini_mcp.store import PoolFull, Store, digest_arguments

RACERS = 6
OPENERS = 12
BARRIER_TIMEOUT_S = support.BARRIER_TIMEOUT_S

# Two of three worker slots are taken before the children start, so there is
# exactly one left for all of them to fight over.
POOL_CAP = 3
VERSIONS_PER_RACER = 3
SHARED_OPERATION = "op-shared"


def race(path: str, index: int, barrier, results) -> None:
    """One racer: take a slot, a name, an operation id and some versions."""
    report: dict[str, object] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        with Store(path) as store:
            barrier.wait(BARRIER_TIMEOUT_S)
            try:
                report["worker"] = store.reserve_worker(cap=POOL_CAP, token=f"token-{index}").alias
            except PoolFull:
                report["worker"] = None
            # Nobody leaves until everybody has asked, so an early exit cannot
            # free a slot that a later racer would then be handed.
            barrier.wait(BARRIER_TIMEOUT_S)

            report["alias"] = store.register_session(
                f"session-{index}", kind="hython", pid=os.getpid(), alias_template="scene-{n}"
            ).alias

            digest = digest_arguments({"node": "/obj/box", "parm": "sx"})
            report["claimed_operation"] = store.begin_operation(SHARED_OPERATION, digest).claimed

            report["versions"] = [
                store.allocate_version(kind="render", name="beauty", hip_family="shot")
                for _ in range(VERSIONS_PER_RACER)
            ]
            # A racer that exits early is a dead owner, and a dead owner's
            # claim may be taken over. Everybody stays alive until the end.
            barrier.wait(BARRIER_TIMEOUT_S)
    except BaseException as error:  # reported, so a failure reads as a message
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def open_fresh(path: str, index: int, barrier, results) -> None:
    """Open a store that does not exist yet, at the same moment as the others."""
    report: dict[str, object] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        barrier.wait(BARRIER_TIMEOUT_S)
        with Store(path) as store:
            report["schema"] = store.schema_version()
            report["alias"] = store.register_session(
                f"session-{index}", kind="hython", pid=os.getpid(), alias_template="scene-{n}"
            ).alias
            # A process that has gone loses its name to the next one, so
            # nobody leaves until every name has been handed out.
            barrier.wait(BARRIER_TIMEOUT_S)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def run_children(target, count: int, path: Path) -> list[dict]:
    """Start `count` spawned children on one store and collect their reports."""
    return support.run_children(target, count, (str(path),), barrier=True)


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    """Run the racers once and let every check read the same results."""
    path = tmp_path_factory.mktemp("contention") / "coord.sqlite"
    with Store(path) as store:
        for slot in range(POOL_CAP - 1):
            store.reserve_worker(cap=POOL_CAP, token=f"held-{slot}")
    return run_children(race, RACERS, path)


def test_the_racers_really_are_separate_processes(reports: list[dict]) -> None:
    pids = {report["pid"] for report in reports}
    assert len(pids) == RACERS
    assert os.getpid() not in pids


def test_exactly_one_racer_takes_the_last_worker_slot(reports: list[dict]) -> None:
    winners = [report for report in reports if report["worker"] is not None]
    assert len(winners) == 1
    assert winners[0]["worker"] == f"w{POOL_CAP}"


def test_every_racer_gets_its_own_alias(reports: list[dict]) -> None:
    aliases = [report["alias"] for report in reports]
    assert sorted(aliases) == sorted(f"scene-{n}" for n in range(1, RACERS + 1))


def test_one_racer_claims_the_shared_operation_id(reports: list[dict]) -> None:
    claims = [report for report in reports if report["claimed_operation"]]
    assert len(claims) == 1


def test_a_version_number_is_never_handed_out_twice(reports: list[dict]) -> None:
    numbers = [version for report in reports for version in report["versions"]]
    assert sorted(numbers) == list(range(1, RACERS * VERSIONS_PER_RACER + 1))


def test_a_crowd_can_create_the_same_store_at_once(tmp_path: Path) -> None:
    """The first move of every process is the one that has to survive a crowd."""
    path = tmp_path / "fresh" / "coord.sqlite"
    collected = run_children(open_fresh, OPENERS, path)
    with Store(path) as opened:
        assert {report["schema"] for report in collected} == {opened.schema_version()}
    assert sorted(report["alias"] for report in collected) == sorted(
        f"scene-{n}" for n in range(1, OPENERS + 1)
    )
