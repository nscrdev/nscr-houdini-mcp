"""Several processes hitting one store at the same moment.

Children are started with the spawn method, which is the only one available on
every supported system, so the child imports this module by name and calls
`race` at module level.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from nscr_houdini_mcp.store import PoolFull, Store, digest_arguments

RACERS = 6
JOIN_TIMEOUT_S = 60.0

# Two of three worker slots are taken before the children start, so there is
# exactly one left for all of them to fight over.
POOL_CAP = 3
VERSIONS_PER_RACER = 3
SHARED_OPERATION = "op-shared"


def race(path: str, index: int, barrier, results) -> None:
    """One racer: take a slot, a name, an operation id and some versions."""
    with Store(path) as store:
        barrier.wait()
        report: dict[str, object] = {"index": index}

        try:
            worker = store.reserve_worker(cap=POOL_CAP, token=f"token-{index}")
        except PoolFull:
            report["worker"] = None
        else:
            report["worker"] = worker.alias

        report["alias"] = store.register_session(
            f"session-{index}", kind="hython", pid=index, alias_template="scene-{n}"
        ).alias

        digest = digest_arguments({"node": "/obj/box", "parm": "sx"})
        _, claimed = store.begin_operation(SHARED_OPERATION, digest)
        report["claimed_operation"] = claimed

        report["versions"] = [
            store.allocate_version(kind="render", name="beauty", hip_family="shot")
            for _ in range(VERSIONS_PER_RACER)
        ]
        results.put(report)


@pytest.fixture(scope="module")
def reports(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    """Run the racers once and let every check read the same results."""
    path = tmp_path_factory.mktemp("contention") / "coord.sqlite"
    with Store(path) as store:
        for slot in range(POOL_CAP - 1):
            store.reserve_worker(cap=POOL_CAP, token=f"held-{slot}")

    context = mp.get_context("spawn")
    barrier = context.Barrier(RACERS)
    queue = context.Queue()
    children = [
        context.Process(target=race, args=(str(path), index, barrier, queue))
        for index in range(RACERS)
    ]
    for child in children:
        child.start()
    collected = [queue.get(timeout=JOIN_TIMEOUT_S) for _ in children]
    for child in children:
        child.join(JOIN_TIMEOUT_S)
        if child.is_alive():
            child.terminate()
            child.join(JOIN_TIMEOUT_S)
            pytest.fail("a racer did not finish")
        assert child.exitcode == 0

    collected.sort(key=lambda report: report["index"])
    return collected


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
