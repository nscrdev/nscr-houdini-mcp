from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.store import (
    AliasInUse,
    OperationMismatch,
    PoolFull,
    Store,
    StoreError,
    UnknownRecord,
    default_home,
    default_store_path,
    digest_arguments,
    write_export,
)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    with Store(tmp_path / "coord.sqlite") as opened:
        yield opened


# -- location and schema --------------------------------------------------


def test_home_follows_the_environment_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(store_module.HOME_ENV_VAR, str(tmp_path / "elsewhere"))
    assert default_home() == tmp_path / "elsewhere"
    assert default_store_path() == tmp_path / "elsewhere" / "coord.sqlite"


def test_home_without_an_override_is_a_per_user_folder(monkeypatch) -> None:
    monkeypatch.delenv(store_module.HOME_ENV_VAR, raising=False)
    home = default_home()
    assert home.name == store_module.APP_DIR_NAME
    assert home.is_absolute()


def test_open_creates_the_file_in_wal_mode_at_the_current_schema(tmp_path) -> None:
    path = tmp_path / "nested" / "coord.sqlite"
    with Store(path) as opened:
        assert path.is_file()
        assert opened.schema_version() == store_module.SCHEMA_VERSION
        mode = opened._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"


def test_opening_an_existing_file_leaves_records_alone(tmp_path) -> None:
    path = tmp_path / "coord.sqlite"
    with Store(path) as first:
        first.register_session("s1", kind="gui", pid=1, alias="scene-1")
    with Store(path) as second:
        assert second.schema_version() == store_module.SCHEMA_VERSION
        assert second.resolve_session("scene-1").session_id == "s1"


def test_a_file_from_an_older_schema_is_migrated_forward(tmp_path) -> None:
    path = tmp_path / "coord.sqlite"
    sqlite3.connect(str(path)).close()
    with Store(path) as opened:
        assert opened.schema_version() == store_module.SCHEMA_VERSION
        assert opened.list_sessions() == []


def test_transactions_do_not_nest(store: Store) -> None:
    with store._txn(write=True):
        with pytest.raises(StoreError):
            with store._txn(write=True):
                pass


# -- sessions -------------------------------------------------------------


def test_register_and_resolve_by_id_or_alias(store: Store) -> None:
    record = store.register_session(
        "s1", kind="gui", pid=42, port=9000, alias="shot-1", hip_path="/scenes/shot.hip"
    )
    assert record.alias == "shot-1"
    assert record.scene_epoch == 0
    assert store.resolve_session("s1") == record
    assert store.resolve_session("shot-1") == record
    assert store.resolve_session("nothing") is None


def test_capabilities_survive_a_round_trip(store: Store) -> None:
    store.register_session(
        "s1", kind="hython", pid=1, alias="w1", capabilities={"gui": False, "routes": ["rop"]}
    )
    assert store.get_session("s1").capabilities == {"gui": False, "routes": ["rop"]}


def test_an_alias_cannot_be_taken_twice_while_it_is_live(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    with pytest.raises(AliasInUse):
        store.register_session("s2", kind="gui", pid=2, alias="shot-1")


def test_an_alias_template_takes_the_lowest_free_name(store: Store) -> None:
    first = store.register_session("s1", kind="gui", pid=1, alias_template="shot-{n}")
    second = store.register_session("s2", kind="gui", pid=2, alias_template="shot-{n}")
    assert [first.alias, second.alias] == ["shot-1", "shot-2"]
    store.end_session("s1")
    third = store.register_session("s3", kind="gui", pid=3, alias_template="shot-{n}")
    assert third.alias == "shot-1"


def test_register_wants_exactly_one_of_alias_or_template(store: Store) -> None:
    with pytest.raises(ValueError):
        store.register_session("s1", kind="gui", pid=1)
    with pytest.raises(ValueError):
        store.register_session("s1", kind="gui", pid=1, alias="a", alias_template="a-{n}")


def test_unknown_session_kind_and_state_are_refused(store: Store) -> None:
    with pytest.raises(ValueError):
        store.register_session("s1", kind="mystery", pid=1, alias="a")
    with pytest.raises(ValueError):
        store.register_session("s2", kind="gui", pid=1, alias="b", state="mystery")


def test_a_session_id_stays_readable_after_the_process_is_gone(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    store.end_session("s1")
    assert store.get_session("s1").state == "gone"
    assert store.resolve_session("shot-1") is None
    assert store.list_sessions() == []
    assert [s.session_id for s in store.list_sessions(include_gone=True)] == ["s1"]


def test_heartbeat_and_state_move_together(store: Store) -> None:
    started = store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    beat = store.touch_session("s1", state="busy")
    record = store.get_session("s1")
    assert record.state == "busy"
    assert record.heartbeat_at == beat >= started.heartbeat_at


def test_scene_epoch_counts_up_and_records_the_new_hip(store: Store) -> None:
    store.register_session("s1", kind="gui", pid=1, alias="shot-1")
    assert store.bump_scene_epoch("s1") == 1
    assert store.bump_scene_epoch("s1", hip_path="/scenes/other.hip") == 2
    assert store.get_session("s1").hip_path == "/scenes/other.hip"


def test_session_helpers_report_an_unknown_id(store: Store) -> None:
    for call in (
        lambda: store.touch_session("nope"),
        lambda: store.bump_scene_epoch("nope"),
        lambda: store.end_session("nope"),
    ):
        with pytest.raises(UnknownRecord):
            call()


# -- workers --------------------------------------------------------------


def test_reservations_fill_the_pool_and_then_refuse(store: Store) -> None:
    first = store.reserve_worker(cap=2, token="t1")
    second = store.reserve_worker(cap=2, token="t2")
    assert [first.alias, second.alias] == ["w1", "w2"]
    assert first.state == "reserved"
    with pytest.raises(PoolFull):
        store.reserve_worker(cap=2, token="t3")


def test_a_starting_worker_still_holds_its_slot(store: Store) -> None:
    store.reserve_worker(cap=1, token="t1")
    store.set_worker_state("t1", "starting")
    with pytest.raises(PoolFull):
        store.reserve_worker(cap=1, token="t2")


def test_a_failed_start_gives_the_slot_back(store: Store) -> None:
    store.reserve_worker(cap=1, token="t1")
    store.release_worker("t1", state="failed")
    replacement = store.reserve_worker(cap=1, token="t2")
    assert replacement.alias == "w1"
    assert store.get_worker("t1").state == "failed"


def test_worker_state_moves_carry_the_session_and_job(store: Store) -> None:
    store.reserve_worker(cap=2, token="t1")
    record = store.set_worker_state("t1", "running", session_id="s9", job_id="j1")
    assert (record.state, record.session_id, record.job_id) == ("running", "s9", "j1")
    kept = store.set_worker_state("t1", "leased")
    assert (kept.session_id, kept.job_id) == ("s9", "j1")


def test_worker_moves_need_a_known_token_and_a_known_state(store: Store) -> None:
    store.reserve_worker(cap=1, token="t1")
    with pytest.raises(UnknownRecord):
        store.set_worker_state("borrowed", "running")
    with pytest.raises(ValueError):
        store.set_worker_state("t1", "mystery")
    with pytest.raises(ValueError):
        store.release_worker("t1", state="running")


def test_listing_shows_active_reservations_only_by_default(store: Store) -> None:
    store.reserve_worker(cap=3, token="t1")
    store.reserve_worker(cap=3, token="t2")
    store.release_worker("t2")
    assert [w.token for w in store.list_workers()] == ["t1"]
    assert {w.token for w in store.list_workers(active_only=False)} == {"t1", "t2"}


def test_an_idle_lease_expires_but_a_worker_on_a_job_never_does(store: Store) -> None:
    store.reserve_worker(cap=3, token="idle")
    store.reserve_worker(cap=3, token="busy", job_id="j1")
    assert [w.token for w in store.idle_workers(max_idle_s=-1)] == ["idle"]
    assert store.idle_workers(max_idle_s=600) == []
    store.touch_worker_lease("idle")
    with pytest.raises(UnknownRecord):
        store.touch_worker_lease("gone")


def test_a_cap_below_one_is_a_mistake(store: Store) -> None:
    with pytest.raises(ValueError):
        store.reserve_worker(cap=0, token="t1")


# -- operation receipts ---------------------------------------------------


def test_argument_digests_ignore_key_order_and_notice_a_change() -> None:
    assert digest_arguments({"a": 1, "b": [2, 3]}) == digest_arguments({"b": [2, 3], "a": 1})
    assert digest_arguments({"a": 1}) != digest_arguments({"a": 2})


def test_a_retry_with_the_same_digest_gets_the_stored_outcome(store: Store) -> None:
    digest = digest_arguments({"node": "/obj/box", "parm": "sx"})
    first, claimed = store.begin_operation("op1", digest, session_id="s1", scene_epoch=3)
    assert claimed is True
    assert first.state == "running"

    store.finish_operation("op1", outcome={"created": ["/obj/box"]})
    again, claimed_again = store.begin_operation("op1", digest)
    assert claimed_again is False
    assert again.state == "done"
    assert again.outcome == {"created": ["/obj/box"]}
    assert again.scene_epoch == 3


def test_the_same_id_with_other_arguments_is_a_mismatch(store: Store) -> None:
    store.begin_operation("op1", digest_arguments({"parm": "sx"}))
    with pytest.raises(OperationMismatch):
        store.begin_operation("op1", digest_arguments({"parm": "sy"}))


def test_a_failed_operation_keeps_its_error(store: Store) -> None:
    digest = digest_arguments({"parm": "sx"})
    store.begin_operation("op1", digest)
    store.finish_operation("op1", state="failed", error={"code": "PARM_NOT_FOUND"})
    stored, claimed = store.begin_operation("op1", digest)
    assert claimed is False
    assert (stored.state, stored.error) == ("failed", {"code": "PARM_NOT_FOUND"})


def test_an_operation_can_point_at_a_job(store: Store) -> None:
    store.begin_operation("op1", digest_arguments({}))
    record = store.finish_operation("op1", outcome={"state": "running"}, job_id="j1")
    assert record.job_id == "j1"
    assert store.get_operation("op1").job_id == "j1"
    assert store.get_operation("missing") is None


def test_finishing_an_unknown_operation_is_refused(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.finish_operation("op1")
    with pytest.raises(ValueError):
        store.begin_operation("op2", "d")
        store.finish_operation("op2", state="mystery")


def test_old_receipts_are_pruned(store: Store) -> None:
    store.begin_operation("op1", "d")
    assert store.prune_operations(max_age_s=600) == 0
    assert store.prune_operations(max_age_s=-1) == 1
    assert store.get_operation("op1") is None


# -- jobs -----------------------------------------------------------------


def test_a_job_records_the_scene_it_consumes(store: Store) -> None:
    scene = {"snapshot": "$HIP/.agent/jobs/j1/shot.hip", "hash": "abc", "scene_epoch": 2}
    record = store.create_job("j1", kind="render", session_id="s1", weight="heavy", scene=scene)
    assert (record.state, record.weight, record.scene) == ("queued", "heavy", scene)
    assert record.finished_at is None


def test_job_updates_keep_the_fields_they_are_not_given(store: Store) -> None:
    store.create_job("j1", kind="render")
    store.update_job("j1", state="running", progress={"frames_done": 1, "frames": 10})
    record = store.update_job("j1", outputs=["/renders/a.exr"])
    assert record.state == "running"
    assert record.progress == {"frames_done": 1, "frames": 10}
    assert record.outputs == ["/renders/a.exr"]
    assert record.finished_at is None


def test_a_final_state_stamps_the_finish_time(store: Store) -> None:
    store.create_job("j1", kind="cook")
    record = store.update_job("j1", state="done")
    assert record.finished_at is not None
    assert record.finished_at >= record.created_at


def test_jobs_are_listed_newest_first_and_can_be_filtered(store: Store) -> None:
    store.create_job("j1", kind="render", session_id="s1")
    store.create_job("j2", kind="render", session_id="s2", state="running")
    assert [j.job_id for j in store.list_jobs()] == ["j2", "j1"]
    assert [j.job_id for j in store.list_jobs(session_id="s1")] == ["j1"]
    assert [j.job_id for j in store.list_jobs(states=["running"])] == ["j2"]
    assert [j.job_id for j in store.list_jobs(limit=1)] == ["j2"]


def test_unknown_jobs_and_states_are_refused(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.update_job("j1", state="done")
    with pytest.raises(ValueError):
        store.create_job("j2", kind="render", state="mystery")
    store.create_job("j3", kind="render")
    with pytest.raises(ValueError):
        store.update_job("j3", state="mystery")


def test_old_jobs_are_pruned(store: Store) -> None:
    store.create_job("j1", kind="render")
    assert store.prune_jobs(max_age_s=-1) == 1
    assert store.get_job("j1") is None


# -- versions and runs ----------------------------------------------------


def test_versions_count_up_per_kind_name_and_hip_family(store: Store) -> None:
    assert store.allocate_version(kind="render", name="beauty", hip_family="shot") == 1
    assert store.allocate_version(kind="render", name="beauty", hip_family="shot") == 2
    assert store.allocate_version(kind="cache", name="beauty", hip_family="shot") == 1
    assert store.allocate_version(kind="render", name="smoke", hip_family="shot") == 1
    assert store.allocate_version(kind="render", name="beauty", hip_family="other") == 1
    assert store.latest_version(kind="render", name="beauty", hip_family="shot") == 2
    assert store.latest_version(kind="render", name="nothing", hip_family="shot") == 0


def test_a_run_freezes_its_expanded_paths(store: Store) -> None:
    paths = {"output": "/scenes/renders/20260921_beauty/v001/beauty_v001.$F4.exr"}
    record = store.create_run(
        "r1",
        kind="render",
        paths=paths,
        name="beauty",
        hip_family="shot",
        version=1,
        session_id="s1",
        source_node="/out/karma1",
        job_id="j1",
        scene={"frames": [1, 10]},
    )
    assert record.paths == paths
    assert store.get_run("r1") == record
    assert store.get_run("missing") is None
    store.create_run("r2", kind="cache", paths={})
    assert [r.run_id for r in store.list_runs()] == ["r2", "r1"]
    assert [r.run_id for r in store.list_runs(kind="render")] == ["r1"]


# -- readable exports -----------------------------------------------------


def test_a_run_exports_as_readable_json_beside_a_scene(store: Store, tmp_path: Path) -> None:
    store.create_run(
        "r1",
        kind="render",
        paths={"output": "$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr"},
        name="beauty",
        version=1,
    )
    export = store.run_export("r1")
    assert export["run_id"] == "r1"
    assert export["paths"]["output"].startswith("$HIP/")
    assert export["created_utc"].endswith("+00:00")

    written = write_export(export, tmp_path / "v001" / "_run.json")
    assert json.loads(written.read_text(encoding="utf-8")) == export


def test_a_job_exports_with_readable_times(store: Store, tmp_path: Path) -> None:
    store.create_job("j1", kind="render", scene={"hash": "abc"})
    store.update_job("j1", state="done", outputs=["/renders/a.exr"])
    export = store.job_export("j1")
    assert export["state"] == "done"
    assert export["scene"] == {"hash": "abc"}
    assert export["finished_utc"] is not None
    write_export(export, tmp_path / ".agent" / "jobs" / "j1" / "job.json")
    assert (tmp_path / ".agent" / "jobs" / "j1" / "job.json").is_file()


def test_exporting_a_record_that_is_not_there_is_refused(store: Store) -> None:
    with pytest.raises(UnknownRecord):
        store.run_export("r1")
    with pytest.raises(UnknownRecord):
        store.job_export("j1")
