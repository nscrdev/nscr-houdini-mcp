"""Managed output paths: the grammar, the config files and the allocation.

No Houdini here. The module is plain Python by design, so every rule in it can
be checked on any machine. The contention test starts real processes with the
spawn method, which is the only one on every supported system, so a child
imports this module by name and calls the function it was handed.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue as queue_module
from datetime import datetime
from pathlib import Path

import pytest

from nscr_houdini_mcp import outputs
from nscr_houdini_mcp.store import Store

WHEN = datetime(2026, 9, 21, 14, 30, 5)
HIP = "/shots/sq010/shot_v002.hip"

RACERS = 6
BARRIER_TIMEOUT_S = 30.0
RESULT_TIMEOUT_S = 60.0
JOIN_TIMEOUT_S = 30.0


@pytest.fixture
def home(tmp_path: Path) -> Path:
    """An empty state folder, so no file on this machine changes a result."""
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


@pytest.fixture
def store(tmp_path: Path):
    with Store(tmp_path / "coord.sqlite") as opened:
        yield opened


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """A scene file on disk, so allocation has a folder to write in."""
    folder = tmp_path / "shots" / "sq010"
    folder.mkdir(parents=True)
    hip = folder / "shot_v002.hip"
    hip.write_text("scene", encoding="utf-8")
    return hip


def plan(kind: str, **kwargs) -> outputs.OutputPlan:
    options = {
        "run_id": "run-abc123",
        "name": "beauty",
        "hip_path": HIP,
        "session_id": "s1",
        "when": WHEN,
    }
    options.update(kwargs)
    return outputs.plan_path(kind, **options)


# -- the default grammar --------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("render", "$HIP/renders/20260921_beauty/v003/beauty_v003.$F4.exr"),
        ("flipbook", "$HIP/flipbook/20260921_beauty/v003/beauty_v003.$F4.png"),
        ("comp", "$HIP/comp/20260921_beauty/v003/beauty_v003.$F4.exr"),
        ("cache", "$HIP/geo/beauty/v003/beauty_v003.$F4.bgeo.sc"),
        ("usd", "$HIP/usd/beauty/v003/beauty_v003.usd"),
        ("hip", "$HIP/beauty_v003.hip"),
    ],
)
def test_the_default_grammar_per_kind(kind: str, expected: str) -> None:
    assert plan(kind, version=3).template == expected


def test_renders_are_dated_and_caches_are_not() -> None:
    """Two questions, two answers: a day to browse, a name a scene reads back."""
    assert "20260921_beauty" in plan("render", version=1).path
    assert "20260921" not in plan("cache", version=1).path


def test_agent_artifacts_live_under_the_agent_folder() -> None:
    for kind in ("capture", "compare"):
        made = plan(kind, version=1)
        assert "/.agent/" in made.path
        assert made.run_id in made.path


def test_the_name_of_a_node_stays_a_variable_in_the_parameter() -> None:
    made = plan("render", name=None, node_name="beauty", version=2)
    assert "${OS}" in made.template
    assert "$HIP" in made.template
    assert made.path.endswith("/beauty_v002.$F4.exr")
    assert "${OS}" not in made.path


def test_a_name_given_by_hand_is_written_out() -> None:
    made = plan("render", name="key_light", node_name="beauty", version=1)
    assert "${OS}" not in made.template
    assert "key_light" in made.template


def test_a_template_never_holds_a_machine_path() -> None:
    for kind in outputs.OUTPUT_KINDS:
        assert plan(kind, version=1).template.startswith("$HIP/")


def test_a_frame_token_is_kept_until_something_asks_for_a_frame() -> None:
    made = plan("render", version=1)
    assert made.path.endswith(".$F4.exr")
    assert outputs.expand(made.path, hip_dir="/shots/sq010", frame=12).endswith(".0012.exr")


def test_names_are_reduced_to_safe_characters() -> None:
    assert outputs.sanitize_name("Beauty Pass/2!") == "Beauty_Pass_2"
    assert outputs.sanitize_name("   ") == "output"


def test_a_name_that_reads_as_a_version_is_flagged() -> None:
    made = plan("render", name="shot_v2", version=1)
    assert any("version" in line for line in made.warnings)


def test_a_missing_version_is_an_error_not_a_guess() -> None:
    with pytest.raises(outputs.OutputError):
        plan("render")


def test_an_unknown_kind_has_no_path() -> None:
    with pytest.raises(outputs.UnknownKind):
        plan("turntable", version=1)


def test_the_marker_node_is_config_and_not_a_rule_in_code() -> None:
    table = outputs.DEFAULT_CONVENTIONS_TABLE
    assert table.output_marker_type == "null"
    assert table.marker_name("beauty") == "OUT_beauty"
    assert table.output_marker_enabled is True


# -- hip paths from other systems -----------------------------------------


def test_a_windows_scene_path_is_split_the_way_windows_wrote_it() -> None:
    folder, stem = outputs.split_hip(r"C:\shots\sq010\shot_v002.hip")
    assert folder == "C:/shots/sq010"
    assert stem == "shot_v002"


def test_a_windows_scene_path_joins_without_a_stray_separator() -> None:
    made = plan("render", hip_path=r"C:\shots\sq010\shot_v002.hip", version=1)
    assert made.path == "C:/shots/sq010/renders/20260921_beauty/v001/beauty_v001.$F4.exr"
    assert "\\" not in made.path
    assert made.template == "$HIP/renders/20260921_beauty/v001/beauty_v001.$F4.exr"


def test_records_are_scoped_to_the_scene_without_its_version() -> None:
    assert outputs.hip_family(r"C:\shots\sq010\shot_v002.hip") == "shot"
    assert outputs.hip_family("/shots/sq010/shot.hipnc") == "shot"
    assert outputs.hip_family(None) == "untitled"


# -- conventions files ----------------------------------------------------


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_with_no_files_the_defaults_are_in_force(home: Path) -> None:
    table = outputs.load_conventions(home=home)
    assert table.grammar == outputs.DEFAULT_GRAMMAR
    assert table.sources == ()


def test_the_user_file_wins_over_the_defaults(home: Path) -> None:
    write(
        home / "config.toml",
        '[outputs]\nproducer = "3d/hip"\n[outputs.extensions]\nrender = "png"\n',
    )
    table = outputs.load_conventions(home=home)
    assert table.producer == "3d/hip"
    assert table.extension_for("render") == "png"
    assert table.extension_for("comp") == "exr"
    made = plan("render", conventions=table, version=1)
    assert made.template == "$HIP/renders/3d/hip/20260921_beauty/v001/beauty_v001.$F4.png"


def test_the_project_file_wins_over_the_user_file(home: Path, tmp_path: Path) -> None:
    write(home / "config.toml", '[outputs]\nproducer = "user"\nversion_width = 4\n')
    write(tmp_path / "scene" / ".agent" / "outputs.toml", '[outputs]\nproducer = "project"\n')
    table = outputs.load_conventions(home=home, hip_path=tmp_path / "scene" / "shot.hip")
    assert table.producer == "project"
    assert table.version_width == 4
    assert len(table.sources) == 2
    assert plan("render", conventions=table, version=7).template.endswith("v0007.$F4.exr")


def test_a_json_file_says_the_same_thing_as_a_toml_one(home: Path) -> None:
    write(home / "config.json", json.dumps({"outputs": {"producer": "3d"}}))
    assert outputs.load_conventions(home=home).producer == "3d"


def test_a_grammar_line_can_be_replaced(home: Path) -> None:
    write(
        home / "config.toml",
        "[outputs.grammar]\n"
        'cache = "<output_root>/cache/<date>_<name>/v<ver>/<name>_v<ver>.<ext>"\n',
    )
    table = outputs.load_conventions(home=home)
    assert plan("cache", conventions=table, version=2).template == (
        "$HIP/cache/20260921_beauty/v002/beauty_v002.bgeo.sc"
    )


def test_caches_can_be_sent_to_another_disk(home: Path) -> None:
    write(home / "config.toml", '[outputs]\ncache_root = "$JOB/scratch"\n')
    table = outputs.load_conventions(home=home)
    assert plan("cache", conventions=table, version=1).template.startswith("$JOB/scratch/geo/")
    assert plan("render", conventions=table, version=1).template.startswith("$HIP/renders/")


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("[outputs]\nversion_width = 0\n", "version_width"),
        ("[outputs]\nspeed = 2\n", "unknown keys"),
        ('[outputs.grammar]\nturntable = "<output_root>/t/"\n', "unknown kinds"),
        ('[outputs.grammar]\nrender = "<output_root>/<nmae>.exr"\n', "unknown tokens"),
        ('[outputs.grammar]\nrender = "/renders/<name>.exr"\n', "drive or a root"),
        ('[outputs.grammar]\nrender = "<output_root>\\\\r\\\\<name>.exr"\n', "forward slashes"),
        ('[outputs]\noutput_root = "C:/renders"\n', "drive"),
        ('[conventions]\noutput_marker_type = ""\n', "node type"),
        ('[conventions]\noutput_marker_enabled = "yes"\n', "true or false"),
        ("[outputs\n", "config.toml"),
    ],
)
def test_a_table_that_cannot_be_used_says_why(home: Path, text: str, says: str) -> None:
    write(home / "config.toml", text)
    with pytest.raises(outputs.ConventionError) as caught:
        outputs.load_conventions(home=home)
    assert says in str(caught.value)


def test_a_project_file_may_only_set_the_two_tables(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "scene" / ".agent" / "outputs.toml", '[store]\npath = "x"\n')
    with pytest.raises(outputs.ConventionError) as caught:
        outputs.load_conventions(home=home, hip_path=tmp_path / "scene" / "shot.hip")
    assert "store" in str(caught.value)


# -- allocation -----------------------------------------------------------


def test_allocation_takes_a_version_and_makes_its_folder(store: Store, scene: Path) -> None:
    made = outputs.allocate(
        store,
        "render",
        node_path="/obj/geo1/OUT_beauty",
        hip_path=scene,
        session_id="s1",
        when=WHEN,
    )
    assert made.version == 1
    assert made.name == "OUT_beauty"
    assert Path(made.version_dir).is_dir()
    assert made.template.startswith("$HIP/renders/")
    assert made.path.startswith(scene.parent.as_posix())

    again = outputs.allocate(store, "render", name="OUT_beauty", hip_path=scene, when=WHEN)
    assert again.version == 2
    assert again.path != made.path


def test_the_run_record_and_the_sidecar_hold_the_same_run(store: Store, scene: Path) -> None:
    made = outputs.allocate(
        store,
        "render",
        name="beauty",
        node_path="/obj/geo1/OUT_beauty",
        hip_path=scene,
        session_id="s1",
        job_id="job-1",
        when=WHEN,
    )
    sidecar = json.loads(Path(made.sidecar).read_text(encoding="utf-8"))
    assert sidecar["run_id"] == made.run_id
    assert sidecar["kind"] == "render"
    assert sidecar["name"] == "beauty"
    assert sidecar["version"] == 1
    assert sidecar["hip_family"] == "shot"
    assert sidecar["session_id"] == "s1"
    assert sidecar["job_id"] == "job-1"
    assert sidecar["source_node"] == "/obj/geo1/OUT_beauty"
    assert sidecar["paths"]["path"] == made.path
    assert sidecar["paths"]["template"] == made.template
    assert sidecar["scene"]["hip_path"] == str(scene)
    assert sidecar["scene"]["unsaved_hip"] is False
    assert sidecar["created_utc"]
    assert store.get_run(made.run_id).version == 1


def test_a_started_run_does_not_move_when_the_node_is_renamed(store: Store, scene: Path) -> None:
    started = outputs.allocate(
        store, "render", node_path="/obj/geo1/OUT_beauty", hip_path=scene, when=WHEN
    )
    later = outputs.allocate(
        store, "render", node_path="/obj/geo1/OUT_hero", hip_path=scene, when=WHEN
    )
    assert store.get_run(started.run_id).paths["path"] == started.path
    assert Path(started.version_dir).is_dir()
    assert "OUT_beauty" in started.path
    assert "OUT_hero" in later.path


def test_two_captures_in_one_second_are_two_files(store: Store, scene: Path) -> None:
    first = outputs.allocate(store, "capture", name="beauty", hip_path=scene, when=WHEN)
    second = outputs.allocate(store, "capture", name="beauty", hip_path=scene, when=WHEN)
    assert first.path != second.path
    assert first.run_id != second.run_id
    assert Path(first.directory).is_dir()


def test_an_unsaved_scene_writes_to_scratch_and_says_so(store: Store, home: Path) -> None:
    made = outputs.allocate(
        store,
        "capture",
        name="beauty",
        hip_path=None,
        session_id="s1",
        when=WHEN,
        scratch_root=home,
    )
    assert made.unsaved_hip is True
    assert made.path.startswith((home / "scratch" / "s1").as_posix())
    assert any("saved" in line for line in made.warnings)
    assert store.get_run(made.run_id).scene["unsaved_hip"] is True
    assert Path(made.sidecar).is_file()


def test_a_folder_left_by_another_machine_is_never_written_into(store: Store, scene: Path) -> None:
    """The version transaction agrees on a number; the mkdir is the last guard."""
    taken = plan("cache", hip_path=scene, version=1)
    Path(taken.version_dir).mkdir(parents=True)
    made = outputs.allocate(store, "cache", name="beauty", hip_path=scene, when=WHEN)
    assert made.version == 2
    assert Path(made.version_dir).is_dir()


def test_a_hip_file_that_is_already_there_takes_the_next_number(store: Store, scene: Path) -> None:
    """A kind with no folder of its own is guarded by the file instead.

    The scene in this test is already `shot_v002.hip`, so the number that would
    land on it is passed over rather than written.
    """
    first = outputs.allocate(store, "hip", name="shot", hip_path=scene, when=WHEN)
    assert first.path.endswith("shot_v001.hip")
    Path(first.path).write_text("scene", encoding="utf-8")
    second = outputs.allocate(store, "hip", name="shot", hip_path=scene, when=WHEN)
    assert second.version == 3
    assert second.path.endswith("shot_v003.hip")


# -- several processes on one store ---------------------------------------


def racer(path: str, hip: str, index: int, barrier, results) -> None:
    """One racer: allocate the same kind and name as everybody else."""
    report: dict[str, object] = {"index": index, "pid": os.getpid(), "error": None}
    try:
        with Store(path) as opened:
            barrier.wait(BARRIER_TIMEOUT_S)
            made = outputs.allocate(opened, "render", name="beauty", hip_path=hip)
            report["version"] = made.version
            report["folder"] = made.version_dir
            report["sidecar"] = made.sidecar
    except BaseException as error:  # reported, so a failure reads as a message
        report["error"] = f"{type(error).__name__}: {error}"
    results.put(report)


def test_versions_are_never_handed_out_twice_across_processes(tmp_path: Path) -> None:
    folder = tmp_path / "shots"
    folder.mkdir()
    hip = folder / "shot.hip"
    hip.write_text("scene", encoding="utf-8")
    path = tmp_path / "coord.sqlite"
    with Store(path):
        pass

    context = mp.get_context("spawn")
    barrier = context.Barrier(RACERS)
    results = context.Queue()
    children = [
        context.Process(
            target=racer,
            args=(str(path), str(hip), index, barrier, results),
            daemon=True,
        )
        for index in range(RACERS)
    ]
    collected: list[dict] = []
    try:
        for child in children:
            child.start()
        for _ in children:
            collected.append(results.get(timeout=RESULT_TIMEOUT_S))
        for child in children:
            child.join(JOIN_TIMEOUT_S)
    except queue_module.Empty:
        pytest.fail(f"only {len(collected)} of {RACERS} children reported back")
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
                child.join(JOIN_TIMEOUT_S)

    assert [report["error"] for report in collected if report["error"]] == []
    assert len({report["pid"] for report in collected}) == RACERS
    assert sorted(report["version"] for report in collected) == list(range(1, RACERS + 1))
    assert len({report["folder"] for report in collected}) == RACERS
    for report in collected:
        assert Path(report["folder"]).is_dir()
        assert Path(report["sidecar"]).is_file()
