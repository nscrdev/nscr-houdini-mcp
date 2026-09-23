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
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from nscr_houdini_mcp import outputs
from nscr_houdini_mcp import store as store_module
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


def test_the_name_of_a_node_stays_a_variable_in_the_line_a_person_reads() -> None:
    made = plan("render", name=None, node_name="beauty", version=2)
    assert "${OS}" in made.template
    assert "$HIP" in made.template
    assert made.path.endswith("/beauty_v002.$F4.exr")
    assert "${OS}" not in made.path


def test_the_parameter_for_a_started_run_holds_the_path_and_no_node_name() -> None:
    """The server owns a run once it starts, so a rename must not move it."""
    made = plan("render", name=None, node_name="beauty", version=2)
    assert made.parm == made.path
    assert "${OS}" not in made.parm


def test_a_name_given_by_hand_is_written_out() -> None:
    made = plan("render", name="key_light", node_name="beauty", version=1)
    assert "${OS}" not in made.template
    assert "key_light" in made.template


def test_a_name_given_by_hand_stays_written_out_when_it_matches_the_node() -> None:
    made = plan("render", name="beauty", node_name="beauty", version=1)
    assert "${OS}" not in made.template
    assert "20260921_beauty" in made.template


def test_a_template_never_holds_a_machine_path() -> None:
    # A spill is the one kind that lives on this machine, and it never goes on
    # a node; it has checks of its own below.
    for kind in outputs.OUTPUT_KINDS:
        if kind != outputs.SPILL_KIND:
            assert plan(kind, version=1).template.startswith("$HIP/")


def test_a_frame_token_is_kept_until_something_asks_for_a_frame() -> None:
    made = plan("render", version=1)
    assert made.path.endswith(".$F4.exr")
    assert outputs.expand(made.path, hip_dir="/shots/sq010", frame=12).endswith(".0012.exr")


def test_only_whole_variable_names_are_filled_in() -> None:
    filled = outputs.expand("$HIP/$HIPNAME/${HIP}/x.exr", hip_dir="/shots/sq010")
    assert filled == "/shots/sq010/$HIPNAME//shots/sq010/x.exr"


def test_names_are_reduced_to_safe_characters() -> None:
    assert outputs.sanitize_name("Beauty Pass/2!") == "Beauty_Pass_2"
    assert outputs.sanitize_name("   ") == "output"


def test_a_name_windows_keeps_for_itself_is_moved_out_of_the_way() -> None:
    assert outputs.sanitize_name("CON") == "CON_out"
    assert outputs.sanitize_name("lpt9") == "lpt9_out"
    assert outputs.sanitize_name("console") == "console"


def test_two_names_in_another_script_stay_two_names() -> None:
    first = outputs.sanitize_name("煙")
    second = outputs.sanitize_name("炎")
    assert first != second
    assert first.startswith("output_")
    assert outputs.sanitize_name("煙") == first


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


def test_caches_can_be_sent_to_another_disk(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JOB", "/mnt/fast/job")
    write(home / "config.toml", '[outputs]\ncache_root = "$JOB/scratch"\n')
    table = outputs.load_conventions(home=home)
    cache = plan("cache", conventions=table, version=1)
    assert cache.template.startswith("$JOB/scratch/geo/")
    assert cache.path.startswith("/mnt/fast/job/scratch/geo/")
    assert plan("render", conventions=table, version=1).template.startswith("$HIP/renders/")


def test_a_variable_with_no_value_is_said_out_loud(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JOB", raising=False)
    write(home / "config.toml", '[outputs]\ncache_root = "$JOB/scratch"\n')
    table = outputs.load_conventions(home=home)
    with pytest.raises(outputs.OutputError) as caught:
        plan("cache", conventions=table, version=1)
    assert "$JOB" in str(caught.value)


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("[outputs]\nversion_width = 0\n", "version_width"),
        ("[outputs]\nspeed = 2\n", "unknown keys"),
        ('[outputs.grammar]\nturntable = "<output_root>/t/"\n', "unknown kinds"),
        ('[outputs.grammar]\nrender = "<output_root>/<nmae>.exr"\n', "unknown tokens"),
        ('[outputs.grammar]\nrender = "/renders/<name>.exr"\n', "drive or a root"),
        ('[outputs.grammar]\nrender = "<output_root>\\\\r\\\\<name>.exr"\n', "forward slashes"),
        ('[outputs.grammar]\nrender = "<output_root>/../<name>.exr"\n', "no .. in it"),
        ('[outputs.grammar]\nrender = "<output_root>/$SHOT/<name>.exr"\n', "$SHOT"),
        ('[outputs]\noutput_root = "C:/renders"\n', "drive"),
        ('[outputs]\noutput_root = "/tmp/renders"\n', "drive or start at a root"),
        ('[outputs]\noutput_root = "renders"\n', "must start at one of"),
        ('[outputs]\noutput_root = "$HIP/../../renders"\n', "step out of a folder"),
        ('[outputs]\ncache_root = "$SCRATCH/x"\n', "must start at one of"),
        ('[outputs]\nproducer = "../../../../tmp/x"\n', "no .. in it"),
        ('[outputs]\nproducer = "/tmp/x"\n', "not a root of its own"),
        ('[outputs.extensions]\nrender = "../../x"\n', "not an extension"),
        ('[outputs.extensions]\nrender = "exr;rm -rf"\n', "not an extension"),
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


def test_a_scene_folder_cannot_send_writes_out_of_itself(home: Path, tmp_path: Path) -> None:
    """A file beside a scene came with the scene, so it is read and checked."""
    scene_dir = tmp_path / "scene"
    write(scene_dir / ".agent" / "outputs.toml", '[outputs]\nproducer = "../../../../tmp/x"\n')
    with pytest.raises(outputs.ConventionError) as caught:
        outputs.load_conventions(home=home, hip_path=scene_dir / "shot.hip")
    assert "no .. in it" in str(caught.value)


def test_an_extension_from_a_caller_is_checked_too() -> None:
    with pytest.raises(outputs.ConventionError):
        plan("render", version=1, ext="../../x")
    assert plan("render", version=1, ext=".jpg").path.endswith(".jpg")


def test_a_path_that_would_leave_the_root_is_refused() -> None:
    """The last guard, for a table that reached the builder unchecked."""
    broken = replace(
        outputs.DEFAULT_CONVENTIONS_TABLE,
        grammar=dict(outputs.DEFAULT_GRAMMAR, render="<output_root>/../../x/<name>_v<ver>.<ext>"),
    )
    with pytest.raises(outputs.ConventionError) as caught:
        plan("render", version=1, conventions=broken)
    assert "leave the output root" in str(caught.value)


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
    assert sidecar["paths"]["parm"] == made.path
    assert sidecar["paths"]["template"] == made.template
    assert sidecar["paths"]["root"] == scene.parent.as_posix()
    assert sidecar["paths"]["under_root"] == made.path[len(sidecar["paths"]["root"]) + 1 :]
    assert sidecar["paths"]["under_root"].startswith("renders/")
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
    assert made.template.startswith("$HOUDINI_TEMP_DIR/nscr-houdini-mcp/s1/")
    assert made.path.startswith((home / "nscr-houdini-mcp" / "s1").as_posix())
    assert str(home) not in made.template
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


def test_a_hip_file_that_is_already_there_is_never_handed_out(store: Store, scene: Path) -> None:
    """The scene in this test is already `shot_v002.hip`, so the sequence goes
    on above it rather than handing out a number that would land on it."""
    first = outputs.allocate(store, "hip", name="shot", hip_path=scene, when=WHEN)
    assert first.path.endswith("shot_v003.hip")
    Path(first.path).write_text("scene", encoding="utf-8")
    second = outputs.allocate(store, "hip", name="shot", hip_path=scene, when=WHEN)
    assert second.version == 4


def test_a_hip_family_goes_on_above_the_versions_beside_it(store: Store, scene: Path) -> None:
    """Versions saved by hand before the first save here are not handed out again."""
    (scene.parent / "shot_v007.hip").write_text("scene", encoding="utf-8")
    (scene.parent / "shot_v009.hipnc").write_text("scene", encoding="utf-8")
    (scene.parent / "other_v020.hip").write_text("scene", encoding="utf-8")
    (scene.parent / "shot_v030.txt").write_text("notes", encoding="utf-8")
    made = outputs.allocate(store, "hip", hip_path=scene, when=WHEN)
    assert made.version == 10
    assert made.path.endswith("shot_v010.hip")
    assert outputs.allocate(store, "hip", hip_path=scene, when=WHEN).version == 11


def test_the_versions_counted_are_the_ones_where_the_output_goes(store: Store, scene: Path) -> None:
    """With the output root in another folder, that folder decides, not the scene's."""
    elsewhere = scene.parent / "versions"
    elsewhere.mkdir()
    for number in range(1, 21):
        (elsewhere / f"shot_v{number:03d}.hip").write_text("scene", encoding="utf-8")
    table = replace(outputs.DEFAULT_CONVENTIONS_TABLE, output_root="$HIP/versions")
    made = outputs.allocate(store, "hip", hip_path=scene, when=WHEN, conventions=table)
    assert made.version == 21
    assert made.path == (elsewhere / "shot_v021.hip").as_posix()


def test_a_version_folder_kind_goes_on_above_the_folders_there(store: Store, scene: Path) -> None:
    taken = plan("cache", hip_path=scene, version=1)
    base = Path(taken.version_dir).parent
    for number in (1, 2, 17):
        (base / f"v{number:03d}").mkdir(parents=True)
    made = outputs.allocate(store, "cache", name="beauty", hip_path=scene, when=WHEN)
    assert made.version == 18


def test_the_version_floor_reads_the_scenes_own_name() -> None:
    assert outputs.hip_version_floor("/nowhere/shot.v012.hip") == 12
    assert outputs.hip_version_floor("/nowhere/shot.hip") == 0
    assert outputs.hip_version_floor(None) == 0


def test_a_linked_output_folder_is_written_through(
    store: Store, scene: Path, tmp_path: Path
) -> None:
    """A render folder that is a link to a bigger disk is a normal setup."""
    bigger = tmp_path / "bigger_disk"
    bigger.mkdir()
    try:
        (scene.parent / "linked").symlink_to(bigger, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system will not make a link here")
    table = replace(outputs.DEFAULT_CONVENTIONS_TABLE, output_root="$HIP/linked")
    made = outputs.allocate(store, "hip", hip_path=scene, when=WHEN, conventions=table)
    assert made.path == (scene.parent / "linked" / "shot_v001.hip").as_posix()
    assert (bigger / "shot_v001.hip.claim").exists()


def test_a_path_that_climbs_out_of_the_root_is_refused(store: Store, scene: Path) -> None:
    table = replace(
        outputs.DEFAULT_CONVENTIONS_TABLE,
        grammar=dict(outputs.DEFAULT_GRAMMAR, hip="<output_root>/../<name>_v<ver>.<ext>"),
    )
    with pytest.raises(outputs.ConventionError):
        outputs.allocate(store, "hip", hip_path=scene, when=WHEN, conventions=table)


def test_an_output_that_is_itself_a_link_is_never_written_through(
    scene: Path, tmp_path: Path
) -> None:
    made = plan("hip", name="shot", hip_path=scene, version=5)
    try:
        Path(made.path).symlink_to(tmp_path / "nowhere.hip")
    except (OSError, NotImplementedError):
        pytest.skip("this system will not make a link here")
    with pytest.raises(outputs.ConventionError) as caught:
        outputs._check_real_place(made)
    assert "is a link" in str(caught.value)


def test_a_scene_on_a_network_share_keeps_its_share() -> None:
    made = plan("hip", name="shot", hip_path=r"\\server\share\proj\shot_v001.hip", version=2)
    assert made.path == "//server/share/proj/shot_v002.hip"
    assert made.root == "//server/share/proj"
    assert outputs._normalize("//server/share/a/../b") == "//server/share/b"
    # Nothing climbs above the share, so such a path fails the containment check.
    assert outputs._normalize("//server/share/../x") == "//server/share/../x"


def test_skipping_versions_never_lowers_the_sequence(store: Store) -> None:
    key = {"kind": "hip", "name": "shot", "hip_family": "shot"}
    assert store.skip_versions_to(**key, version=4) == 4
    assert store.allocate_version(**key) == 5
    assert store.skip_versions_to(**key, version=2) == 5
    assert store.allocate_version(**key) == 6


def test_two_stores_over_one_scene_folder_do_not_take_the_same_hip_number(
    tmp_path: Path, scene: Path
) -> None:
    """Two machines keep their own store, so the file on disk settles it."""
    with Store(tmp_path / "one.sqlite") as one, Store(tmp_path / "two.sqlite") as two:
        first = outputs.allocate(one, "hip", name="shot", hip_path=scene, when=WHEN)
        second = outputs.allocate(two, "hip", name="shot", hip_path=scene, when=WHEN)
    assert first.path != second.path
    # Both go on above the scene, v002. The first store's claim on v003 is on
    # disk, so the second store steps past it.
    assert first.version == 3
    assert second.version == 4
    for made in (first, second):
        assert json.loads(Path(made.sidecar).read_text(encoding="utf-8"))["run_id"] == made.run_id


def version_rows(path: Path) -> list[tuple]:
    """Version rows straight from the store file, for what has no reader yet."""
    with sqlite3.connect(str(path)) as db:
        return db.execute("SELECT version, run_id FROM versions ORDER BY version").fetchall()


def test_folders_left_by_other_machines_cost_no_run_names(tmp_path: Path, scene: Path) -> None:
    path = tmp_path / "coord.sqlite"
    with Store(path) as store:
        for number in range(1, 9):
            taken = plan("cache", hip_path=scene, version=number)
            Path(taken.version_dir).mkdir(parents=True)
        made = outputs.allocate(store, "cache", name="beauty", hip_path=scene, when=WHEN)
    assert made.version == 9
    # The folders there are counted before any number is taken, so the
    # sequence skips to them in one step and none of them carries this run.
    rows = version_rows(path)
    assert rows == [(8, None), (9, made.run_id)]


def test_a_number_whose_run_never_arrived_loses_its_name(tmp_path: Path) -> None:
    """A process can stop between taking a number and recording the run."""
    path = tmp_path / "coord.sqlite"
    with Store(path) as store:
        store.allocate_version(kind="cache", name="beauty", hip_family="shot", run_id="run-ghost")
        assert store.reap_versions(0) == 1
        assert store.latest_version(kind="cache", name="beauty", hip_family="shot") == 1
    assert version_rows(path) == [(1, None)]


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


def test_a_run_id_used_twice_keeps_the_first_run_record(store: Store, scene: Path) -> None:
    first = outputs.allocate(store, "hip", hip_path=scene, when=WHEN, run_id="run-op1")
    with pytest.raises(store_module.DuplicateRecord):
        outputs.allocate(store, "hip", hip_path=scene, when=WHEN, run_id="run-op1")
    kept = store.get_run("run-op1")
    assert kept is not None
    assert kept.version == first.version
    assert list(scene.parent.glob("*.claim")) == [Path(f"{first.path}.claim")]


def test_a_place_whose_run_could_not_be_recorded_is_given_back(
    store: Store, scene: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args, **rest):
        raise store_module.StoreError("the disk is full")

    monkeypatch.setattr(store, "create_run", refuse)
    with pytest.raises(store_module.StoreError):
        outputs.allocate(store, "hip", hip_path=scene, when=WHEN, run_id="run-lost")
    assert list(scene.parent.glob("*.claim")) == []
    assert version_rows(Path(store.path)) == [(2, None), (3, None)]


# -- records --------------------------------------------------------------


def test_a_job_record_goes_beside_the_scene_and_is_never_handed_out(
    store: Store, tmp_path: Path
) -> None:
    folder = tmp_path / "shots"
    folder.mkdir()
    made = outputs.record_path(
        "job", "job-op-1", hip_path=str(folder / "shot.hip"), session_id="s1"
    )
    assert made.template == "$HIP/.agent/jobs/job-op-1.json"
    assert Path(made.path) == folder / ".agent" / "jobs" / "job-op-1.json"
    assert Path(made.directory).is_dir()
    with pytest.raises(outputs.UnknownKind):
        outputs.allocate(store, "job", name="x", hip_path=str(folder / "shot.hip"))
    with pytest.raises(outputs.UnknownKind):
        outputs.record_path("capture", "x", hip_path=None, session_id="s1")


def test_a_job_record_of_a_scene_whose_folder_is_gone_is_refused(tmp_path: Path) -> None:
    missing = tmp_path / "moved" / "shot.hip"
    with pytest.raises(outputs.ConventionError):
        outputs.record_path("job", "job-1", hip_path=str(missing), session_id="s1")
    assert not missing.parent.exists()


def test_a_job_record_of_an_untitled_scene_goes_to_the_scratch_folder(tmp_path: Path) -> None:
    made = outputs.record_path(
        "job", "job-1", hip_path=None, session_id="s1", scratch_root=tmp_path / "temp"
    )
    assert made.unsaved_hip is True
    assert Path(made.path).parent.is_dir()
    assert Path(made.path).is_relative_to(tmp_path / "temp")


# -- references, checks and spills ----------------------------------------


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("reference", "$HIP/.agent/reference/beauty_run-abc123.png"),
        ("check", "$HIP/.agent/checks/20260921/143005_beauty_run-abc123.png"),
    ],
)
def test_references_and_checks_are_agent_artifacts_beside_the_scene(
    kind: str, expected: str
) -> None:
    made = plan(kind)
    assert made.template == expected
    assert made.version is None
    assert made.path == expected.replace("$HIP", "/shots/sq010")


def test_a_spill_goes_to_the_servers_spill_folder_whatever_the_scene(tmp_path: Path) -> None:
    folder = tmp_path / "spill"
    saved = plan("spill", spill_root=folder)
    unsaved = plan("spill", hip_path=None, spill_root=folder)
    for made in (saved, unsaved):
        assert Path(made.path).is_relative_to(folder)
        assert made.path.endswith("/2026-09-21/143005-beauty-run-abc123.json")
        assert made.template == made.path
        assert made.root == folder.as_posix()
    # A scene never saved still spills where the server spills, with no word
    # about a scratch folder.
    assert unsaved.warnings == ()


def test_a_spill_on_a_windows_drive_keeps_its_drive() -> None:
    made = plan("spill", spill_root="C:\\Users\\a\\spill")
    assert made.path.startswith("C:/Users/a/spill/2026-09-21/")


def test_a_spill_is_claimed_and_recorded_like_any_other_run(store: Store, tmp_path: Path) -> None:
    made = outputs.allocate(
        store,
        "spill",
        name="big answer",
        hip_path=None,
        session_id="s1",
        spill_root=tmp_path / "spill",
        when=WHEN,
    )
    assert Path(made.sidecar).is_file()
    assert store.get_run(made.run_id).kind == "spill"
    assert "big_answer" in made.path


def test_code_is_handed_every_kind_but_records_and_spills() -> None:
    assert set(outputs.CODE_KINDS) == set(outputs.OUTPUT_KINDS) - {"job", "spill"}
    assert {"reference", "check"} <= set(outputs.CODE_KINDS)


@pytest.mark.parametrize(
    ("line", "why"),
    [
        ('spill = "$HIP/spill/<name>.<ext>"', "must start at <spill_root>"),
        ('spill = "<spill_root>/$HIP/<name>.<ext>"', "must not use Houdini variables"),
        ('render = "<spill_root>/<name>_v<ver>.<ext>"', "only the spill template"),
    ],
)
def test_the_spill_line_is_the_only_one_that_starts_at_the_spill_folder(
    home: Path, line: str, why: str
) -> None:
    (home / "config.toml").write_text(f"[outputs.grammar]\n{line}\n", encoding="utf-8")
    with pytest.raises(outputs.ConventionError, match=why):
        outputs.load_conventions(home=home)


def test_a_reference_line_can_be_moved_like_any_other(home: Path) -> None:
    text = '[outputs.grammar]\nreference = "<output_root>/refs/<name>.<ext>"\n'
    (home / "config.toml").write_text(text, encoding="utf-8")
    table = outputs.load_conventions(home=home)
    assert plan("reference", conventions=table).template == "$HIP/refs/beauty.png"


# -- reading outputs back -------------------------------------------------


def test_a_scene_key_reads_one_file_the_same_however_it_is_written() -> None:
    assert outputs.scene_key(None) is None
    assert outputs.scene_key("") is None
    assert outputs.scene_key("/a/b/shot.hip") == "/a/b/shot.hip"
    key = outputs.scene_key("C:\\Shots\\Shot.hip")
    assert key == ("c:/shots/shot.hip" if os.name == "nt" else "C:/Shots/Shot.hip")


@pytest.mark.parametrize(
    ("value", "machine"),
    [
        ("/Users/somebody/render/a.exr", True),
        ("C:/render/a.exr", True),
        ("\\\\server\\share\\a.exr", True),
        ("$HIP/render/a.exr", False),
        ("render/a.exr", False),
    ],
)
def test_a_machine_path_starts_at_a_root_or_a_drive(value: str, machine: bool) -> None:
    assert outputs.is_machine_path(value) is machine


def test_the_managed_roots_of_a_scene_are_its_output_and_cache_roots(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("JOB", raising=False)
    assert outputs.managed_roots(None, hip_path="/shots/shot.hip") == ["/shots"]
    (home / "config.toml").write_text('[outputs]\ncache_root = "$JOB/cache"\n', encoding="utf-8")
    table = outputs.load_conventions(home=home)
    # A root whose variable has no value here is left out, not guessed at.
    assert outputs.managed_roots(table, hip_path="/shots/shot.hip") == ["/shots"]
    monkeypatch.setenv("JOB", "/jobs/show")
    assert outputs.managed_roots(table, hip_path="/shots/shot.hip") == [
        "/shots",
        "/jobs/show/cache",
    ]
    scratch = outputs.managed_roots(None, hip_path=None, session_id="s1", scratch_root="/tmp/h")
    assert scratch == ["/tmp/h/nscr-houdini-mcp/s1"]


def test_a_path_is_inside_the_roots_only_when_it_stays_there() -> None:
    roots = ["/shots"]
    assert outputs.inside_roots(roots, "/shots/render/a.exr")
    assert not outputs.inside_roots(roots, "/shots/../elsewhere/a.exr")
    assert not outputs.inside_roots(roots, "/shotsmore/a.exr")
    assert not outputs.inside_roots(roots, "render/a.exr")


def test_an_output_is_on_disk_when_its_file_or_any_frame_is(tmp_path: Path) -> None:
    folder = tmp_path / "v001"
    folder.mkdir()
    sequence = f"{folder.as_posix()}/beauty_v001.$F4.exr"
    assert outputs.on_disk(sequence) is False
    (folder / "beauty_v001.1001.exr").write_bytes(b"x")
    assert outputs.on_disk(sequence) is True
    assert outputs.on_disk(f"{folder.as_posix()}/other.$F.exr") is False
    assert outputs.on_disk(f"{folder.as_posix()}/beauty_v001.1001.exr") is True
    assert outputs.on_disk(f"{tmp_path.as_posix()}/gone/$F4/x.exr") is False
    assert outputs.on_disk("") is False
