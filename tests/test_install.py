"""The package file, the commands that write it, and what status reports.

Every test here points Houdini's preference folder at a temporary one, so a
run never reads or writes the folder the person at this machine works in.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from nscr_houdini_mcp import cli
from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.bridge.launcher import find_hython, hython_available

# Houdini only honours the preference folder setting when the version token is
# in it, so the temporary one carries the token too.
PREF_TEMPLATE = f"prefs{install_module.VERSION_TOKEN}"

# Its own range, away from anything a person on this machine may be running.
PORT_RANGE = (18400, 18449)

START_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 60.0


@pytest.fixture(autouse=True)
def temp_pref_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    """Point every command in this file at a preference folder of its own.

    The package folder variable is cleared, and no Houdini is asked anything,
    so a unit test decides the lookup with the preference folder alone. A test
    that wants a Houdini answer sets one with `answers`, and the test that
    needs a real Houdini is marked and left to ask for itself.
    """
    root = tmp_path / PREF_TEMPLATE
    monkeypatch.setenv(install_module.PREF_DIR_ENV_VAR, str(root))
    monkeypatch.delenv(install_module.PACKAGE_DIR_ENV_VAR, raising=False)
    if request.node.get_closest_marker("houdini") is None:
        answers(monkeypatch, None, "no Houdini found to ask")
    return root


def answers(
    monkeypatch: pytest.MonkeyPatch,
    answer: install_module.HoudiniAnswer | None,
    note: str = "asked a Houdini",
) -> None:
    """Say what a Houdini would answer, without starting one."""
    monkeypatch.setattr(install_module, "ask_houdini", lambda version=None: (answer, note))


def houdini_answer(
    home: Path,
    *,
    package_dirs: list[Path] | None = None,
    hsite: Path | None = None,
) -> install_module.HoudiniAnswer:
    return install_module.HoudiniAnswer(
        home=home,
        package_dirs=package_dirs or [],
        user_pref_dir=home,
        hsite=hsite,
        version="22.0.368",
        hython=Path("/somewhere/bin/hython"),
    )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "state"
    folder.mkdir(mode=0o700)
    return folder


def package_file() -> Path:
    return install_module.package_path()


def written(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def env_value(document: dict, name: str) -> object:
    for item in document["env"]:
        if name in item:
            return item[name]
    raise AssertionError(f"{name} is not in the package")


# Section: where the package goes


def test_the_preference_folder_setting_wins_and_takes_the_version(tmp_path: Path) -> None:
    assert install_module.user_pref_dir("22.0") == tmp_path / "prefs22.0"
    assert install_module.user_pref_dir("21.5") == tmp_path / "prefs21.5"


def test_each_system_has_its_own_preference_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(install_module.PREF_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(install_module.Path, "home", classmethod(lambda cls: Path("/u/me")))

    monkeypatch.setattr(install_module.sys, "platform", "darwin")
    assert install_module.packages_dir("22.0") == Path(
        "/u/me/Library/Preferences/houdini/22.0/packages"
    )

    monkeypatch.setattr(install_module.sys, "platform", "linux")
    assert install_module.packages_dir("22.0") == Path("/u/me/houdini22.0/packages")

    monkeypatch.setattr(install_module.sys, "platform", "win32")
    monkeypatch.setenv("USERPROFILE", str(Path("/u/win")))
    assert install_module.packages_dir("22.0") == Path("/u/win/Documents/houdini22.0/packages")


# Section: which folder the package goes into


def test_a_named_folder_wins_over_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(install_module.PACKAGE_DIR_ENV_VAR, str(tmp_path / "from-shell"))
    answers(monkeypatch, houdini_answer(tmp_path / "from-houdini"))
    found = install_module.resolve(override=tmp_path / "named")
    assert found.path == tmp_path / "named"
    assert found.source == install_module.SOURCE_GIVEN


def test_the_package_folder_variable_in_this_shell_comes_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    monkeypatch.setenv(
        install_module.PACKAGE_DIR_ENV_VAR, os.pathsep.join([str(first), str(second)])
    )
    answers(monkeypatch, houdini_answer(tmp_path / "from-houdini"))
    found = install_module.resolve()
    assert found.path == first
    assert found.source == install_module.SOURCE_PACKAGE_ENV
    # Houdini scans every folder on the list, so the rest are reported.
    assert [(item.path, item.used) for item in found.candidates] == [
        (first, True),
        (second, False),
    ]
    assert "also scanned" in found.candidates[1].note


def test_houdini_is_asked_before_the_preference_variable_in_this_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "houdini-home"
    answers(monkeypatch, houdini_answer(home), "asked /somewhere/bin/hython")
    found = install_module.resolve()
    assert found.path == home / "packages"
    assert found.source == install_module.SOURCE_HOUDINI_HOME
    assert found.notes == ["asked /somewhere/bin/hython"]


def test_the_package_folder_houdini_itself_reads_wins_over_its_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = tmp_path / "studio-packages"
    site = tmp_path / "site"
    answers(
        monkeypatch,
        houdini_answer(tmp_path / "houdini-home", package_dirs=[named], hsite=site),
    )
    found = install_module.resolve()
    assert found.path == named
    assert found.source == install_module.SOURCE_HOUDINI_PACKAGE
    # The site folder is other people's, so it is reported and never written to.
    site_candidates = [
        item for item in found.candidates if item.source == install_module.SOURCE_HSITE
    ]
    assert [item.path for item in site_candidates] == [site / "houdini22.0" / "packages"]
    assert site_candidates[0].used is False


def test_a_houdini_that_cannot_be_asked_leaves_a_note_and_the_lookup_carries_on(
    temp_pref_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    answers(monkeypatch, None, "no hython at /nowhere/bin/hython")
    found = install_module.resolve()
    assert found.path == Path(str(temp_pref_dir).replace("__HVER__", "22.0")) / "packages"
    assert found.source == install_module.SOURCE_PREF_ENV
    assert found.notes == ["no hython at /nowhere/bin/hython"]


def test_the_last_word_is_the_usual_folder_for_this_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(install_module.PREF_DIR_ENV_VAR)
    monkeypatch.setattr(install_module.sys, "platform", "darwin")
    monkeypatch.setattr(install_module.Path, "home", classmethod(lambda cls: Path("/u/me")))
    found = install_module.resolve()
    assert found.path == Path("/u/me/Library/Preferences/houdini/22.0/packages")
    assert found.source == install_module.SOURCE_DEFAULT


def test_a_redirected_documents_folder_on_windows_is_where_the_profile_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(install_module.PREF_DIR_ENV_VAR)
    monkeypatch.setattr(install_module.sys, "platform", "win32")
    # A synced documents folder is the usual reason the profile is not the
    # place the folder really is.
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "Users" / "me" / "OneDrive"))
    found = install_module.resolve()
    assert (
        found.path
        == tmp_path / "Users" / "me" / "OneDrive" / "Documents" / ("houdini22.0") / "packages"
    )
    assert found.source == install_module.SOURCE_DEFAULT


def test_a_houdini_answer_is_read_from_its_marked_line() -> None:
    printed = (
        "Licence line\n"
        + install_module.ANSWER_MARKER
        + json.dumps(
            {
                "home": "/u/me/houdini22.0",
                "package_dir": os.pathsep.join(["/a", "/b"]),
                "user_pref_dir": "/u/me/houdini22.0",
                "hsite": "/studio",
                "version": "22.0.368",
            }
        )
    )
    answer = install_module._read_answer(printed, Path("/bin/hython"))
    assert answer is not None
    assert answer.home == Path("/u/me/houdini22.0")
    assert answer.package_dirs == [Path("/a"), Path("/b")]
    assert answer.hsite == Path("/studio")


def test_noise_on_its_own_is_no_answer() -> None:
    assert install_module._read_answer("warning: something\n", Path("/bin/hython")) is None


def test_install_and_uninstall_take_the_named_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "somewhere-else"
    assert cli.main(["bridge", "install", "--packages-dir", str(folder)]) == 0
    assert (folder / install_module.PACKAGE_FILE_NAME).exists()
    assert not (tmp_path / "prefs22.0").exists()
    capsys.readouterr()

    assert cli.main(["bridge", "uninstall", "--packages-dir", str(folder)]) == 0
    assert "removed" in capsys.readouterr().out
    assert not (folder / install_module.PACKAGE_FILE_NAME).exists()


# Section: what travels with an installed copy


def test_the_payload_travels_inside_the_package() -> None:
    """An installed copy has no project tree, so the payload ships in it."""
    settings = tomllib.loads(
        (Path(install_module.__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    included = settings["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["houdini"] == "nscr_houdini_mcp/houdini"


def test_the_payload_next_to_the_package_is_preferred(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    beside = tmp_path / "nscr_houdini_mcp"
    (beside / "houdini" / install_module.PACKAGES_DIR_NAME).mkdir(parents=True)
    monkeypatch.setattr(install_module, "__file__", str(beside / "install.py"))
    assert install_module.payload_root() == beside / "houdini"


# Section: install


def test_install_writes_one_file_with_both_paths_and_no_auto_start() -> None:
    result = install_module.install()
    assert result.written is True
    assert result.replaced is False
    assert result.path == package_file()

    document = written(result.path)
    assert document[install_module.MARKER_KEY] == install_module.MARKER_VALUE
    assert document["enable"] is True
    assert env_value(document, install_module.PAYLOAD_ENV_VAR) == str(install_module.payload_root())
    assert env_value(document, install_module.SOURCE_ENV_VAR) == str(install_module.source_root())
    assert env_value(document, install_module.AUTOSTART_ENV_VAR) == "0"
    assert env_value(document, "PYTHONPATH") == {
        "value": f"${install_module.SOURCE_ENV_VAR}",
        "method": "prepend",
    }
    # `hpath` is the key Houdini reads for its own path; `path` is deprecated.
    assert document["hpath"] == f"${install_module.PAYLOAD_ENV_VAR}"
    assert "path" not in document
    # The paths in it are real folders of this copy of the project.
    assert install_module.payload_root().is_dir()
    assert (install_module.source_root() / "nscr_houdini_mcp").is_dir()


def test_install_can_be_asked_for_auto_start() -> None:
    result = install_module.install(autostart=True)
    assert env_value(written(result.path), install_module.AUTOSTART_ENV_VAR) == "1"


def test_a_dry_run_writes_nothing_and_says_what_it_would_write(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["bridge", "install", "--dry-run"]) == 0
    assert not package_file().exists()
    printed = capsys.readouterr().out
    assert "would write" in printed
    assert str(package_file()) in printed
    assert str(install_module.source_root()) in printed


def test_installing_again_replaces_our_own_file() -> None:
    install_module.install()
    again = install_module.install(autostart=True)
    assert again.replaced is True
    assert env_value(written(again.path), install_module.AUTOSTART_ENV_VAR) == "1"


def test_a_package_this_tool_did_not_write_is_left_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = package_file()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"enable": True, "path": "/somebody/else"}), encoding="utf-8")

    with pytest.raises(install_module.NotOurs):
        install_module.install()
    assert cli.main(["bridge", "install"]) == 1
    assert "not written by this tool" in capsys.readouterr().out
    assert written(path)["path"] == "/somebody/else"


def test_a_file_that_is_not_json_is_not_ours_either() -> None:
    path = package_file()
    path.parent.mkdir(parents=True)
    path.write_text("not json at all", encoding="utf-8")
    assert install_module.is_ours(path) is False
    with pytest.raises(install_module.NotOurs):
        install_module.install()


def test_a_link_where_the_package_goes_is_never_written_through(tmp_path: Path) -> None:
    path = package_file()
    path.parent.mkdir(parents=True)
    elsewhere = tmp_path / "somewhere" / "else.json"
    path.symlink_to(elsewhere)

    with pytest.raises(install_module.NotOurs):
        install_module.install()
    # The link pointed nowhere, and still points nowhere.
    assert path.is_symlink()
    assert not elsewhere.exists()


def test_a_link_to_a_package_of_ours_is_left_alone_too(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text(
        json.dumps(install_module.document(source=tmp_path, payload=tmp_path)), encoding="utf-8"
    )
    path = package_file()
    path.parent.mkdir(parents=True)
    path.symlink_to(real)

    with pytest.raises(install_module.NotOurs):
        install_module.install()
    assert install_module.is_ours(path) is False
    removed = install_module.uninstall()
    assert [(item.removed, item.reason) for item in removed] == [(False, "a link, kept")]
    assert path.is_symlink()
    assert real.exists()


def test_a_path_houdini_would_expand_is_refused(tmp_path: Path) -> None:
    with pytest.raises(install_module.InstallError) as raised:
        install_module.document(source=tmp_path / "$WORK" / "src", payload=tmp_path)
    assert "$" in str(raised.value)
    with pytest.raises(install_module.InstallError):
        install_module.document(source=tmp_path, payload=tmp_path / "back`tick")


def test_a_marker_on_its_own_is_not_enough_to_be_ours() -> None:
    path = package_file()
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({install_module.MARKER_KEY: install_module.MARKER_VALUE}), encoding="utf-8"
    )
    assert install_module.is_ours(path) is False


def test_install_writes_nowhere_but_the_packages_folder(tmp_path: Path) -> None:
    install_module.install()
    made = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert made == [
        Path("prefs22.0"),
        Path("prefs22.0/packages"),
        Path("prefs22.0/packages/nscr_houdini_mcp.json"),
    ]


# Section: uninstall


def test_uninstall_removes_what_it_wrote_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    install_module.install()
    assert cli.main(["bridge", "uninstall"]) == 0
    printed = capsys.readouterr().out
    assert "removed" in printed
    assert str(package_file()) in printed
    assert not package_file().exists()


def test_uninstall_is_happy_when_there_is_nothing_there(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["bridge", "uninstall"]) == 0
    assert "nothing to remove" in capsys.readouterr().out


def test_uninstall_keeps_a_package_it_did_not_write(capsys: pytest.CaptureFixture[str]) -> None:
    path = package_file()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"enable": True}), encoding="utf-8")
    assert cli.main(["bridge", "uninstall"]) == 1
    assert "kept" in capsys.readouterr().out
    assert path.exists()


def test_uninstall_takes_away_the_folders_install_made(tmp_path: Path) -> None:
    install_module.install()
    made = package_file().parent
    assert made.is_dir()
    install_module.uninstall()
    # The packages folder and the preference folder were both made here.
    assert not made.exists()
    assert not (tmp_path / "prefs22.0").exists()


def test_uninstall_keeps_a_folder_it_did_not_make(tmp_path: Path) -> None:
    folder = tmp_path / "already-there"
    folder.mkdir()
    install_module.install(packages=folder)
    install_module.uninstall(packages=folder)
    assert folder.is_dir()


def test_uninstall_with_no_version_looks_at_every_version_present(tmp_path: Path) -> None:
    install_module.install("22.0")
    install_module.install("21.5")
    removed = sorted(item.path for item in install_module.uninstall())
    assert removed == sorted(
        [install_module.package_path("21.5"), install_module.package_path("22.0")]
    )


def test_uninstall_can_be_pointed_at_one_version() -> None:
    install_module.install("22.0")
    install_module.install("21.5")
    removed = install_module.uninstall("21.5")
    assert [item.path for item in removed] == [install_module.package_path("21.5")]
    assert install_module.package_path("22.0").exists()


# Section: what is installed


def test_installed_reports_the_file_and_its_auto_start() -> None:
    assert install_module.installed("22.0")[0].present is False
    install_module.install(autostart=True)
    state = install_module.installed("22.0")[0]
    assert (state.present, state.ours, state.autostart) == (True, True, True)


def test_installed_reports_somebody_elses_package_as_not_ours() -> None:
    path = package_file()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"enable": True}), encoding="utf-8")
    state = install_module.installed("22.0")[0]
    assert (state.present, state.ours, state.autostart) == (True, False, None)


# Section: the Houdini installs on this machine


def fake_installs(monkeypatch: pytest.MonkeyPatch, root: Path, names: list[str]) -> None:
    for name in names:
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(install_module, "install_roots", lambda: [root])
    monkeypatch.delenv(install_module.HFS_ENV_VAR, raising=False)


def test_houdini_installs_are_found_newest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_module.sys, "platform", "linux")
    fake_installs(monkeypatch, tmp_path / "opt", ["hfs22.0.368", "hfs21.5.100", "notahoudini"])
    found = install_module.find_installs()
    assert [item.version for item in found] == ["22.0.368", "21.5.100"]
    assert found[0].short_version == "22.0"
    assert found[0].hfs == tmp_path / "opt" / "hfs22.0.368"


def test_a_named_install_comes_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_module.sys, "platform", "linux")
    fake_installs(monkeypatch, tmp_path / "opt", ["hfs22.0.368"])
    found = install_module.find_installs(tmp_path / "elsewhere" / "hfs22.0.400")
    assert found[0].version == "22.0.400"
    assert [item.version for item in found] == ["22.0.400", "22.0.368"]


def test_the_mac_install_holds_its_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_module.sys, "platform", "darwin")
    fake_installs(monkeypatch, tmp_path / "Applications", ["Houdini22.0.368", "Current"])
    found = install_module.find_installs()
    assert [item.version for item in found] == ["22.0.368"]
    assert found[0].hfs.name == "Resources"
    assert "Houdini.framework" in str(found[0].hfs)


# Section: snippet


def test_the_snippet_names_the_source_folder_it_was_printed_from(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["bridge", "snippet"]) == 0
    printed = capsys.readouterr().out
    assert str(install_module.source_root()) in printed
    assert "from nscr_houdini_mcp.bridge import Bridge" in printed
    assert "bridge.start()" in printed
    # It has to run as it stands in a Houdini, so it has to parse here.
    compile(printed, "snippet", "exec")


# Section: status


def test_status_says_when_there_is_nothing_running(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["bridge", "status", "--home", str(home)]) == 0
    printed = capsys.readouterr().out
    assert "sessions: none" in printed
    assert "not installed" in printed
    assert f"decided by {install_module.SOURCE_PREF_ENV}" in printed
    assert str(package_file()) in printed


def test_status_reports_the_package_it_wrote(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_module.install(autostart=True)
    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert "installed, autostart on" in printed
    assert "considered:" in printed


def test_status_lists_a_session_from_the_store(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with store_module.Store(home / store_module.STORE_FILE_NAME) as store:
        store.register_session(
            "s-1",
            kind="hython",
            pid=os.getpid(),
            pid_start=store_module.process_start_stamp(),
            alias="w1",
            port=18999,
            hip_path="/scenes/one.hip",
            scene_epoch=2,
        )
    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert "w1 s-1" in printed
    assert "kind hython" in printed
    assert "port 18999" in printed
    assert "epoch 2" in printed
    assert "/scenes/one.hip" in printed
    # With no session file there is no token, so nothing is sent anywhere.
    assert "health not asked" in printed


def test_status_says_when_a_session_does_not_answer(
    home: Path, silent_port: int, capsys: pytest.CaptureFixture[str]
) -> None:
    registry.ensure_registry_dir(home)
    registry.write_entry(
        home,
        {
            "session_id": "s-2",
            "alias": "w2",
            "kind": "hython",
            "pid": os.getpid(),
            "pid_start": store_module.process_start_stamp(),
            "port": silent_port,
            "token": "nothing-is-listening",
            "scene_epoch": 0,
            "hip_path": None,
        },
    )
    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert "w2 s-2" in printed
    assert "health no answer" in printed


@pytest.fixture
def silent_port() -> Iterator[int]:
    """A port held open and never listened on, so a connection is refused.

    Holding it for the length of the test is what keeps something else from
    taking it in between, which a port merely found free cannot promise.
    """
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        yield int(sock.getsockname()[1])


def test_the_health_sweep_stops_when_its_time_is_spent(
    home: Path,
    silent_port: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.ensure_registry_dir(home)
    for number in (1, 2):
        registry.write_entry(
            home,
            {
                "session_id": f"s-{number}",
                "alias": f"w{number}",
                "kind": "hython",
                "pid": os.getpid(),
                "pid_start": store_module.process_start_stamp(),
                "port": silent_port,
                "token": "nothing-is-listening",
                "scene_epoch": 0,
                "hip_path": None,
            },
        )
    monkeypatch.setattr(cli, "HEALTH_BUDGET_S", 0.0)
    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert printed.count("health not asked, the time for asking was spent") == 2


# Section: the startup module that ships with the payload


def autostart_module():
    """The payload module, loaded from where Houdini would load it."""
    import importlib.util

    path = install_module.payload_root() / "python3.13libs" / "nscr_mcp_autostart.py"
    spec = importlib.util.spec_from_file_location("nscr_mcp_autostart_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_port_range_that_makes_no_sense_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = autostart_module()
    default = (18100, 18199)

    monkeypatch.setenv(module.PORT_VAR, "18400")
    monkeypatch.setenv(module.MAX_PORT_VAR, "18449")
    assert module.port_range(default) == (18400, 18449)

    monkeypatch.setenv(module.PORT_VAR, "18500")
    monkeypatch.setenv(module.MAX_PORT_VAR, "18400")
    assert module.port_range(default) == default

    monkeypatch.setenv(module.PORT_VAR, "80")
    monkeypatch.setenv(module.MAX_PORT_VAR, "90")
    assert module.port_range(default) == default

    monkeypatch.setenv(module.PORT_VAR, "not a number")
    assert module.port_range(default) == default
    # Every refusal says so where a person can see it.
    assert capsys.readouterr().err.count("\n") == 3


def test_a_start_that_fails_says_so_and_writes_the_story_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = autostart_module()
    monkeypatch.setenv(module.AUTOSTART_VAR, "1")
    monkeypatch.setenv(module.HOME_VAR, str(tmp_path))
    monkeypatch.setattr(module, "start", _raise_for_test)

    assert module.start_if_wanted() is None
    assert (
        "nscr bridge did not start: RuntimeError: no port for this one" in capsys.readouterr().err
    )
    log = (tmp_path / "logs" / module.LOG_NAME).read_text(encoding="utf-8")
    assert "RuntimeError: no port for this one" in log
    assert "Traceback" in log


def _raise_for_test() -> None:
    raise RuntimeError("no port for this one")


# Section: a real Houdini


@pytest.fixture
def houdini_lookup(temp_pref_dir: Path) -> install_module.Lookup:
    """Where a real Houdini on this machine says its packages folder is."""
    return install_module.resolve()


@pytest.fixture
def hython_session(
    tmp_path: Path, home: Path, temp_pref_dir: Path, houdini_lookup: install_module.Lookup
) -> Iterator[subprocess.Popen[str]]:
    """One hython started with the installed package, and stopped again."""
    install_module.install(autostart=True, lookup=houdini_lookup)
    script = tmp_path / "hold.py"
    # The bridge runs on a thread of its own, so the process only has to stay
    # alive. It ends as soon as anything arrives on its input.
    script.write_text("import sys\nsys.stdin.readline()\n", encoding="utf-8")
    environment = dict(os.environ)
    environment[install_module.PREF_DIR_ENV_VAR] = str(temp_pref_dir)
    environment[store_module.HOME_ENV_VAR] = str(home)
    environment["NSCR_MCP_PORT"] = str(PORT_RANGE[0])
    environment["NSCR_MCP_MAX_PORT"] = str(PORT_RANGE[1])
    log = (tmp_path / "hython.log").open("w", encoding="utf-8")
    process = subprocess.Popen(  # noqa: S603 - the binary is this machine's Houdini
        [str(find_hython()), str(script)],
        stdin=subprocess.PIPE,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write("\n")
                process.stdin.flush()
                process.stdin.close()
            try:
                process.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=STOP_TIMEOUT_S)
        log.close()


@pytest.mark.houdini
@pytest.mark.skipif(not hython_available(), reason="no hython on this machine")
def test_the_installed_package_starts_a_bridge_in_a_fresh_houdini(
    tmp_path: Path,
    home: Path,
    temp_pref_dir: Path,
    capsys: pytest.CaptureFixture[str],
    houdini_lookup: install_module.Lookup,
    hython_session: subprocess.Popen[str],
) -> None:
    # The folder was not guessed: a real Houdini was asked and named its own
    # home, which is the temporary one this test set.
    assert houdini_lookup.source == install_module.SOURCE_HOUDINI_HOME
    expected = Path(str(temp_pref_dir).replace(install_module.VERSION_TOKEN, "22.0"))
    assert houdini_lookup.path == expected / "packages"

    entry = wait_for_entry(home, hython_session)
    assert PORT_RANGE[0] <= int(entry["port"]) <= PORT_RANGE[1]

    with store_module.Store(home / store_module.STORE_FILE_NAME) as store:
        listed = store.list_sessions()
    assert [record.session_id for record in listed] == [entry["session_id"]]
    assert listed[0].kind == "hython"

    answer = client.health(client.Session.from_entry(entry), timeout_s=10.0)
    assert answer.status == 200
    assert answer.payload["data"]["status"] == "ok"
    assert answer.payload["data"]["busy"] is False

    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert entry["session_id"] in printed
    assert "health ok" in printed
    assert "busy no" in printed

    # Stop it, and leave the preference folder as it was found.
    assert hython_session.stdin is not None
    hython_session.stdin.write("\n")
    hython_session.stdin.flush()
    hython_session.stdin.close()
    assert hython_session.wait(timeout=STOP_TIMEOUT_S) == 0

    removed = install_module.uninstall(lookup=houdini_lookup)
    assert [item.removed for item in removed] == [True]
    assert list(houdini_lookup.path.glob("*")) == []


def wait_for_entry(home: Path, process: subprocess.Popen[str]) -> dict:
    """Wait for the session file the started bridge writes."""
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        for entry in registry.list_entries(home):
            if entry.get("pid") == process.pid:
                return entry
        if process.poll() is not None:
            raise AssertionError(f"hython exited with {process.returncode} and no bridge")
        time.sleep(0.25)
    raise AssertionError("no bridge after waiting")
