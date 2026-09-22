"""The package file, the commands that write it, and what status reports.

Every test here points Houdini's preference folder at a temporary one, so a
run never reads or writes the folder the person at this machine works in.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
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
def temp_pref_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every command in this file at a preference folder of its own."""
    root = tmp_path / PREF_TEMPLATE
    monkeypatch.setenv(install_module.PREF_DIR_ENV_VAR, str(root))
    return root


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
    assert document["path"] == f"${install_module.PAYLOAD_ENV_VAR}"
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
    assert "22.0 not installed" in printed


def test_status_reports_the_package_it_wrote(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    install_module.install(autostart=True)
    cli.main(["bridge", "status", "--home", str(home)])
    assert "22.0 installed, autostart on" in capsys.readouterr().out


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
    home: Path, capsys: pytest.CaptureFixture[str]
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
            "port": free_port(),
            "token": "nothing-is-listening",
            "scene_epoch": 0,
            "hip_path": None,
        },
    )
    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert "w2 s-2" in printed
    assert "health no answer" in printed


def free_port() -> int:
    """A port nothing is on, which is a port nothing will answer."""
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


# Section: a real Houdini


@pytest.fixture
def hython_session(
    tmp_path: Path, home: Path, temp_pref_dir: Path
) -> Iterator[subprocess.Popen[str]]:
    """One hython started with the installed package, and stopped again."""
    install_module.install(autostart=True)
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
    capsys: pytest.CaptureFixture[str],
    hython_session: subprocess.Popen[str],
) -> None:
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

    removed = install_module.uninstall()
    assert [item.removed for item in removed] == [True]
    assert list(install_module.packages_dir().glob("*")) == []


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
