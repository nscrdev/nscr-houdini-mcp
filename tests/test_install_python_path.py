"""The folder Houdini's Python imports this package from, and nothing else.

An installed copy sits in a site-packages folder beside every other library of
its environment. Putting that folder in front of Houdini's own would load, say,
numpy built for another Python in place of the one Houdini ships, and break
Houdini's own tools on start. These tests build such a folder by hand and check
that Houdini is only ever given a folder holding this package alone, that the
copies made for it are never half there, and that old ones are taken away.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import support
from nscr_houdini_mcp import cli, pool
from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry
from nscr_houdini_mcp.bridge.launcher import HythonBridge, find_hython, hython_available

PREF_TEMPLATE = f"prefs{install_module.VERSION_TOKEN}"

# Its own range, away from anything a person on this machine may be running.
PORT_RANGE = support.INSTALL_PORTS

START_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 60.0

# What the libraries in a pretend environment say when they are imported, so a
# Houdini that loads one in place of its own says so out loud.
SHADOWED = "loaded from the environment instead of from Houdini"

ANSWER = "nscr-python-path-answer "

REAL_SOURCE_ROOT = install_module.source_root
REAL_PACKAGE_ROOT = install_module.package_root
CHECKOUT_PAYLOAD = install_module.payload_root()

EIGHT_DAYS_S = 8 * 24 * 3600.0


@pytest.fixture(autouse=True)
def temp_pref_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / PREF_TEMPLATE
    monkeypatch.setenv(install_module.PREF_DIR_ENV_VAR, str(root))
    monkeypatch.delenv(install_module.PACKAGE_DIR_ENV_VAR, raising=False)
    # Nothing here may reach the state folder of the person at this machine.
    monkeypatch.setenv(store_module.HOME_ENV_VAR, str(tmp_path / "default-home"))
    monkeypatch.setattr(
        install_module, "ask_houdini", lambda version=None: (None, "no Houdini asked")
    )
    return root


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "state"
    folder.mkdir(mode=0o700)
    return folder


def make_site(folder: Path, marker: str = "v1") -> Path:
    """A site-packages folder as a wheel install leaves it: this package, with
    its startup files inside it, and the libraries around it."""
    folder.mkdir(parents=True)
    ours = folder / install_module.PACKAGE_NAME
    shutil.copytree(
        REAL_PACKAGE_ROOT(),
        ours,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(CHECKOUT_PAYLOAD, ours / "houdini")
    (ours / "_marker.py").write_text(f"V = {marker!r}\n", encoding="utf-8")
    (ours / "__pycache__").mkdir()
    (ours / "__pycache__" / "cached.cpython-311.pyc").write_bytes(b"not a module")
    for name in ("numpy", "PIL", "mcp", "pydantic", "anyio"):
        (folder / name).mkdir()
        (folder / name / "__init__.py").write_text(
            f"raise ImportError({name!r} + ' ' + {SHADOWED!r})\n", encoding="utf-8"
        )
    (folder / "typing_extensions.py").write_text("", encoding="utf-8")
    (folder / "numpy-2.3.0.dist-info").mkdir()
    (folder / "distutils-precedence.pth").write_text("", encoding="utf-8")
    return folder


def run_from(monkeypatch: pytest.MonkeyPatch, site: Path) -> Path:
    """Run this module as the copy of the package inside `site` would."""
    package = site / install_module.PACKAGE_NAME
    monkeypatch.setattr(install_module, "source_root", lambda: site)
    monkeypatch.setattr(install_module, "package_root", lambda: package)
    monkeypatch.setattr(install_module, "payload_root", lambda: package / "houdini")
    return package


@pytest.fixture
def site(tmp_path: Path) -> Path:
    return make_site(tmp_path / "venv" / "lib" / "site-packages")


@pytest.fixture
def from_site(site: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    run_from(monkeypatch, site)
    return site


def stamp_of(site: Path) -> str:
    return install_module.fingerprint(site / install_module.PACKAGE_NAME)


def package_file() -> Path:
    return install_module.package_path()


def written(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def env_value(document: dict, name: str) -> object:
    for item in document["env"]:
        if name in item:
            return item[name]
    raise AssertionError(f"{name} is not in the package")


def names_in(folder: Path) -> list[str]:
    return sorted(child.name for child in folder.iterdir()) if folder.is_dir() else []


def age(path: Path, seconds: float) -> None:
    """Make a file or folder look as if it was last touched that long ago."""
    then = time.time() - seconds
    os.utime(path, (then, then))


# Section: what counts as something else to import


def test_a_folder_holding_only_this_package_has_nothing_else(tmp_path: Path) -> None:
    (tmp_path / install_module.PACKAGE_NAME).mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "nscr_houdini_mcp-0.1.0.dist-info").mkdir()
    (tmp_path / "nscr_houdini_mcp.egg-info").mkdir()
    (tmp_path / ".DS_Store").write_bytes(b"")
    (tmp_path / "README.txt").write_text("", encoding="utf-8")
    assert install_module.strays(tmp_path) == []


def test_every_way_a_python_imports_something_else_is_found(tmp_path: Path) -> None:
    (tmp_path / install_module.PACKAGE_NAME).mkdir()
    (tmp_path / "numpy").mkdir()
    # A folder with no __init__.py is still a namespace package.
    (tmp_path / "google").mkdir()
    for name in (
        "six.py",
        "compiled.pyc",
        "_cffi_backend.cpython-312-darwin.so",
        "_speedups.cp312-win_amd64.pyd",
        "gui.pyw",
        "distutils-precedence.pth",
    ):
        (tmp_path / name).write_bytes(b"")
    assert install_module.strays(tmp_path) == [
        "_cffi_backend.cpython-312-darwin.so",
        "_speedups.cp312-win_amd64.pyd",
        "compiled.pyc",
        "distutils-precedence.pth",
        "google",
        "gui.pyw",
        "numpy",
        "six.py",
    ]


def test_a_checkout_source_folder_holds_only_this_package() -> None:
    assert install_module.strays(install_module.source_root()) == []


def test_a_fingerprint_follows_what_the_files_hold_not_where_they_are(tmp_path: Path) -> None:
    one = make_site(tmp_path / "one") / install_module.PACKAGE_NAME
    two = make_site(tmp_path / "two") / install_module.PACKAGE_NAME
    assert install_module.fingerprint(one) == install_module.fingerprint(two)
    (two / "_marker.py").write_text("V = 'v2'\n", encoding="utf-8")
    assert install_module.fingerprint(one) != install_module.fingerprint(two)


# Section: a checkout is named as it stands


def test_a_checkout_install_names_its_source_folder_and_copies_nothing(home: Path) -> None:
    result = install_module.install(home=home)
    document = written(result.path)
    assert env_value(document, install_module.SOURCE_ENV_VAR) == str(install_module.source_root())
    assert env_value(document, install_module.PAYLOAD_ENV_VAR) == str(CHECKOUT_PAYLOAD)
    assert install_module.COPY_KEY not in document
    assert result.copied_from is None
    assert not install_module.copies_dir(home).exists()


# Section: an installed copy is copied


def test_an_installed_copy_gives_houdini_a_folder_with_only_this_package(
    from_site: Path, home: Path
) -> None:
    result = install_module.install(home=home)
    document = written(result.path)
    source = Path(str(env_value(document, install_module.SOURCE_ENV_VAR)))

    assert source != from_site
    assert source == install_module.copy_for_package_file(result.path, home, stamp_of(from_site))
    assert source.parent == install_module.copies_dir(home)
    assert result.source == source
    assert result.copied_from == from_site / install_module.PACKAGE_NAME
    assert document[install_module.COPY_KEY] == str(source)
    assert env_value(document, "PYTHONPATH") == {
        "value": f"${install_module.SOURCE_ENV_VAR}",
        "method": "prepend",
    }

    # This package and nothing else: no numpy, no PIL, no mcp, no .pth.
    assert install_module.strays(source) == []
    assert [name for name in names_in(source) if not name.startswith(".")] == [
        install_module.PACKAGE_NAME
    ]
    copied = source / install_module.PACKAGE_NAME
    assert (copied / "__init__.py").is_file()
    assert (copied / "bridge" / "app.py").is_file()
    assert not (copied / "__pycache__").exists()
    assert install_module.copy_is_current(source) is True


def test_houdini_s_path_takes_the_startup_files_from_the_same_copy(
    from_site: Path, home: Path
) -> None:
    """The startup files and the bridge they start are always one version."""
    result = install_module.install(home=home)
    document = written(result.path)
    payload = Path(str(env_value(document, install_module.PAYLOAD_ENV_VAR)))
    assert payload == result.source / install_module.PACKAGE_NAME / "houdini"
    assert result.payload == payload
    assert not payload.is_relative_to(from_site)
    assert (payload / "python3.13libs" / "nscr_mcp_autostart.py").is_file()
    assert document["hpath"] == f"${install_module.PAYLOAD_ENV_VAR}"


def test_the_install_lines_say_where_the_copy_came_from(
    from_site: Path, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.setenv(store_module.HOME_ENV_VAR, str(home))
    assert cli.main(["bridge", "install"]) == 0
    printed = capsys.readouterr().out
    copy = install_module.copy_for_package_file(package_file(), home, stamp_of(from_site))
    assert f"pythonpath     {copy}" in printed
    assert f"copied from    {from_site / install_module.PACKAGE_NAME}" in printed


def test_a_dry_run_names_the_copy_and_makes_nothing(from_site: Path, home: Path) -> None:
    result = install_module.install(home=home, dry_run=True)
    assert result.source == install_module.copy_for_package_file(
        result.path, home, stamp_of(from_site)
    )
    assert not result.path.exists()
    assert not install_module.copies_dir(home).exists()


def test_an_upgrade_gets_a_copy_of_its_own_and_the_old_one_stays_a_while(
    from_site: Path, home: Path
) -> None:
    first = install_module.install(home=home)
    package = from_site / install_module.PACKAGE_NAME
    (package / "added_later.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "helptext.py").unlink()

    assert install_module.copy_is_current(first.source) is False
    again = install_module.install(home=home, autostart=True)

    assert again.source != first.source
    assert written(again.path)[install_module.COPY_KEY] == str(again.source)
    copied = again.source / install_module.PACKAGE_NAME
    assert (copied / "added_later.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (copied / "helptext.py").exists()
    assert install_module.copy_is_current(again.source) is True
    # A Houdini started before still imports from the old copy, whole.
    assert (first.source / install_module.PACKAGE_NAME / "helptext.py").is_file()
    assert (first.source / install_module.SUPERSEDED_FILE_NAME).is_file()
    # Nothing is left beside them from the build.
    assert names_in(install_module.copies_dir(home)) == sorted(
        [first.source.name, again.source.name]
    )


def test_installing_the_same_version_again_keeps_the_same_copy(from_site: Path, home: Path) -> None:
    first = install_module.install(home=home)
    again = install_module.install(home=home)
    assert again.source == first.source
    assert not (again.source / install_module.SUPERSEDED_FILE_NAME).exists()
    assert names_in(install_module.copies_dir(home)) == [first.source.name]


def test_an_old_copy_goes_a_week_after_an_install_replaced_it(from_site: Path, home: Path) -> None:
    first = install_module.install(home=home)
    (from_site / install_module.PACKAGE_NAME / "added_later.py").write_text("", encoding="utf-8")
    second = install_module.install(home=home)
    age(first.source / install_module.SUPERSEDED_FILE_NAME, EIGHT_DAYS_S)
    # The copy a package file names is never taken for its age.
    age(second.source / install_module.USED_FILE_NAME, EIGHT_DAYS_S)
    install_module.sweep(install_module.copies_dir(home))
    assert not first.source.exists()
    assert second.source.is_dir()


def test_uninstall_takes_every_copy_away_with_the_file(from_site: Path, home: Path) -> None:
    first = install_module.install(home=home)
    (from_site / install_module.PACKAGE_NAME / "added_later.py").write_text("", encoding="utf-8")
    second = install_module.install(home=home)
    removed = install_module.uninstall(home=home)
    assert [(item.path, item.reason) for item in removed] == [
        (second.path, "removed"),
        *sorted(
            [(first.source, "removed its copy"), (second.source, "removed its copy")],
        ),
    ]
    assert not second.path.exists()
    assert not install_module.copies_dir(home).exists()
    # The state folder itself was not made by the install and stays.
    assert home.is_dir()


def test_uninstall_takes_the_copies_for_started_sessions_away_too(
    from_site: Path, home: Path
) -> None:
    install_module.install(home=home)
    run_copy = install_module.python_path_for_run(home)
    removed = install_module.uninstall(home=home)
    assert (run_copy, "removed a copy for started sessions") in [
        (item.path, item.reason) for item in removed
    ]
    assert not run_copy.exists()


def test_two_package_files_have_copies_of_their_own(
    from_site: Path, home: Path, tmp_path: Path
) -> None:
    one = install_module.install("22.0", home=home)
    other = install_module.install(home=home, packages=tmp_path / "elsewhere")
    assert one.source != other.source

    install_module.uninstall("22.0", home=home)
    assert not one.source.exists()
    assert install_module.strays(other.source) == []
    assert (other.source / install_module.PACKAGE_NAME / "__init__.py").is_file()


def test_a_folder_where_the_copy_goes_that_is_not_ours_is_left_alone(
    from_site: Path, home: Path
) -> None:
    folder = install_module.copy_for_package_file(package_file(), home, stamp_of(from_site))
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(install_module.InstallError, match="not made by this tool"):
        install_module.install(home=home)
    assert (folder / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not package_file().exists()


def test_uninstall_leaves_a_copy_folder_it_cannot_recognise(from_site: Path, home: Path) -> None:
    result = install_module.install(home=home)
    (result.source / install_module.STAMP_FILE_NAME).unlink()
    install_module.uninstall(home=home)
    assert not result.path.exists()
    assert result.source.is_dir()


def test_moving_to_a_checkout_marks_the_old_copy_replaced(
    from_site: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = install_module.install(home=home)
    monkeypatch.setattr(install_module, "source_root", REAL_SOURCE_ROOT)
    monkeypatch.setattr(install_module, "package_root", REAL_PACKAGE_ROOT)
    monkeypatch.setattr(install_module, "payload_root", lambda: CHECKOUT_PAYLOAD)
    again = install_module.install(home=home)
    assert again.path == first.path
    assert again.source == install_module.source_root()
    assert (first.source / install_module.SUPERSEDED_FILE_NAME).is_file()


def test_a_copy_is_private_to_the_person_who_made_it(from_site: Path, home: Path) -> None:
    result = install_module.install(home=home)
    if sys.platform != "win32":
        assert result.source.stat().st_mode & 0o777 == 0o700


# Section: what status says


def test_status_says_whether_the_copy_is_this_server_s_version(
    from_site: Path, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = install_module.install(home=home)
    cli.main(["bridge", "status", "--home", str(home)])
    assert f"python copy {result.source} (the same version as this server)" in (
        capsys.readouterr().out
    )

    (from_site / install_module.PACKAGE_NAME / "added_later.py").write_text("", encoding="utf-8")
    cli.main(["bridge", "status", "--home", str(home)])
    assert "another version than this server, run bridge install again" in (capsys.readouterr().out)


def test_status_compares_the_copy_with_the_running_server_not_its_source(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new environment runs the server while the old one is still on disk:
    the copy still matches its own source, and is still the wrong version."""
    site_a = make_site(tmp_path / "a", "v1")
    site_b = make_site(tmp_path / "b", "v2")
    run_from(monkeypatch, site_a)
    result = install_module.install(home=home)
    run_from(monkeypatch, site_b)
    [state] = [state for state in install_module.installed() if state.copy is not None]
    assert state.copy_current is False
    assert "v1" in (result.source / install_module.PACKAGE_NAME / "_marker.py").read_text()


def test_status_flags_a_package_file_that_names_a_whole_site_packages(
    site: Path, capsys: pytest.CaptureFixture[str], home: Path
) -> None:
    """A package file written before the copy existed, naming the environment."""
    path = package_file()
    path.parent.mkdir(parents=True)
    body = install_module.document(source=site, payload=site / "nscr_houdini_mcp" / "houdini")
    path.write_text(json.dumps(body), encoding="utf-8")

    [state] = [state for state in install_module.installed() if state.present]
    assert state.source == site
    assert "numpy" in state.source_strays
    cli.main(["bridge", "status", "--home", str(home)])
    printed = capsys.readouterr().out
    assert f"{install_module.SOURCE_ENV_VAR} {site} holds other libraries too" in printed
    assert "run bridge install again" in printed


def test_status_says_nothing_of_the_kind_for_a_checkout(
    capsys: pytest.CaptureFixture[str], home: Path
) -> None:
    install_module.install(home=home)
    cli.main(["bridge", "status", "--home", str(home)])
    assert "holds other libraries" not in capsys.readouterr().out


# Section: Windows paths


def test_a_copy_folder_name_is_short_and_safe_on_every_system(home: Path) -> None:
    windows_file = r"C:\Users\artist\Documents\houdini22.0\packages\nscr_houdini_mcp.json"
    stamp = "0123456789abcdef0123456789abcdef"
    folder = install_module.copy_for_package_file(Path(windows_file), home, stamp)
    assert folder.parent == install_module.copies_dir(home)
    name = folder.name
    assert name.startswith("package-")
    assert len(name) == len("package-") + 16 + 1 + 16
    assert all(character.isalnum() or character == "-" for character in name)
    # The same file and version always get the same folder.
    assert install_module.copy_for_package_file(Path(windows_file), home, stamp) == folder
    other = windows_file.replace("22.0", "21.5")
    assert install_module.copy_for_package_file(Path(other), home, stamp) != folder


def test_a_windows_copy_path_is_written_into_the_package_as_it_stands() -> None:
    source = Path(r"C:\Users\artist\AppData\Local\nscr-houdini-mcp\houdini-python\package-0a1b")
    payload = source / "nscr_houdini_mcp" / "houdini"
    body = install_module.document(source=source, payload=payload, copy=source)
    assert env_value(body, install_module.SOURCE_ENV_VAR) == str(source)
    assert env_value(body, install_module.PAYLOAD_ENV_VAR) == str(payload)
    assert body[install_module.COPY_KEY] == str(source)
    assert install_module.copy_of(json.loads(json.dumps(body))) == source


def test_a_windows_extension_module_is_something_else_to_import(tmp_path: Path) -> None:
    (tmp_path / install_module.PACKAGE_NAME).mkdir()
    (tmp_path / "_multiarray_umath.cp313-win_amd64.pyd").write_bytes(b"")
    assert install_module.strays(tmp_path) == ["_multiarray_umath.cp313-win_amd64.pyd"]


def test_a_rename_windows_refuses_for_a_moment_is_tried_again(
    from_site: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scanner or indexer holding the new folder open makes Windows say no."""
    real = os.rename
    refusals = [PermissionError(5, "Access is denied")] * 2

    def rename(source: object, target: object) -> None:
        if refusals and ".part-" in str(source):
            raise refusals.pop()
        real(source, target)

    monkeypatch.setattr(install_module.os, "rename", rename)
    monkeypatch.setattr(install_module, "RENAME_PAUSE_S", 0.0)
    folder = install_module.python_path_for_run(home)
    assert refusals == []
    assert install_module.is_our_copy(folder)


def test_a_rename_windows_keeps_refusing_is_an_error_and_leaves_nothing(
    from_site: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rename(source: object, target: object) -> None:
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(install_module.os, "rename", rename)
    monkeypatch.setattr(install_module, "RENAME_PAUSE_S", 0.0)
    with pytest.raises(PermissionError):
        install_module.python_path_for_run(home)
    assert names_in(install_module.copies_dir(home)) == []


# Section: sessions this tool starts itself


def test_a_started_session_from_a_checkout_gets_the_source_folder(home: Path) -> None:
    assert install_module.python_path_for_run(home) == install_module.source_root()
    assert not install_module.copies_dir(home).exists()


def test_a_started_session_from_an_installed_copy_gets_a_copy(from_site: Path, home: Path) -> None:
    package = from_site / install_module.PACKAGE_NAME
    folder = install_module.python_path_for_run(home)
    assert folder == install_module.copy_for_source(package, home, stamp_of(from_site))
    assert install_module.strays(folder) == []
    made = (folder / install_module.STAMP_FILE_NAME).stat().st_mtime_ns

    # A current copy is used as it is.
    assert install_module.python_path_for_run(home) == folder
    assert (folder / install_module.STAMP_FILE_NAME).stat().st_mtime_ns == made

    # A changed source gets a copy of its own; the old one is not touched, so
    # a session importing from it never finds it gone.
    (package / "added_later.py").write_text("", encoding="utf-8")
    newer = install_module.python_path_for_run(home)
    assert newer != folder
    assert (newer / install_module.PACKAGE_NAME / "added_later.py").is_file()
    assert (folder / install_module.PACKAGE_NAME / "__init__.py").is_file()


RACER = textwrap.dedent(
    """
    import sys
    import time
    from pathlib import Path

    from nscr_houdini_mcp import install

    site, home, go = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    install.source_root = lambda: site
    install.package_root = lambda: site / install.PACKAGE_NAME
    while time.time() < go:
        pass
    try:
        print("OK", install.python_path_for_run(home))
    except BaseException as error:
        print("ERR", type(error).__name__, error)
    """
)


def test_servers_making_the_same_copy_at_once_never_leave_it_missing(
    tmp_path: Path, home: Path
) -> None:
    site = make_site(tmp_path / "a")
    script = tmp_path / "racer.py"
    script.write_text(RACER, encoding="utf-8")
    folder = install_module.copy_for_source(site / install_module.PACKAGE_NAME, home, "x")
    answers: list[str] = []
    for _round in range(3):
        shutil.rmtree(install_module.copies_dir(home), ignore_errors=True)
        go = time.time() + 1.0
        racers = [
            subprocess.Popen(
                [sys.executable, str(script), str(site), str(home), str(go)],
                stdout=subprocess.PIPE,
                text=True,
            )
            for _ in range(6)
        ]
        seen_whole = False
        gaps = 0
        while any(racer.poll() is None for racer in racers):
            done = [
                path
                for path in install_module.copies_dir(home).glob("run-*")
                if ".part-" not in path.name and ".old-" not in path.name
            ]
            marker = Path(install_module.PACKAGE_NAME) / "__init__.py"
            whole = any((path / marker).exists() for path in done)
            if seen_whole and not whole:
                gaps += 1
            seen_whole = seen_whole or whole
        answers.extend(racer.stdout.read().strip() for racer in racers)  # type: ignore[union-attr]
        assert gaps == 0
        # One copy, and nothing half built beside it.
        [made] = names_in(install_module.copies_dir(home))
        assert made.startswith(folder.name.rsplit("-", 1)[0])
    assert all(answer.startswith("OK") for answer in answers), answers
    assert len({answer.split(" ", 1)[1] for answer in answers}) == 1


def test_what_a_process_that_died_left_half_built_is_swept_away(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = make_site(tmp_path / "a")
    code = textwrap.dedent(
        f"""
        import os
        import shutil
        from pathlib import Path

        from nscr_houdini_mcp import install

        real = shutil.copytree

        def dying(*arguments, **named):
            real(*arguments, **named)
            os._exit(9)

        install.shutil.copytree = dying
        install.source_root = lambda: Path({str(site)!r})
        install.package_root = lambda: Path({str(site)!r}) / install.PACKAGE_NAME
        install.python_path_for_run(Path({str(home)!r}))
        """
    )
    for _ in range(2):
        subprocess.run([sys.executable, "-c", code], check=False)
    parts = [path for path in install_module.copies_dir(home).iterdir() if ".part-" in path.name]
    assert len(parts) == 2
    # One is old enough to be left behind for good; the other could still be
    # somebody's copy in the making, so it stays.
    age(parts[0], install_module.LEFTOVER_AFTER_S + 60)

    run_from(monkeypatch, site)
    folder = install_module.python_path_for_run(home)
    assert names_in(install_module.copies_dir(home)) == sorted([folder.name, parts[1].name])


def test_copies_no_session_has_asked_for_in_a_week_are_taken_away(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every uvx cache or venv a server ever ran from left a copy behind."""
    made = []
    for index in range(4):
        site = make_site(tmp_path / f"cache-{index}", f"v{index}")
        run_from(monkeypatch, site)
        made.append(install_module.python_path_for_run(home))
    for folder in made[:3]:
        age(folder / install_module.USED_FILE_NAME, EIGHT_DAYS_S)
    # Used again recently, so it stays whatever its age.
    run_from(monkeypatch, tmp_path / "cache-0")
    install_module.python_path_for_run(home)
    assert names_in(install_module.copies_dir(home)) == sorted([made[0].name, made[3].name])


def test_a_pool_worker_is_never_given_the_whole_environment(from_site: Path, home: Path) -> None:
    config = pool.PoolConfig(home=home)
    given = pool.worker_env(config, base={"PYTHONPATH": "/already/there"})
    first, rest = given["PYTHONPATH"].split(os.pathsep, 1)
    assert Path(first) == install_module.copy_for_source(
        from_site / install_module.PACKAGE_NAME, home, stamp_of(from_site)
    )
    assert rest == "/already/there"
    assert str(from_site) not in given["PYTHONPATH"].split(os.pathsep)


def test_a_launched_hython_is_never_given_the_whole_environment(
    from_site: Path, home: Path
) -> None:
    # Any file stands in for hython: only the environment it would get is read.
    stand_in = home / "hython"
    stand_in.write_text("")
    given = HythonBridge(home=home, env={}, hython=stand_in)._child_env()
    assert given[install_module.NO_AUTOSTART_ENV_VAR] == "1"
    assert given["PYTHONPATH"] == str(
        install_module.copy_for_source(
            from_site / install_module.PACKAGE_NAME, home, stamp_of(from_site)
        )
    )


def test_the_snippet_names_a_copy_for_an_installed_copy(from_site: Path, home: Path) -> None:
    printed = install_module.snippet(home=home)
    copy = install_module.copy_for_source(
        from_site / install_module.PACKAGE_NAME, home, stamp_of(from_site)
    )
    assert repr(str(copy)) in printed
    assert repr(str(from_site)) not in printed


# Section: a real Houdini with a crowded environment beside the package


PROBE = f"""
import json
import os
import sys

import numpy

import nscr_houdini_mcp

print(
    {ANSWER!r}
    + json.dumps(
        {{
            "numpy": numpy.__file__,
            "package": nscr_houdini_mcp.__file__,
            "hfs": os.environ.get("HFS", ""),
            "prefix": sys.prefix,
        }}
    ),
    flush=True,
)
sys.stdin.readline()
"""


@pytest.fixture
def crowded_hython(
    tmp_path: Path, home: Path, temp_pref_dir: Path, from_site: Path
) -> Iterator[tuple[subprocess.Popen[str], Path, install_module.InstallResult]]:
    """A hython reading a package installed from a site folder full of other libraries."""
    packages = tmp_path / "packages"
    result = install_module.install(autostart=True, packages=packages, home=home)
    script = tmp_path / "probe.py"
    script.write_text(PROBE, encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    }
    environment[install_module.PREF_DIR_ENV_VAR] = str(temp_pref_dir)
    environment[install_module.PACKAGE_DIR_ENV_VAR] = str(packages)
    environment[store_module.HOME_ENV_VAR] = str(home)
    environment["NSCR_MCP_PORT"] = str(PORT_RANGE[0])
    environment["NSCR_MCP_MAX_PORT"] = str(PORT_RANGE[1])
    log_path = tmp_path / "hython.log"
    log = log_path.open("w", encoding="utf-8")
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
        yield process, log_path, result
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
def test_houdini_loads_its_own_numpy_and_our_bridge_beside_a_crowded_environment(
    home: Path,
    crowded_hython: tuple[subprocess.Popen[str], Path, install_module.InstallResult],
) -> None:
    process, log_path, result = crowded_hython
    answer = wait_for_answer(process, log_path)
    printed = log_path.read_text(encoding="utf-8", errors="replace")

    assert SHADOWED not in printed
    assert "Traceback" not in printed
    # Houdini's own numpy lives under $HFS, or under the Python Houdini ships
    # with, which on macOS sits beside $HFS inside the same install.
    assert answer["hfs"], "hython set no HFS"
    own = [Path(answer["hfs"]).resolve(), Path(answer["prefix"]).resolve()]
    numpy_file = Path(answer["numpy"]).resolve()
    assert any(numpy_file.is_relative_to(root) for root in own), answer["numpy"]
    assert not numpy_file.is_relative_to(result.source.resolve())
    assert Path(answer["package"]).resolve().is_relative_to(result.source.resolve())

    entry = wait_for_entry(home, process)
    assert PORT_RANGE[0] <= int(entry["port"]) <= PORT_RANGE[1]
    assert "nscr bridge" in log_path.read_text(encoding="utf-8", errors="replace")


def wait_for_answer(process: subprocess.Popen[str], log_path: Path) -> dict:
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith(ANSWER):
                return json.loads(line[len(ANSWER) :])
        if process.poll() is not None:
            raise AssertionError(
                f"hython exited with {process.returncode}:\n"
                + log_path.read_text(encoding="utf-8", errors="replace")
            )
        time.sleep(0.25)
    raise AssertionError("hython printed no answer")


def wait_for_entry(home: Path, process: subprocess.Popen[str]) -> dict:
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        for entry in registry.list_entries(home):
            if entry.get("pid") == process.pid:
                return entry
        if process.poll() is not None:
            raise AssertionError(f"hython exited with {process.returncode} and no bridge")
        time.sleep(0.25)
    raise AssertionError("no bridge after waiting")
