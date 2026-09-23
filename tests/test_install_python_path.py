"""The folder Houdini's Python imports this package from, and nothing else.

An installed copy sits in a site-packages folder beside every other library of
its environment. Putting that folder in front of Houdini's own would load, say,
numpy built for another Python in place of the one Houdini ships, and break
Houdini's own tools on start. These tests build such a folder by hand and check
that Houdini is only ever given a folder holding this package alone.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from nscr_houdini_mcp import cli, pool
from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import registry
from nscr_houdini_mcp.bridge.launcher import HythonBridge, find_hython, hython_available

PREF_TEMPLATE = f"prefs{install_module.VERSION_TOKEN}"

# Its own range, away from anything a person on this machine may be running.
PORT_RANGE = (21110, 21119)

START_TIMEOUT_S = 180.0
STOP_TIMEOUT_S = 60.0

# What the libraries in a pretend environment say when they are imported, so a
# Houdini that loads one in place of its own says so out loud.
SHADOWED = "loaded from the environment instead of from Houdini"

ANSWER = "nscr-python-path-answer "

REAL_SOURCE_ROOT = install_module.source_root
REAL_PACKAGE_ROOT = install_module.package_root


@pytest.fixture(autouse=True)
def temp_pref_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    root = tmp_path / PREF_TEMPLATE
    monkeypatch.setenv(install_module.PREF_DIR_ENV_VAR, str(root))
    monkeypatch.delenv(install_module.PACKAGE_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(
        install_module, "ask_houdini", lambda version=None: (None, "no Houdini asked")
    )
    return root


@pytest.fixture
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "state"
    folder.mkdir(mode=0o700)
    return folder


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A site-packages folder: this package, and the libraries around it."""
    folder = tmp_path / "venv" / "lib" / "site-packages"
    folder.mkdir(parents=True)
    ours = folder / install_module.PACKAGE_NAME
    shutil.copytree(
        install_module.package_root(),
        ours,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
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


@pytest.fixture
def from_site(site: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run install as the copy of this package inside `site` would."""
    payload = install_module.payload_root()
    monkeypatch.setattr(install_module, "source_root", lambda: site)
    monkeypatch.setattr(install_module, "package_root", lambda: site / install_module.PACKAGE_NAME)
    monkeypatch.setattr(install_module, "payload_root", lambda: payload)
    return site


def package_file() -> Path:
    return install_module.package_path()


def written(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def env_value(document: dict, name: str) -> object:
    for item in document["env"]:
        if name in item:
            return item[name]
    raise AssertionError(f"{name} is not in the package")


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


# Section: a checkout is named as it stands


def test_a_checkout_install_names_its_source_folder_and_copies_nothing(home: Path) -> None:
    result = install_module.install(home=home)
    document = written(result.path)
    assert env_value(document, install_module.SOURCE_ENV_VAR) == str(install_module.source_root())
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
    assert source == install_module.copy_for_package_file(result.path, home)
    assert source.parent == install_module.copies_dir(home)
    assert result.source == source
    assert result.copied_from == from_site / install_module.PACKAGE_NAME
    assert document[install_module.COPY_KEY] == str(source)
    # Houdini's path still gets the folder, prepended, and the payload as it was.
    assert env_value(document, "PYTHONPATH") == {
        "value": f"${install_module.SOURCE_ENV_VAR}",
        "method": "prepend",
    }
    assert env_value(document, install_module.PAYLOAD_ENV_VAR) == str(install_module.payload_root())

    # This package and nothing else: no numpy, no PIL, no mcp, no .pth.
    assert install_module.strays(source) == []
    assert sorted(child.name for child in source.iterdir() if not child.name.startswith(".")) == [
        install_module.PACKAGE_NAME
    ]
    copied = source / install_module.PACKAGE_NAME
    assert (copied / "__init__.py").is_file()
    assert (copied / "bridge" / "app.py").is_file()
    assert not (copied / "__pycache__").exists()
    assert install_module.copy_is_current(source) is True


def test_the_install_lines_say_where_the_copy_came_from(
    from_site: Path, home: Path, capsys: pytest.CaptureFixture[str], monkeypatch
) -> None:
    monkeypatch.setenv(store_module.HOME_ENV_VAR, str(home))
    assert cli.main(["bridge", "install"]) == 0
    printed = capsys.readouterr().out
    copy = install_module.copy_for_package_file(package_file(), home)
    assert f"pythonpath     {copy}" in printed
    assert f"copied from    {from_site / install_module.PACKAGE_NAME}" in printed


def test_a_dry_run_names_the_copy_and_makes_nothing(from_site: Path, home: Path) -> None:
    result = install_module.install(home=home, dry_run=True)
    assert result.source == install_module.copy_for_package_file(result.path, home)
    assert not result.path.exists()
    assert not install_module.copies_dir(home).exists()


def test_installing_again_refreshes_the_copy(from_site: Path, home: Path) -> None:
    first = install_module.install(home=home)
    package = from_site / install_module.PACKAGE_NAME
    (package / "added_later.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "helptext.py").unlink()

    assert install_module.copy_is_current(first.source) is False
    again = install_module.install(home=home, autostart=True)

    assert again.source == first.source
    copied = again.source / install_module.PACKAGE_NAME
    assert (copied / "added_later.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not (copied / "helptext.py").exists()
    assert install_module.copy_is_current(again.source) is True
    # Nothing is left beside it from the build or the swap.
    assert sorted(child.name for child in install_module.copies_dir(home).iterdir()) == [
        again.source.name
    ]


def test_uninstall_takes_the_copy_away_with_the_file(from_site: Path, home: Path) -> None:
    result = install_module.install(home=home)
    removed = install_module.uninstall()
    assert [(item.path, item.reason) for item in removed] == [
        (result.path, "removed"),
        (result.source, "removed its copy"),
    ]
    assert not result.path.exists()
    assert not result.source.exists()
    assert not install_module.copies_dir(home).exists()
    # The state folder itself was not made by the install and stays.
    assert home.is_dir()


def test_two_package_files_have_copies_of_their_own(
    from_site: Path, home: Path, tmp_path: Path
) -> None:
    one = install_module.install("22.0", home=home)
    other = install_module.install(home=home, packages=tmp_path / "elsewhere")
    assert one.source != other.source

    install_module.uninstall("22.0")
    assert not one.source.exists()
    assert install_module.strays(other.source) == []
    assert (other.source / install_module.PACKAGE_NAME / "__init__.py").is_file()


def test_a_folder_where_the_copy_goes_that_is_not_ours_is_left_alone(
    from_site: Path, home: Path
) -> None:
    folder = install_module.copy_for_package_file(package_file(), home)
    folder.mkdir(parents=True)
    (folder / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(install_module.InstallError, match="not made by this tool"):
        install_module.install(home=home)
    assert (folder / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert not package_file().exists()


def test_uninstall_leaves_a_copy_folder_it_cannot_recognise(from_site: Path, home: Path) -> None:
    result = install_module.install(home=home)
    (result.source / install_module.STAMP_FILE_NAME).unlink()
    install_module.uninstall()
    assert not result.path.exists()
    assert result.source.is_dir()


def test_moving_to_a_checkout_takes_the_old_copy_away(
    from_site: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = install_module.install(home=home)
    monkeypatch.setattr(install_module, "source_root", REAL_SOURCE_ROOT)
    monkeypatch.setattr(install_module, "package_root", REAL_PACKAGE_ROOT)
    again = install_module.install(home=home)
    assert again.path == first.path
    assert again.source == install_module.source_root()
    assert not first.source.exists()


def test_status_says_whether_the_copy_is_current(
    from_site: Path, home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = install_module.install(home=home)
    cli.main(["bridge", "status", "--home", str(home)])
    assert f"python copy {result.source} (up to date)" in capsys.readouterr().out

    (from_site / install_module.PACKAGE_NAME / "added_later.py").write_text("", encoding="utf-8")
    cli.main(["bridge", "status", "--home", str(home)])
    assert "older than its source, run bridge install again" in capsys.readouterr().out


# Section: Windows paths


def test_a_copy_folder_name_is_short_and_safe_on_every_system(home: Path) -> None:
    windows_file = r"C:\Users\artist\Documents\houdini22.0\packages\nscr_houdini_mcp.json"
    folder = install_module.copy_for_package_file(Path(windows_file), home)
    assert folder.parent == install_module.copies_dir(home)
    name = folder.name
    assert name.startswith("package-")
    assert len(name) == len("package-") + 16
    assert all(character.isalnum() or character == "-" for character in name)
    # The same file always gets the same folder, so an install refreshes it.
    assert install_module.copy_for_package_file(Path(windows_file), home) == folder
    other = windows_file.replace("22.0", "21.5")
    assert install_module.copy_for_package_file(Path(other), home) != folder


def test_a_windows_copy_path_is_written_into_the_package_as_it_stands() -> None:
    source = Path(r"C:\Users\artist\AppData\Local\nscr-houdini-mcp\houdini-python\package-0a1b")
    payload = Path(r"C:\venv\Lib\site-packages\nscr_houdini_mcp\houdini")
    body = install_module.document(source=source, payload=payload, copy=source)
    assert env_value(body, install_module.SOURCE_ENV_VAR) == str(source)
    assert env_value(body, install_module.PAYLOAD_ENV_VAR) == str(payload)
    assert body[install_module.COPY_KEY] == str(source)
    assert install_module.copy_of(json.loads(json.dumps(body))) == source


def test_a_windows_extension_module_is_something_else_to_import(tmp_path: Path) -> None:
    (tmp_path / install_module.PACKAGE_NAME).mkdir()
    (tmp_path / "_multiarray_umath.cp313-win_amd64.pyd").write_bytes(b"")
    assert install_module.strays(tmp_path) == ["_multiarray_umath.cp313-win_amd64.pyd"]


# Section: sessions this tool starts itself


def test_a_started_session_from_a_checkout_gets_the_source_folder(home: Path) -> None:
    assert install_module.python_path_for_run(home) == install_module.source_root()
    assert not install_module.copies_dir(home).exists()


def test_a_started_session_from_an_installed_copy_gets_a_copy(from_site: Path, home: Path) -> None:
    folder = install_module.python_path_for_run(home)
    assert folder == install_module.copy_for_source(from_site / install_module.PACKAGE_NAME, home)
    assert install_module.strays(folder) == []
    stamp = folder / install_module.STAMP_FILE_NAME
    made = stamp.stat().st_mtime_ns

    # A current copy is used as it is.
    assert install_module.python_path_for_run(home) == folder
    assert stamp.stat().st_mtime_ns == made

    # A changed source is copied again.
    added = from_site / install_module.PACKAGE_NAME / "added_later.py"
    added.write_text("", encoding="utf-8")
    assert install_module.python_path_for_run(home) == folder
    assert (folder / install_module.PACKAGE_NAME / "added_later.py").is_file()


def test_a_pool_worker_is_never_given_the_whole_environment(from_site: Path, home: Path) -> None:
    config = pool.PoolConfig(home=home)
    given = pool.worker_env(config, base={"PYTHONPATH": "/already/there"})
    first, rest = given["PYTHONPATH"].split(os.pathsep, 1)
    assert Path(first) == install_module.copy_for_source(
        from_site / install_module.PACKAGE_NAME, home
    )
    assert rest == "/already/there"
    assert str(from_site) not in given["PYTHONPATH"].split(os.pathsep)


def test_a_launched_hython_is_never_given_the_whole_environment(
    from_site: Path, home: Path
) -> None:
    given = HythonBridge(home=home, env={})._child_env()
    assert given["PYTHONPATH"] == str(
        install_module.copy_for_source(from_site / install_module.PACKAGE_NAME, home)
    )


def test_the_snippet_names_a_copy_for_an_installed_copy(from_site: Path, home: Path) -> None:
    printed = install_module.snippet(home=home)
    copy = install_module.copy_for_source(from_site / install_module.PACKAGE_NAME, home)
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
