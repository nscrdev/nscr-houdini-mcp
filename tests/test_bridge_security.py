from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from nscr_houdini_mcp.bridge import security


def test_a_token_is_long_and_never_the_same_twice() -> None:
    first = security.mint_token()
    second = security.mint_token()
    assert first != second
    assert len(first) >= 32
    assert first.isascii()


def test_header_names_are_lowercased_so_casing_cannot_hide_one() -> None:
    headers = security.normalise_headers({"X-Nscr-Mcp-Session": "abc", "ORIGIN": "http://x"})
    assert headers == {"x-nscr-mcp-session": "abc", "origin": "http://x"}
    assert security.normalise_headers(None) == {}


@pytest.mark.parametrize("name", ["origin", "referer"])
def test_a_browser_only_header_is_named_back(name: str) -> None:
    assert security.browser_header({name: "http://evil.example"}) == name
    assert security.browser_header({name: ""}) == name


def test_a_plain_request_carries_no_browser_header() -> None:
    assert security.browser_header({"x-nscr-mcp-nonce": "abc", "host": "127.0.0.1"}) is None


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:18100", "localhost:18100", "LOCALHOST:18100", "[::1]:18100", " 127.0.0.1:18100 "],
)
def test_this_machine_on_this_port_is_the_only_host_accepted(host: str) -> None:
    assert security.host_allowed({"host": host}, 18100) is True


@pytest.mark.parametrize(
    "host",
    [
        "evil.example:18100",
        "127.0.0.1:18101",
        "127.0.0.1",
        "127.0.0.1:",
        "",
        "192.168.1.5:18100",
        "127.0.0.1.evil.example:18100",
        "[::1]",
    ],
)
def test_any_other_host_is_refused(host: str) -> None:
    assert security.host_allowed({"host": host}, 18100) is False


def test_a_request_with_no_host_at_all_is_refused() -> None:
    assert security.host_allowed({}, 18100) is False


def test_a_private_file_is_readable_by_its_owner_alone(tmp_path: Path) -> None:
    path = security.write_private(tmp_path / "sessions" / "one.json", "{}\n")
    assert path.read_text(encoding="utf-8") == "{}\n"
    assert security.is_private(path)
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == security.PRIVATE_FILE_MODE
        assert path.parent.stat().st_mode & 0o777 == security.PRIVATE_DIR_MODE


def test_writing_again_replaces_the_file_and_keeps_the_mode(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    security.write_private(path, "first")
    security.write_private(path, "second")
    assert path.read_text(encoding="utf-8") == "second"
    assert security.is_private(path)
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.skipif(sys.platform == "win32", reason="there are no mode bits here")
def test_a_folder_open_to_other_users_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "sessions"
    folder.mkdir()
    folder.chmod(0o755)
    with pytest.raises(security.InsecureLocation):
        security.check_private_dir(folder)


@pytest.mark.skipif(sys.platform == "win32", reason="there are no symbolic links to test here")
def test_a_link_in_place_of_the_folder_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    link = tmp_path / "sessions"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(security.InsecureLocation):
        security.check_private_dir(link)


@pytest.mark.skipif(sys.platform == "win32", reason="there are no mode bits here")
def test_a_folder_left_open_is_closed_again_when_it_is_made(tmp_path: Path) -> None:
    folder = tmp_path / "sessions"
    folder.mkdir(mode=0o777)
    security.private_dir(folder)
    assert folder.stat().st_mode & 0o777 == security.PRIVATE_DIR_MODE


@pytest.mark.skipif(sys.platform == "win32", reason="there are no mode bits here")
def test_a_file_left_in_place_of_the_new_one_is_replaced(tmp_path: Path) -> None:
    path = tmp_path / "one.json"
    waiting = path.with_name(path.name + ".part")
    waiting.write_text("planted", encoding="utf-8")
    os.chmod(waiting, 0o666)
    security.write_private(path, "mine")
    assert path.read_text(encoding="utf-8") == "mine"
    assert security.is_private(path)


def test_a_shared_state_folder_is_refused(tmp_path: Path) -> None:
    shared = tmp_path / "Dropbox" / "nscr-houdini-mcp"
    shared.mkdir(parents=True)
    with pytest.raises(security.InsecureLocation):
        security.check_home(shared)
    security.check_home(tmp_path)
