from __future__ import annotations

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


@pytest.mark.parametrize(
    "presented",
    [None, "", "wrong", "secret ", " secret", "secret\n", "secre", "secretsecret", 7, b"secret"],
)
def test_anything_but_the_token_is_refused(presented: object) -> None:
    assert security.token_matches(presented, "secret") is False


def test_the_token_itself_is_accepted() -> None:
    token = security.mint_token()
    assert security.token_matches(token, token) is True


def test_a_token_with_characters_that_cannot_compare_is_refused() -> None:
    assert security.token_matches("secrét", "secret") is False


def test_header_names_are_lowercased_so_casing_cannot_hide_one() -> None:
    headers = security.normalise_headers({"X-Nscr-Mcp-Token": "abc", "ORIGIN": "http://x"})
    assert headers == {"x-nscr-mcp-token": "abc", "origin": "http://x"}
    assert security.normalise_headers(None) == {}


@pytest.mark.parametrize("name", ["origin", "referer"])
def test_a_browser_only_header_is_named_back(name: str) -> None:
    assert security.browser_header({name: "http://evil.example"}) == name
    assert security.browser_header({name: ""}) == name


def test_a_plain_request_carries_no_browser_header() -> None:
    assert security.browser_header({"x-nscr-mcp-token": "abc", "host": "127.0.0.1"}) is None


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
