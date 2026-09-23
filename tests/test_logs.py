"""The server's log: where it goes and how much of it there is."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from nscr_houdini_mcp import logs


@pytest.fixture
def package() -> Iterator[logging.Logger]:
    logger = logging.getLogger(logs.ROOT_LOGGER)
    kept = (list(logger.handlers), logger.level, logger.propagate)
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            if handler not in kept[0]:
                logger.removeHandler(handler)
                handler.close()
        logger.setLevel(kept[1])
        logger.propagate = kept[2]


def test_the_default_level_is_warning_and_lines_reach_the_file(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(logs.LEVEL_ENV_VAR, raising=False)
    logs.setup(tmp_path)
    assert package.level == logging.WARNING
    logging.getLogger("nscr_houdini_mcp.server").info("not written")
    logging.getLogger("nscr_houdini_mcp.server").warning("hou_ping refused: NO_SESSION: none")
    text = (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")
    assert "WARNING" in text and "hou_ping refused: NO_SESSION" in text
    assert "not written" not in text


@pytest.mark.parametrize("given", ["debug", "INFO", " error "])
def test_the_level_follows_the_environment(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch, given: str
) -> None:
    monkeypatch.setenv(logs.LEVEL_ENV_VAR, given)
    logs.setup(tmp_path)
    assert package.level == logs.LEVELS[given.strip().lower()]


def test_a_level_it_does_not_know_is_the_default_and_says_so(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(logs.LEVEL_ENV_VAR, "loud")
    logs.setup(tmp_path)
    assert package.level == logging.WARNING
    text = (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")
    assert "'loud' is not one of debug, info, warning, error" in text


def test_setting_up_twice_writes_each_line_once(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(logs.LEVEL_ENV_VAR, raising=False)
    logs.setup(tmp_path)
    logs.setup(tmp_path)
    package.warning("once")
    text = (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")
    assert text.count("once") == 1


def test_a_big_file_is_started_again_at_start(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logs, "ROLL_OVER_BYTES", 10)
    path = tmp_path / "logs" / "server.log"
    path.parent.mkdir()
    path.write_text("an old long log\n", encoding="utf-8")
    logs.setup(tmp_path)
    assert (tmp_path / "logs" / "server.log.1").read_text(encoding="utf-8") == "an old long log\n"


def test_a_long_level_it_does_not_know_is_quoted_short(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(logs.LEVEL_ENV_VAR, "x" * 500)
    logs.setup(tmp_path)
    text = (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")
    assert "x" * logs.QUOTED_CHARS in text
    assert "x" * (logs.QUOTED_CHARS + 1) not in text


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_the_folder_and_the_file_are_this_users_alone(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(logs.LEVEL_ENV_VAR, raising=False)
    old = os.umask(0o022)
    try:
        logs.setup(tmp_path)
    finally:
        os.umask(old)
    assert (tmp_path / "logs").stat().st_mode & 0o777 == 0o700
    assert (tmp_path / "logs" / "server.log").stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_a_file_made_open_by_an_earlier_build_is_closed_off(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "logs" / "server.log"
    path.parent.mkdir()
    path.write_text("old\n", encoding="utf-8")
    path.chmod(0o644)
    logs.setup(tmp_path)
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="no O_NOFOLLOW here")
def test_a_link_where_the_file_should_be_is_not_followed(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("", encoding="utf-8")
    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "server.log").symlink_to(elsewhere)
    logs.setup(tmp_path)
    package.warning("where does this go")
    assert elsewhere.read_text(encoding="utf-8") == ""


def test_a_folder_that_is_not_private_leaves_standard_error_alone(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(path: Path) -> Path:
        raise logs.security.InsecureLocation(f"{path} is open to other users")

    monkeypatch.setattr(logs.security, "private_dir", refuse)
    logs.setup(tmp_path)
    assert [type(handler) for handler in package.handlers] == [logging.StreamHandler]
    assert not (tmp_path / "logs" / "server.log").exists()


def test_a_roll_over_another_server_holds_the_lock_for_is_left_alone(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logs, "ROLL_OVER_BYTES", 10)
    path = tmp_path / "logs" / "server.log"
    path.parent.mkdir()
    path.write_text("an old long log\n", encoding="utf-8")
    with logs._roll_lock(path.with_name(logs.ROLL_LOCK_NAME)) as held:
        assert held is True
        logs.setup(tmp_path)
    assert not path.with_name("server.log.1").exists()
    assert path.read_text(encoding="utf-8").startswith("an old long log")


def test_two_starts_roll_over_once(
    tmp_path: Path, package: logging.Logger, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(logs, "ROLL_OVER_BYTES", 10)
    path = tmp_path / "logs" / "server.log"
    path.parent.mkdir()
    path.write_text("an old long log\n", encoding="utf-8")
    logs.setup(tmp_path)
    logs.setup(tmp_path)
    assert path.with_name("server.log.1").read_text(encoding="utf-8") == "an old long log\n"
