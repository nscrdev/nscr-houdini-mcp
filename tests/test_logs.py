"""The server's log: where it goes and how much of it there is."""

from __future__ import annotations

import logging
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
