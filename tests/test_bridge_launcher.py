from __future__ import annotations

from pathlib import Path

import pytest

from nscr_houdini_mcp.bridge import launcher


def test_a_configured_path_is_used_as_given(tmp_path: Path) -> None:
    named = tmp_path / "hython"
    named.write_text("", encoding="utf-8")
    assert launcher.find_hython(named) == named


def test_a_configured_path_that_is_not_there_is_an_error_not_a_search(tmp_path: Path) -> None:
    with pytest.raises(launcher.HythonNotFound):
        launcher.find_hython(tmp_path / "missing")


def test_the_environment_override_is_read_when_nothing_was_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = tmp_path / "hython"
    named.write_text("", encoding="utf-8")
    monkeypatch.setenv(launcher.HYTHON_ENV_VAR, str(named))
    assert launcher.find_hython() == named


def test_nowhere_to_look_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(launcher.HYTHON_ENV_VAR, raising=False)
    monkeypatch.delenv(launcher.HFS_ENV_VAR, raising=False)
    monkeypatch.setattr(launcher, "install_roots", lambda: [tmp_path / "nothing"])
    assert launcher.hython_available() is False
    with pytest.raises(launcher.HythonNotFound):
        launcher.find_hython()


def test_the_hfs_variable_points_at_a_binary_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(launcher.HYTHON_ENV_VAR, raising=False)
    monkeypatch.setenv(launcher.HFS_ENV_VAR, str(tmp_path))
    monkeypatch.setattr(launcher, "install_roots", list)
    binary = tmp_path / "bin" / launcher._executable("hython")
    binary.parent.mkdir(parents=True)
    binary.write_text("", encoding="utf-8")
    assert launcher.find_hython() == binary


def test_the_candidate_list_holds_no_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    found = launcher.candidates()
    assert len(found) == len(set(found))
