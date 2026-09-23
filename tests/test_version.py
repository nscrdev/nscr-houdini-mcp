"""The version the server and the bridge report, and when they disagree."""

from __future__ import annotations

import tomllib
from pathlib import Path

from nscr_houdini_mcp import version as version_module


def test_the_version_is_the_one_the_project_ships() -> None:
    settings = tomllib.loads(
        (Path(version_module.__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert version_module.VERSION == settings["project"]["version"]


def test_the_same_version_and_protocol_is_no_warning() -> None:
    assert version_module.mismatch(version_module.VERSION, version_module.PROTOCOL) is None


def test_another_version_or_protocol_or_none_at_all_is_a_warning() -> None:
    for version, protocol in (
        ("0.0.9", version_module.PROTOCOL),
        (version_module.VERSION, version_module.PROTOCOL + 1),
        (None, None),
    ):
        warning = version_module.mismatch(version, protocol)
        assert warning is not None
        assert warning["code"] == version_module.MISMATCH_CODE
        assert warning["hint"] == version_module.MISMATCH_HINT


def test_the_version_module_imports_nothing_houdini_lacks() -> None:
    source = Path(version_module.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    assert imports == ["from __future__ import annotations", "from typing import Any"]
