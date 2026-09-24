"""The plugin and marketplace manifests at the top of the repo.

They are read by a client, never by this package, so nothing else notices when
one stops parsing or points at a folder that moved.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / ".claude-plugin"  # lint-allow: client-names
PLUGIN = MANIFESTS / "plugin.json"
MARKETPLACE = MANIFESTS / "marketplace.json"

KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entry_points() -> dict[str, str]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def test_both_manifests_parse_and_are_named() -> None:
    plugin = load(PLUGIN)
    market = load(MARKETPLACE)
    assert KEBAB.match(plugin["name"])
    assert KEBAB.match(market["name"])
    assert market["owner"]["name"]


def test_the_marketplace_lists_this_plugin_at_the_repo_root() -> None:
    plugin = load(PLUGIN)
    entries = load(MARKETPLACE)["plugins"]
    assert [entry["name"] for entry in entries] == [plugin["name"]]
    source = entries[0]["source"]
    assert source.startswith("./")
    assert (ROOT / source).resolve() == ROOT


def test_every_shipped_skill_is_where_the_plugin_looks() -> None:
    skills = ROOT / "skills"
    found = sorted(path.parent.name for path in skills.glob("*/SKILL.md"))
    assert "houdini-artist" in found


def test_the_server_command_starts_the_real_entry_point() -> None:
    servers = load(PLUGIN)["mcpServers"]
    assert len(servers) == 1
    (server,) = servers.values()
    assert server["command"] == "uvx"
    args = server["args"]
    assert args[:2] == ["--from", "${CLAUDE_PLUGIN_ROOT}"]
    assert args[2] in entry_points()
    assert args[3:] == []


def test_no_version_is_pinned_so_each_commit_is_an_update() -> None:
    # A version string here would hold every install at it until it is bumped.
    assert "version" not in load(PLUGIN)
    assert all("version" not in entry for entry in load(MARKETPLACE)["plugins"])
