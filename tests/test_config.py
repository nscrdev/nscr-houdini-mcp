"""The server config file: reading, checking, the defaults and the commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from nscr_houdini_mcp import cli, install, pool
from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp.config import (
    DEFAULT_SPILL_OVER_BYTES,
    ConfigError,
    load_config,
    resolve_hython,
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "home"
    folder.mkdir()
    monkeypatch.setenv("NSCR_MCP_HOME", str(folder))
    monkeypatch.delenv(config_module.CONFIG_ENV_VAR, raising=False)
    return folder


def write(home: Path, text: str) -> Path:
    path = home / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_no_file_means_every_default(home: Path) -> None:
    config = load_config()
    assert config.path == home / "config.toml"
    assert config.exists is False
    assert config.hython is None
    assert config.houdini_build is None
    assert config.default_session is None
    assert config.pool_cap == pool.DEFAULT_CAP
    assert config.state_home == home
    assert config.spill_folder == home / "spill"
    assert config.spill_over_bytes == DEFAULT_SPILL_OVER_BYTES == 64 * 1024
    assert config.transport == "stdio"


def test_the_template_reads_back_as_the_defaults(home: Path) -> None:
    config_module.write_template(home / "config.toml")
    config = load_config()
    assert config.exists is True
    assert config.pool_cap == pool.DEFAULT_CAP
    assert config.spill_over_bytes == DEFAULT_SPILL_OVER_BYTES
    assert config.hython is None
    assert config.state_home == home


def test_every_key_is_read(home: Path, tmp_path: Path) -> None:
    state = tmp_path / "state"
    spill = tmp_path / "spilled"
    write(
        home,
        f"""
houdini_build = "22.0.368"
default_session = "w1"
pool_cap = 5
state_home = "{state.as_posix()}"
spill_dir = "{spill.as_posix()}"
spill_over_bytes = 4096
transport = "stdio"
""",
    )
    config = load_config()
    assert config.houdini_build == "22.0.368"
    assert config.default_session == "w1"
    assert config.pool_cap == 5
    assert config.state_home == state
    assert config.spill_folder == spill
    assert config.spill_over_bytes == 4096
    assert config.store_path == state / "coord.sqlite"
    assert "pool_cap" in config.from_file
    assert "hython" not in config.from_file


def test_the_config_env_var_names_another_file(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "elsewhere.toml"
    other.write_text("pool_cap = 2\n", encoding="utf-8")
    monkeypatch.setenv(config_module.CONFIG_ENV_VAR, str(other))
    assert load_config().pool_cap == 2


@pytest.mark.parametrize(
    ("text", "key", "words"),
    [
        ("poolcap = 2", "poolcap", "did you mean pool_cap"),
        ("pool_cap = 0", "pool_cap", "from 1 to 16"),
        ("pool_cap = 2.5", "pool_cap", "whole number"),
        ("pool_cap = true", "pool_cap", "whole number"),
        ("spill_over_bytes = 10", "spill_over_bytes", "from 1024"),
        ('transport = "http"', "transport", "one of stdio"),
        ('houdini_build = "latest"', "houdini_build", "22.0.368"),
        ('state_home = "relative/place"', "state_home", "absolute path"),
        ("default_session = 3", "default_session", "must be a string"),
    ],
)
def test_a_wrong_value_is_refused_with_the_key(home: Path, text: str, key: str, words: str) -> None:
    write(home, text + "\n")
    with pytest.raises(ConfigError) as caught:
        load_config()
    assert caught.value.key == key
    assert words in caught.value.message
    assert caught.value.path == home / "config.toml"


def test_hython_and_a_build_together_are_refused(home: Path, tmp_path: Path) -> None:
    named = (tmp_path / "hython").as_posix()
    write(home, f'hython = "{named}"\nhoudini_build = "22.0"\n')
    with pytest.raises(ConfigError) as caught:
        load_config()
    assert caught.value.key == "houdini_build"
    assert "not both" in caught.value.message


def test_a_file_that_is_not_toml_is_refused(home: Path) -> None:
    write(home, "pool_cap = = 3\n")
    with pytest.raises(ConfigError) as caught:
        load_config()
    assert "not valid TOML" in caught.value.message


def test_an_empty_string_means_the_default(home: Path) -> None:
    write(home, 'hython = ""\ndefault_session = "  "\n')
    config = load_config()
    assert config.hython is None
    assert config.default_session is None


# Section: which hython


def fake_installs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, versions: list[str]) -> None:
    found = [
        install.HoudiniInstall(version, tmp_path / version, tmp_path / version / "hfs")
        for version in versions
    ]
    monkeypatch.setattr(install, "find_installs", lambda configured=None: found)


def test_a_build_picks_the_matching_install(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_installs(monkeypatch, tmp_path, ["22.0.429", "22.0.368"])
    write(home, 'houdini_build = "22.0.368"\n')
    chosen = resolve_hython(load_config())
    assert chosen is not None
    assert chosen.parent == tmp_path / "22.0.368" / "hfs" / "bin"
    assert chosen.stem == "hython"


def test_a_short_build_picks_the_newest_of_that_line(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_installs(monkeypatch, tmp_path, ["22.0.429", "22.0.368"])
    write(home, 'houdini_build = "22.0"\n')
    chosen = resolve_hython(load_config())
    assert chosen is not None and chosen.parent.parent.parent.name == "22.0.429"


def test_a_build_that_is_not_installed_names_the_ones_that_are(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_installs(monkeypatch, tmp_path, ["22.0.429"])
    write(home, 'houdini_build = "21.5"\n')
    with pytest.raises(ConfigError) as caught:
        resolve_hython(load_config())
    assert "22.0.429" in caught.value.message
    assert caught.value.key == "houdini_build"


def test_a_named_hython_is_taken_as_given(home: Path, tmp_path: Path) -> None:
    named = tmp_path / "bin" / "hython"
    write(home, f'hython = "{named.as_posix()}"\n')
    config = load_config()
    assert resolve_hython(config) == named
    assert config.pool_config().hython == named
    assert config.pool_config().cap == pool.DEFAULT_CAP


def test_nothing_named_leaves_the_search_to_the_pool(home: Path) -> None:
    assert resolve_hython(load_config()) is None


# Section: the commands


def test_config_init_writes_once_then_needs_force(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["config", "init"]) == 0
    assert (home / "config.toml").is_file()
    assert cli.main(["config", "init"]) == 1
    assert "--force" in capsys.readouterr().out
    assert cli.main(["config", "init", "--force"]) == 0


def test_config_show_prints_every_key_and_its_source(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(home, "pool_cap = 2\n")
    assert cli.main(["config", "show"]) == 0
    out = capsys.readouterr().out
    for key in config_module.KEYS:
        assert f"  {key} = " in out
    assert "pool_cap = 2  (file)" in out
    assert "transport = stdio  (default)" in out
    assert "hython in use:" in out


def test_config_show_reports_a_bad_file_and_fails(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(home, "poolcap = 2\n")
    assert cli.main(["config", "show"]) == 1
    assert "did you mean pool_cap" in capsys.readouterr().out
