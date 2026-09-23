"""The server config file: reading, checking, the defaults and the commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from nscr_houdini_mcp import cli, install, outputs, pool
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
    assert config.worker_ports == pool.DEFAULT_PORT_RANGE
    assert config.state_home == home
    assert config.spill_folder == home / "spill"
    assert config.spill_over_bytes == DEFAULT_SPILL_OVER_BYTES == 64 * 1024
    assert config.spill_keep_days == 7
    assert config.python_timeout_cap_s == 3600
    assert config.inline_wait_s == 10
    assert config.transport == "stdio"


def test_the_template_reads_back_as_the_defaults(home: Path) -> None:
    config_module.write_template(home / "config.toml")
    config = load_config()
    assert config.exists is True
    assert config.pool_cap == pool.DEFAULT_CAP
    assert config.worker_ports == pool.DEFAULT_PORT_RANGE
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
worker_ports = [18830, 18839]
state_home = "{state.as_posix()}"
spill_dir = "{spill.as_posix()}"
spill_over_bytes = 4096
python_timeout_cap_s = 120
inline_wait_s = 30
transport = "stdio"
""",
    )
    config = load_config()
    assert config.houdini_build == "22.0.368"
    assert config.default_session == "w1"
    assert config.pool_cap == 5
    assert config.worker_ports == (18830, 18839)
    assert config.state_home == state
    assert config.spill_folder == spill
    assert config.spill_over_bytes == 4096
    assert config.python_timeout_cap_s == 120
    assert config.inline_wait_s == 30
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
        ("pool_cap = 2.5", "pool_cap", "got a decimal"),
        ("pool_cap = 3.0", "pool_cap", "got a decimal"),
        ("pool_cap = true", "pool_cap", "whole number"),
        ("spill_over_bytes = 10", "spill_over_bytes", "from 1024"),
        ("spill_keep_days = 0", "spill_keep_days", "from 1 to 365"),
        ("python_timeout_cap_s = 0", "python_timeout_cap_s", "from 1 to 3600"),
        ("inline_wait_s = 0", "inline_wait_s", "from 1 to 50"),
        ("inline_wait_s = 51", "inline_wait_s", "from 1 to 50"),
        ("spill_dir = '//server/share/spill'", "spill_dir", "cannot hold private results"),
        ('transport = "http"', "transport", "one of stdio"),
        ('houdini_build = "latest"', "houdini_build", "22.0.368"),
        ('state_home = "relative/place"', "state_home", "absolute path"),
        ("default_session = 3", "default_session", "must be a string"),
        ("worker_ports = 18100", "worker_ports", "two port numbers"),
        ("worker_ports = [18100]", "worker_ports", "two port numbers"),
        ("worker_ports = [80, 90]", "worker_ports", "from 1024 to 65535"),
        ("worker_ports = [18199, 18100]", "worker_ports", "lower port first"),
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


def test_the_output_tables_in_the_same_file_are_left_to_the_output_paths(home: Path) -> None:
    path = write(
        home,
        """
pool_cap = 2

[outputs]
version_width = 4

[conventions]
output_marker_prefix = "OUTPUT_"
""",
    )
    config = load_config()
    assert config.pool_cap == 2
    assert config.path == path
    # The output paths read the very same file, tables and all.
    conventions = outputs.load_conventions(home=home)
    assert conventions.output_marker_prefix == "OUTPUT_"
    assert conventions.version_width == 4
    assert str(path) in conventions.sources


def test_a_config_named_by_the_environment_must_be_there(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(config_module.CONFIG_ENV_VAR, str(tmp_path / "missing.toml"))
    with pytest.raises(ConfigError) as caught:
        load_config()
    assert "not there" in caught.value.message


def test_a_config_named_on_the_command_line_must_be_there(home: Path, tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml")


def test_a_windows_path_in_double_quotes_gets_a_hint(home: Path) -> None:
    write(home, 'hython = "C:\\Users\\me\\hython.exe"\n')
    with pytest.raises(ConfigError) as caught:
        load_config()
    assert "single quotes" in caught.value.message


def test_the_error_names_the_file_relative_to_the_state_folder(home: Path) -> None:
    write(home, "poolcap = 1\n")
    with pytest.raises(ConfigError) as caught:
        load_config()
    assert caught.value.details() == {"path": "config.toml", "key": "poolcap"}


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


def test_worker_start_takes_the_state_folder_cap_and_hython_from_config(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    named = tmp_path / "bin" / "hython"
    write(
        home,
        f'state_home = "{state.as_posix()}"\npool_cap = 2\nhython = "{named.as_posix()}"\n'
        "worker_ports = [18830, 18839]\n",
    )
    seen: list[pool.PoolConfig] = []

    def start_worker(config: pool.PoolConfig, store: object, **rest: object) -> object:
        seen.append(config)
        raise pool.WorkerStartFailed("stopped here")

    monkeypatch.setattr(pool, "start_worker", start_worker)
    assert cli.main(["bridge", "worker", "start"]) == 1
    [config] = seen
    assert config.home == state
    assert config.cap == 2
    assert config.hython == named
    assert config.port_range == (18830, 18839)
    # A flag given on the command line still wins.
    assert cli.main(["bridge", "worker", "start", "--cap", "1", "--port", "18831"]) == 1
    assert seen[1].cap == 1
    assert seen[1].port_range == (18831, 18839)


def test_bridge_status_reads_the_state_folder_from_config(
    home: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    write(home, f'state_home = "{state.as_posix()}"\n')
    monkeypatch.setattr(install, "resolve", _no_packages)
    monkeypatch.setattr(install, "installed", lambda lookup=None: [])
    monkeypatch.setattr(install, "find_installs", lambda configured=None: [])
    assert cli.main(["bridge", "status"]) == 0
    assert f"home {state}" in capsys.readouterr().out


def _no_packages(*args: object, **rest: object) -> install.Lookup:
    return install.Lookup(path=Path("packages"), source="test", candidates=[], notes=[])


def test_a_worker_command_with_a_bad_config_says_so(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(home, "poolcap = 2\n")
    assert cli.main(["bridge", "worker", "list"]) == 1
    assert "did you mean pool_cap" in capsys.readouterr().out


def test_the_template_is_the_one_config_init_writes(home: Path) -> None:
    assert cli.main(["config", "init"]) == 0
    text = (home / "config.toml").read_text(encoding="utf-8")
    assert "state_home = ''" in text
    assert "spill_keep_days = 7" in text
