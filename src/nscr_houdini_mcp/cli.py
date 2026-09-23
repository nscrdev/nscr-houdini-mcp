"""Command line entry point.

With no arguments this runs the MCP server on stdio, which is how a client
starts it. The `bridge` group is for a person at a terminal: it puts the
Houdini package in place, takes it away again, says what is running, and
prints the few lines that start a bridge inside a Houdini that is already
open. The `worker` commands under it drive the pool of hython workers: start
one, stop one, list what is there, and take or hand back a warm worker. The
`config` group writes the server's config file and shows what it resolves to.
The `skills` group says where the shipped agent skills are and copies them into
a folder the person names.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import agent_skills as skills_module
from nscr_houdini_mcp import config as config_module
from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import pool
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.server import SERVER_NAME, package_version, run

# How long `status` waits on one session before it says it got no answer. A
# session busy with a long cook still answers health, so anything slower than
# this is a session that is not there.
HEALTH_TIMEOUT_S = 2.0

# How long the whole sweep of sessions may take. Twenty dead sessions must not
# turn a status into a minute of waiting, so the budget is shared: once it is
# spent, the sessions left are reported as not asked.
HEALTH_BUDGET_S = 5.0

# What `worker start` exits with when the pool has no room. It is its own
# code, so a script can tell it from a start that went wrong.
POOL_FULL_EXIT = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=SERVER_NAME,
        description="Run the Houdini MCP server on stdio, or manage the Houdini side.",
    )
    parser.add_argument("--version", action="version", version=package_version())
    commands = parser.add_subparsers(dest="command")

    bridge = commands.add_parser("bridge", help="the Houdini side of this tool")
    actions = bridge.add_subparsers(dest="action", required=True)

    add = actions.add_parser("install", help="write the Houdini package file")
    add.add_argument(
        "--houdini-version",
        default=install_module.DEFAULT_HOUDINI_VERSION,
        help="which Houdini preference folder to write into",
    )
    add.add_argument(
        "--autostart",
        action="store_true",
        help="start a bridge with every Houdini that reads this package",
    )
    add.add_argument("--dry-run", action="store_true", help="print the file, write nothing")
    add.add_argument(
        "--packages-dir",
        type=Path,
        default=None,
        help="the packages folder to write into, ahead of everything else",
    )
    add.set_defaults(handler=_install)

    remove = actions.add_parser("uninstall", help="take away package files this tool wrote")
    remove.add_argument(
        "--houdini-version",
        default=None,
        help="one version, or every preference folder found when left out",
    )
    remove.add_argument(
        "--packages-dir",
        type=Path,
        default=None,
        help="the packages folder to look in, ahead of everything else",
    )
    remove.set_defaults(handler=_uninstall)

    status = actions.add_parser("status", help="sessions, package state and Houdini installs")
    status.add_argument("--home", type=Path, default=None, help="state folder to read")
    status.add_argument(
        "--packages-dir",
        type=Path,
        default=None,
        help="the packages folder to report on, ahead of everything else",
    )
    status.set_defaults(handler=_status)

    snippet = actions.add_parser("snippet", help="Python that starts a bridge in an open Houdini")
    snippet.set_defaults(handler=_snippet)

    _add_worker_commands(actions)
    _add_config_commands(commands)
    _add_skills_commands(commands)
    return parser


def _add_skills_commands(commands: Any) -> None:
    """The commands for the agent skills that ship with this package."""
    skills = commands.add_parser("skills", help="the agent skills that ship with this package")
    actions = skills.add_subparsers(dest="skills_action", required=True)

    where = actions.add_parser("path", help="print the folder the shipped skills are in")
    where.set_defaults(handler=_skills_path)

    copy = actions.add_parser(
        "install",
        help="copy the skills into a folder you name",
        description=(
            "Copy each shipped skill into <dest>/<skill name>. A skill you have edited "
            "is kept as it is unless --force, and a link in its place is never written "
            "through. Exit status: 0 when every skill was installed, left unchanged, "
            "replaced or kept; 1 when the skills could not be found or copied."
        ),
    )
    copy.add_argument("dest", type=Path, help="the folder your client reads skills from")
    copy.add_argument(
        "--force",
        action="store_true",
        help="write the shipped files over a copy there that differs",
    )
    copy.set_defaults(handler=_skills_install)


def _add_config_commands(commands: Any) -> None:
    """The commands for the server's own config file."""
    config = commands.add_parser("config", help="the server's config file")
    actions = config.add_subparsers(dest="config_action", required=True)

    show = actions.add_parser("show", help="every setting, where it came from, and which hython")
    show.add_argument("--path", type=Path, default=None, help="the config file to read")
    show.set_defaults(handler=_config_show)

    init = actions.add_parser("init", help="write a commented config file with the defaults")
    init.add_argument("--path", type=Path, default=None, help="where to write it")
    init.add_argument("--force", action="store_true", help="replace a file that is there")
    init.set_defaults(handler=_config_init)


def _add_worker_commands(actions: Any) -> None:
    """The commands for the hython workers this machine may run."""
    worker = actions.add_parser("worker", help="the pool of hython workers")
    jobs = worker.add_subparsers(dest="worker_action", required=True)

    start = jobs.add_parser("start", help="start one worker, if the pool has room")
    _add_home(start)
    start.add_argument(
        "--cap", type=int, default=None, help="how many workers may run, pool_cap from config"
    )
    start.add_argument(
        "--weight",
        default=pool.DEFAULT_WEIGHT,
        choices=sorted(pool.WEIGHTS),
        help="how much of the machine this worker is for",
    )
    start.add_argument(
        "--weight-budget",
        type=float,
        default=None,
        help="how much weight the whole pool may hold, the cap when left out",
    )
    start.add_argument(
        "--max-idle-s",
        type=float,
        default=pool.DEFAULT_MAX_IDLE_S,
        help="how long the worker stays warm with no job before it ends itself",
    )
    start.add_argument("--max-threads", type=int, default=None, help="thread cap for this worker")
    start.add_argument(
        "--hython", type=Path, default=None, help="the hython to start, from config when left out"
    )
    start.add_argument(
        "--port", type=int, default=None, help="the first port to try, from config when left out"
    )
    start.add_argument(
        "--max-port", type=int, default=None, help="the last port to try, from config when left out"
    )
    start.add_argument("--job", default=None, help="take the worker for this job at once")
    start.add_argument(
        "--timeout-s",
        type=float,
        default=pool.DEFAULT_START_TIMEOUT_S,
        help="how long to wait for the worker's bridge",
    )
    start.set_defaults(handler=_worker_start)

    stop = jobs.add_parser("stop", help="ask one worker to stop, then make sure it has")
    _add_home(stop)
    stop.add_argument("alias", help="the worker's name, token or session id")
    stop.add_argument(
        "--grace-s",
        type=float,
        default=pool.DEFAULT_STOP_GRACE_S,
        help="how long it may take to go before it is ended here",
    )
    stop.set_defaults(handler=_worker_stop)

    listing = jobs.add_parser("list", help="what the pool is holding")
    _add_home(listing)
    listing.set_defaults(handler=_worker_list)

    reserve = jobs.add_parser("reserve", help="take a warm worker for one job")
    _add_home(reserve)
    reserve.add_argument("alias", nargs="?", default=None, help="a worker, or the first free one")
    reserve.add_argument("--job", required=True, help="the job the worker is taken for")
    reserve.set_defaults(handler=_worker_reserve)

    give_back = jobs.add_parser("release", help="hand a worker back to the pool, warm")
    _add_home(give_back)
    give_back.add_argument("alias", help="the worker's name, token or session id")
    give_back.set_defaults(handler=_worker_release)


def _add_home(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--home", type=Path, default=None, help="state folder to use, state_home from config"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        run()
        return 0
    try:
        return handler(args)
    except config_module.ConfigError as error:
        return _config_failed(error)


# Section: the bridge commands


def _install(args: argparse.Namespace) -> int:
    home, problem = _state_home()
    if problem:
        print(problem)
    try:
        result = install_module.install(
            args.houdini_version,
            autostart=args.autostart,
            dry_run=args.dry_run,
            packages=args.packages_dir,
            home=home,
        )
    except (install_module.InstallError, OSError) as error:
        print(str(error))
        return 1
    print("would write" if result.dry_run else ("replaced" if result.replaced else "wrote"))
    for line in result.lines:
        print(f"  {line}")
    for note in result.lookup.notes if result.lookup else []:
        print(f"  note: {note}")
    if not result.autostart:
        print("  a Houdini reading this opens no port until the bridge is started by hand")
    return 0


def _uninstall(args: argparse.Namespace) -> int:
    home, problem = _state_home()
    if problem:
        print(problem)
    results = install_module.uninstall(args.houdini_version, packages=args.packages_dir, home=home)
    if not results:
        print("nothing to remove")
        return 0
    kept = False
    for item in results:
        print(f"{item.reason}: {item.path}")
        kept = kept or not item.removed
    return 1 if kept else 0


def _status(args: argparse.Namespace) -> int:
    if args.home:
        home = Path(args.home)
    else:
        home, problem = _state_home()
        if problem:
            print(problem)
    print(f"home {home}")
    _print_sessions(home)
    _print_packages(install_module.resolve(override=args.packages_dir))
    return 0


def _snippet(_args: argparse.Namespace) -> int:
    home, problem = _state_home()
    if problem:
        print(f"# {problem}")
    print(install_module.snippet(home=home))
    return 0


def _state_home() -> tuple[Path, str]:
    """The state folder from config, or the usual one and why, when it is broken.

    These commands still do their work from the usual folder rather than
    refusing, and say so.
    """
    try:
        return config_module.load_config().state_home, ""
    except config_module.ConfigError as error:
        return store_module.default_home(), f"config invalid: {error.message}"


# Section: the worker commands


def _settings() -> config_module.Config:
    """The server's config, so these commands and the server agree on the
    state folder, the cap and the hython. Raises `ConfigError`."""
    return config_module.load_config()


def _home_of(args: argparse.Namespace) -> Path:
    return Path(args.home) if args.home else _settings().state_home


def _config_failed(error: config_module.ConfigError) -> int:
    where = f" ({error.key})" if error.key else ""
    print(f"config invalid{where}: {error.message}")
    print(f"  {error.path}")
    return 1


def _worker_start(args: argparse.Namespace) -> int:
    try:
        settings = _settings()
        hython = args.hython or config_module.resolve_hython(settings)
    except config_module.ConfigError as error:
        return _config_failed(error)
    config = pool.PoolConfig(
        home=Path(args.home) if args.home else settings.state_home,
        cap=args.cap if args.cap is not None else settings.pool_cap,
        max_idle_s=args.max_idle_s,
        weight_budget=args.weight_budget,
        max_threads=args.max_threads,
        hython=hython,
        port_range=(
            args.port if args.port is not None else settings.worker_ports[0],
            args.max_port if args.max_port is not None else settings.worker_ports[1],
        ),
        start_timeout_s=args.timeout_s,
    )
    try:
        with pool.open_store(config.home) as store:
            record = pool.start_worker(config, store, weight=args.weight, job_id=args.job)
    except store_module.PoolFull as error:
        # One word first, so a person and an agent read the same thing, and an
        # exit code of its own, so a script can tell a full pool from a
        # failure without reading the line.
        print(f"{pool.POOL_FULL}: {error}")
        return POOL_FULL_EXIT
    except (pool.PoolError, store_module.StoreError) as error:
        print(str(error))
        return 1
    print(f"started {record.alias} {record.session_id} pid {record.pid}")
    print(f"  log {pool.log_path(config.home, record.alias)}")
    print(f"  state {record.state}  weight {record.weight:g}  job {record.job_id or '-'}")
    print(f"  {pool.capability_summary(record.capabilities)}")
    return 0


def _worker_stop(args: argparse.Namespace) -> int:
    config = pool.PoolConfig(home=_home_of(args))
    try:
        with pool.open_store(config.home) as store:
            stopped = pool.stop_worker(config, store, args.alias, grace_s=args.grace_s)
    except (pool.PoolError, store_module.StoreError) as error:
        print(str(error))
        return 1
    how = "ended here" if stopped.killed else "stopped on request"
    print(f"{stopped.record.alias} {how}")
    if stopped.note:
        print(f"  {stopped.note}")
    if not stopped.ended:
        print("  the process is still there")
        return 1
    return 0


def _worker_list(args: argparse.Namespace) -> int:
    home = _home_of(args)
    path = pool.store_path(home)
    if not path.exists():
        print("workers: none")
        return 0
    with pool.open_store(home) as store:
        rows = pool.list_workers(store)
    if not rows:
        print("workers: none")
        return 0
    print("workers:")
    for row in rows:
        print(f"  {row['alias']} {row['session_id']}")
        print(
            f"    pid {row['pid']}  state {row['state']}  job {row['job']}"
            f"  weight {row['weight']:g}  lease age {row['lease_age_s']:.0f}s"
        )
        print(f"    {row['capabilities']}")
    return 0


def _worker_reserve(args: argparse.Namespace) -> int:
    home = _home_of(args)
    try:
        with pool.open_store(home) as store:
            handle = args.alias or _first_free(store)
            record = pool.reserve(store, handle, job_id=args.job)
    except (pool.PoolError, store_module.StoreError) as error:
        print(str(error))
        return 1
    print(f"{record.alias} taken for job {record.job_id}")
    return 0


def _first_free(store: store_module.Store) -> str:
    """The warm worker that has been waiting longest, or nothing doing."""
    for record in store.list_workers():
        if record.state == "running" and record.job_id is None:
            return record.alias
    raise pool.UnknownWorker("no warm worker is free")


def _worker_release(args: argparse.Namespace) -> int:
    home = _home_of(args)
    try:
        with pool.open_store(home) as store:
            record = pool.release(store, args.alias)
    except (pool.PoolError, store_module.StoreError) as error:
        print(str(error))
        return 1
    print(f"{record.alias} is back in the pool")
    return 0


# Section: the config commands


def _config_show(args: argparse.Namespace) -> int:
    path = args.path or config_module.config_path()
    try:
        config = config_module.load_config(path)
    except config_module.ConfigError as error:
        print(f"config {path}")
        where = f" ({error.key})" if error.key else ""
        print(f"  invalid{where}: {error.message}")
        return 1
    print(f"config {config.path}{'' if config.exists else '  (not there, defaults in use)'}")
    for key, value in config.shown().items():
        source = "file" if key in config.from_file else "default"
        shown = "-" if value is None else value
        print(f"  {key} = {shown}  ({source})")
    print(f"  hython in use: {config_module.describe_hython(config)}")
    return 0


def _config_init(args: argparse.Namespace) -> int:
    path = args.path or config_module.config_path()
    try:
        written = config_module.write_template(path, force=args.force)
    except FileExistsError as error:
        print(str(error))
        return 1
    print(f"wrote {written}")
    return 0


# Section: the skills commands


def _skills_path(_args: argparse.Namespace) -> int:
    try:
        print(skills_module.skills_root())
    except skills_module.SkillsError as error:
        print(str(error))
        return 1
    return 0


def _skills_install(args: argparse.Namespace) -> int:
    try:
        results = skills_module.install(args.dest, force=args.force)
    except (skills_module.SkillsError, OSError) as error:
        print(str(error))
        return 1
    # A kept skill is the person's own edit, or a link they placed, so it is
    # reported and is not a failure.
    for item in results:
        print(f"{item.outcome} {item.name}: {item.path}")
        if item.note:
            print(f"  {item.note}")
    return 0


# Section: what status prints


def _print_sessions(home: Path) -> None:
    # Nothing is cleared up on the way past: status only reports.
    live = registry.live_entries(home, remove_stale=False)
    entries = {str(entry.get("session_id")): entry for entry in live}
    rows = _session_rows(home, entries, time.monotonic() + HEALTH_BUDGET_S)
    if not rows:
        print("sessions: none")
        return
    print("sessions:")
    for row in rows:
        print(f"  {row['alias']} {row['session_id']}")
        print(
            f"    kind {row['kind']}  pid {row['pid']}  port {row['port']}"
            f"  epoch {row['scene_epoch']}  busy {row['busy']}"
        )
        print(f"    hip {row['hip_path']}")
        print(f"    health {row['health']}")


def _session_rows(
    home: Path, entries: dict[str, dict[str, Any]], deadline: float
) -> list[dict[str, Any]]:
    """One row per session, from the store where it can be read, else the files."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in _store_sessions(home):
        seen.add(record.session_id)
        rows.append(
            _row(
                session_id=record.session_id,
                alias=record.alias,
                kind=record.kind,
                pid=record.pid,
                port=record.port,
                hip_path=record.hip_path,
                scene_epoch=record.scene_epoch,
                entry=entries.get(record.session_id),
                deadline=deadline,
            )
        )
    for session_id, entry in entries.items():
        if session_id in seen:
            continue
        rows.append(
            _row(
                session_id=session_id,
                alias=entry.get("alias"),
                kind=entry.get("kind"),
                pid=entry.get("pid"),
                port=entry.get("port"),
                hip_path=entry.get("hip_path"),
                scene_epoch=entry.get("scene_epoch"),
                entry=entry,
                deadline=deadline,
            )
        )
    return rows


def _store_sessions(home: Path) -> list[Any]:
    """Live sessions the coordination store knows, or none when it has no file."""
    path = home / store_module.STORE_FILE_NAME
    if not path.exists():
        return []
    try:
        with store_module.Store(path) as store:
            store.reclaim_sessions()
            return store.list_sessions()
    except store_module.StoreError:
        return []


def _row(
    *,
    session_id: str,
    alias: Any,
    kind: Any,
    pid: Any,
    port: Any,
    hip_path: Any,
    scene_epoch: Any,
    entry: dict[str, Any] | None,
    deadline: float,
) -> dict[str, Any]:
    health, busy = _health(entry, deadline)
    return {
        "session_id": session_id,
        "alias": alias or "-",
        "kind": kind or "-",
        "pid": pid if pid is not None else "-",
        "port": port if port is not None else "-",
        "hip_path": hip_path or "-",
        "scene_epoch": scene_epoch if scene_epoch is not None else "-",
        "busy": busy,
        "health": health,
    }


def _health(entry: dict[str, Any] | None, deadline: float) -> tuple[str, str]:
    """What the session says about itself, and whether it is working.

    The session file holds the token, so a session with no file cannot be
    asked. No session is waited on longer than the short timeout, and no sweep
    longer than what is left of the budget the caller passed.
    """
    if entry is None or not entry.get("token") or not entry.get("port"):
        return "not asked", "-"
    left = deadline - time.monotonic()
    if left <= 0:
        return "not asked, the time for asking was spent on the sessions before it", "-"
    try:
        answer = client.health(
            client.Session.from_entry(entry), timeout_s=min(HEALTH_TIMEOUT_S, left)
        )
    except client.BridgeNotAuthentic:
        return "answered by something else", "-"
    except client.BridgeUnreachable:
        return "no answer", "-"
    data = answer.payload.get("data") if isinstance(answer.payload, dict) else None
    if not isinstance(data, dict):
        return f"http {answer.status}", "-"
    busy = data.get("busy")
    return "ok", ("-" if busy is None else ("yes" if busy else "no"))


def _print_packages(lookup: install_module.Lookup) -> None:
    """Which folder was chosen, why, and what is in the ones considered."""
    print(f"packages folder {lookup.path}")
    print(f"  decided by {lookup.source}")
    for note in lookup.notes:
        print(f"  note: {note}")
    print("  considered:")
    for candidate in lookup.candidates:
        mark = "->" if candidate.used else "  "
        tail = f"  ({candidate.note})" if candidate.note else ""
        print(f"    {mark} {candidate.source}: {candidate.path}{tail}")

    print("packages:")
    for state in install_module.installed(lookup=lookup):
        if not state.present:
            word = "not installed"
        elif not state.ours:
            word = "a package of that name is there, written by something else"
        else:
            word = f"installed, autostart {'on' if state.autostart else 'off'}"
        print(f"  {word}")
        print(f"    {state.path}")
        if state.copy is not None:
            if not install_module.is_our_copy(state.copy):
                how = "missing, run bridge install again"
            elif state.copy_current:
                how = "the same version as this server"
            else:
                how = "another version than this server, run bridge install again"
            print(f"    python copy {state.copy} ({how})")
        if state.source_strays:
            shown = ", ".join(state.source_strays[:5])
            more = ", ..." if len(state.source_strays) > 5 else ""
            print(
                f"    {install_module.SOURCE_ENV_VAR} {state.source} holds other libraries"
                f" too ({shown}{more}), which load in place of Houdini's own:"
                " run bridge install again"
            )

    installs = install_module.find_installs()
    if not installs:
        print("houdini: none found")
        return
    print("houdini:")
    for found in installs:
        print(f"  {found.version or 'unknown version'} {found.hfs}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
