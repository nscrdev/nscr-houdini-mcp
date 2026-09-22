"""Command line entry point.

With no arguments this runs the MCP server on stdio, which is how a client
starts it. The `bridge` group is for a person at a terminal: it puts the
Houdini package in place, takes it away again, says what is running, and
prints the few lines that start a bridge inside a Houdini that is already
open.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from nscr_houdini_mcp import install as install_module
from nscr_houdini_mcp import store as store_module
from nscr_houdini_mcp.bridge import client, registry
from nscr_houdini_mcp.server import SERVER_NAME, package_version, run

# How long `status` waits on one session before it says it got no answer. A
# session busy with a long cook still answers health, so anything slower than
# this is a session that is not there.
HEALTH_TIMEOUT_S = 2.0


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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        run()
        return 0
    return handler(args)


# Section: the bridge commands


def _install(args: argparse.Namespace) -> int:
    try:
        result = install_module.install(
            args.houdini_version,
            autostart=args.autostart,
            dry_run=args.dry_run,
            packages=args.packages_dir,
        )
    except install_module.InstallError as error:
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
    results = install_module.uninstall(args.houdini_version, packages=args.packages_dir)
    if not results:
        print("nothing to remove")
        return 0
    kept = False
    for item in results:
        print(f"{item.reason}: {item.path}")
        kept = kept or not item.removed
    return 1 if kept else 0


def _status(args: argparse.Namespace) -> int:
    home = Path(args.home) if args.home else store_module.default_home()
    print(f"home {home}")
    _print_sessions(home)
    _print_packages(install_module.resolve(override=args.packages_dir))
    return 0


def _snippet(_args: argparse.Namespace) -> int:
    print(install_module.snippet())
    return 0


# Section: what status prints


def _print_sessions(home: Path) -> None:
    # Nothing is cleared up on the way past: status only reports.
    live = registry.live_entries(home, remove_stale=False)
    entries = {str(entry.get("session_id")): entry for entry in live}
    rows = _session_rows(home, entries)
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


def _session_rows(home: Path, entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
) -> dict[str, Any]:
    health, busy = _health(entry)
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


def _health(entry: dict[str, Any] | None) -> tuple[str, str]:
    """What the session says about itself, and whether it is working.

    The session file holds the token, so a session with no file cannot be
    asked. Nothing here waits longer than the short timeout above.
    """
    if entry is None or not entry.get("token") or not entry.get("port"):
        return "not asked", "-"
    try:
        answer = client.health(client.Session.from_entry(entry), timeout_s=HEALTH_TIMEOUT_S)
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

    installs = install_module.find_installs()
    if not installs:
        print("houdini: none found")
        return
    print("houdini:")
    for found in installs:
        print(f"  {found.version or 'unknown version'} {found.hfs}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
