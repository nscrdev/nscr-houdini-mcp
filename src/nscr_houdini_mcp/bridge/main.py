"""Start a bridge in this Houdini process and hold the process open.

Run with hython:

    hython -m nscr_houdini_mcp.bridge.main --home <folder>

The process stays alive until its input closes or the word `stop` arrives on
it. Tying the lifetime to the input means a launcher that dies takes its
worker with it, instead of leaving a Houdini running with nobody to talk to.

A pool worker is the other case. It is started detached, with no input at all,
because it has to outlive the server that asked for it. Such a worker is given
`--worker-token`, and its lifetime is its lease in the coordination store: it
watches its own row from a thread of its own and ends itself when a server
asks it to or when nobody has wanted it for `--max-idle-s`.

The input is watched on a thread of its own, because the thread that owns the
process has work to do: it runs the scene edits. Headless, an undo group only
records on the main thread, so a bridge whose main thread sat in a read would
give the artist no single step to undo and no way to roll a failed call back.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

from nscr_houdini_mcp import pool
from nscr_houdini_mcp.bridge.app import (
    DEFAULT_DROP_REPLY_S,
    DEFAULT_HEARTBEAT_S,
    Bridge,
    BridgeConfig,
)
from nscr_houdini_mcp.bridge.net import DEFAULT_PORT_RANGE

STOP_WORD = "stop"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nscr-houdini-mcp bridge",
        description="Run the Houdini side bridge until told to stop.",
    )
    parser.add_argument("--home", type=Path, help="state folder, for the store and session files")
    parser.add_argument("--alias", help="fixed name for this session")
    parser.add_argument("--alias-template", help="name pattern containing {n}")
    parser.add_argument("--kind", choices=("gui", "hython"), help="override the session kind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT_RANGE[0])
    parser.add_argument("--max-port", type=int, default=DEFAULT_PORT_RANGE[1])
    parser.add_argument("--heartbeat", type=float, default=DEFAULT_HEARTBEAT_S)
    parser.add_argument(
        "--drop-reply-s",
        type=float,
        default=DEFAULT_DROP_REPLY_S,
        help="how long an answer is held when a call asks to lose it",
    )
    parser.add_argument(
        "--worker-token",
        help="the pool reservation this process is, which it then watches instead of its input",
    )
    parser.add_argument(
        "--max-idle-s",
        type=float,
        default=pool.DEFAULT_MAX_IDLE_S,
        help="how long a pool worker with no job stays warm before it ends itself",
    )
    parser.add_argument(
        "--skip-loopback-check",
        action="store_true",
        help="do not prove the port is unreachable from this machine's own addresses",
    )
    return parser


def wait_for_stop(stream) -> None:
    """Block until the word `stop` arrives or the input closes."""
    while True:
        line = stream.readline()
        if not line or line.strip().lower() == STOP_WORD:
            return


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BridgeConfig(
        home=args.home,
        port_range=(args.port, args.max_port),
        alias=args.alias,
        alias_template=args.alias_template,
        kind=args.kind,
        heartbeat_s=args.heartbeat,
        drop_reply_s=args.drop_reply_s,
        verify_loopback=not args.skip_loopback_check,
        owns_process=True,
    )
    bridge = Bridge(config)
    record = bridge.start()
    # The token is not printed here and never is. Whoever may read the session
    # file may read the token.
    print(f"bridge {record.alias} {record.session_id} on port {bridge.port}", flush=True)
    watcher = _start_watcher(args, bridge)
    watcher.start()
    try:
        bridge.main_loop.run_until(bridge.stopping)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stopping.set()
        bridge.stop()
    return 0


def _start_watcher(args: argparse.Namespace, bridge: Bridge) -> threading.Thread:
    """The thread that decides when this process has done its job.

    A worker watches its lease, everything else watches its input. Either way
    the answer arrives as the same stop flag, and the thread that owns the
    process is left free to run scene edits.
    """
    if args.worker_token:
        return threading.Thread(
            target=_watch_lease,
            args=(bridge, args.worker_token, args.max_idle_s),
            name="nscr-mcp-lease",
            daemon=True,
        )
    return threading.Thread(
        target=_watch, args=(sys.stdin, bridge.stopping), name="nscr-mcp-stdin", daemon=True
    )


def _watch(stream, stop: threading.Event) -> None:
    """Watch the input, and say so when it asks the process to end."""
    try:
        wait_for_stop(stream)
    finally:
        stop.set()


def _watch_lease(bridge: Bridge, token: str, max_idle_s: float) -> None:
    """Watch this worker's own row, and end the process when it says to.

    The store handle belongs to this thread alone, which is the rule for
    store handles. A store that cannot be read is not a reason to end a
    worker that is otherwise working, so the watcher says so and stops
    watching.
    """
    try:
        with pool.open_store(bridge.home) as store:
            reason = pool.watch_lease(store, token, max_idle_s=max_idle_s, stop=bridge.stopping)
    except Exception as error:  # noqa: BLE001 - reported, never fatal to the worker
        print(f"worker lease not watched: {error}", flush=True)
        return
    print(f"worker lease: {reason}", flush=True)
    bridge.stopping.set()


if __name__ == "__main__":  # pragma: no cover - the process entry point
    raise SystemExit(main())
