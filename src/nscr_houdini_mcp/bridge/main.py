"""Start a bridge in this Houdini process and hold the process open.

Run with hython:

    hython -m nscr_houdini_mcp.bridge.main --home <folder>

The process stays alive until its input closes or the word `stop` arrives on
it. Tying the lifetime to the input means a launcher that dies takes its
worker with it, instead of leaving a Houdini running with nobody to talk to.

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

from nscr_houdini_mcp.bridge.app import DEFAULT_HEARTBEAT_S, Bridge, BridgeConfig
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
        verify_loopback=not args.skip_loopback_check,
    )
    bridge = Bridge(config)
    record = bridge.start()
    # The token is not printed here and never is. Whoever may read the session
    # file may read the token.
    print(f"bridge {record.alias} {record.session_id} on port {bridge.port}", flush=True)
    stop = threading.Event()
    watcher = threading.Thread(
        target=_watch, args=(sys.stdin, stop), name="nscr-mcp-stdin", daemon=True
    )
    watcher.start()
    try:
        bridge.main_loop.run_until(stop)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        bridge.stop()
    return 0


def _watch(stream, stop: threading.Event) -> None:
    """Watch the input, and say so when it asks the process to end."""
    try:
        wait_for_stop(stream)
    finally:
        stop.set()


if __name__ == "__main__":  # pragma: no cover - the process entry point
    raise SystemExit(main())
