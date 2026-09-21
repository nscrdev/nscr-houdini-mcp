"""Telling one process from another that happens to have the same number.

A pid on its own is not an identity. Numbers are handed out again, so a
session file left behind by a process that crashed can name a pid that now
belongs to something else entirely. Each system can say when a process
started, and a pid plus a start time is an identity that holds.

Every answer here is a string, so it goes into a session file as it is and
comes back comparable. `None` means this system would not say, and a caller
that gets `None` has learned nothing and must not pretend otherwise.
"""

from __future__ import annotations

import os
import subprocess
import sys

from nscr_houdini_mcp.store import process_is_alive

__all__ = ["process_is_alive", "process_start_stamp", "same_process"]

PS_TIMEOUT_S = 5.0


def process_start_stamp(pid: int | None = None) -> str | None:
    """When a process started, in whatever form this system reports it."""
    number = os.getpid() if pid is None else pid
    if number <= 0:
        return None
    if sys.platform == "win32":
        return _windows_start(number)
    if sys.platform == "linux":
        return _linux_start(number)
    if sys.platform == "darwin":
        return _ps_start(number)
    return None


def same_process(pid: int | None, stamp: str | None) -> bool | None:
    """Whether this pid is still the process that recorded that stamp.

    `True` and `False` are answers. `None` says the question could not be
    settled here, which happens when nothing recorded a stamp or when the
    system will not give one.
    """
    if not process_is_alive(pid):
        return False
    if not stamp:
        return None
    current = process_start_stamp(pid)
    if current is None:
        return None
    return current == stamp


def _linux_start(pid: int) -> str | None:
    """Field 22 of the process stat file: start time in clock ticks.

    The name of the program sits in brackets and may itself contain brackets
    and spaces, so the fields are counted from the last closing bracket.
    """
    try:
        text = (
            open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace")  # noqa: SIM115
            .read()
            .strip()
        )
    except OSError:
        return None
    tail = text.rpartition(")")[2].split()
    # After the name come state and 19 more fields before start time.
    if len(tail) < 20:
        return None
    return tail[19]


def _ps_start(pid: int) -> str | None:
    """Ask the process listing, which every system of this kind ships."""
    try:
        finished = subprocess.run(  # noqa: S603 - a fixed command with a number
            ["/bin/ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    stamp = finished.stdout.strip()
    return stamp or None


def _windows_start(pid: int) -> str | None:
    """Creation time from the kernel, as a plain number."""
    import ctypes
    import ctypes.wintypes

    process_query_limited_information = 0x1000
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return None
    try:
        created = ctypes.wintypes.FILETIME()
        exited = ctypes.wintypes.FILETIME()
        kernel = ctypes.wintypes.FILETIME()
        user = ctypes.wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(
            handle,
            ctypes.byref(created),
            ctypes.byref(exited),
            ctypes.byref(kernel),
            ctypes.byref(user),
        )
        if not ok:
            return None
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)
