"""Fixtures every test file may ask for.

The machinery itself is in `support`, so a script outside the suite can use the
same helpers. What is here is only the wiring pytest needs: a state folder
nobody else shares, and a Houdini that is always stopped again.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import pytest

import support
from nscr_houdini_mcp.bridge.launcher import HythonBridge


@pytest.fixture
def isolated_home(tmp_path: Path) -> Path:
    """A state folder of this test's own, with nothing else in it."""
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


@pytest.fixture
def hython_sessions() -> Iterator[Callable[..., HythonBridge]]:
    """Start sessions that are stopped again when the test ends.

    Ask for as many as the check needs, in whatever state folder and port range
    it wants. Every one of them is stopped in the order they were started,
    including after a failure, so no check can leave a Houdini running.
    """
    with ExitStack() as stack:

        def start(home: Path, **rest: object) -> HythonBridge:
            return stack.enter_context(support.hython_session(home, **rest))  # type: ignore[arg-type]

        yield start
