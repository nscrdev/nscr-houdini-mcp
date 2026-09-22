"""A stand in for `hou` that remembers which thread touched it.

The rule it exists to check is that no thread answering a request calls into
`hou` in a graphical session. Every attribute read is recorded with the name of
the thread that made it, so a test can say which threads were involved in an
answer rather than trusting that none were.
"""

from __future__ import annotations

import threading
from typing import Any


class Guard:
    """Wrap a module stand in and note every attribute read against a thread."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self._lock = threading.Lock()
        self.touched: list[tuple[str, str]] = []

    def __getattr__(self, name: str) -> Any:
        with self._lock:
            self.touched.append((threading.current_thread().name, name))
        return getattr(self._module, name)

    def forget(self) -> None:
        with self._lock:
            self.touched.clear()

    def threads(self) -> set[str]:
        with self._lock:
            return {thread for thread, _ in self.touched}

    def touched_by(self, thread_name: str) -> list[str]:
        with self._lock:
            return [name for thread, name in self.touched if thread == thread_name]
