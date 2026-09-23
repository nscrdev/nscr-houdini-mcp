"""The tool list cost script: which counter it uses, and what it reports."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import types
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tools_list_cost.py"


def load() -> Any:
    spec = importlib.util.spec_from_file_location("tools_list_cost", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_without_the_tokenizer_it_estimates_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load()
    # A None entry makes the import fail, as it does where the package is missing.
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    counter = script.make_counter()
    assert counter.method.startswith("estimate")
    assert counter.note == "tiktoken is not installed"
    assert counter.count("x" * 9) == 3

    assert script.main(["--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counted_with"].startswith("estimate")
    assert report["fallback_reason"] == "tiktoken is not installed"
    assert report["budget_tokens"] == 4000
    assert report["total_bytes"] > 0
    assert len(report["tools"]) >= 1
    assert all(row["tokens"] == -(-row["bytes"] // 4) for row in report["tools"])


def test_an_encoding_that_will_not_load_falls_back_to_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load()

    class Broken:
        @staticmethod
        def get_encoding(name: str) -> Any:
            raise ValueError(f"no {name}")

    monkeypatch.setitem(sys.modules, "tiktoken", Broken)
    counter = script.make_counter("nothing_base")
    assert counter.method.startswith("estimate")
    assert "nothing_base" in counter.note


def test_the_tokenizer_counts_when_it_is_there(monkeypatch: pytest.MonkeyPatch) -> None:
    script = load()

    class Words:
        def encode(self, text: str, **_: Any) -> list[str]:
            return text.split()

    class Stub:
        @staticmethod
        def get_encoding(name: str) -> Any:
            return Words()

    monkeypatch.setitem(sys.modules, "tiktoken", Stub)
    counter = script.make_counter()
    assert counter.method == f"tiktoken {script.DEFAULT_ENCODING}"
    assert counter.note is None
    assert counter.count("one two three") == 3


def test_over_a_given_budget_it_fails(capsys: pytest.CaptureFixture[str]) -> None:
    script = load()
    assert script.main(["--estimate", "--max-tokens", "1"]) == 1
    assert "over budget" in capsys.readouterr().err


def laid_out_like_tiktoken(
    monkeypatch: pytest.MonkeyPatch, cached: set[str], fetch: Any
) -> tuple[Any, list[str]]:
    """A package shaped the way the script reads tiktoken: `get_encoding`
    reads its tables through `load.read_file` unless they are cached."""
    package = types.ModuleType("stand_in_tokenizer")
    loader = types.ModuleType("stand_in_tokenizer.load")
    loader.read_file = fetch
    fetched: list[str] = []

    class Words:
        def encode(self, text: str, **_: Any) -> list[str]:
            return text.split()

    def get_encoding(name: str) -> Any:
        if name not in cached:
            fetched.append(name)
            loader.read_file(f"blob://{name}")
        return Words()

    package.get_encoding = get_encoding
    package.load = loader
    monkeypatch.setitem(sys.modules, "stand_in_tokenizer", package)
    monkeypatch.setitem(sys.modules, "stand_in_tokenizer.load", loader)
    return package, fetched


def test_a_cached_encoding_never_reaches_for_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    script = load()

    def never(path: str) -> bytes:
        raise AssertionError(f"fetched {path}")

    package, fetched = laid_out_like_tiktoken(monkeypatch, {"o200k_base"}, never)
    tokenizer, note = script.load_encoding(package, "o200k_base")
    assert note is None
    assert len(tokenizer.encode("one two")) == 2
    assert fetched == []
    assert package.load.read_file is never


def test_an_encoding_not_cached_is_fetched_with_a_limit_and_says_so_when_it_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = load()
    limits: list[float | None] = []

    def unreachable(path: str) -> bytes:
        limits.append(socket.getdefaulttimeout())
        raise OSError("no route")

    package, fetched = laid_out_like_tiktoken(monkeypatch, set(), unreachable)
    before = socket.getdefaulttimeout()
    tokenizer, note = script.load_encoding(package, "o200k_base", timeout_s=3.0)
    assert tokenizer is None
    # Tried from the cache first, then fetched once, under the limit.
    assert fetched == ["o200k_base", "o200k_base"]
    assert limits == [3.0]
    assert socket.getdefaulttimeout() == before
    assert package.load.read_file is unreachable
    assert note.startswith("tiktoken is installed, but the o200k_base tables are not in its cache")
    assert "could not be fetched within 3 seconds" in note
    assert "not installed" not in note
