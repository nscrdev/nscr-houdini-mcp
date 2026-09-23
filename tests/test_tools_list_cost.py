"""The tool list cost script: which counter it uses, and what it reports."""

from __future__ import annotations

import importlib.util
import json
import sys
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
