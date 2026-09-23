#!/usr/bin/env python3
"""Print the token cost of this server's `tools/list` payload.

How it counts
-------------
1. Build the server and ask it for its tool list.
2. Serialise each tool the way the protocol sends it: the JSON object a client
   receives, with the wire names (`inputSchema`, not `input_schema`), `None`
   fields dropped and no extra whitespace.
3. Count tokens with a real tokenizer when one is importable: `tiktoken`, an
   optional dev dependency, with the `o200k_base` encoding unless `--encoding`
   names another. The encoding's tables are read from tiktoken's own cache
   first; only when they are not there are they fetched, once, with a ten
   second limit on the network. Without the package, or when the tables are
   neither cached nor fetched, tokens are estimated as
   `ceil(len(json_text) / 4)`. The report says which of the two was used, and
   why when it fell back.

A tokenizer's count is exact for that encoding only; a client on another
encoding pays a somewhat different number. The estimate is rougher still.
Treat either as a budget signal to watch across commits. The byte figures are
exact.

The total is reported against the budget for the whole tool list. Going over
the budget is reported, not failed: a schema is never thinned to fit it. Exit
code is 1 only when `--max-tokens` is given and the count goes over that, so
the script can gate a build when someone asks it to.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import socket
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nscr_houdini_mcp.server import build_server  # noqa: E402

CHARS_PER_TOKEN = 4

# The encoding used when the tokenizer is there and no other is named.
DEFAULT_ENCODING = "o200k_base"

# The longest any one step of fetching an encoding's tables may take, so a
# network that swallows requests costs seconds, not minutes.
DOWNLOAD_TIMEOUT_S = 10.0

# The goal for the whole `tools/list` payload.
BUDGET_TOKENS = 4000


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class Counter:
    """Counts tokens one way, and says which way in `method`."""

    def __init__(self, count: Callable[[str], int], method: str, note: str | None = None) -> None:
        self.count = count
        self.method = method
        self.note = note


def make_counter(encoding: str = DEFAULT_ENCODING, *, estimate_only: bool = False) -> Counter:
    """The tokenizer when it imports and its encoding loads, else the estimate."""
    estimate = Counter(estimate_tokens, f"estimate, bytes / {CHARS_PER_TOKEN} rounded up")
    if estimate_only:
        return estimate
    try:
        import tiktoken
    except ImportError:
        estimate.note = "tiktoken is not installed"
        return estimate
    tokenizer, note = load_encoding(tiktoken, encoding)
    if tokenizer is None:
        estimate.note = note
        return estimate
    return Counter(
        lambda text: len(tokenizer.encode(text, disallowed_special=())),
        f"tiktoken {encoding}",
    )


class NotCached(Exception):
    """The tables are not in tiktoken's cache, and reading them was held back."""


def load_encoding(
    tiktoken: Any, encoding: str, *, timeout_s: float = DOWNLOAD_TIMEOUT_S
) -> tuple[Any, str | None]:
    """The encoding and nothing, or nothing and why not.

    The cache is tried first with every read of the tables held back, so a
    cached encoding never touches the network. Only then are they fetched,
    under a socket timeout that is put back afterwards.
    """
    try:
        loader = importlib.import_module(f"{tiktoken.__name__}.load")
    except Exception:  # noqa: BLE001 - a tokenizer laid out otherwise is only tried
        loader = None
    fetch = getattr(loader, "read_file", None)
    if loader is not None and fetch is not None:

        def cache_only(path: str) -> bytes:
            raise NotCached(path)

        loader.read_file = cache_only
        try:
            return tiktoken.get_encoding(encoding), None
        except NotCached:
            pass
        except Exception as error:  # noqa: BLE001 - no table means fall back
            return None, f"the {encoding} encoding could not be loaded ({type(error).__name__})"
        finally:
            loader.read_file = fetch
    before = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    if loader is not None and fetch is not None:
        # tiktoken's own fetch waits on the network with no limit of its own,
        # which the socket default does not reach, so a web address is read
        # here instead, with the limit on every step. Anything else is its.
        def bounded(path: str) -> bytes:
            if not path.startswith(("http://", "https://")):
                return fetch(path)
            with urllib.request.urlopen(path, timeout=timeout_s) as answer:  # noqa: S310
                return answer.read()

        loader.read_file = bounded
    try:
        return tiktoken.get_encoding(encoding), None
    except Exception as error:  # noqa: BLE001 - no table means fall back, whatever the reason
        return None, (
            f"tiktoken is installed, but the {encoding} tables are not in its cache and"
            f" could not be fetched within {timeout_s:g} seconds a step"
            f" ({type(error).__name__}); run once with a network to cache them"
        )
    finally:
        socket.setdefaulttimeout(before)
        if loader is not None and fetch is not None:
            loader.read_file = fetch


def tool_payloads() -> list[dict]:
    tools = asyncio.run(build_server().list_tools())
    return [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="print machine readable output")
    parser.add_argument(
        "--encoding", default=DEFAULT_ENCODING, help="the tokenizer encoding to count with"
    )
    parser.add_argument(
        "--estimate", action="store_true", help="skip the tokenizer and use the estimate"
    )
    args = parser.parse_args(argv)

    counter = make_counter(args.encoding, estimate_only=args.estimate)
    payloads = tool_payloads()
    whole = json.dumps({"tools": payloads}, separators=(",", ":"))
    total = counter.count(whole)

    rows = []
    for payload in payloads:
        text = json.dumps(payload, separators=(",", ":"))
        rows.append({"name": payload["name"], "bytes": len(text), "tokens": counter.count(text)})
    rows.sort(key=lambda row: row["tokens"], reverse=True)

    if args.json:
        report = {
            "total_tokens": total,
            "total_bytes": len(whole),
            "budget_tokens": BUDGET_TOKENS,
            "counted_with": counter.method,
            "fallback_reason": counter.note,
            "tools": rows,
        }
        print(json.dumps(report, indent=2))
    else:
        print(f"tools: {len(rows)}")
        print(f"payload bytes: {len(whole)}")
        print(f"tokens: {total}  (counted with {counter.method})")
        if counter.note:
            print(f"estimated, not counted: {counter.note}")
        share = total * 100 / BUDGET_TOKENS
        state = "within" if total <= BUDGET_TOKENS else "OVER"
        print(f"budget: {BUDGET_TOKENS} tokens, {share:.0f} percent used, {state} budget")
        if rows:
            width = max(len(row["name"]) for row in rows)
            print()
            print(f"{'tool'.ljust(width)}  {'bytes':>7}  {'tokens':>7}")
            for row in rows:
                print(f"{row['name'].ljust(width)}  {row['bytes']:>7}  {row['tokens']:>7}")

    if args.max_tokens is not None and total > args.max_tokens:
        print(f"over budget: {total} > {args.max_tokens}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
