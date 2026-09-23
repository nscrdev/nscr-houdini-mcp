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
   names another. Without it, or when the encoding cannot be loaded (its tables
   are fetched once and then cached), tokens are estimated as
   `ceil(len(json_text) / 4)`. The report says which of the two was used.

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
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nscr_houdini_mcp.server import build_server  # noqa: E402

CHARS_PER_TOKEN = 4

# The encoding used when the tokenizer is there and no other is named.
DEFAULT_ENCODING = "o200k_base"

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
    try:
        tokenizer = tiktoken.get_encoding(encoding)
    except Exception as error:  # noqa: BLE001 - no table means fall back, whatever the reason
        estimate.note = f"the {encoding} encoding could not be loaded ({type(error).__name__})"
        return estimate
    return Counter(
        lambda text: len(tokenizer.encode(text, disallowed_special=())),
        f"tiktoken {encoding}",
    )


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
            print(f"no tokenizer: {counter.note}; install the dev extras for a real count")
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
