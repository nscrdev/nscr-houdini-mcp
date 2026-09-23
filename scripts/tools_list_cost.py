#!/usr/bin/env python3
"""Print an estimate of the token cost of this server's `tools/list` payload.

How it counts
-------------
1. Build the server and ask it for its tool list.
2. Serialise each tool the way the protocol sends it: the JSON object a client
   receives, with the wire names (`inputSchema`, not `input_schema`), `None`
   fields dropped and no extra whitespace.
3. Estimate tokens as `ceil(len(json_text) / 4)`.

The divisor of 4 is a rough characters-per-token figure for English prose and
JSON. It is an estimate, not a tokeniser: treat it as a budget signal and as a
number to watch across commits, not as an exact count. The byte figures are
exact.

The total is reported against the budget for the whole tool list. Going over
the budget is reported, not failed: a schema is never thinned to fit it. Exit
code is 1 only when `--max-tokens` is given and the estimate goes over that, so
the script can gate a build when someone asks it to.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nscr_houdini_mcp.server import build_server  # noqa: E402

CHARS_PER_TOKEN = 4

# The goal for the whole `tools/list` payload.
BUDGET_TOKENS = 4000


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def tool_payloads() -> list[dict]:
    tools = asyncio.run(build_server().list_tools())
    return [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in tools]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="print machine readable output")
    args = parser.parse_args(argv)

    payloads = tool_payloads()
    whole = json.dumps({"tools": payloads}, separators=(",", ":"))
    total = estimate_tokens(whole)

    rows = []
    for payload in payloads:
        text = json.dumps(payload, separators=(",", ":"))
        rows.append({"name": payload["name"], "bytes": len(text), "tokens": estimate_tokens(text)})
    rows.sort(key=lambda row: row["tokens"], reverse=True)

    if args.json:
        report = {
            "total_tokens": total,
            "total_bytes": len(whole),
            "budget_tokens": BUDGET_TOKENS,
            "tools": rows,
        }
        print(json.dumps(report, indent=2))
    else:
        print(f"tools: {len(rows)}")
        print(f"payload bytes: {len(whole)}")
        print(f"estimated tokens: {total}  (bytes / {CHARS_PER_TOKEN}, rounded up)")
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
