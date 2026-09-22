# nscr-houdini-mcp

An MCP server and a small set of agent skills for SideFX Houdini 22.

Status: early. Nothing here is usable yet.

## Goals

- A small tool set with direct Python access to Houdini, so the context cost stays low.
- Works with any MCP client. No client-specific rules.
- One setup talks to many Houdini sessions at once, both open GUI sessions and headless workers.
- Agents check their work against a reference image before they call it done.
- Renders, caches and captures go to managed folders next to the scene file.
- Plain `SKILL.md` skills that help an agent build scenes a person can read, change and reuse. You can edit them to fit how you work.
- Runs on macOS, Windows and Linux.

## Development

Python 3.11 or newer. The server process never imports `hou`.

Setup, with [uv](https://docs.astral.sh/uv/):

```sh
uv venv
uv pip install -e ".[dev]"
```

Or with the standard library (`.venv\Scripts\activate` on Windows):

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Checks, the same ones CI runs:

```sh
ruff check . && ruff format --check .
pytest -q
python scripts/lint_client_names.py   # shipped text must not name any client or vendor
python scripts/tools_list_cost.py     # estimated token cost of the tools/list payload
```

`scripts/tools_list_cost.py` serialises the tool list the way a client receives
it and estimates tokens as bytes divided by four, rounded up. That is a budget
signal to watch across commits, not an exact count. Pass `--max-tokens N` to
make it fail over a budget.

Git hooks:

```sh
python scripts/install_hooks.py
```

This sets `core.hooksPath` to the tracked `hooks/` directory. The hooks check
staged content and the commit message against a private term list at
`.context/leak-terms.txt`, which is ignored by git and kept only on your own
machine. They fail closed: with no list, every commit is refused.

Run the server on stdio:

```sh
nscr-houdini-mcp
```

### The Houdini side

`nscr_houdini_mcp.bridge` runs inside Houdini's own Python. It serves two
endpoints on loopback, registers the session in the coordination store and
takes it out again when the process quits.

What the bridge guards against: a web page in a browser on this machine,
another person with an account on it, and the network. Not the owner's own
programs. Anything running as the same person can read that person's files,
token included, and could drive Houdini with or without this.

The token is never sent. Each request is signed with it and each answer is
signed back, so nothing on the port learns anything it could use again, and a
caller can tell the bridge from something that took its port after a crash. A
request is also refused if it carries `Origin` or `Referer`, names a host
other than this bridge, is not JSON, is over the size cap, or nests too deeply
for an envelope. The bridge registers no built in API route, so the one that
parses a posted form before any handler runs does not exist on its server.

### Addressing a session, and sending a call twice

A session has two handles. `session_id` is random, minted at start and never
reused. `alias` is the readable name: a session with a scene is named after
the scene file, a worker is `w1`, `w2` and so on, and two Houdinis on the same
file get different names. Either handle addresses a session. The name never
moves once it is settled, so a scene saved under a different name leaves the
session's name out of date, which health and every reply say and nothing
silently corrects.

`scene_epoch` counts the times a session has replaced its scene, which is any
open, new scene or reload. A call carrying an older epoch is refused with
`SCENE_REPLACED` and a summary of the scene there is now, before the tool
runs. A call addressed to a session whose process has gone is refused by the
calling end with `SESSION_DEAD` and the id of whatever answers to the same
name now. Nothing is sent to the new process.

A call that changes the scene carries an `operation_id`. The bridge takes a
receipt under that id before the work runs and finishes it with the answer, so
the same id arriving again is answered from the receipt rather than doing the
work twice. The same id with different arguments is `OPERATION_MISMATCH`, and
an id whose first attempt left no answer is `OUTCOME_UNKNOWN` rather than a
guess. Reads take no receipt.

That is what makes a retry safe, and the only retry the client does by itself:
one, on a lost reply (the connection closed, or the read ran out of time), and
only for a call that carries an operation id, using that same id. A reply that
arrived is never sent again, whatever it says.

Start one in a headless Houdini:

```sh
hython -m nscr_houdini_mcp.bridge.main --home <state folder>
```

It stops when its input closes, or on the word `stop`.

For a session with an interface, copy `houdini/packages/nscr_houdini_mcp.json`
into a Houdini packages folder and replace the two paths in it. Installing the
package opens no port: `houdini/scripts/456.py` starts a bridge only when
`NSCR_MCP_AUTOSTART` is `1`.

Tests marked `houdini` need a Houdini on the machine and skip when there is
none, so `pytest -q` is complete everywhere. Run only those with `pytest -m
houdini`, or skip them with `pytest -m "not houdini"`. The bridge is found
through `NSCR_MCP_HYTHON`, then `HFS`, then the usual install folder.

## License

MIT
