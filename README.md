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

A reply is usually lost because this end gave up on the socket first, so the
work is often still running when the second send arrives. That send queues
behind it, asking for a longer turn than the first, and can still come back
`SESSION_BUSY`. The answer is in the receipt as soon as the work ends, so the
way on from there is the same id again, not a fresh call.

Start one in a headless Houdini:

```sh
hython -m nscr_houdini_mcp.bridge.main --home <state folder>
```

It stops when its input closes, or on the word `stop`.

### Installing the Houdini side

One command writes the Houdini package that puts this on a session's path:

```sh
nscr-houdini-mcp bridge install                  # for Houdini 22.0
nscr-houdini-mcp bridge install --houdini-version 22.0 --dry-run
nscr-houdini-mcp bridge install --autostart      # every Houdini opens a port
nscr-houdini-mcp bridge uninstall
nscr-houdini-mcp bridge status
nscr-houdini-mcp bridge snippet
```

It writes one file, `nscr_houdini_mcp.json`, into the packages folder Houdini
reads. That folder is often not where a shell would guess, because a launcher,
a line in `houdini.env`, another package or a synced documents folder can move
it, so it is worked out in this order:

1. `--packages-dir`, on `install`, `uninstall` and `status`.
2. `HOUDINI_PACKAGE_DIR` in this shell. Houdini reads every folder it names,
   so the first is written into and the rest are reported.
3. A real Houdini, asked. One short script runs in `hython` with a ten second
   cap and reports the home folder, `HOUDINI_PACKAGE_DIR`, `HOUDINI_USER_PREF_DIR`
   and `HSITE` as Houdini itself sees them, after `houdini.env` and packages.
   A Houdini that is missing or slow leaves a note and the lookup carries on.
   `HSITE` is other people's, so it is reported and never written to.
4. `HOUDINI_USER_PREF_DIR` in this shell, with `__HVER__` filled in.
5. The usual folder for the system: `~/Library/Preferences/houdini/<version>`
   on macOS, `Documents\houdini<version>` under the profile on Windows,
   `~/houdini<version>` on Linux.

Nothing is remembered between runs, and `bridge status` prints which of those
decided and every folder it considered.

The file points `HOUDINI_PATH` at this project's `houdini/` folder and
`PYTHONPATH` at its `src/`, both worked out from where this copy is running,
so there is nothing to edit by hand. Every file it writes carries a marker: a
package of the same name that this did not write, or a link where the file
should be, is reported and left exactly as it is, and `uninstall` takes away
only its own, plus any folder it had to make and nothing else.

Installing opens no port. Auto start is off unless you ask for it with
`--autostart`, which sets `NSCR_MCP_AUTOSTART` to `1` in the package. The
start hangs off `python3.13libs/ready.py`, which Houdini runs for every folder
on its path, in a session with an interface and without, so this package takes
nobody else's startup script away.

`bridge status` lists the sessions running now with their ids, names, ports,
scenes and whether each is busy, asks each one whether it is healthy under one
shared time budget, and says which folders have the package and where Houdini
is installed.

To start a bridge inside a Houdini that is already open, paste what `bridge
snippet` prints into its Python shell. It works out the source path as it
prints, so the lines run as they stand.

`houdini/packages/nscr_houdini_mcp.json` is the same file as a template, for
anyone who would rather place it themselves.

### The worker pool

A worker is a headless Houdini of its own with a bridge in it, kept warm so
the next piece of work does not pay a cold start:

```sh
nscr-houdini-mcp bridge worker start             # one worker, if there is room
nscr-houdini-mcp bridge worker start --weight heavy --max-threads 8
nscr-houdini-mcp bridge worker list
nscr-houdini-mcp bridge worker reserve w1 --job job-1
nscr-houdini-mcp bridge worker release w1
nscr-houdini-mcp bridge worker stop w1
```

Three workers may run at once by default. The slot is taken in one
transaction in the coordination store, which counts the workers that are
still starting, so several servers on one machine cannot hand out the same
last slot, and `POOL_FULL` is the answer when there is none. A start that
fails gives its slot straight back. Each reservation also carries a weight,
and the pool holds a budget of its own, so a heavy job is refused while the
machine is busy even when a slot is free.

A worker does not belong to the process that started it. It is started
detached, with its output in `logs/worker-<name>.log` under the state folder,
and it stays when its server exits, so a client restart does not throw the
warm pool away. What ends it is its lease: it watches its own row and goes
when a server asks it to, or when nobody has wanted it for half an hour.
Routing a call to it renews the lease. `stop` asks first and ends the process
itself only if the ask was not enough.

Each worker is asked once, when it comes up, what it can do: the build, the
license it got, the renderers that are really installed, how it can make a
picture, and that it can be cancelled. The answer is kept beside the worker,
so another process can pick one without asking it anything.

Tests marked `houdini` need a Houdini on the machine and skip when there is
none, so `pytest -q` is complete everywhere. Run only those with `pytest -m
houdini`, or skip them with `pytest -m "not houdini"`. The bridge is found
through `NSCR_MCP_HYTHON`, then `HFS`, then the usual install folder.

## Output folders

No tool takes an output path. You name a kind and a name, and the server
builds the path from a token table. Renders, flipbooks and comps go to dated
folders, because a person browses them by the day they were made. Caches, USD
layers and hip files keep a stable folder under the name, because the scene
reads them back and only the version should change between runs. Files the
agent writes for itself go under `$HIP/.agent/`.

The defaults, with `<ver>` three digits and `<date>` as `YYYYMMDD`:

```
render    $HIP/renders/<date>_<name>/v<ver>/<name>_v<ver>.$F4.exr
flipbook  $HIP/flipbook/<date>_<name>/v<ver>/<name>_v<ver>.$F4.png
comp      $HIP/comp/<date>_<name>/v<ver>/<name>_v<ver>.$F4.exr
cache     $HIP/geo/<name>/v<ver>/<name>_v<ver>.$F4.bgeo.sc
usd       $HIP/usd/<name>/v<ver>/<name>_v<ver>.usd
hip       $HIP/<name>_v<ver>.hip
capture   $HIP/.agent/captures/<date>/<time>_<name>_<run_id>.png
compare   $HIP/.agent/compare/<date>_<name>/<ver>_<run_id>/
```

Each output comes with two lines. The template keeps its Houdini variables and
uses `${OS}` for a name that came from the node, so it reads well and a scene
carries it to another machine:
`$HIP/renders/20260921_${OS}/v003/${OS}_v003.$F4.exr`. That is the line to show
a person and to leave on a node with no run yet. What the server writes on the
node for a run it has started is the expanded path, because the run is the
server's from then on: renaming the node mid render must not send half the
frames somewhere else, and the record and the scene have to say the same thing.
A rename still carries through to the next run, which takes its own version.

`$HIP`, `$HOUDINI_TEMP_DIR` and `$JOB` are the variables this fills in, the
first two from the session and `$JOB` from the environment. A table naming any
other variable is refused when it is read, so no run ever creates a folder
called `$SHOT`. Roots have to start at one of those three, and templates,
roots, producer levels and extensions may not step out of a folder with `..`;
the finished path is checked against the root once more before anything is
created.

A version number is taken inside one store transaction and the version folder
is then created with an exclusive `mkdir`, so a number is used once even with
several processes and several machines on the same scene folder. A hip file has
no folder of its own, so it claims a small `.claim` file beside it instead,
which is what stops two machines with their own stores from writing the same
`v001`. Every file the agent writes for itself carries the run id, so two
captures in the same second are two files. Each run leaves a readable
`_run.json` beside its output with the scene, session, node, version and paths,
absolute for the machine that made the run and again relative to the root, so
the same folder read from somewhere else still makes sense.

Edit the table where it suits you. Built in defaults come first, then
`config.toml` in the state folder (`NSCR_MCP_HOME` moves it), then
`.agent/outputs.toml` beside the scene, which may set only `[outputs]` and
`[conventions]`. Later wins key by key, and a file that cannot be used says
which key and why.

```toml
[outputs]
producer = "3d/hip"        # an extra level under each kind, off by default
cache_root = "$JOB/cache"  # heavy caches on a fast local disk
version_width = 3

[outputs.grammar]
comp = "<output_root>/comp/<name>/v<ver>/<name>_v<ver>.<frame>.<ext>"

[outputs.extensions]
flipbook = "jpg"

[conventions]
output_marker_type = "null"
output_marker_prefix = "OUT_"
```

A scene that has never been saved has no `$HIP`, so its runs go under
`$HOUDINI_TEMP_DIR/nscr-houdini-mcp/<session>/` and every result says
`unsaved_hip`, which is the cue to save and run again. The template keeps the
variable, so saving that scene later leaves nothing about this machine in it.

## License

MIT
