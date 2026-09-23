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

Seven tools so far. `hou_ping` says which session a call reaches and that it
answers. `hou_sessions` lists every session with its state (`live`, `busy`,
`unresponsive`, `crashed` or `gone`) and starts and stops workers under the
pool's rules; it never closes a Houdini with a user interface. `hou_scene`
reads the scene, opens a file and reports what it could not resolve as data,
saves in place, and saves the next `<name>_v###` without writing over
anything. A scene open in a user interface with unsaved changes is not
replaced unless the call says to throw them away. `info` says whether the
scene has unsaved changes and where that answer came from: Houdini itself in
a session with a user interface, and in a worker, which answers yes whatever
it holds, the bridge's own mark. That mark is clean after a save or a load,
dirty after any call that changes the scene or a merge, and unknown at start
or after code that saved or loaded by itself, when it cannot tell.
Opening a scene runs the code that scene file carries, as Houdini always
does, so only open files you trust.

`hou_inspect` reads nodes, networks and parameters. `tree` lists what is
under a path, `node` reads one node or a batch of up to fifty, `parms` reads
parameter tables, `find` searches by a glob on the name or path and by type,
and `selection` lists what is selected in a session with a user interface.
Rows come sorted by path, in every mode. A read with more rows than `limit`
(200 unless you say, up to 2000) hands back `next_page`; send it back as
`page`, with the same arguments, to carry on after the last path. A tree or a
search stops walking once its page is full, so `total` comes only with a
first page that holds everything. If the scene changed in between, the page
still comes back and says `scene_changed`. A batch never fails for one
missing path: that entry carries its own error, with the closest paths that
are there. A tree's network boxes and sticky notes come whole on its first
page, up to 500 of each, with `boxes_truncated` or `notes_truncated` past
that. A multiparm row carries up to 200 instances and says how many there
are; name the multiparm itself to page through all of them.

A read cooks nothing unless it passes `evaluate`. Without it, a value that
could only be had by cooking is left out and `not_cooked` says why: a number
a channel operator's export may drive, a parameter with several keyframes, a
Python expression, a backtick, a variable such as `$NPT` that only means
something inside a cook, or an expression that reads the scene, such as one
that counts another node's points, directly or through `ch()`. A node's
errors are the ones the last cook left, marked `not_cooked` when there has
been none and `stale` with the reason when the node has changed since. With
`evaluate` the read may cook, under the same `wait_s` and `timeout_s` as any
other call, and one that runs out of time is asked to stop between rows; a
render node or a task network is never cooked by a read.

Every tool that reads takes the same three `detail` levels:

- `summary`, the default: who each item is and how many of things it has,
  one compact row per item.
- `standard`: what a person sees without digging, on the node and in the
  parameter pane: the parameters that differ from their defaults, the wires
  by input label, the flags, error text and comments.
- `full`: everything else: parameters at their defaults, expressions with
  their text and values, code, spare parameter templates, cook times and user
  data.

`include` brings single items into a lower level: `wires`, `flags`, `errors`,
`expressions`, `code`, `notes` or `cook_time`.

`hou_python` runs Python inside a session with the whole `hou` API. What the
code leaves in a variable named `result` comes back, turned into JSON on the
session's own thread the way every answer is: a node or a parameter as its
path, a vector or an array as a list, a numpy scalar as its number, up to the
usual caps. A collection is read only as far as its cap. Variables stay
between calls in a namespace, one dict per session and name, seeded with
`hou` and `mcp` and nothing else. A call that names none uses this server's
own, `c_` and an id drawn when the server starts; `shared` is the name to use
when agents should share. A namespace goes when a call passes `reset`, when
the session ends, or when nobody has used it for an hour (a `c_` default) or
a day (any other name). A session keeps at most 32, dropping the least
recently used, and its health says how many there are and roughly how big.
Separate namespaces keep variables apart, never the scene: every one of them
works on the same node graph.

What the code prints, on either stream, comes back as `stdout_tail`. Only
what the call's own thread prints is taken, and whatever the code does to
`sys.stdout` or `sys.stderr` is undone when the call ends. `max_chars`
(12,000 unless you say) is the budget for the result and the printed text
together; what does not fit is counted in `elided_chars`, and what the
session kept is written to the spill folder, named in `spill_path`. The
session keeps the last 512,000 characters of output and the result as it was
encoded, which is bounded too, so the spill holds that much and no more. Text
UTF-8 cannot carry, such as a lone surrogate in a file name, comes back as its
escape, and `lossy` with `cut` says where anything was changed or cut.

An exception in the code is not a failed call: the result carries `error`
with the type, the message and the last twenty lines of the traceback, places
on disk taken out, and the client sees it marked as an error. Code that will
not compile is the same, with a syntax error's line and offset. The namespace
and the scene are left as the code left them, on purpose: what the code did
before it raised stays as one undo entry, for a person to keep or take back.

Every call counts as a change. It runs in one undo group, named `undo_label`
or `hou_python` and the operation id, takes a receipt under that id and
follows the same busy and retry rules as any other change. Because the call
is inside its own undo group, `hou.undos.performUndo()` in the code raises;
taking a whole call back is for a person in a session with a user interface.
The receipt is bound to the arguments the caller sent, so a lost reply sent
again with the same operation id is answered from it, even by a server
started since and even when the call named no namespace, and the answer names
the namespace the code ran in.

Every call is also a job, with a `job_id` made from its operation id, so a
reply that never arrived can still be followed. `background` says how long
the call itself waits. `auto`, the default, waits up to `inline_wait_s` from
`config.toml` (ten seconds unless you say, from 1 to 50, and never longer
than `timeout_s`): code that finishes in that time answers as usual with its
`state`, and slower code answers with the job to follow at that moment and
carries on. `true` answers with the job as soon as the session has taken the
call. `false` waits up to `timeout_s` (a minute unless you say; anything above
`python_timeout_cap_s`, at most an hour, is lowered to it) and then answers
`TIMEOUT` with `still_running` and the `job_id`. The code is never stopped by
any of these, and the same operation id fetches its answer once it ends, also
when sent while the code is still running.

`mcp` in every namespace has four things, made afresh for each call and
answering only on that call's thread while it runs. `mcp.output_path(kind,
name, ext)` hands out a managed path for this session and scene from the
output table below, for `render`, `flipbook`, `comp`, `cache`, `usd`, `hip`,
`capture`, `compare`, `reference` or `check`; never a `spill` or a `job`,
which are the server's own. `mcp.freeze_parm(parm, path)` puts a path the call
was handed on an output parameter, given as a `hou.Parm` or its path, for as
long as the call runs; when the call ends, however it ends, the parameter
gets back what it held before, its value or its expression, and the answer
says so in `restored_parms`.
`mcp.progress(done, total, message)` leaves a note,
finite numbers only, that health and `hou_ping` show while the call runs.
The same note is written to the call's job, at most once a second.
`mcp.cancelled()` says whether the call should stop, for a long loop to look
at between pieces of work. It turns true when `hou_jobs` cancels the job or
the session is going down. Code that stops once it has seen a cancel ends
`cancelled`; code that never looks runs to the end and ends `done`, with the
request still on the job. Code that stops because its session is going down
ends `lost`, as it would had the session gone first.

`hou_jobs` follows long running work by id. `status`, the default, reads one
job: its state (`queued`, `running`, `done`, `failed`, `cancelled` or
`lost`), progress, outputs, error, when it started and ended, the scene epoch
it ran against and whether a cancel was asked for. With `wait_s` (up to 50)
the call is held until the state or the progress changes, and says so with
`changed`, so there is no need to poll with sleeps. For a Python job that has
ended, `outputs` is the answer the call would have given. `cancel` writes the
request where the session reads it within two seconds and also asks the
session directly, and says whether that direct request got there. In a
session with a user interface a cancel is best effort: the code runs on
Houdini's main thread, and only code that looks at `mcp.cancelled()` stops.
`list` gives jobs newest first, by session and state, with `next_page`.

Jobs live in the coordination store, so any server can answer for any job,
including one started after the call that began it. A job whose session ends,
whether stopped or found gone, is `lost` with the progress and outputs it had
written, and so is one its session has said nothing about for fifteen
minutes. A job that ends leaves a readable copy of itself beside the scene,
in `.agent/jobs/`. Jobs are kept for 7 days.

`hou_outputs` hands out managed paths and reads back what a scene made.
`resolve` takes a place for a `kind` and a `name`, with `ext` when the kind's
own extension is not the one wanted and `node` for the node the output
belongs to, whose name is used when none is given. It answers with
`parm_string`, the line with `$HIP` that belongs on the node,
`expanded_path`, the absolute path for this run, and the `version`, `folder`,
`run_id` and `sidecar` beside them. A version is taken on every call; with
`operation_id` the same id sent again answers with the place it took. `$JOB`
and `$HOUDINI_TEMP_DIR` are read from the session, once, never from the
server's own environment. `job` and `spill` paths belong to the server and are
refused. `list` reads the runs this scene has made, newest first, with their
paths, kinds, versions, sessions, times and whether anything is on disk for
them; `filter` narrows it by `kind`, a `name` glob and `since`. `lint` reads
every output parameter under `node` (the whole scene unless named), walking
the network a node at a time and stopping when its page is full or the call is
asked to stop. An output is a file parameter a node type marks as written to,
a multiparm's instances included, or an unmarked one on a render node, as
Alembic and USD render nodes have; hooks, a renderer's logs, a render it reads
back and folders are not. One row per problem: `absolute_path`, `outside_hip`
(outside the folders the output table manages for this scene),
`unversioned`, `missing_on_disk` (for a sequence, at the current frame and at
both ends of the frame range, and with `empty` for a node's main output, or
one a toggle of its own turns on, that names nothing), `frozen_after_run`, `expression` for a value only an evaluation
could give and `unexpanded` for a variable it cannot fill in. Nothing is
evaluated and nothing cooks: values are expanded from the table's variables
and the node's and scene's own names. `list` and `lint` page with `limit` and
`next_page`, the way reads do.

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

A worker started from a client through `hou_sessions` takes `pool_cap`,
`hython` or `houdini_build`, and `worker_ports` from `config.toml`, and so
does `worker start` for any flag it is not given.

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
warm pool away. The log is private to its owner and is rolled over when it
grows, keeping the last two. What ends a worker is its lease: it watches its
own row and goes when a server asks it to, or when nobody has wanted it for
half an hour. Routing a call to it renews the lease. `stop` asks first and
ends the process itself only if the ask was not enough, and never unless that
process can be shown to still be the worker. `start` exits 3 when the pool is
full and 1 when the start went wrong.

A worker inherits this process's environment on purpose: it has to see the
same licensing, path and package settings as the shell the tool was started
from, or it is a different Houdini from the artist's. Three things are decided
rather than inherited. Its state folder is the one the pool is using, its
reservation token arrives in the environment and never on the command line,
which every account on the machine can read, and its thread cap is either the
one you named with `--max-threads` or the one the weight implies: a heavy
worker gets the machine, a light one is left at Houdini's own default.

The Windows side of this, the detached start, the kill and the start stamp,
is written and read but has not been run on Windows yet.

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
reference $HIP/.agent/reference/<name>_<run_id>.png
check     $HIP/.agent/checks/<date>/<time>_<name>_<run_id>.png
job       $HIP/.agent/jobs/<job_id>.json
spill     <spill folder>/<YYYY-MM-DD>/<time>-<name>-<run_id>.json
```

A `reference` is an image an agent keeps to check its work against, and a
`check` is what it made to compare with one. A `job` path is only ever written
by a session or the server, for the record of a job that has ended. A `spill`
goes to the server's own spill folder, where results too large to return go,
whatever the scene is: its line starts at `<spill_root>` and holds no Houdini
variable, it is never handed to code and it never goes on a node. The row is
kept for the server's own use; the spill writer does not read it yet. Neither
is ever handed out, by `mcp.output_path` or by `hou_outputs resolve`.

Each output comes with two lines. The template keeps its Houdini variables and
uses `${OS}` for a name that came from the node, so it reads well and a scene
carries it to another machine:
`$HIP/renders/20260921_${OS}/v003/${OS}_v003.$F4.exr`. That is the line to show
a person and to leave on a node with no run yet. What the server writes on the
node for a run it has started is the expanded path, because the run is the
server's from then on: renaming the node mid render must not send half the
frames somewhere else, and the record and the scene have to say the same thing.
A rename still carries through to the next run, which takes its own version.

When the run is over, whether it ended done, failed or cancelled, the node
gets back what it held before, the `$HIP` line a person set or the expression
that fed it, so a scene saved afterwards never carries a path from this
machine. A save while the run is going writes that same value and puts the
run's path back straight after. Every parameter a run freezes is recorded in
the coordination store before it is touched, with what it held, and the
record goes once that is back. Each record carries a token: a parameter held
by one run is refused to another with `PARM_FROZEN`, naming the run, and only
the holder gives it back. The call that froze it gives it back when it ends,
finding a node renamed in the meantime by its session id. A session that dies
with a parameter frozen cannot, so the next session to load that scene with
`hou_scene open`, or to save it with `save` or `save_increment`, does; its
copy may be older than the file, so the record stays while the file on disk
still holds the path. A record from a scene that was never saved is dropped,
because no session can open it again. A parameter someone changed in between
keeps their value. `lint` reports anything left over as `frozen_after_run`.

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
