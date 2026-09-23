# nscr-houdini-mcp

An MCP server and a small set of agent skills for SideFX Houdini 22.

Status: 0.1.0, the first release. The eleven tools described below work
against a real Houdini 22 on macOS; see [Tested on](#tested-on) for what has
and has not been run where. Names and arguments may still change before 1.0,
and [CHANGELOG.md](CHANGELOG.md) says what changed.

## Goals

- A small tool set with direct Python access to Houdini, so the context cost stays low.
- Works with any MCP client. No client-specific rules.
- One setup talks to many Houdini sessions at once, both open GUI sessions and headless workers.
- Agents check their work against a reference image before they call it done.
- Renders, caches and captures go to managed folders next to the scene file.
- Plain `SKILL.md` skills that help an agent build scenes a person can read, change and reuse. You can edit them to fit how you work.
- Runs on macOS, Windows and Linux.

## Tested on

- macOS on Apple silicon (arm64), with Houdini 22.0.368 and 22.0.429: the unit
  tests, the tests that start a real hython, and GUI sessions driven through
  one MCP client.
- Windows and Linux: the unit tests run in CI, on Python 3.11 and 3.13, for
  every push to main and every pull request. Neither has been run against a
  real Houdini yet, so treat Houdini on either as untried.

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
python scripts/tools_list_cost.py     # token cost of the tools/list payload
```

Before a release, check a clean install from a built wheel in a throwaway
environment, with no checkout in reach. It needs `uv`, writes only inside one
temporary folder, and prints each check:

```sh
python scripts/check_install.py
```

`scripts/tools_list_cost.py` serialises the tool list the way a client receives
it and counts its tokens, in total and per tool, against a budget of 4,000.
With the dev extras installed it counts with `tiktoken` and the `o200k_base`
encoding (`--encoding` names another), read from tiktoken's cache, or fetched
once with a ten second limit when it is not there. Without the package, or
with the encoding neither cached nor fetched, it falls back to bytes divided
by four, rounded up, and says which of the two happened. `tiktoken` is only ever a dev
dependency. Either count is a budget signal to watch across commits, since a
client on another encoding pays a somewhat different number. Pass
`--max-tokens N` to make it fail over a budget.

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

Eleven tools. `hou_ping` says which session a call reaches and that it
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
which are the server's own. The run it records names the call's job, so
`hou_compare` can find an image the job wrote by its `job_id`.
`mcp.freeze_parm(parm, path)` puts a path the call
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
ended, `outputs` is the answer the call would have given, fitted to
`max_chars`, and what does not fit is written to one spill file per job.
`cancel` writes the
request where the session reads it within two seconds and also asks the
session directly, and says whether that direct request got there. In a
session with a user interface a cancel is best effort: the code runs on
Houdini's main thread, and only code that looks at `mcp.cancelled()` stops.
`list` gives jobs newest first, by session and state, with `next_page`.

Jobs live in the coordination store, so any server can answer for any job,
including one started after the call that began it. A job's row is written
before its code may run, and the call is refused with `STORE_UNAVAILABLE`
when it cannot be; an operation id whose job is still kept cannot start
another (`JOB_ID_TAKEN`). The same call sent again while it runs is answered
at once with its job rather than queued behind itself. A job whose session ends,
whether stopped or found gone, is `lost` with the progress and outputs it had
written. Silence alone never ends a job, since a long cook can hold Houdini's
interpreter and a machine can sleep; a job still going says how long its
session has said nothing in `silent_s`. A job whose caller was handed the
job to follow, rather than its answer, leaves a readable copy of itself
beside the scene file when it ends, in `.agent/jobs/`, and nothing is made
beside a scene file that is not there. Jobs are kept for 7 days after they
end, and their copies go with them.

`hou_node_type` reads a node type from the running Houdini before any node of
it is made, so installed assets and this build's own versions are the ones
described. Name a `type` and a `context` (`obj`, `sop`, `lop`, `cop`, `dop`,
`top`, `chop`, `out`, `mat` or `vop`, or a category name such as `Sop`) for
its inputs and outputs with their labels and its real parameter names with
their defaults; `full` adds menus, ranges, folders, hidden parameters and the
first line of its help. A bare name is the version Houdini would make for it,
and `resolved_from` says which name was asked for. A multiparm carries its
instance template under `instances`, and a menu that a script fills in says
`dynamic` rather than running the script. `query` searches every type by
keyword: the exact name first, then a name that starts with it, one that
holds it, the label and the help line. Hidden types come only with `include`
`hidden`. A name that is not there is `TYPE_NOT_FOUND` with up to five near
names, or with the contexts that have it. Parameters and search rows page the
way `hou_inspect` does, and a page after any row or the asset's library
changed says `changed`. A page token is state the caller holds and can edit,
as with the other tools, not proof that this tool wrote it: it is checked for
shape and tied to its session and arguments, and no more.

Everything comes from the node type itself: nothing is cooked or made. An
asset's input labels come from its dialog script, and a type built into
Houdini takes them from the headings of its help page, which only
approximate what the node shows; `labels_from` says which it was. A ramp
says how many points it starts with in `default_points`. Help is the
asset's own embedded help, or a page from the `nodes.zip` Houdini ships or
from any `help/nodes` folder on Houdini's search path, which is where
packages keep theirs. A search with no context leaves out data recipes,
managers and the networks that only hold other contexts, and equally close
matches come in context order, SOPs first. Search ranks by the shipped and
package help pages only: an asset's embedded help is read for one card, or
for the rows a page returns, and never changes the order. A query is at most
200 characters and 16 different words, and a search asked to stop hands back
what it found with `stopped`. When the shipped help is missing or cannot be
read, the answer says `help_available: false`, and it is tried again a
minute later.

`hou_docs` reads Houdini's own documentation for the build in use. `search`
looks for `query` in page titles and first paragraphs and ranks a title that
is the query first, then one that starts with it, then one that holds it,
then a page whose first paragraph holds every word. For nodes, VEX functions
and Python classes the internal name, such as `attribwrangle`, counts as a
title. Among equals the current version of a page comes before older and
deprecated ones, and release notes come last. `page` reads one page by its
help path, such as `nodes/sop/attribwrangle`, or by a node type with its
namespace or version, such as `nodes/sop/copytopoints::2.0`; `vex` reads one
VEX function's page by its name. A page says its `version` when it has one.
Text comes `plain`, or `markdown` with its headings, lists and code kept;
table rows stay on one line. Past `max_chars` (20,000 unless you say) the
text is cut, says `truncated`, and the whole page goes to the spill folder,
named in `spill_path`.

Pages come from the install's own `houdini/help` folder first, which answers
in a few milliseconds and needs nothing from the session. With a session, that
is only a folder of exactly the session's build; with no session, the install
`hython` or `houdini_build` names in `config.toml`, or the newest on this
machine. The session's help server is read only for a build with no folder
here, or for a page the folder does not have. It runs inside the session, so
it is slow and stops answering while the session cooks or runs code: asking
where it is never waits behind other work, each request has two seconds from
start to finish, redirects are refused, and a help server that timed out is
left alone for a minute. `source` says which answered and `build` which build
was read; `HELP_UNAVAILABLE` says neither could. Pages from both are kept in
a small cache under the state folder, and search uses an index of every page,
built once per build and kept beside it; the first result's `note` says how
long that took. The cache holds at most 64 MB across builds, the builds used
least recently going first, and a build whose install has gone is removed.

`hou_compare` puts a candidate image beside a reference and says how far
apart they are, as pictures and as numbers. It never says pass or fail: there
is no match flag and no built in threshold. The `candidate` is one of four
sources:

- `file`: an image given by its absolute path; a path with nothing there is
  `FILE_NOT_FOUND`.
- `viewport` or `node`: a picture captured for the compare, through the same
  code as `hou_capture` and with the capture arguments the candidate carries
  (`path`, `camera`, `frame_target`, `display`, `resolution`, `frame`,
  `region`, and `timeout_s` for how long to wait for it). It is a capture
  like any other, under an operation id made from the compare's own with
  `:capture` added: a run in the `capture` folder and a job `hou_jobs` can
  read. The reference and any `mask` are read first, so one that will not do
  costs no capture. When the reference was registered with a camera, that
  camera frames the capture unless the candidate names its own camera, and
  the capture is made at the reference's aspect (its own size, or 2048
  pixels on the long edge when larger) unless the candidate names its own
  size, which then sets the size alone. A camera the reference names that is
  gone or is not a camera is the reference's error (`argument: reference`,
  with the camera in the details). A capture whose aspect is off the
  reference's by more than a pixel says so in `warnings`, beside the
  capture's own warnings. A capture that runs past its time answers
  `TIMEOUT` with its `job_id`: wait for it with `hou_jobs`, then compare
  with a `render` candidate of that job.
- `render`: the newest image on disk that a finished job wrote (`job_id`),
  or that runs of a node wrote (`path`, the node), read from the run records.
  A capture job's image is the one its answer names; any other job's is the
  newest image among the runs it took, such as a path a `hou_python` job took
  with `mcp.output_path`. A job not yet ended is `JOB_RUNNING`, with a hint
  to wait on it with `hou_jobs`; a job or node with no image on disk is
  `NO_OUTPUT`.

The result's `sources.candidate` says which: for a capture its `run_id`,
`job_id`, `path`, `route`, `frame`, the `camera` that framed it and
`framed_by` (`reference`, `candidate` or `capture`), and `unsaved_hip` when
the scene had no file; for a render the run, the job and the node.
`result.json` keeps the same, with the path written relative to itself. The work runs in the server with NumPy and Pillow, in a fixed order,
and the result records every step:

1. Colour. A PNG, JPEG or TIFF is converted to sRGB through its embedded ICC
   profile; with none, sRGB is assumed and the result says `assumed_srgb`.
   The full scale of a sample comes from the file's bit depth, never from its
   pixels: 16 bit greys and 16 bit RGB or RGBA PNGs are read at 16 bits, and
   the colour record says how many bits the file had and how many were read.
   An EXR or HDR file is read by the session, with OpenImageIO when the
   session has it and otherwise through a COP `file` node made and removed
   for the purpose (which a session with a user interface counts as a change
   to the scene, and the result says so). Its colour is divided by alpha and
   brought to display values with the session's OpenColorIO display and
   view. The result names the configuration by file name and content hash,
   the display, view, exposure, channels and what was done with alpha.
   `transfer_mismatch_possible` is set when a scene linear side went through
   a view that is not a plain sRGB display, when two scene linear sides went
   through different views, or when the mean luminance of the two sides is
   more than 25 percent apart after alignment.
2. Alignment at native size: `align` (`fit` letterboxes to the reference's
   aspect, `fill` crops to cover it, `stretch`, `none` keeps the candidate's
   pixels one for one), then `adjust` (`dx` and `dy` from -1 to 1, in shares
   of the frame, and `scale` from 0.05), then `auto_shift`, a translation
   found by phase correlation and applied only when it lowers the mean
   difference where the candidate covers; otherwise `shift_px` says
   `applied: false` with the reason, the estimate and both errors. A candidate with more pixels than the
   reference keeps them: the grid grows instead, up to four times. Only the
   part of an enlarged candidate that lands on the frame is ever made. There
   is no fixed largest `scale`: one that would place the candidate over four
   times the frame's area is refused, which for a candidate that already
   fills the frame is a little over 2.
3. Crops before any shrinking. Named regions and `region`, each
   `[x0, y0, x1, y1]` in shares of the frame, are cut from the aligned native
   images, and a detail crop's numbers are counted at that size, a block of
   rows at a time.
4. The overview: both sides shrunk to a long edge of 1024, or kept as they
   are when smaller, for the whole image numbers, the difference map and the
   sheet.

An image over 64 million pixels is shrunk by a whole factor as it is read, and
the result says `resized_on_read`. One past Pillow's own size limit is
`IMAGE_TOO_LARGE`.

The numbers are mean absolute error and RMSE per channel and overall, PSNR,
the share of the counted area whose difference is over `tolerance` (0.05
unless you say), and a rough box around the largest area of difference.
`match_exposure` adds the same numbers after one gain evens out the mean
luminance. In `likeness` mode, the default, they are labelled secondary,
because lighting and framing move them; in `regression` mode they are
primary. `mask` limits the counted area to the reference's registered mask or
alpha, or to a mask file; the candidate's alpha never decides it, and a
candidate whose alpha covers between 5 and 95 percent of it, with no mask,
gets a warning, because its transparent pixels count as the colour they
store. Numbers that cannot be
counted are named in `missing` with the reason, and the aligned pair and the
sheet are written all the same.

Every compare writes `candidate.png` and `reference.png` (the aligned pair),
`diff.png`, `overview.jpg` (candidate, reference and difference, labelled),
`crops/<region>.png` and `result.json` to a `compare` folder. `result.json`
names every file relative to itself and holds no place on the machine other
than the scene's own path. The call returns the overview as an image for the
person as well as the agent (`return_image` `thumb`, `full` or `none`). The
structured result, the text and the image of a reply stay within one
megabyte together: a full sheet that would not fit is sent as the thumbnail,
with `image_downgraded`.

`set_reference` copies an image into `$HIP/.agent/reference/` and writes one
record beside it, `<ref_id>.json`, that is never written again: the content
hash, the colour profile or the assumption made (an EXR or HDR is read as far
as its header), and optionally the camera it was framed for, named `regions`
and a mask. The record names its image and mask relative to its own folder,
so a project that moves still finds them. A name may hold letters, digits,
underscore and dash; any other name is refused rather than changed.
Registering a name again makes a new record and id. `list_references` lists
the names, and `hou_scene` `info` at `full` detail lists them too, so a fresh
context finds the goal. A name is looked up before a path, and only an
absolute path is read as a file. The reference folder is fixed per scene: a
`reference` template whose folder part uses `<name>`, `<run_id>`, `<date>`,
`<date_iso>`, `<time>`, `<ver>` or `<session>` is refused when the
conventions are read. Compares with the same reference id and hash, camera,
crop, mask, colour records and settings form a series, and for a captured
candidate the same size, display, frame and region too; each run of a series
is one file under `reference/series/<series_id>/`, so two servers writing to
one shared folder never interleave. A result in a series with earlier runs
carries the trend, and one that starts a new series says which of those
changed.

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

`hou_capture` saves a picture of what a session shows: the `viewport`, one
`node` on its own, the `network` editor, a `cop` output or a `pane` by name.
Every file goes under the `capture` kind of the output table below, and the
answer carries the path, the width and height, the frame, the camera, the
`route` that made it and `image_stats`: the mean, least and most of each
channel at the image's own depth, `flat` for an image that is one value in
every channel, and `non_empty`, which is false for an empty file and, for the
viewport and a node, for an alpha that is zero everywhere. A flat image is a
picture, with a note. A capture whose every image is empty is
`CAPTURE_EMPTY`. A thumbnail of at most 512 pixels on its long edge comes back
as image content beside the text; `return_image` says `thumb`, `full` or
`none`.

The routes for the viewport, in order: the Scene Viewer that is showing,
flipbooked with settings of its own (no MPlay, the beauty pass only unless
`guides`); an existing Scene Viewer made the current tab for the capture and
then put back; and a flipbook render node made for the capture and taken
away after. A `camera`, `display` or `frame_target` asked for is applied for
the capture and the view is put back as it was, camera, pivot and width
included. `frame_target: "all"` (the default for a view turned or fitted for
the capture) frames the geometry of the objects shown at the frames captured
(the first and last of a sequence together), never a camera, light or a null
that draws its stock cross; a simulation network is drawn but not framed, and
with no geometry the origin is framed with a warning. Without `guides` both
routes draw every object but those guides, so a null's cross stays out of the
picture. A Scene Viewer inside a geometry network or showing a stage keeps
Houdini's own frame all. The render node looks through a camera made for the capture:
one that follows a named camera and reads its lens by reference, so the named
camera is never written, or one fitted to the target's bounds from `persp`,
`top`, `front`, `right` or an `{orbit, elevation}`. A worker is started on
Qt's offscreen screen plugin, so it draws at one pixel to a point on any
display. In a session with a user interface that
route cannot know what the artist's view frames, so it says
`framing_unverified`; it is the only route a worker has. `node` draws that
node's object alone with the node carrying the display flag, and puts the
flag back. Whatever a capture makes for itself is made and taken away with
undo turned off. A route that fails part way takes its frames with it; a
capture stopped on request keeps the frames it wrote and lists them, and the
run record names each file, the capture's job and, for a `node` or `cop`
capture, the node, which is how `hou_compare` finds a node's newest picture.
The network editor and panes are made the
current tab and grabbed from their own window, which needs a user interface;
a worker answers `UI_UNAVAILABLE`. A render Houdini stops with an error is
`CAPTURE_FAILED`, with that error in the details. Every step that puts the
scene or the view back is tried, and one that fails makes a capture that
worked `CLEANUP_FAILED`, naming the step. Crops and the contact sheet are
made once per operation, so a reply sent again and a job read at the same
moment do not both write them. A viewport sequence goes a few frames at a
time, so it can be stopped between them, and its job row names each run and
the frames written so far.

`views: quad` captures persp, top, front and right, `turntable4` four orbits
a quarter turn apart, and both add a two by two contact sheet, which is then
`path`. `region` crops each saved image to `[x0, y0, x1, y1]`, fractions from
the top left. `frames: [start, end, step]` captures a sequence, which is a
job: it answers inline when it is done within `inline_wait_s`, and with the
job to follow in `hou_jobs` when it is not.

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

It stops when its input closes, or on the word `stop`. A start like this puts
Qt on its offscreen screen plugin unless `QT_QPA_PLATFORM` already names one,
as the pool does for its workers. On another plugin and a dense display a
render node draws larger than asked; a capture then reads the scale and makes
up for it, or says `framing_unverified` when it cannot.

### Pacing a Houdini with a user interface

In a session with a user interface every call runs on Houdini's main thread,
the one that draws the interface. Calls sent back to back leave it no room,
and an agent in a loop can send a great many, so the server paces the calls it
sends to such a session: one call out at a time, a pause after each one ends,
and a cap on how many start in a second. Two keys in `config.toml` set the pace:

```toml
gui_min_pause_ms = 50     # least time from one call ending to the next starting
gui_max_calls_per_s = 10  # most calls that may start in any one second
```

A value of 0 turns that rule off, and both at 0 turn pacing off altogether;
a negative value is refused.

The wait for a turn comes out of the call's own `wait_s` (a second unless it
says), and the bridge gets what is left. Calls sent side by side queue and go
out one after another, however long the call out may run. A call is answered
at once with `SESSION_BUSY` when the pause and the cap alone would hold it
past its wait, when it passed `skip_if_busy`, or when the queue is full; one
whose wait runs out in the queue gets the same answer then. So a call may be
refused up front when the calls queued ahead of it, at the rate cap, already
fill its `wait_s`, and `retry_after_s` says when to come back. Nothing of a
refused call is sent, so a caller that has given up never has its change made
later. `retry_after_s` is an estimate from the calls queued ahead and how
long the session's last few calls took, between the pause and 30 seconds; a
call out longer than usual is guessed to run as long again. The answer also
carries `queued_ahead`, how many calls waited ahead. A call that did wait
says how long in its trace, as `throttled_ms`, and when it was let through,
as `admitted_at`.

The pace is kept per server process and per session. Two agents sharing one
server process share its one allowance; two server processes, one per
client, each have their own, so together they get no more than twice it.
A call sent again under the operation id of the call that is out goes
straight through, since the bridge answers it from that call's receipt. At
most 32 calls wait on one session's turn; one more is answered
`SESSION_BUSY` at once, and a call whose client cancels leaves the queue
without being sent. Workers are headless and are not paced. A cancel is never
paced either, since it runs beside the call it stops.

The bridge has a limit of its own, whoever sends: it remembers the signed
requests of the last two minutes, 20,000 at most, and refuses one more with
`FLOOD_GUARD`, naming the limit and how many seconds to wait.

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

The file points `HOUDINI_PATH` at this copy's `houdini/` folder and
`PYTHONPATH` at the folder it is imported from: `src/` in a checkout,
site-packages in an installed copy, where `houdini/` travels inside the
package. Both are worked out from where this copy is running, so there is
nothing to edit by hand. Every file it writes carries a marker: a
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
half an hour. Routing a call to it renews the lease, and a worker running a
call or a job is never idle, however long the work takes and whether or not
anybody routes to it meanwhile. `stop` asks first and
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

### Running the sequence yourself

`tests/sequence.py` is one scripted pass over every tool, the way a client
would go about a first piece of work: list the tools, find a session, start a
worker, read its scene, build three nodes with a lost reply sent again under
the same operation id, read them back, look up `attribwrangle` and its help
page, take a cache path, save the next version, follow slow code that became a
job, compare two images and stop the worker. Every step checks the shape of
its result and records the tool, how long it took, and whether the reply
carried structured content, a text block and an image.

```sh
pytest -m houdini -s tests/test_sequence_hython.py
```

runs it three times, each with a server process of its own over stdio: under
the current protocol revision, under the older handshake revision, and through
the SDK's lower level client class. `-s` prints each run's steps. A last check
holds the three to the same shapes. The only difference the revision makes is
its own: the current one stamps every result with the server's details in its
metadata. Everything the pass writes stays in pytest's temporary folder.

`tests/test_storm_cap_hython.py` has two server processes hammer one worker
that is paced as if it had an interface, which only a test can ask for, through
the server's constructor, and prints the rates with and without the pace. Both
files carry their own deadline of 900 seconds, longer than the suite's usual
limit, for a slow Houdini to start in.

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
reference $HIP/.agent/reference/<name>_<run_id>.png
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

## Logs

Every log lives in the `logs` folder of the state folder:

| System  | Folder                                                                       |
|---------|------------------------------------------------------------------------------|
| macOS   | `~/Library/Application Support/nscr-houdini-mcp/logs`                        |
| Windows | `%LOCALAPPDATA%\nscr-houdini-mcp\logs`                                       |
| Linux   | `$XDG_STATE_HOME/nscr-houdini-mcp/logs`, or `~/.local/state/nscr-houdini-mcp/logs` |

`NSCR_MCP_HOME` moves the whole state folder, and `state_home` in the config
file moves it for the server. Keep the server and the bridges pointing at the
same one.

What goes where:

- `server.log`: the MCP server. It also writes the same lines to standard
  error, which a client that starts the server over stdio usually keeps.
  Every call that ends in an error leaves one warning here with the tool and
  the error code and nothing else, for example `hou_scene refused:
  SESSION_BUSY`. A failure a tool reports without a code of ours, such as
  code in `hou_python` that raised, is `TOOL_REPORTED_ERROR`. The message,
  which can hold text from your own code, is written only at `debug`. Each
  line carries the process id, since several servers can share the file. The
  file starts again at `server.log.1` when a server starts and finds it over
  five megabytes.
- `<session id>.log`: one bridge inside one Houdini. What went wrong in a
  call, with its trace, stays here and never reaches the caller.
- `worker-<name>.log`: what a worker's hython printed, for a worker started
  from the pool.
- `autostart.log`: a bridge that failed to start with Houdini, when
  `NSCR_MCP_HOME` is set in that Houdini. Without it the one line goes to
  Houdini's own console.

`NSCR_MCP_LOG_LEVEL` sets how much the server writes: `debug`, `info`,
`warning` (the default) or `error`. Set it in the environment the client
starts the server with. It does not reach the bridge, worker and autostart
logs, which have no level and are written whatever it says.

## Skills

A skill is a plain `SKILL.md` file that tells an agent how to work: what to
plan before building, what to look at before calling something finished, and
when to stop and report. The first one, `houdini-artist`, covers modelling,
lookdev, cleaning up a network, matching a reference, finding out why a scene
is slow and getting a scene ready to hand on. It works best with this server
connected, and still helps with planning when it is not.

The skills ship with the package:

```sh
nscr-houdini-mcp skills path                # where the shipped skills are
nscr-houdini-mcp skills install <folder>    # copy them into a folder you name
```

Clients that read skills usually look in a skills folder in your user settings
or in the project, with one subfolder per skill. Point `install` at the one
yours uses. A copy already there that differs from the shipped one is left
alone, so your edits survive an upgrade; `--force` writes the shipped files
over it.

The skills are yours to edit. The block at the top, House conventions, is the
dial: the output marker, when to save an increment, how many attempts one
detail gets, what good enough means for this pass, what node colours mean and
where files may be written. Change those lines to fit how you work, and keep
the output marker in step with the `[conventions]` table above.

## License

MIT
