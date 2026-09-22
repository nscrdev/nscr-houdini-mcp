# Evaluation method

The question being measured: does an agent working through this server leave a Houdini scene that
a person can understand and change? A good looking image is necessary, not sufficient.

## Fixed material

Everything lives in `tests/fixtures/brief/` and is fixed before any run:

- `BRIEF.md`: one small modeling task with repeated parts, one adjustable relationship and one
  small detail that needs a close look, with units, camera presets and what "done" means.
- `references/`: synthetic reference images for the brief's views and detail crop, drawn by a
  script from the brief's numbers.
- `EXISTING_SCENE.md`: a second task that edits a small saved scene in place.
- `HANDOFF.md`: the protocol and record sheet for the handoff edit.

The material does not change between conditions. If it must change, runs before and after the
change are not compared with each other.

## Conditions and runs

Each setup under test is run with the skills installed and without them. Every condition gets at
least three runs, each from a fresh agent context and a fresh Houdini session, with the same brief
text, the same references and the same model settings. Failed and abandoned runs are recorded
like any other.

After each run: the handoff edit by a person, the handoff edit by a fresh agent given only the
sheet, and a reopen of the saved file in a fresh process.

## What each measurement means

- Compare before done: the closing message names a saved compare result for every acceptance
  view. Without it the run is not done, whatever the message claims.
- Shared construction: the repeated part is built once and placed many times. If a change to the
  part or its count needs edits in more than one place, the construction is duplicated, whatever
  the node count.
- Attempts on the detail: how many separate passes the run made on the chamfer. Many passes on a
  small detail usually mean the reference was not looked at closely before building it.
- Checked before use: node types and parameters new to the run were looked up in the installed
  Houdini first. Errors from wrong names are counted.
- Control change: the count change in the brief was made through the promoted control alone.
- Handoff edit, the primary measure: time to find the controls, edits needed, breakage, and
  whether the result still looks right, for each of the three changes on the sheet.
- Reopen: whether the fresh process loads the file with a clean dependency report.
- Cost: wall time, tool calls, tool errors and model tokens where the client reports them. This
  puts the other results in context; it is not a score on its own.
- Readability observations: named boundaries, named outputs, repeats shared where they should
  change together, tuned values promoted, notes where they help, debug branches kept off the
  output path. These are observations for a person to weigh, not a pass mark. A large readable
  graph can be better than a small one that hides everything in one code node, so node count
  alone measures nothing.

## Reading the results

Report every run, and for each condition give all three results rather than the best one. No
conclusion rests on a single run. When the difference between two conditions is smaller than the
spread inside one condition, the report says that no difference was shown.
