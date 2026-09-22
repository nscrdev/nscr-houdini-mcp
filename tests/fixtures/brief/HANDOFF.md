# Handoff edit: protocol and record sheet

This sheet is used after an agent has finished the key strip brief. It measures one thing: can
someone who was not there open the agent's scene and change it. It is written to be run from this
page alone. Do not read the brief, the agent's messages or its notes outside the scene first.

## What you need

- The agent's saved `key_strip.hip` from one run.
- The expected result images in `references/variants/`, named in each step below.
- A timer, and Houdini started fresh: a new process, not the session the agent used.

## Steps

1. Open `key_strip.hip`. Start the timer when the file has finished loading.
2. Make the three changes below in order, each on top of the one before. Use whatever the scene
   offers: promoted controls, notes, node names. For each change, note the time when you make the
   first edit that turns out to be the right one, then the time when the result is correct.
3. Judge each result in the `three_quarter` camera against the named image. Shape, count,
   spacing and proportion should match; lighting and colour will not.
4. Give each change at most 10 minutes. If it is not done by then, record it as not done. Go on
   to the next change only if you can still reach the expected state; otherwise stop and record
   the remaining changes as not attempted.

| # | Change | Expected image |
| --- | --- | --- |
| 1 | Part count: 6 keys become 5. The row still fills the same length, so the keys spread out. | `handoff_1_count_three_quarter_shaded.png` |
| 2 | Spacing: bring the keys closer, 0.04 centre to centre, the row still centred on the housing. Under the brief's rule this is an end margin of 0.06. | `handoff_2_spacing_three_quarter_shaded.png` |
| 3 | Overall proportion: housing depth from 0.09 to 0.12 and height from 0.035 to 0.025. The keys stay seated on the top and centred in depth. | `handoff_3_proportion_three_quarter_shaded.png` |

What to record per change:

- Time to find the controls: from the start of this change to the first right edit, in seconds.
- Edits needed: parameters changed, plus any node created, deleted or rewired, and any code
  edited. One parameter per change is the best case.
- Breakage: errors or warnings, anything that stopped cooking, keys floating, sunk or
  overlapping, the chamfer lost or changed, the wrong key count.
- Still looks right: yes, partly or no against the expected image, with a few words on why.

## Reopen in a fresh process

After change 3, save as a new file. Quit Houdini. Open the new file in a new process (a GUI
session or `hython`) and record its dependency report: whatever report the setup provides, or one
gathered by hand. Record load errors and warnings, node types that are not installed, every file
path a parameter refers to and whether it exists, whether `OUT` cooks without error, and the point
and primitive count at `OUT`.

## Record sheet

Run id: ______  Condition (setup, skill on or off): ______  Handoff by (person or fresh agent): ______
Date: ______  Houdini build: ______  Scene file: ______

| # | Change | Time to find controls (s) | Time to done (s) | Edits needed | Breakage | Still looks right | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Part count |  |  |  |  |  |  |
| 2 | Spacing |  |  |  |  |  |  |
| 3 | Overall proportion |  |  |  |  |  |  |

| Reopen check | Result |
| --- | --- |
| Load errors and warnings |  |
| Node types not installed |  |
| File paths referenced, and whether each exists |  |
| `OUT` cooks without error |  |
| Points and primitives at `OUT` |  |

## A single run proves nothing

Fill one sheet per run. A condition is only described once all three of its runs have a sheet,
and the report gives all three results, not the best one or an average alone. One good or bad run
is an observation, never a conclusion.
