# Task two: edit an existing scene

The agent is handed a small saved scene and asked for one change. The point of the task is to read
what is already there and change it in place. Deleting the setup and building a new one fails the
task even when the result looks right.

## The starter scene

`vent_panel_start.hip`, built once from this description and then kept fixed. Units are metres,
Y is up, the panel faces +Z.

`/obj/vent_panel`, a geometry object, contains one chain from top to bottom:

| Node | Type | Setup |
| --- | --- | --- |
| `CONTROLS` | null | Spare parameters: `panel_width` 0.30, `panel_height` 0.18, `thickness` 0.01, `slot_count` 5, `slot_length` 0.20, `slot_width` 0.012 |
| `plate` | box | Size from `CONTROLS`: width, height, thickness. Centred on the origin |
| `slot_cutter` | box | One slot: `slot_length` by `slot_width` by 0.03 deep, so it passes through the plate |
| `slot_positions` | line | Along +Y, point count from `CONTROLS` `slot_count`, length typed in as 0.096, origin Y set by expression to minus half the length so the column stays centred |
| `copy_slots` | copytopoints | `slot_cutter` onto `slot_positions` |
| `cut_slots` | boolean | `plate` minus `copy_slots` |
| `soften_edges` | polybevel | The seam edges from `cut_slots`, distance 0.001 |
| `OUT` | null | Display and render flags |

A sticky note beside `CONTROLS` reads: "Vent: 5 slots at 0.024 pitch."

`/obj/backplate`, a second geometry object: one box sized from `/obj/vent_panel/CONTROLS`
`panel_width` and `panel_height`, 0.004 thick, 0.01 behind the panel. It is not part of the
request and must come through the change unchanged.

`/obj/cam_front`: position 0, 0, 0.8, looking at the origin, focal length 50 mm, 960 x 540.

The spacing is deliberately not a control. It is baked into the line length: 4 gaps of 0.024
make 0.096. Changing only the count squeezes or stretches the slots within that fixed length.

## The request, as given to the agent

"Change the vent to 7 slots with a centre to centre spacing of 0.018, still centred on the
panel. Edit the existing setup rather than rebuilding it. Save the result as a new version next
to the original and show me the front view before and after."

## What a correct result looks like

- 7 slots, pitch 0.018, the column 0.12 tall from the top edge of the top slot to the bottom
  edge of the bottom slot, centred on the panel.
- The existing nodes are still there, with the same names. New nodes or parameters only where
  they make the relationship clearer, such as a promoted spacing control that drives the line
  length as (count - 1) x spacing.
- `/obj/backplate` is unchanged.
- The sticky note matches the new state, or says where the numbers now live.
- `vent_panel_start.hip` is untouched; the result is a new file beside it.
- A before and after compare of `cam_front`, with the same camera and settings on both sides,
  saved on disk, whose difference sits only in the slot area.

## What to record

Nodes deleted, nodes created, parameters changed, whether spacing became a control, whether the
note was updated, whether anything outside the slot area changed, whether the original file was
overwritten, and whether the closing message names the compare result.
