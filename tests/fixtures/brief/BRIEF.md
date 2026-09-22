# Brief: key strip

Build one small hard surface object in Houdini as a procedural setup that a person who has not
seen this brief can read and change. After the run, a person will change the key count, the
spacing and the overall proportion on their own, using only what you leave in the scene.

## Units and axes

One Houdini unit is one metre. Y is up. The object rests on the ground plane (Y = 0) and is
centred on the origin in X and Z. Length runs along X, depth along Z, and the front faces +Z.

## The object

- Housing: a box 0.32 long (X), 0.09 deep (Z) and 0.035 high (Y), resting on Y = 0.
- Keys: 6 identical square keys, 0.028 by 0.028, standing 0.012 above the housing top and
  centred on the housing in depth. Every key is the same part.
- Relationship: the key row starts and ends 0.03 in from each end of the housing (the end
  margin). The row between the margins is split into equal cells, one per key, and each key sits
  at the centre of its cell: pitch = (length - 2 x end_margin) / key_count, which is 0.04333 at
  the defaults, leaving a gap of 0.01533 between keys. Changing the count changes the spacing.
  The housing never grows to fit the keys.
- Detail: a 45 degree chamfer, 0.002 along each face, on the four top edges of every key. Not on
  the vertical key edges, not on the housing. It is about 4 pixels wide in the three quarter view,
  which is why it has its own crop.

## Controls

Promote these to the top of the setup, on the object or on one clearly named controls node at the
head of the network: key count, end margin, key size, key height, chamfer size, housing length,
depth and height. Names are yours; a person must find them without reading the graph.

## Views

All presets: focal length 50 mm, horizontal aperture 41.4214 mm (the Houdini default), 960 x 540,
square pixels, no roll. Position and look-at define each camera. The rotation column is the same
camera as Houdini rx and ry with the default rotate order (Rx Ry Rz) and rz = 0.

| Preset | Position | Look-at | Rotate rx, ry |
| --- | --- | --- | --- |
| front | 0, 0.06, 0.70 | 0, 0.03, 0 | -2.454, 0 |
| top | 0, 0.75, 0 | 0, 0, 0 | -90, 0 (image up is -Z) |
| three_quarter | 0.32, 0.24, 0.40 | 0, 0.02, 0 | -23.242, 38.660 |

Detail crop: the three_quarter camera rendered at 3840 x 2160, cropped to x 2388 to 3060 and
y 971 to 1349 (normalized 0.6219, 0.4495, 0.7969, 0.6245). It frames the key nearest the camera.

Acceptance views: the three presets and the detail crop at the default values, plus the
three_quarter view with 8 keys.

## References

In `references/`: `<preset>_line.png`, `<preset>_shaded.png` and `<preset>_mask.png` for each
preset, `detail_chamfer_line.png` and `detail_chamfer_shaded.png`, and
`variants/count_8_three_quarter_shaded.png` for the control change below. They are flat shaded
drawings, so compare shape, framing and proportion. Lighting and material will differ.

## Required in the run (not an order of work)

1. Before building, say which parts are shared and which controls express the design.
2. Check every node type and parameter you have not used in this run against the installed
   Houdini before relying on it.
3. Make a preview of each acceptance view.
4. Compare each acceptance view with its reference.
5. Change the key count from 6 to 8 through the promoted control alone, with no other edit,
   compare the three_quarter view with the 8 key variant, then set the count back to 6.

## Deliverable

- `key_strip.hip`, saved in the working folder you are given.
- `/obj/key_strip` whose final geometry node is a null named `OUT`, with the controls promoted.
- Three cameras, `cam_front`, `cam_top` and `cam_three_quarter`, set to the presets.
- The file opens in a fresh Houdini process with no missing node types and no missing files.

## What done means

Done is a compare result, not a description. Before claiming completion, the closing message
names, for every acceptance view, the compare result saved on disk (candidate image, reference
image and difference numbers), lists the largest remaining differences in plain words, and gives
the path of the saved hip. A closing message without these is an unfinished run.
