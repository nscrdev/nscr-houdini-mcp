# Reference images for the key strip brief

These images are synthetic. `make_references.py` builds the object in `../BRIEF.md` from its
numbers, projects it through the brief's camera presets with the Houdini camera model (50 mm
focal length, 41.4214 mm horizontal aperture) and draws it with Pillow. No Houdini render, photo
or outside image is involved, so the set carries no outside content.

## Files

| File | What it shows |
| --- | --- |
| `<preset>_line.png` | Visible edges in black on white, 960 x 540 |
| `<preset>_shaded.png` | Flat shaded faces, one fixed light, 960 x 540 |
| `<preset>_mask.png` | Object silhouette, white on black, 960 x 540 |
| `detail_chamfer_line.png`, `detail_chamfer_shaded.png` | The detail crop named in the brief, cut from the three_quarter view drawn at 3840 x 2160 |
| `variants/count_8_three_quarter_shaded.png` | The brief's control change: 8 keys |
| `variants/handoff_<n>_<change>_three_quarter_shaded.png` | The expected result after each step of `../HANDOFF.md`, applied in order |

Presets are `front`, `top` and `three_quarter`. Edges and silhouettes are anti aliased by drawing
at four times the size and averaging down. Masks are thresholded back to pure black and white.

## Regenerate

Pillow is the only dependency. From this folder:

```sh
uv run --no-project --with pillow python make_references.py --print
```

or, in any environment with Pillow installed, `python make_references.py --print`. The `--print`
flag reports each camera's rotation and the detail crop box, which are the numbers the brief
quotes. The output is deterministic for a given Pillow version.

The numbers live in two places: the `Design` and `CAMERAS` values in the script and the text of
`../BRIEF.md`. Change both together, regenerate, and commit the images with the change.
