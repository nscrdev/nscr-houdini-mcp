---
name: image-to-cops
description: "Turn a material seen in a photo, film still or batch of reference images into a tileable procedural PBR material built in Houdini Copernicus, then prove it on a shader ball beside its reference. Use whenever someone wants to recreate, extract, match or 'make a material from' an image, asks for a texture or material pack from references, wants procedural or tileable maps for a surface in a picture (concrete, metal, wood, plaster, fabric, stone, paint, anything), or asks for a reference versus material contact sheet. Also use when a procedural material looks flat, samey, fake or low in detail and needs realism, bump and specular work."
compatibility: "Written for the nscr-houdini-mcp server and Houdini 22. The memory guard and contact sheet scripts are written for macOS."
---

## What this produces

For each material: a Copernicus network built from layered procedural signals, seamless over one tile, baked to six EXR maps, rendered in Karma XPU on Houdini's own shader ball scene under one of its HDRIs, and laid beside the reference on a contact sheet the person can mark up. The loop is build, bake, render, compare, revise, until the sheet convinces.

Read `houdini-artist` first if it is available; this skill is that way of working applied to one job.

## Before building: know which material

An image usually holds several surfaces. If the person did not say which one they want, ask before doing anything else, and ask with concrete choices taken from what is in the image ("the floor", "the left wall", "the painted column"), one question per image or one for the batch. Guessing wastes a full build on the wrong surface, and when a later note says "look at the floor here" the whole material has to be rebuilt.

Then look hard at that one surface and write down, in plain words, what happened to it: how it was made (cast, poured, sawn, rolled, brushed, sprayed), what its joints and fixings are and how they repeat, what has aged it (water, dust, traffic, rust, sun, salt, flaking), and how it takes light (matte, burnished, wet in places). Those sentences become the recipe. Also fix the real size of one tile in metres; every later number hangs off it.

## Build each material in layers

Use `scripts/cops_lib.py`. A material is a function that takes a parent node and a resolution and returns the builder. `B` wraps node creation, noises, small OpenCL kernels and controls; `Material` carries the running state: height in millimetres, colour layers, tone multipliers, roughness deltas and specular weight deltas. `examples/concrete_recipes.py` is a worked set of nineteen; read two or three recipes there before writing your own.

- One layer function per physical cause, and a recipe that reads as the list of causes. Never one node or one preset. A stock grunge node on its own reads as a preset the moment it tiles.
- Carry height in millimetres and sizes in metres, and convert at the end. Real numbers keep twenty materials consistent and make a note like "deeper seams" a one line change.
- Every signal must tile. Use the tiling noises and the periodic kernels in the library; check with a 3x3 view, not by eye on one tile.
- Give each material its own damage. The same scatter of holes on every material is the first thing a viewer notices. Put damage where it physically gathers: along joints, at edges, under leaks, in traffic lanes.
- Put the controls a person would reach for on the yellow `CONTROLS` null and name the outputs `OUT_basecolor`, `OUT_roughness`, `OUT_specular`, `OUT_height`, `OUT_normal`, `OUT_ao`.

Read `references/realism.md` before the first build and again when a render looks fake. It holds what actually moved these materials from procedural to believable.

## Bake and render, one material per process

Run from a shell, with `hython` from the Houdini install:

```text
hython scripts/bake.py <recipes.py> <index or function> <resolution> <texdir> b16
hython scripts/lookdev.py <texdir> <MaterialName> <renderdir>
scripts/sheet.sh <renderdir> <refdir> <out.jpg>
```

`scripts/run_all.sh` does bake then render for a whole recipe file, each step under `scripts/guard.sh`. Iterate at 2K or 4K; confirm at the delivery resolution before calling a material done, because fine grain that is invisible at 2K is the point at 8K.

Each bake and each render is its own process that exits. This is not tidiness: a long-lived session baking material after material grows until the machine hangs. Read `references/memory.md` before any batch, and before changing how the bake works.

The maps: base colour as linear Rec.709, tagged so the renderer converts it to the working space; roughness, specular weight, height, ambient occlusion as single channel data; normal as data. Height is stored 0 to 1 around 0.5, with the range in millimetres written to `meta.json` next to the maps.

## Look-dev that shows the truth

`lookdev.py` renders three views, because each answers a different question:

- `ball`: Houdini's shipped material look-dev scene, with its own light box switched off and an HDRI that ships with Houdini lighting it. The tile repeats 2x2 over the ball. This is where specular character shows.
- `close`: a raking close-up of a wall. This is where bump shows.
- `front`: an orthographic 3x3 of the tile. This is where repetition shows.

Rotate the HDRI until the ball carries a visible highlight and the close-up is lit across the surface, not at it. A material judged under flat light looks flat whatever its maps say. Render in the scene's working colour space and convert with the display view the person actually uses; the script reads both from the active OCIO config. Use at least four diffuse bounces, since the renderer's default of one leaves shadows black.

## Compare, critique, revise

Build the contact sheet with `sheet.sh`: reference, ball, close-up, 3x3, one row per material. Keep the reference crops in the project, not a temp folder. Then look at each row and write the differences, largest first, before touching anything. Send the sheet to the person and, where a markup tool is available, open it there so they can point at things.

When notes come back, change the layer that owns the cause, re-bake only the materials touched, and show before and after. If the notes are about a quality every material shares, get it right on two contrasting materials first, have that approved, then roll it out. It is far cheaper than re-rendering twenty materials per round.

Keep to the revision budget: three attempts on one detail, then show what you have and say what each attempt taught.

## Before you say done

- The contact sheet path, and for each material the largest remaining difference from its reference in plain words.
- The 3x3 view checked for seams and for repetition, and which materials still read as repeating.
- The resolution and bit depth of the maps, the tile size in metres and the displacement range in millimetres per material.
- Every process started and confirmation that each has exited.
- What is unverified: anything not rendered at the delivery resolution, any material not shown to the person.
