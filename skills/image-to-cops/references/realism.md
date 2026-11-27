# What makes a procedural material read as real

These are the lessons from matching a set of materials to film stills, in the order they mattered. They are about surfaces in general; the examples happen to be concrete.

## Contents
1. Roughness and specular carry the material
2. Bump at every scale
3. Colour: small breakup beats big clouds
4. Damage belongs somewhere
5. Formwork, joints and fixings
6. Staining follows water and gravity
7. Judging: light it to see it
8. Faults to look for in your own work

## 1. Roughness and specular carry the material

A surface is recognised by how it takes light more than by its colour. Two materials with similar base colour, one chalky and one burnished, must look opposite on the ball. If they do not, the roughness and specular maps are doing nothing.

- Build roughness as its own structure, not as a copy of the colour. Give it a broad drift, a cloudy middle scale and a grain, none of them correlated with the base colour mask. Then let each physical layer add or subtract: polish lowers it, dust and exposed aggregate raise it.
- Drive specular weight with a map as well. Dust, open pores, flaked patches and crumbled edges kill the highlight; sealed, wet or trafficked areas restore it. Roughness alone cannot make a dusty patch go dead.
- Set the base values from the material, and be bold. Chalky cast concrete sat near roughness 0.78 and specular 0.45; a trafficked floor near 0.6 and 0.8 with polished lanes falling to 0.2. Mid values everywhere give the plastic look.
- Specular detail is the product of three things: bump, roughness and specular weight. A highlight that breaks up convincingly needs all three to vary at the same place.

## 2. Bump at every scale

Flat surfaces are never flat. Stack scales, each a fraction of the last: slab undulation over a metre, trowel or float marks over centimetres, sand grain at a few millimetres, paste texture around a millimetre, and pin pores below that. The finest two are a light touch, but they are what makes a close-up tactile, and they only exist in the bake at 4K and above.

Give features a profile, not a step. A seam is a narrow slot inside a shallow chamfered trough. A fixing hole has a proud, rough collar round it. A crack has crumbled shoulders either side. A flaked patch has a crisp edge and a rougher floor. Without the profile the feature reads as a drawn line; with it, it catches light.

When someone asks for "more bump", they usually mean more contrast at the small scales and at features, not a bigger overall height.

## 3. Colour: small breakup beats big clouds

Large soft clouds are what makes a tile read as a tile. Weight mottling towards small scales, keep the largest tonal feature well under a quarter of the tile, and lower the contrast of anything large. A material can look perfect in close-up and wrong at distance for exactly this reason, so always judge the 3x3 view.

Keep base colour in a believable range and let lighting do the rest. If the reference is graded, match the surface, not the grade.

## 4. Damage belongs somewhere

Scattering the same holes across every material is the fastest way to make a set look generated. Ask where this surface would actually fail and put the damage there: voids strung along board joints, chipped arrises at panel seams, honeycombing at pour lifts, flaking where a skin has lifted, cracks with spalled shoulders, pin holes in loose drifts. Keep it finer than feels natural; real damage is mostly small.

Give each material a different mix. If two materials share a damage setup, change one.

## 5. Formwork, joints and fixings

Made surfaces show how they were made, and it repeats regularly: panel grids, board widths, fixing holes on a set-out, day joints, saw cuts. Build these as periodic layouts so they tile exactly, then vary within them: a per panel shift in height, tone and gloss, slightly warped lines, the odd patch. Regular layout with irregular condition is what reads as real.

## 6. Staining follows water and gravity

Runs start at a joint, a fixing or a ledge and fall. Rust is a broad brown bleed, not a thin orange line. Mineral bloom is pale, soft and wide. Dirt gathers in pores and low spots. Wet areas are darker and glossier at once. Stretch and blur stains along the fall direction and break them with a fine streak so they do not look airbrushed.

## 7. Judging: light it to see it

- Flat, frontal light hides bump and specular completely. Use an HDRI and rotate it until the ball has a highlight and the close-up is raked.
- Use the same scene for every material so differences are the materials', not the lighting's.
- Check the reference for which surface was meant. A still often shows walls, floor and columns; build the one asked for.
- Renders in a wide working space need the viewer's display transform, or everything looks washed out or oversaturated and gets "fixed" in the wrong place.
- Bake at the delivery resolution before the final judgement. Detail finer than the look-dev resolution cannot be judged at it.

## 8. Faults to look for in your own work

- Every material has the same holes, the same clouds or the same streaks.
- The ball shows no highlight, or the same highlight on every material.
- Seams, cracks and holes read as dark lines with no relief.
- Big blotches that repeat in the 3x3 view.
- A stain that is a single thin saturated line.
- A surface that is uniformly rough: no lanes, no sheen drift, no dead patches.
- A read of the reference that the person corrects ("that is not wood, it is brown concrete"). When unsure what a surface is, ask.
