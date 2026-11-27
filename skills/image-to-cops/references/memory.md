# Baking and rendering without taking the machine down

Copernicus cooks on the GPU and keeps every cooked layer. On a machine with shared memory this does not show up in the process's own numbers, and when it runs out the whole system stalls. These rules come from two hard crashes and several killed bakes.

## What went wrong, and what fixed it

| Symptom | Cause | Fix |
|---|---|---|
| Memory climbs and each bake gets slower across a batch, then the machine hangs | One long-lived session baking many materials; cached layers never leave | One process per material that exits when done |
| A single 8K bake of a large network exhausts memory | Every node's layers alive at once | Streamed cook: cook in dependency order and unload each node once nothing downstream needs it |
| Memory still explodes at write time, after the cook | The Image ROP re-cooks the network when it runs, and an output null only references its source's layer, so freeing the source forces a full re-cook | Write each map straight from the layer the moment its output cooks, before its source is freed |
| 32 bit layers use far more than expected | Default precision | 16 bit float layers; the maps were indistinguishable |

`scripts/bake.py` does all four. If you change it, keep those properties and measure again.

## How to run a batch

- One heavy job at a time. Never a bake and a render together, never two workers.
- Before starting, look for leftover Houdini processes and at free memory and swap.
- Run each step under `scripts/guard.sh`. It refuses to start below 45 percent free, samples every few seconds, and kills the job below 25 percent free or when swap grows by more than 3 GB. Judge swap by its growth: swap already in use stays allocated long after the memory is free.
- Watch the per item log. Steady time and steady free memory from item to item is healthy. Growth in either is the signal to stop, even if nothing has failed yet.
- A test that passes on one item does not clear a batch, and a test at one resolution does not clear the next. Trial one material at the delivery resolution under the guard before committing.
- When finished, confirm no Houdini process is left and say so.

## Figures to expect

On a 64 GB machine, with the streamed 16 bit bake: an 8K bake of a network of about 150 nodes took 45 to 85 seconds and never went below 76 percent free. An 8K look-dev render of three views took about 170 seconds at close to 90 percent free. If your numbers are far from these, something is holding memory.

## Other things that cost time

- Project folders on a syncing drive can be silently reverted to older copies after a crash. Keep a copy of edited scripts outside the synced folder, and when a working script fails oddly, check its size and modification time before debugging it.
- A reboot empties temp folders. Keep reference crops and anything the contact sheet needs inside the project.
- A reboot may also sign Houdini out of its licence server. A bake that dies in seconds with a licence message needs the person to sign in again; ask them.
