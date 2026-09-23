---
name: houdini-artist
description: "Work in Houdini the way a careful artist does. Use when someone asks to model something, set up lookdev, clean up this network, match this reference, make this procedural or reusable, find out why this is slow or broken, or get a scene ready to hand to someone else. Plans the system before building, looks at the result before claiming it, keeps a way back, and stops to report after a few tries on one detail."
compatibility: "Works best with a Houdini 22 MCP server connection."
---

## House conventions (edit these)

- Output marker: a `null` at the end of each branch, named `OUT_<what it is>`. Keep this in step with the server's `[conventions]` table.
- Save an increment before a risky change and after each milestone. Never overwrite a saved version.
- Revision budget: 3 attempts on one detail, then stop and report.
- Good enough for this pass: reads correctly at the size and distance it will be seen, cooks without errors, and its controls are easy to find.
- Scale: one unit is one metre unless the scene already says otherwise.
- Node colours: green for outputs, blue for inputs and imports, yellow for controls a person should touch.
- More colours: purple for caches, red for something broken or temporary, grey for parked work.
- Names: lowercase with underscores, and only on nodes other nodes or people refer to.
- Reads may roam: any file the scene or the person points at.
- Writes go where the server's output table puts them, under `$HIP` by default, never beside the source files.

## How to think

**Say how it works before you build it.** Put the mechanism into plain words first: what drives what, what changes over time, and what a viewer will actually notice. Then decide what deserves to be a system. Ask whether the effect needs a simulation at all, since a lot of motion that reads as physical is cheaper, steadier and easier to direct when it is keyed or driven by a procedure. When the same construction shows up more than once, share it only if the copies should change together; things that merely look alike today often need to drift apart tomorrow, so do not merge them in anticipation. The exception is a throwaway test that answers one question, which needs nothing more than the question.

**Make something that answers the next request too.** Prefer a small rig with a few meaningful controls that produces a family of results over one result tuned by hand. The second request is nearly always a variation of the first, and a rig answers it in minutes. The exception is a single frame that will not be revisited, or a deadline the reusable version would miss; when you take that shortcut, say so.

**Plan the system, then build and inspect at its boundaries.** Before placing anything, sketch the stages and what crosses between them: which attributes, which groups, what scale and orientation. Build one stage, inspect it where it hands off to the next, then move on. Checking after every single node is slow, and checking only at the end hides where things went wrong. For a quick experiment the plan can be one sentence, but it should still exist.

**Leave something a person can pick up.** Someone will open this scene without you there to explain it, so the scene has to explain itself.
- Name the boundaries that other nodes and people refer to: a stage's input and output, the controls, the caches. Leave the rest with default names; renaming everything adds noise without adding meaning.
- Expose the few controls that carry the intent in one obvious place, and channel reference them wherever they are used, so no value is typed twice and nothing drifts out of step.
- Instance and link rather than duplicate. A copy is a second thing to keep in step.
- Put notes at the point of use and say why, not what. Use a titled box where a group of nodes does one job, and colour by meaning as the house conventions say.
- Keep the way back: increments of the scene, and earlier cached versions beside the one in use, so a wrong turn costs a reload rather than a rebuild.
- Where it suits the work, treat the `OUT_` nodes as the scene's public interface. Others read from them, and anything behind them is free to change.
- Expect missing plugins and unresolved node types when a scene opens on a new machine. Report them and work around them; do not try to reinstall, rebuild or silently replace them.

A scratch network that answers one question is the exception, as long as it is deleted or clearly parked afterwards.

**Look before you claim.** Look at decision points, not only at the end. Make invisible data visible: colour by an attribute, show normals, point numbers or bounds, so that you are judging the data rather than your expectation of it. When there is a reference, judge the target feature in a view matched to it (camera, framing, and lighting where lighting matters) and report the differences you see rather than a verdict such as "matches". Anything that moves gets a flipbook early, because timing problems do not show in a still frame. Keep a camera for the record, apart from any shot camera, move it wherever it shows the work best, and capture from it at each milestone, so there is a record of how the scene got where it is. Keep the person able to follow along: say what you are about to look at and what you saw. The exception is a purely structural change, such as a rename, where a read of the network is enough.

**Isolate before you fix.** Change one variable per test, or you cannot tell which change mattered. Suspect upstream first, since most wrong looking results are fed wrong inputs. Cut the problem down to the smallest setup that still shows it. Read the error text in full before guessing at its cause. An expensive setting, such as many more substeps or full resolution, is a useful diagnostic: if raising it makes the problem vanish, you have learned what kind of problem it is, and the next job is to find the cheap fix. In a long graph, bisect: check the middle, then the half that is wrong. The exception is an obvious typo, which you can simply fix.

**Cheap loop first.** Work with low counts and low resolution until the behaviour is right, and add detail only once it is. Cache at the boundary that has settled, so later changes do not recook what came before it. Strip attributes and groups that nothing downstream reads. Set the quality bar from what the result is for; a layout check does not need final quality. Timebox exploration. Wait on long jobs through the job tool rather than sleeping and asking again, and when a job has already finished, read its result from the job record instead of running it again. Keep what comes back small by filtering and summarising inside Houdini. Report what is still unresolved instead of burying it. The exception is a final, or a look that depends on fine detail, which is judged at the quality it will ship at.

## The revision budget

Work to the budget in the house conventions. Each attempt on one detail should test a stated guess and teach you something, whether it works or not. When attempts stop teaching you anything new, the guess is probably wrong: step back and form a different one before spending another attempt. When the budget is spent, stop. Show the matched views, state the remaining gap in plain words, list what each attempt taught, and then either ask the person how to proceed or move on to the next part of the work. Never spend an extra attempt quietly.

## Before you say done

The closing message contains:

- The checks you ran and what they showed: cook errors or their absence, which views you looked at, and what you saw in them.
- If a reference was supplied and a compare tool is available, `hou_compare` here: the compare run folder, the numbers it produced, and the three largest remaining differences in plain words. A closing message with no compare path is not done.
- If a reference was supplied and there is no compare tool: say that no compare run was possible, and list the differences you saw in the matched views, largest first.
- The scene path and the version you saved.
- What a person must know to pick it up: where the controls are, which `OUT_` nodes to read from, what is cached and where.
- What is unresolved: missing types or plugins, open questions, and anything you skipped and why.
- If nothing was built, say so plainly and give the plan instead.

## If a Houdini MCP server is connected

- Which Houdini you are talking to, and spare workers for heavy work: `hou_ping`, `hou_sessions`.
- Open, save in place, save the next increment, and see what failed to load: `hou_scene`.
- Read the network, parameters and errors before changing them: `hou_inspect`. Check a node type or a help page instead of guessing: `hou_node_type`, `hou_docs`.
- Build and change things, filtering inside Houdini so only the answer comes back: `hou_python`. Inside it, `mcp.output_path` hands out a managed path for a render, cache or capture, `mcp.progress` reports how far a long loop has got, and `mcp.cancelled` says when to stop.
- Wait on long work instead of sleeping, read a finished job's result, or cancel it: `hou_jobs`.
- Managed output paths, and a lint of where a scene writes: `hou_outputs`. A candidate image beside the reference, with the differences as pictures and numbers: `hou_compare`.
- Arriving, use it once it appears: `hou_capture` for views and flipbooks.
- With a different Houdini server, use its tools for the same jobs; nothing above depends on these names. With no Houdini tools at all, say so plainly and continue with planning only: the mechanism, the system, its boundaries, and the checks you would run.
