# Changelog

## 0.1.0

The first release: an MCP server for SideFX Houdini 22, the bridge that runs
inside Houdini, and one agent skill.

### The server

- Runs over stdio on Python 3.11 or newer and never imports `hou`. One server
  reaches every Houdini session on the machine, open GUI sessions and headless
  workers alike.
- Eleven tools, in a fixed order so a client can cache the list:
  - `hou_ping`: which session a call reaches, and whether it answers.
  - `hou_sessions`: every session with its state, and starting and stopping
    workers. It never closes a Houdini with a user interface.
  - `hou_scene`: read the scene, open a file, save in place, save the next
    increment, and report what failed to load.
  - `hou_inspect`: nodes, networks and parameters, summary first, paged, and
    without cooking unless asked to.
  - `hou_python`: Python inside a session with the whole `hou` API, with
    namespaces kept between calls, progress notes and cancelling.
  - `hou_jobs`: follow, wait on, cancel and list long running work.
  - `hou_node_type`: what a node type takes, from the running Houdini.
  - `hou_docs`: search and read Houdini's own documentation for the build in use.
  - `hou_compare`: a candidate image beside a reference, as pictures and numbers.
  - `hou_outputs`: managed output paths, the record of what a scene made, and
    a lint of where a scene writes.
  - `hou_capture`: pictures of what a session shows, from one view, four views,
    a turntable or a frame range.
- Errors come back with a code, a message and a hint, and never with a place
  on disk. Large results are written to a file and returned as its path.
- A change sent again under the same operation id is answered from its receipt
  rather than run twice.
- Calls to a Houdini with a user interface are paced so an agent cannot keep
  its interface busy without a break.
- A log in the state folder and on standard error, with one warning for every
  call that ends in an error, naming the tool and the code only.
  `NSCR_MCP_LOG_LEVEL` sets the level.

### The Houdini side

- A bridge on loopback only, with every request and answer signed, one call at
  a time per Houdini, and every call in a GUI session run on its main thread.
- A scene counter that moves whenever the scene is replaced, so a call written
  against a scene that has gone is refused rather than run.
- A pool of hython workers kept warm for heavy work, within a cap.

### The command line

- `bridge install`, `uninstall`, `status` and `snippet` write and check the
  Houdini package, with optional autostart.
- `bridge worker start`, `stop`, `list`, `reserve` and `release` run the pool.
- `skills path` and `skills install` find and copy the shipped skills.
- `config show` and `config init` read and write the server's config file.

### Fixed before release

- `bridge install` from an installed wheel put the environment's whole
  site-packages folder in front of Houdini's `PYTHONPATH`, so numpy built for
  another Python, and every other library there, loaded in place of Houdini's
  own and broke Houdini's own tools on start. Houdini is now given a folder
  holding this package alone: `src/` in a checkout, and otherwise a copy made
  under the state folder, refreshed on every install and removed on uninstall.
  Workers the pool starts, and `bridge snippet`, get the same.

### The skill

- `houdini-artist`: how an agent plans, builds, checks its work against a
  reference and hands a scene on, with a block of house conventions to edit.

### Known limits

- A coordination store held by another process can stall a call in a GUI
  session, and Houdini's interface with it, for up to the store's busy
  timeout of ten seconds.

### Tested on

- macOS on Apple silicon with Houdini 22.0.368 and 22.0.429.
- Windows and Linux run the unit tests in CI only, not yet against a real
  Houdini.
