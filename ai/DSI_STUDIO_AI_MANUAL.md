# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` first. This manual is intentionally concise so an
agent can read it without losing the end of the file to output truncation. The
complete command inventory and source-verified examples are divided by topic and
linked near the end.

## Command routing reference — read this first

A `CMD` must target a numeric window ID returned by top-level `LIST`. Commands
are accepted only by the window type that implements them.

| Window type | What it represents | Common valid commands | File-opening role |
|---|---|---|---|
| **main** | Main DSI Studio window | `list_recent_fib`, `list_recent_src`, `open_image`, `hub ...`, `run_cli` | `open_image` is primarily for opening NIfTI and other image files for image viewing, modification, and editing. Do not use it to open `.fz` when the fiber-tracking interface is needed. |
| **image** | General image viewer | Image-viewer inspection and display commands | Used for NIfTI, DICOM, NRRD, and other ordinary image data opened by the main window. |
| **tracking** | A loaded FIB/FZ tracking window | `list_slice`, `list_region`, `list_tract`, `run_tracking`, `open_fib`, tract/region/slice/device/rendering commands | Use `open_fib` to open `.fz` or `*fib.gz` in the fiber-tracking interface. It is a tracking-window command and therefore requires an existing tracking window. |

Always call top-level `LIST` first and use the returned numeric ID. Never use a
window title, filename, type name, guessed number, or stale number as `window`.

## Opening files: three distinct routes

These routes are intentionally different and should not be treated as aliases.

### 1. Raw absolute path — simplest for one existing local file

Send one existing absolute path as raw non-JSON pipe text:

```text
C:\data\subject.fz
```

DSI Studio routes the file by extension. `.fz` and `*fib.gz` open as tracking
data; `.sz` and `*src.gz` open reconstruction; ordinary image formats open an
image window. Raw text is reserved for one local file path. Use this route to
create the first tracking window when no tracking window is currently open.

### 2. Tracking-window `open_fib` — fiber-tracking interface

Target an existing **tracking** window:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["open_fib","C:/data/second_subject.fz"]}
```

Use `open_fib` to open `.fz` or `*fib.gz` in a new fiber-tracking window. Because
`open_fib` is handled by a tracking window, create the first tracking window by
sending its absolute path as raw pipe text, then use `open_fib` for additional
FIB/FZ files.

### 3. Main-window `open_image` — NIfTI/image editing route

Target the **main** window:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"1","command":["open_image","C:/data/T1w.nii.gz"]}
```

`open_image` is primarily for opening NIfTI and other ordinary image files in an
image window for viewing, modification, and editing. It may receive multiple
image paths when those files should open together. Do not use `open_image` for
`.fz` when the fiber-tracking interface is required.

## Recommended request sequence

1. Send one concise `TITLE` after understanding the task.
2. Send top-level `LIST`.
3. Choose the numeric ID for the correct window type.
4. Run discovery commands before mutation.
5. Use `LIST` for routine polling and targeted `list_*` commands for detail.
6. Verify output files and report the result with `chat` or standalone `CHAT`.

## Request formats

### LIST

```json
{"agent":"Codex","session":"<uuid>","cwd":"C:/work","request":"LIST"}
```

The first line reports application-wide activity. Following lines contain:

```text
type    id    busy    tracking-jobs    title
```

Window `id` is the quoted numeric value required by every `CMD`.

### CMD

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["list_region"]}
```

Every command name and parameter must be a JSON string. Use `"7"`, not numeric
`7`.

A meaningful command should normally include a useful progress update:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","SynthSeg V2","7"],"chat":"I verified that the T1w slice is ready. I am starting SynthSeg now."}
```

The top-level `chat` field is shown to the user and does not change the command.
Silent polling may omit it.

### CHAT

```json
{"agent":"Codex","session":"<uuid>","request":"CHAT","chat":"Tracking completed and the output file was verified."}
```

Use standalone `CHAT` when no other request is needed.

### TITLE

```json
{"agent":"Codex","session":"<uuid>","request":"TITLE","title":"Corticospinal tract analysis"}
```

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

Use `LOG` only when `LIST` and targeted discovery cannot explain a failure.

## Critical command syntax

### `list_tract` does not require a numeric parameter

Full tract table:

```json
["list_tract"]
```

Compact tracking status:

```json
["list_tract","status"]
```

`list_tract` without a second element is valid and returns every bundle with
index, running state, shown state, name, tract count, deleted count, and seeds.
The optional literal string `"status"` returns only running-job and bundle
counts. A numeric tract index is **not** required for `list_tract`.

If a client reports `need-param1` for `["list_tract"]`, the request was likely
sent through an incompatible wrapper or malformed command interface. Send the
standard JSON `CMD` array directly to a tracking window.

### `run_tracking` requires a new bundle name

Minimum form:

```json
["run_tracking","CST"]
```

The second command element is mandatory and becomes the new tract-bundle name.
An empty name fails with `missing tract-bundle name`. With the two-element form,
DSI Studio uses the current tracking parameters and checked region settings.
Before running, use `["list_param","tracking"]` to show all tracking parameters
and their current values. Review these values before changing them or starting
tracking.

Typical sequence:

```json
["list_region"]
["list_param","tracking"]
["set_params","fa_threshold=0.08&min_length=20"]
["run_tracking","CST"]
```

Do not resend `run_tracking` merely because a client timeout occurred. Poll
`LIST`; fiber tracking is asynchronous after the command is accepted.

## Discovery quick reference

| Need | Command | Window |
|---|---|---|
| Open windows and activity | top-level `LIST` | none |
| Recent FIB/FZ paths | `["list_recent_fib"]` | main |
| Recent SRC/SZ paths | `["list_recent_src"]` | main |
| Slice names and readiness | `["list_slice"]` | tracking |
| Regions and ROI types | `["list_region"]` | tracking |
| Full tract table | `["list_tract"]` | tracking |
| Compact tract status | `["list_tract","status"]` | tracking |
| Parameter IDs | `["list_param"]` | tracking |
| Tracking parameters and current values | `["list_param","tracking"]` | tracking |
| One parameter value | `["list_param","fa_threshold"]` | tracking |
| Atlases | `["list_atlas"]` | tracking |
| Segmentation models | `["list_unet"]` | tracking |
| AutoTrack names | `["list_auto_tract"]` | tracking |

## Operational rules

- Each named-pipe connection sends one request, reads the complete reply, and closes.
- Reuse the exact nonempty `agent` and `session` values for the conversation.
- Native identities are `Codex` and `Claude`.
- Ollama-backed identities include the host, for example `Codex/Ollama(192.168.1.14)`.
- Inspect `LIST` before substantial loading, registration, segmentation, reconstruction, or tracking.
- Discover names, indices, and parameter IDs rather than guessing.
- Confirm destructive actions and overwrites.
- Do not answer modal dialogs remotely; tell the user what must be selected.
- `okay:true` means the command was accepted; asynchronous work may still be active.
- A disappeared window or `window not found` means the user likely closed it. Do not reopen it automatically.
- Do not expose private chain-of-thought. Report conclusions, actions, progress, and blockers.

## Complete command inventory and examples

The complete inventory is split into smaller files so agents can retrieve only
the relevant section without truncation:

- [Main window, Hub, FIB, workspace, settings, and parameters](DSI_STUDIO_AI_COMMAND_EXAMPLES_GENERAL.md)
- [Slices and segmentation](DSI_STUDIO_AI_COMMAND_EXAMPLES_SLICE.md)
- [Regions and tract-to-region analysis](DSI_STUDIO_AI_COMMAND_EXAMPLES_REGION.md)
- [Tracts, tracking, AutoTrack, clustering, recognition, and TDI](DSI_STUDIO_AI_COMMAND_EXAMPLES_TRACT.md)
- [Devices and AC-PC locators](DSI_STUDIO_AI_COMMAND_EXAMPLES_DEVICE.md)
- [Rendering, camera, surfaces, and display](DSI_STUDIO_AI_COMMAND_EXAMPLES_RENDERING.md)
- [Image-window and TIPL generic image operations](DSI_STUDIO_AI_COMMAND_EXAMPLES_IMAGE.md)

Rows with examples provide recommended or source-verified syntax. Blank example
cells preserve commands from the previous complete manual without inventing
parameters. Search the appropriate topic file for the exact command and inspect
current source before using any blank-example command.