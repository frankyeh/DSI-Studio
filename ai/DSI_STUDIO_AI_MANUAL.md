# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` first. This manual is intentionally concise so an
agent can read it without losing the end of the file to output truncation. The
complete command inventory and source-verified examples are divided by topic and
linked near the end.

## Start with `TITLE`

After understanding the task, send one concise `TITLE` as the first request:

```json
{"session":"<session-uuid>","request":"TITLE","title":"Corticospinal tract analysis"}
```

Send another `TITLE` whenever the active task changes substantially. Repeated
`TITLE` requests update the displayed chat name while keeping the same session.
The `title` field is required; do not put it in `chat` or `text`, or include it
in `CMD`, `CHAT`, `LIST`, or `LOG`.

## Tracking and image window IDs: `LIST`

`main` is fixed and can be targeted directly. Call top-level `LIST` only when a
tracking or image window ID is needed:

```json
{"session":"<session-uuid>","request":"LIST"}
```

Example reply:

```json
{
  "status":"success",
  "application":{"status":"busy"},
  "windows":{
    "main":{"status":"idle","title":"DSI Studio"},
    "tracking7ff6ab123410":{"status":"busy","title":"subject.fz"},
    "image7ff6ab456780":{"status":"idle","title":"T1w.nii.gz"}
  }
}
```

Tracking and image keys append the window pointer address in lowercase
hexadecimal without `0x`. Copy the exact current key; do not construct an ID or
use a window title, filename, guessed ID, or stale ID. The ID is valid only while
that window remains open.

## Command routing reference

A `CMD` targets either the fixed `main` ID or an exact tracking/image ID returned
by `LIST`. Commands are accepted only by the window type that implements them.

| Window ID | What it represents | Common valid commands | File-opening role |
|---|---|---|---|
| `main` | Main DSI Studio window | `list_recent_fib`, `list_recent_src`, `reset_settings`, `set_work_dir`, `rename_dicom`, `rename_dicom_dir`, `convert_dicom_dir`, `bids_to_src`, `nifti_dir_to_src`, `collect_network_measures`, `open_src`, `open_dwi_nifti`, `open_dwi_dicom`, `open_dwi_2dseq`, `open_src_dir`, `open_fib`, `open_structural_tracking`, `open_template`, `create_db`, `create_average`, `open_db`, `open_connectometry`, `open_auto_track`, `open_nonlinear_registration`, `open_xnat`, `open_console`, `clear_recent_src`, `clear_recent_fib`, `qc_nii`, `qc_src`, `qc_fib`, `run_cli`, `open_image`, `open_ai`, `open_hub`, `hub_*` | Main-window opening commands may accept paths as command parameters. Without parameters, many open a local picker. Never send a filesystem path as the complete named-pipe request. |
| `image<hex-address>` | General image viewer | Image inspection, editing, and `segmentation` commands | Used mainly for standalone NIfTI editing and batch image processing, not as the default T1w-segmentation route when a related FIB is already open. |
| `tracking<hex-address>` | A loaded FIB/FZ tracking window | `list_slice`, `set_slice`, `list_unet`, `segment_brain`, `list_region`, `list_tract`, `run_tracking`, `open_fib`, tract/region/slice/device/rendering commands | Tracking-window `open_fib` requires an explicit `.fz` or `*fib.gz` path and creates another tracking window. |

Do not invent command names. To discover recent files, target **main** and use
these exact commands:

```json
["list_recent_fib"]
["list_recent_src"]
```

Use `list_recent_fib` for recent FIB/FZ files and `list_recent_src` for recent
SRC/SZ files. Do not substitute guessed names such as `recent_list`.

## Opening files: documented command routes

Use the documented command interface for file opening. Do not send a filesystem
path by itself as a named-pipe request.

### 1. Main-window open commands

These commands accept an optional path parameter. Without a path, they open a
local GUI dialog:

```json
["open_fib","C:/data/subject.fz"]
["open_structural_tracking","C:/data/T1w.nii.gz"]
["open_src","C:/data/subject.sz"]
["open_image","C:/data/T1w.nii.gz"]
```

Omit the path to use the corresponding local file picker:

```json
["open_fib"]
["open_structural_tracking"]
["open_src"]
["open_image"]
```

- Main-window `open_fib` opens a supplied `.fz`, `*fib.gz`, or `.dz` path, or opens a FIB picker when no path is supplied.
- `open_structural_tracking` passes a supplied `.nii.gz`, `.nii`, or `2dseq` image to `loadFib`, or opens a structural-image picker.
- `open_src` opens one or more supplied `.sz`, `*src.gz`, `.jpg`, or `.tif` inputs, or opens a file picker.
- `open_image` opens one or more supplied ordinary image paths, or opens an image picker.

Picker-based forms require local user interaction. See footnote 1 regarding
cancellation and verification.

After a FIB opens, use the
[fiber-tracking skill](DSI_STUDIO_AI_SKILL_FIBER_TRACKING.md) to choose the
tracking strategy. A newly opened FIB normally has no regions; do not call
`list_region` unless the task uses regions or regions were created, loaded, or
restored.

### 2. Tracking-window `open_fib` — explicit additional FIB

Call `LIST` to obtain the tracking-window ID, then target that exact ID:

```json
{"session":"<session-uuid>","request":"CMD","window":"tracking7ff6ab123410","command":{"cmd":"open_fib","param":"C:/data/second_subject.fz"}}
```

Both main and tracking windows accept `open_fib` with a path, but they are
different command implementations. Always use the exact current window ID.
Main-window `open_fib` opens the supplied FIB as a primary tracking window;
tracking-window `open_fib` opens an additional FIB from an existing tracking
window.

### 3. Main-window `open_image` — explicit image paths

Target **main** directly:

```json
{"session":"<session-uuid>","request":"CMD","window":"main","command":{"cmd":"open_image","param":"C:/data/T1w.nii.gz"}}
```

With one or more paths, `open_image` passes the files to a `view_image` window
for viewing, modification, and editing. Do not use this route for `.fz` when the
fiber-tracking interface is required. See footnote 3 regarding error reporting.

After opening a related FIB, do not open its T1w again in an image window merely
to segment it. Use tracking-window `segment_brain` so the generated regions stay
in the fiber-tracking workflow. Use the image-window route mainly for standalone
image editing or batch processing.

## Recommended request sequence

1. Send one concise `TITLE` after understanding the task.
2. Target `main` directly, or call `LIST` only when a tracking/image ID is needed.
3. Run relevant discovery commands before mutation.
4. Verify output files or created objects before reporting completion.
5. Send another `TITLE` if the active task changes substantially.

## Request formats

Send the exact resumable `session` UUID with each request. A session not already
known to DSI Studio must include `agent` in its first request; existing sessions
do not need it. An optional `chat` may accompany any request.
Attach an update directly to `CMD` when it describes that command; use standalone
`CHAT` otherwise.

### CMD

```json
{"session":"<session-uuid>","request":"CMD","window":"tracking7ff6ab123410","command":{"cmd":"list_region"}}
```

The `command` field accepts one command object or an array of command objects.
Each command object requires `cmd`. Omit `param` when the command has no
parameter. Use a scalar `param` for one parameter and an array for multiple
parameters in command order. Command names and text or path parameters are
strings. Send standalone numeric parameters as JSON numbers, for example `7`,
not `"7"`.

For main-window commands that accept multiple files, pass each path as a
separate element in the `param` array:

```json
{"cmd":"open_src","param":["C:/data/a.sz","C:/data/b.sz"]}
{"cmd":"open_image","param":["C:/data/T1w.nii.gz","C:/data/T2w.nii.gz"]}
{"cmd":"qc_fib","param":["C:/data/a.fz","C:/data/b.fz"]}
```

Do not combine multiple paths into one `&`-separated string.

Multiple commands execute sequentially in the same targeted window and stop
after the first error:

```json
[
  {"cmd":"list_slice"},
  {"cmd":"set_slice","param":7}
]
```

A meaningful command should normally include a useful progress update:

```json
{"session":"<session-uuid>","request":"CMD","window":"tracking7ff6ab123410","command":{"cmd":"segment_brain","param":["human_synthseg",7]},"chat":"I verified that the T1w slice is ready. I am starting SynthSeg now."}
```

The top-level `chat` field is shown to the user and does not change the command.
Silent polling may omit it.

Every reply has `status`; `CMD` puts one result per executed command in
`result`. Each result has its own `status`, and `cmd` identifies the command.

A command that produces text returns:

```json
{"status":"success","result":[{"cmd":"list_region","status":"success","output":"<command output>"}]}
```

A successful command with no captured text returns:

```json
{"status":"success","result":[{"cmd":"set_slice","status":"success","output":"completed"}]}
```

An executed command that fails includes `error`:

```json
{"status":"error","result":[{"cmd":"set_slice","status":"error","error":"<reason>"}]}
```

A request rejected before execution returns `status:"error"` with an `error`
field. Interpret the fields as follows:

- `status` is `success`, `error`, or `busy`.
- `cmd` identifies the executed command.
- `output` contains captured text or `completed` when no text was captured.
- `error` explains a failed request or command.
- A command batch stops after the first error.

A `success` response does not prove that asynchronous work has finished or
that a GUI-backed operation created the expected object. Verify the resulting
window, file, region, tract, slice status, or other documented state before
reporting completion.

For `list_*` commands, actual rows appear in `output`. If the response is
`completed`, the command produced no rows or no textual output.

### CHAT

```json
{"session":"<session-uuid>","request":"CHAT","chat":"Tracking completed and the output file was verified."}
```

Use standalone `CHAT` when no other request is needed.

### LOG

```json
{"session":"<session-uuid>","request":"LOG"}
```

Use `LOG` only when the direct `CMD` response and targeted discovery cannot
explain a failure.

## Main-window command reference

All commands in this table target **main**.

| Command | Behavior |
|---|---|
| `["list_recent_fib"]` | List saved recent FIB/FZ paths. |
| `["list_recent_src"]` | List saved recent SRC/SZ paths. |
| `["reset_settings"]` | Clear all application settings. Takes no arguments. |
| `["set_work_dir","C:/work"]` | Add a work directory. Without a parameter, open a directory picker. |
| `["rename_dicom","<file1>","<file2>"]` | Rename one or more DICOM files. Without file parameters, open a picker. |
| `["rename_dicom_dir","<directory>"]` | Rename DICOM files recursively. Without a parameter, open a directory picker. |
| `["convert_dicom_dir","<directory>"]` | Recursively convert DICOM series to SRC/SZ or NIfTI without overwriting existing output. |
| `["bids_to_src","<BIDS-directory>"]` | Find BIDS diffusion data, ask the local user for an output directory, and create SRC/SZ files. |
| `["nifti_dir_to_src","<directory>"]` | Find diffusion NIfTI data and create SRC/SZ files in the directory. |
| `["collect_network_measures","<file1>","<file2>"]` | Collect network-measure text files into `<first-file>.collected.txt`. |
| `["open_src","<file1>","<file2>"]` | Open supplied SRC/SZ or histology files. Without parameters, open a picker. |
| `["open_dwi_nifti","<file1>"]` | Open diffusion NIfTI input through `open_DWI`. |
| `["open_dwi_dicom","<file1>","<file2>"]` | Open one or more DICOM inputs through `open_DWI`. |
| `["open_dwi_2dseq","<file1>"]` | Open 2dseq, FDF, or NRRD diffusion input through `open_DWI`. |
| `["open_src_dir","<directory>"]` | Load `*src.gz` and `.sz` files found in a directory. |
| `["open_fib","<file>"]` | Open a supplied FIB/FZ/DZ file. Without a parameter, open a FIB picker. |
| `["open_structural_tracking","<file>"]` | Open a supplied structural image for tracking. Without a parameter, open a picker. |
| `["open_template","<template-name>"]` | Open an exact built-in template. Invalid names return an error. |
| `["create_db"]` | Open the database-creation dialog. |
| `["create_average"]` | Open the average-database creation dialog. |
| `["open_db","<file>"]` | Load a connectometry database and open a database window. |
| `["open_connectometry","<file>"]` | Load a connectometry database and open a group-connectometry window. |
| `["open_auto_track"]` | Open AutoTrack. |
| `["open_nonlinear_registration"]` | Open nonlinear registration. |
| `["open_xnat"]` | Open XNAT. |
| `["open_console"]` | Open the console. |
| `["clear_recent_src"]` | Immediately clear saved recent SRC/SZ history. |
| `["clear_recent_fib"]` | Immediately clear saved recent FIB/FZ history. |
| `["qc_nii","<file1>","<file2>"]` | Run NIfTI quality checks. |
| `["qc_src","<file1>","<file2>"]` | Run SRC/SZ quality checks. |
| `["qc_fib","<file1>","<file2>"]` | Run FIB/FZ quality checks. |
| `["run_cli","<command-line>"]` | Run one CLI command line containing a valid `--action`. |
| `["open_image","<file1>","<file2>"]` | Open ordinary image files in an image window. |
| `["open_ai"]` | Show and activate the AI Agent window. |
| `["open_hub"]` | Show and activate Fiber Data Hub. |

Use the General examples file for the complete `hub_*` query syntax.

## Critical command syntax

### `list_slice` uses one readable status column

```json
["list_slice"]
```

The reply columns are:

```text
index    current    name    status
```

Interpret `status` directly:

- `available` — a URL-backed custom slice is listed but has not yet been loaded locally. Select it with `set_slice`.
- `registering` — registration is still running. Poll `list_slice` again.
- `ready` — the slice is local or built in and is not registering. It is ready for a dependent operation.

The `current` column is only the selected-state flag (`1` or `0`). It does not
mean the slice is ready. After `set_slice`, poll until the selected row reports
`ready`.

### T1w segmentation: prefer the tracking window for FIB workflows

T1w segmentation is available in both the **tracking** window and the **image**
window, but they serve different workflows.

When an `.fz`/FIB is already open, the normal and most common route is to segment
the T1w directly in that **tracking window**. Do not call main-window
`open_image` to create a separate image window for the same T1w. Keeping the
segmentation in the FIB workflow makes the resulting regions available in the
tracking interface.

Use this robust sequence on the tracking-window ID:

```json
["list_slice"]
["set_slice",7]
["list_slice"]
["list_unet"]
["segment_brain","<model-ID>",7]
```

`set_slice` may start slice loading or registration asynchronously and return
before it is finished. Poll `list_slice` and proceed only when the selected row's
`status` is `ready`. Do not proceed while it is `available` or `registering`.

`list_unet` returns these columns:

```text
index    available    model    name    description
```

The second `segment_brain` element must be the internal ID from the **`model`**
column, such as `human_synthseg`, not the display text from the **`name`** column,
such as `SynthSeg V2`. Use only a row with `available=1`.

The optional third element selects the slice by its exact name or numeric index
from `list_slice`. Supplying it causes `segment_brain` to select that slice;
prechecking `status=ready` still avoids waiting or failure during segmentation.

Segmentation inference may outlast the named-pipe client's wait time. A client
timeout does not prove that `segment_brain` failed. Do not immediately resend the
command. Use `list_slice` and `list_region` to verify that processing finished
and segmentation regions were created. Call `LIST` only if the tracking-window
ID must be obtained again.

Use the image-window `segmentation` command mainly when processing a standalone
NIfTI image or applying an image-processing workflow to multiple files. Open a
T1w with `open_image` for this route only when the task is explicitly image
editing or batch processing, rather than work on an already-open FIB.

### Fiber Data Hub uses separate `hub_*` commands

All Hub commands target the **main** window. `["open_hub"]` only opens the Hub
window. The former subcommand form such as `["hub","files",...]` is no longer
accepted. Hub commands are routed before regular main-window command handling
and may use their full documented argument lists. Use this discovery sequence:

```json
["hub_repo"]
["hub_tags","<repo>"]
["hub_files","<repo>","<tag>",".fz",0,20]
["hub_open","<repo>","<tag>",12]
```

`hub_repo` lists an index and the exact `owner/repository` identifier. Pass that
exact identifier to `hub_tags`. The release list may still be loading; when
`hub_tags` reports `repository data is loading; retry`, repeat the same command
after the metadata finishes loading.

`hub_files` syntax is:

```json
["hub_files","<repo>","<tag>","<optional-text>",0,20]
```

The text filter is a case-insensitive substring match. Filtering occurs before
offset and limit are applied. The first output column remains the actual row
index in the full file table, not the ordinal position within the filtered
results. Use that returned index or the exact filename for `hub_open` and
`hub_download`. Send numeric offsets, limits, and returned indices as JSON
numbers.

To persist a file without opening it:

```json
["hub_download","<repo>","<tag>",12,"C:/data"]
```

`hub_download` requires exactly five elements. It creates the destination
directory when needed, disables overwrite, and skips an existing destination
file.

When opening FIB data, verify that the selected file is `.fz` or `*fib.gz`. After
`hub_open`, call `LIST` to obtain the new tracking-window ID when needed. Hub
open/download routines are GUI-backed; verify the created window or output file
rather than treating a response without an error as proof of completion.

### `list_tract` uses `running` or `done`

Full tract table:

```json
["list_tract"]
```

The full reply columns are:

```text
index    status    shown    name    tracts    deleted    seeds
```

Each row reports `status=running` while its tracking thread is active and
`status=done` after that thread has finished. The `shown` column remains a
separate `1`/`0` visibility flag.

Compact tracking status:

```json
["list_tract","status"]
```

The compact reply columns are:

```text
status    bundles
```

`status=running` means at least one tracking thread is active. `status=done`
means no tracking thread remains active and tracking is complete. `bundles` is
the total number of tract rows, not a running-job count. Poll until
`status=done` before starting a dependent step.

`list_tract` does not require a numeric tract index. If `["list_tract"]` reports
`need-param1`, the request was likely sent through an incompatible wrapper or a
malformed command interface. Send `{"cmd":"list_tract"}` as the `command` field
to a tracking window.

### `run_tracking` requires a new bundle name

Minimum form:

```json
["run_tracking","CST"]
```

The command requires `param:"CST"`, which becomes the new tract-bundle name. An
empty name fails with `missing tract-bundle name`. Without additional parameters,
DSI Studio uses the current tracking parameters and checked region settings.
Follow the [fiber-tracking skill](DSI_STUDIO_AI_SKILL_FIBER_TRACKING.md) when
choosing tracking strategy, parameters, region roles, and quality control.
Before running, use `["list_param","tracking"]` to show all tracking parameters
and their current values. Review these values before changing them or starting
tracking.

Typical sequence without region constraints:

```json
["list_param","tracking"]
["run_tracking","CST"]
```

Use `list_region` only for a region-based workflow after regions were created,
loaded, segmented, or restored. Do not use it as a routine step after opening a
FIB because the initial region list is normally empty. Change tracking
parameters only for a documented reason.

Do not resend `run_tracking` merely because a client timeout occurred. Poll
`["list_tract","status"]` for completion. `status=done` is the definitive
completion signal.

## Discovery quick reference

| Need | Command | Window |
|---|---|---|
| Tracking or image window IDs | top-level `LIST` | none |
| Recent FIB/FZ paths | `["list_recent_fib"]` | main |
| Recent SRC/SZ paths | `["list_recent_src"]` | main |
| Interactive FIB picker | `["open_fib"]` | main |
| Interactive structural-tracking picker | `["open_structural_tracking"]` | main |
| Interactive SRC/reconstruction picker | `["open_src"]` | main |
| Interactive image picker | `["open_image"]` | main |
| Fiber Data Hub window | `["open_hub"]` | main |
| Hub repositories | `["hub_repo"]` | main |
| Hub release tags | `["hub_tags","<repo>"]` | main |
| Hub release files and exact indices | `["hub_files","<repo>","<tag>"]` | main |
| Slice names and `available`/`registering`/`ready` status | `["list_slice"]` | tracking |
| Segmentation model IDs | `["list_unet"]` | tracking |
| Regions and ROI types | `["list_region"]` | tracking |
| Full tract table with per-bundle `running`/`done` status | `["list_tract"]` | tracking |
| Tracking completion (`status=done`) | `["list_tract","status"]` | tracking |
| Parameter IDs and values by domain | `["list_param"]` | tracking |
| Tracking parameters and current values | `["list_param","tracking"]` | tracking |
| One parameter value | `["list_param","fa_threshold"]` | tracking |
| Atlases | `["list_atlas"]` | tracking |
| AutoTrack names | `["list_auto_tract"]` | tracking |

## Operational rules

- Each named-pipe connection sends one request, reads the complete reply, and closes.
- Reuse the exact nonempty `session` UUID for the conversation.
- Send `TITLE` first and update it when the active task changes substantially.
- Call `LIST` only when a tracking or image window ID is needed; `main` is fixed.
- Copy exact command names, current window IDs, indices, internal model IDs, and parameter IDs rather than guessing.
- For `run_auto_track`, call `list_auto_tract` first and use an exact internal atlas label such as `ProjectionBrainstem_CorticospinalTractL`.
- Main-window GUI picker commands require local user interaction; do not claim completion from the response alone.
- Confirm `clear_recent_src` and `clear_recent_fib` because they erase saved history immediately without another prompt.
- Confirm other destructive actions and overwrites.
- Do not answer modal dialogs remotely; tell the user what must be selected.
- A response without `error` means the command handler returned success; asynchronous work may still be active.
- A client timeout does not prove failure; verify application state before retrying a long command.
- For a selected slice, `list_slice` with `status=ready` is the readiness signal.
- For fiber tracking, `list_tract status` with `status=done` is the completion signal.
- A disappeared window or `window not found` means the user likely closed it. Call `LIST` again to obtain a current tracking/image ID; do not reopen it automatically.
- Do not expose private chain-of-thought. Report conclusions, actions, progress, and blockers.

## Footnotes

1. `set_work_dir`, `open_src`, main-window `open_fib`, `open_structural_tracking`, `open_db`, `open_connectometry`, and parameterless `open_image` use local GUI dialogs. The current command branches may report completion when the user cancels. Verify the resulting directory or window with the GUI or, when a tracking/image ID is needed, top-level `LIST`. Database-loading failures from `open_db` and `open_connectometry` are returned through the `CMD` error field.
2. `open_template` now returns failure when the supplied name does not match a built-in template or when `loadFib()` fails.
3. `open_image` now returns the image-window error through the `CMD` response if the supplied files cannot be opened.

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