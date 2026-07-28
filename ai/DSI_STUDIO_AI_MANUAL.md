# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` first. Use this file for the protocol and critical
routing rules. Use the topic-specific command inventories for complete syntax.

## Command routing

A `CMD` must target the quoted numeric window ID returned by top-level `LIST`.
Never use a window title, filename, type name, guessed number, or stale number.

| Window type | Use it for |
|---|---|
| **main** | Recent files, file opening, reconstruction, DICOM/BIDS conversion, QC, databases, Hub, AI, and main tools |
| **image** | General image viewing, editing, and image-window segmentation |
| **tracking** | FIB/FZ slices, regions, tracts, tracking, devices, parameters, and rendering |

Call `LIST` before every substantial operation and again after a command that
may create or close a window.

## Exact recent-file commands

Target the numeric **main** window ID:

```json
["list_recent_fib"]
["list_recent_src"]
```

Use these exact names. Do not invent aliases such as `recent_list`.

## Request formats

### LIST

```json
{"agent":"Codex","session":"<uuid>","request":"LIST"}
```

The reply begins with application activity, followed by rows containing window
type, numeric ID, busy state, tracking jobs, and title.

### CMD

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["list_region"]}
```

Every command name and parameter must be a JSON string.

A command may include a top-level progress message:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","human_synthseg","7"],"chat":"Starting SynthSeg after verifying the selected T1w slice."}
```

### CHAT

```json
{"agent":"Codex","session":"<uuid>","request":"CHAT","chat":"Tracking completed and the output was verified."}
```

### TITLE

```json
{"agent":"Codex","session":"<uuid>","request":"TITLE","title":"Corticospinal tract analysis"}
```

Send one concise title after understanding the task and before the first `LIST`
or `CMD`.

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

Use `LOG` only when `LIST`, the direct `CMD` response, and targeted discovery do
not explain a failure.

## CMD response format

Every `CMD` returns a JSON array with one result object per command.

Command with captured text:

```json
[{"index":0,"output":"<command output>"}]
```

Successful command with no captured text:

```json
[{"index":0,"output":"command completed"}]
```

Failed command:

```json
[{"index":0,"error":"<reason>"}]
```

The presence of `error` means that command failed. A batch stops after the first
error. A response without `error` does not prove that asynchronous or GUI-backed
work finished; verify the resulting state.

## Main-window file opening

Use documented commands. Never send a filesystem path by itself as a named-pipe
request.

The current main-window router accepts an optional path parameter for these
commands:

```json
["open_fib","C:/data/subject.fz"]
["open_structural_tracking","C:/data/T1w.nii.gz"]
["open_src","C:/data/subject.sz"]
["open_image","C:/data/T1w.nii.gz"]
["open_db","C:/data/group.db.fz"]
["open_connectometry","C:/data/group.db.fz"]
```

Omitting the path opens the corresponding local picker. Picker cancellation may
return without an immediate error, so verify the resulting window.

Main-window `open_fib` now supports either an explicit FIB/FZ path or no path for
the picker. Tracking-window `open_fib` remains a separate command implemented by
the tracking window.

Do not use `open_image` for FIB/FZ tracking data.

## Multiple-file commands

The latest main-window router accepts one command-array element per file:

```json
["open_src","C:/data/a.sz","C:/data/b.sz"]
["open_dwi_dicom","C:/dicom/a.dcm","C:/dicom/b.dcm"]
["open_image","C:/data/T1w.nii.gz","C:/data/T2w.nii.gz"]
["qc_fib","C:/data/a.fz","C:/data/b.fz"]
["rename_dicom","C:/dicom/a.dcm","C:/dicom/b.dcm"]
```

Do not combine paths into one `&`-separated parameter.

## Main-window discovery and utility commands

Common commands include:

```json
["list_recent_fib"]
["list_recent_src"]
["set_work_dir","C:/work"]
["reset_settings"]
["open_console"]
["open_ai"]
["open_auto_track"]
["open_nonlinear_registration"]
["open_xnat"]
["clear_recent_fib"]
["clear_recent_src"]
```

Confirm `reset_settings`, `clear_recent_fib`, and `clear_recent_src` before use
because they immediately modify saved application state.

## DICOM, BIDS, reconstruction, and QC

Current main-window commands include:

```json
["rename_dicom","<file1>","<file2>"]
["rename_dicom_dir","<directory>"]
["convert_dicom_dir","<directory>"]
["bids_to_src","<BIDS-directory>"]
["nifti_dir_to_src","<directory>"]
["collect_network_measures","<file1>","<file2>"]
["open_dwi_nifti","<file1>","<file2>"]
["open_dwi_dicom","<file1>","<file2>"]
["open_dwi_2dseq","<file1>","<file2>"]
["open_src_dir","<directory>"]
["qc_nii","<file1>","<file2>"]
["qc_src","<file1>","<file2>"]
["qc_fib","<file1>","<file2>"]
```

`bids_to_src` still asks the local user to choose an output directory. Commands
without explicit inputs may open local pickers.

## Templates and databases

```json
["open_template","<exact-template-name>"]
["create_db"]
["create_average"]
["open_db","<database-path>"]
["open_connectometry","<database-path>"]
```

Invalid template names, template loading failures, and database loading failures
are returned through `error`.

## Fiber Data Hub

Hub commands are routed before the regular main-window command handling and may
use their full argument lists:

```json
["open_hub"]
["hub_repo"]
["hub_tags","<repo>"]
["hub_files","<repo>","<tag>",".fz","0","20"]
["hub_open","<repo>","<tag>","<exact-filename-or-returned-index>"]
["hub_download","<repo>","<tag>","<exact-filename-or-returned-index>","C:/data"]
```

Use exact repository names, tags, filenames, and row indices returned by the Hub
commands. Verify the created window or downloaded file.

## Tracking-window critical syntax

### Slice readiness

```json
["list_slice"]
```

Proceed only when the selected row reports `status=ready`. Do not proceed while
it reports `available` or `registering`.

### Segmentation

```json
["list_slice"]
["set_slice","<slice-index>"]
["list_slice"]
["list_unet"]
["segment_brain","<model-ID>","<slice-index>"]
```

Use the internal model ID returned by `list_unet`, not its display name.

### Tracking

```json
["list_param","tracking"]
["run_tracking","CST"]
["list_tract","status"]
```

`run_tracking` requires a new bundle name. Poll until `list_tract status` reports
`status=done`.

## Operational rules

- Reuse the exact nonempty `agent` and `session` values.
- Discover names, indices, model IDs, and parameter IDs instead of guessing.
- Do not repeatedly resend a long-running command after a client timeout.
- Do not answer modal dialogs remotely; tell the local user what must be selected.
- Verify windows, files, regions, tracts, and status commands before reporting completion.
- Do not expose private chain-of-thought. Report actions, conclusions, progress, and blockers.

## Complete command inventory and examples

- [Main window, Hub, FIB, workspace, settings, and parameters](DSI_STUDIO_AI_COMMAND_EXAMPLES_GENERAL.md)
- [Slices and segmentation](DSI_STUDIO_AI_COMMAND_EXAMPLES_SLICE.md)
- [Regions and tract-to-region analysis](DSI_STUDIO_AI_COMMAND_EXAMPLES_REGION.md)
- [Tracts, tracking, AutoTrack, clustering, recognition, and TDI](DSI_STUDIO_AI_COMMAND_EXAMPLES_TRACT.md)
- [Devices and AC-PC locators](DSI_STUDIO_AI_COMMAND_EXAMPLES_DEVICE.md)
- [Rendering, camera, surfaces, and display](DSI_STUDIO_AI_COMMAND_EXAMPLES_RENDERING.md)
- [Image-window and TIPL generic image operations](DSI_STUDIO_AI_COMMAND_EXAMPLES_IMAGE.md)
