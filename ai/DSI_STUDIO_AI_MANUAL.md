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
| **main** | Main DSI Studio window | `list_recent_fib`, `list_recent_src`, `set_work_dir`, `open_src`, `open_fib`, `open_structural_tracking`, `open_template`, `open_db`, `open_connectometry`, `open_auto_track`, `open_nonlinear_registration`, `open_xnat`, `open_console`, `clear_recent_src`, `clear_recent_fib`, `open_image`, `open_hub`, `hub_*`, `run_cli` | Main-window `open_fib` takes no path and opens a picker. Main-window `open_image` accepts explicit image paths or no path for a picker. |
| **image** | General image viewer | Image inspection, editing, and `segmentation` commands | Used mainly for standalone NIfTI editing and batch image processing, not as the default T1w-segmentation route when a related FIB is already open. |
| **tracking** | A loaded FIB/FZ tracking window | `list_slice`, `set_slice`, `list_unet`, `segment_brain`, `list_region`, `list_tract`, `run_tracking`, `open_fib`, tract/region/slice/device/rendering commands | Tracking-window `open_fib` requires an explicit `.fz` or `*fib.gz` path and creates another tracking window. |

Always call top-level `LIST` first and use the returned numeric ID. Never use a
window title, filename, type name, guessed number, or stale number as `window`.

## Opening files: four distinct routes

These routes have different argument contracts and should not be treated as
aliases.

### 1. Raw absolute path — direct local-file routing

Send one existing absolute path as raw non-JSON pipe text:

```text
C:\data\subject.fz
```

DSI Studio routes the file by extension. `.fz` and `*fib.gz` open as tracking
data; `.sz` and `*src.gz` open reconstruction; ordinary image formats open an
image window. Raw text is reserved for one local file path and can create the
first tracking window without a GUI file dialog.

### 2. Main-window interactive open commands

These commands take no path and open a local GUI dialog:

```json
["open_fib"]
["open_structural_tracking"]
["open_src"]
["open_image"]
```

- Main-window `open_fib` selects `.fz`, `*fib.gz`, or `.dz` and creates a tracking window.
- `open_structural_tracking` selects `.nii.gz`, `.nii`, or `2dseq` and passes it to `loadFib`.
- `open_src` selects `.sz`, `*src.gz`, `.jpg`, or `.tif` input and creates a reconstruction window.
- Parameterless `open_image` selects ordinary image data and creates an image window.

These are interactive commands. A local user must operate the file dialog. See
footnote 1 regarding cancellation and verification.

### 3. Tracking-window `open_fib` — explicit additional FIB

Target an existing **tracking** window:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["open_fib","C:/data/second_subject.fz"]}
```

This `open_fib` implementation requires the explicit path and creates another
tracking window. It is a separate command contract from main-window
`["open_fib"]`.

### 4. Main-window `open_image` — explicit image paths

Target the **main** window:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"1","command":["open_image","C:/data/T1w.nii.gz"]}
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
2. Send top-level `LIST`.
3. Choose the numeric ID for the correct window type.
4. Run discovery commands before mutation.
5. Use `LIST` for routine polling and targeted `list_*` commands for detail.
6. Verify output files or created objects before reporting completion.

## Request formats

An optional `chat` may accompany any request. Attach an update directly to
`CMD` when it describes that command; use standalone `CHAT` otherwise.

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
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","human_synthseg","7"],"chat":"I verified that the T1w slice is ready. I am starting SynthSeg now."}
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

Send one concise title after understanding the initial task and before the
first `LIST` or `CMD`. The `title` field is required; do not put the title in
`chat` or `text`, or include it in `CMD`, `CHAT`, `LIST`, or `LOG`. Reuse the
exact agent and session identity, and rename later only with the user's
permission.

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

Use `LOG` only when `LIST` and targeted discovery cannot explain a failure.

## Main-window command reference

All commands in this table target the numeric **main** window ID.

| Command | Behavior |
|---|---|
| `["set_work_dir"]` | Open a directory picker and add the selected directory. It does not accept a path argument. |
| `["open_src"]` | Open the SRC/histology picker and create a reconstruction window. |
| `["open_fib"]` | Open the FIB picker and create a tracking window. Do not add a path on a main-window target. |
| `["open_structural_tracking"]` | Select a structural image and pass it to `loadFib`. |
| `["open_template"]` | Open the template currently selected in the main-window template list. See footnote 2. |
| `["open_db"]` | Select and load a connectometry database, then open a database window. |
| `["open_connectometry"]` | Select and load a connectometry database, then open a group-connectometry window. |
| `["open_auto_track"]` | Open the main AutoTrack window. |
| `["open_nonlinear_registration"]` | Open the nonlinear-registration toolbox. |
| `["open_xnat"]` | Open the XNAT dialog. |
| `["open_console"]` | Show the application console. |
| `["clear_recent_src"]` | Immediately clear the recent SRC/SZ list and saved setting, without confirmation. |
| `["clear_recent_fib"]` | Immediately clear the recent FIB/FZ list and saved setting, without confirmation. |
| `["open_image"]` | Open the image picker. |
| `["open_image","C:/data/T1w.nii.gz"]` | Open one or more explicit paths in an image window. |
| `["open_hub"]` | Show, raise, and activate the Fiber Data Hub window without running a Hub query. |

Use the General examples file for `list_recent_*`, `run_cli`, and the complete
`hub_*` query syntax.

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
["set_slice","<T1w-slice-index>"]
["list_slice"]
["list_unet"]
["segment_brain","<model-ID>","<T1w-slice-index>"]
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

The optional third element selects the slice by its exact name or quoted numeric
index from `list_slice`. Supplying it causes `segment_brain` to select that slice;
prechecking `status=ready` still avoids waiting or failure during segmentation.

Segmentation inference may outlast the named-pipe client's wait time. A client
timeout does not prove that `segment_brain` failed. Do not immediately resend the
command. Poll top-level `LIST`, then use `list_slice` and `list_region` to verify
that processing finished and segmentation regions were created.

Use the image-window `segmentation` command mainly when processing a standalone
NIfTI image or applying an image-processing workflow to multiple files. Open a
T1w with `open_image` for this route only when the task is explicitly image
editing or batch processing, rather than work on an already-open FIB.

### Fiber Data Hub uses separate `hub_*` commands

All Hub commands target the **main** window. `["open_hub"]` only opens the Hub
window. The former subcommand form such as `["hub","files",...]` is no longer
accepted. Use this discovery sequence:

```json
["hub_repo"]
["hub_tags","<repo>"]
["hub_files","<repo>","<tag>",".fz","0","20"]
["hub_open","<repo>","<tag>","<exact-FIB-filename-or-returned-index>"]
```

`hub_repo` lists an index and the exact `owner/repository` identifier. Pass that
exact identifier to `hub_tags`. The release list may still be loading; when
`hub_tags` reports `repository data is loading; retry`, repeat the same command
after the metadata finishes loading.

`hub_files` syntax is:

```json
["hub_files","<repo>","<tag>","<optional-text>","<offset>","<limit>"]
```

The text filter is a case-insensitive substring match. Filtering occurs before
offset and limit are applied. The first output column remains the actual row
index in the full file table, not the ordinal position within the filtered
results. Use that returned index or the exact filename for `hub_open` and
`hub_download`.

To persist a file without opening it:

```json
["hub_download","<repo>","<tag>","<exact-filename-or-returned-index>","C:/data"]
```

`hub_download` requires exactly five elements. It creates the destination
directory when needed, disables overwrite, and skips an existing destination
file.

When opening FIB data, verify that the selected file is `.fz` or `*fib.gz`. After
`hub_open`, call top-level `LIST` and verify that a new `tracking` window appeared.
Hub open/download routines are GUI-backed; verify the created window or output
file rather than treating `okay:true` alone as proof of completion.

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
malformed command interface. Send the standard JSON `CMD` array directly to a
tracking window.

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
`LIST` for general application activity and `["list_tract","status"]` for
tract completion. `status=done` is the definitive completion signal.

## Discovery quick reference

| Need | Command | Window |
|---|---|---|
| Open windows and activity | top-level `LIST` | none |
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
- Reuse the exact nonempty `agent` and `session` values for the conversation.
- Native identities are `Codex` and `Claude`.
- Ollama-backed identities include the host, for example `Codex/Ollama(192.168.1.14)`.
- Inspect `LIST` before substantial loading, registration, segmentation, reconstruction, or tracking.
- Discover names, indices, internal model IDs, and parameter IDs rather than guessing.
- Main-window GUI picker commands require local user interaction; do not claim completion from `okay:true` alone.
- Confirm `clear_recent_src` and `clear_recent_fib` because they erase saved history immediately without another prompt.
- Confirm other destructive actions and overwrites.
- Do not answer modal dialogs remotely; tell the user what must be selected.
- `okay:true` means the command was accepted; asynchronous work may still be active.
- A client timeout does not prove failure; verify application state before retrying a long command.
- For a selected slice, `list_slice` with `status=ready` is the readiness signal.
- For fiber tracking, `list_tract status` with `status=done` is the completion signal.
- A disappeared window or `window not found` means the user likely closed it. Do not reopen it automatically.
- Do not expose private chain-of-thought. Report conclusions, actions, progress, and blockers.

## Footnotes

1. `set_work_dir`, `open_src`, main-window `open_fib`, `open_structural_tracking`, `open_db`, `open_connectometry`, and parameterless `open_image` use local GUI dialogs. The current command branches return success when the user cancels. `open_db` and `open_connectometry` also return success when `load_db()` rejects the selected database after showing its own error dialog. Verify the resulting directory or window with the GUI or top-level `LIST`.
2. `open_template` checks that a template-list item is selected, but then calls a `void` helper and returns success. The helper silently returns when the selected text does not match a template stem and does not propagate `loadFib()` failure. Verify that a new tracking window appears.
3. Explicit-path main-window `open_image` calls the `void` `loadNii()` helper and then returns success. `loadNii()` deletes the image window and returns silently when `view_image::open()` fails, so verify that an image window appears.

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
