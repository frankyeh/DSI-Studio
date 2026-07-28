# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` completely. Read the operating rules and common
syntax below, then search this manual only for commands needed by the request.
Do not reread the entire file for each action.

## Operating rules

- Each named-pipe connection sends exactly one request and then closes.
- Send JSON for every AI request. Raw non-JSON text is reserved for one existing
  local file path. If DSI Studio rejects direct text, reread this manual and
  resend a valid JSON request.
- Use a native client for `\\.\pipe\dsi-studio` first. Do not use
  `dsi_agent.ps1`, `dsi_studio.exe`, or another wrapper unless direct access is
  unavailable or fails and the user approves the fallback.
- Send separate non-empty `agent` and `session` fields and reuse the exact pair.
  `agent` must include `Codex` or `Claude` and must not contain `@`.
- For a new Codex chat launched by DSI Studio, process the initiating task in
  the first run. There is no bootstrap run and DSI Studio does not launch the
  same task twice. Use `CODEX_THREAD_ID` as `session` immediately; do not wait
  for a second launch or generate another ID.
- DSI Studio also reads `thread.started.thread_id` to store and later resume the
  same Codex thread with `codex exec resume`.
- For an externally initiated Codex Desktop chat, use a task UUID only when it
  is explicitly present in runtime context. Never guess or scan for one.
- For Claude Code, read `~/.claude/sessions/<pid>.json` and use `sessionId`, not
  `name`. DSI Studio resumes it with `claude -p --resume <sessionId>`.
- Ollama is a model provider, not an agent identity.
- After understanding the initiating prompt, send one concise `TITLE`. Send a
  later `TITLE` only when the user permits renaming.
- Call top-level `LIST` first. It returns global activity plus every window's
  numeric ID and busy state; it does not use a window ID.
- Before starting loading, registration, reconstruction, segmentation, batch
  processing, fiber tracking, or other substantial work, inspect `LIST`. When
  DSI Studio is already busy, follow the wait etiquette below instead of adding
  another large task.
- Every `CMD`, including every `list_*` command, requires a numeric `window`
  returned by `LIST`. Never use a type, title, filename, guessed ID, or stale ID.
- Every command name and parameter inside `command` must be a JSON string.
  Write slice index `"7"`, not numeric `7`.
- Use GUI commands. Do not use `run_cli` unless the user explicitly requests
  CLI execution.
- Discover names, indices, and parameter IDs before mutation. Never guess them.
- `okay:true` means the handler accepted the command; asynchronous work may
  still be running.
- Poll top-level `LIST` for global and per-window activity. Use the relevant
  `list_*` command only when detailed state or output verification is needed.
- Use `LOG` only for failures or states that `LIST` and targeted discovery
  cannot explain.
- User-facing text uses the top-level `chat` field. A standalone message uses
  top-level request `CHAT`, not a `CMD` command.
- If a required window disappears or returns `window not found`, assume the
  user closed it. Do not reopen or retry it. Ask whether to continue.
- Confirm destructive actions and overwrites. Verify every output file.
- Do not answer modal dialogs remotely; tell the user what action is required.
- Attach concise progress `chat` to an already-needed request. Avoid separate
  status-only requests except for a required decision, blocked/waiting state,
  or the final reply.
- Inspect every complete reply. A queued `PROMPT` may follow `LIST`, `LOG`,
  `CHAT`, `TITLE`, or appear in the last `CMD` result.

## Discovery

| Need | Command | Note |
|---|---|---|
| Global and per-window activity | Top-level JSON `LIST` | Use for initial discovery and all routine status polling. |
| Recent FIB files | Main: `list_recent_fib` | FIB means `.fz`. |
| Recent SRC files | Main: `list_recent_src` | SRC means `.sz`. |
| Slices/readiness/registration | Tracking: `list_slice` | Use after loading or registration when exact fields are needed. |
| Regions and ROI types | Tracking: `list_region` | Discover indices before region mutation or tracking. |
| Full tract-bundle details | Tracking: `list_tract` | Prefer after tracking completes. |
| Targeted tract polling | Tracking: `list_tract status` | Use only when a targeted window reply is useful. |
| Valid tracking/GUI parameter IDs | Tracking: `list_param` | Call without an ID once before setting values. |
| One parameter value | Tracking: `list_param <id>` | Query only discovered IDs. |
| Atlases | Tracking: `list_atlas` | Discover template/atlas/label IDs before adding regions. |
| Segmentation models | Tracking: `list_unet` | Use returned model names exactly. |
| Automatic tract names | Tracking: `list_auto_tract` | Use returned tract names exactly. |
| Incremental diagnostics | Top-level JSON `LOG` | Use only when compact state cannot explain a problem. |

`list_slice` fields are:

```text
index current name ready registering downloaded registered
```

Built-in/native volumes report `ready=1`. For custom volumes,
`registering=1` means registration is still running; `registered=0` does not
mean the image is broken.

`list_tract status` returns only:

```text
running bundles
```

`LIST` already reports the active tracking-job count for every tracking window.
Use `list_tract status` only when a targeted tracking-window reply is useful;
request full `list_tract` after completion when bundle details are needed.

`list_param` without an ID lists all valid IDs. Query only needed values after
that discovery call. Change tracking behavior with `set_param` or `set_params`
before starting tracking.

## Request formats

### LIST

```json
{"agent":"Codex","session":"<uuid>","cwd":"C:/work","request":"LIST"}
```

The reply is compact tab-separated text:

```text
OKAY<TAB>busy<TAB>level<TAB>status
type<TAB>id<TAB>busy<TAB>tracking-jobs<TAB>title
...
```

Example:

```text
OKAY	1	2	segment_brain (3/5)
main	1	0	0	DSI Studio
tracking	2	1	0	C:/data/a.fz
tracking	3	1	2	C:/data/b.fz
tracking	4	0	0	C:/data/c.fz
```

The first line reports application-wide state:

- `busy=1` when a TIPL operation or any supported window is busy.
- `level` is the active TIPL nesting depth, excluding the persistent application
  root. It is `1` when only asynchronous work such as fiber tracking is active.
- `status` is the deepest available TIPL status. When asynchronous tracking is
  the only known activity, it is `fiber tracking`.

Each following line reports a window:

- `type` is `main`, `tracking`, or `image`.
- `id` is the numeric value required by `CMD`.
- `busy=1` indicates an active AI command, command sequence, fiber tracking, or
  custom-slice registration associated with that window.
- `tracking-jobs` is the number of active tracking bundles in that window.
- `title` is the normalized window title/path.

Global `busy` can be `1` while all window rows are `0` when TIPL reports work
that cannot be reliably assigned to one window. Reuse numeric IDs until a
window opens or closes, but call `LIST` while waiting to obtain current activity.
Silent `LIST` requests without `chat` are not written to AI history or console,
so they are preferred for polling.

### CMD

A single command:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["list_region"]}
```

All command elements must be strings:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["set_slice","7"]}
```

Use `"7"`, not `7`.

A safe same-window batch:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":[["list_param"],["list_slice"]]}
```

Do not batch destructive, asynchronous, output-dependent, or modal-opening
commands. Do not send an empty command array. A second synchronous AI `CMD` is
rejected while another synchronous AI command is running; use `LIST` instead of
repeatedly retrying it.

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

`LOG` returns at most 4096 new console characters and advances the session's
cursor. AI-facing `LOG` and `CMD` text has ANSI escape sequences removed.
`LOG` reads diagnostics; it does not publish a message to the user and ignores
an invented top-level `log` field.

### CHAT

```json
{"agent":"Codex","session":"<uuid>","request":"CHAT","chat":"The requested fiber tracking has completed successfully."}
```

`CHAT` is a standalone top-level request. It uses no `window` or `command`
field. A non-empty message returns `OKAY`. Use it for the final answer, a
required user decision, or one blocked/waiting update. Do not send
`["chat","..."]` as a `CMD`, and do not use `LOG` to publish text.

### TITLE

```json
{"agent":"Codex","session":"<uuid>","request":"TITLE","title":"Concise task name"}
```

Another `TITLE` replaces the current title. Send it later only when the user
permits renaming.

### Open one local file

Send one absolute path as raw pipe text. Raw text is accepted only when it
resolves to an existing file:

```text
C:\data\subject.fz
```

## Reply formats

- `LIST`, `LOG`, `CHAT`, `TITLE`, file-open replies, and validation errors are
  text.
- `LIST` begins with
  `OKAY<TAB>busy<TAB>level<TAB>status`; each following line is
  `type<TAB>numeric-id<TAB>busy<TAB>tracking-jobs<TAB>title`.
- A `CMD` reply is a JSON array of `{index,okay,output,error?}` objects.
- List-command data remains tab-separated text inside `output`; do not invent
  properties such as `.windows`, `.tracks`, or `.regions`.
- `CHAT` returns `OKAY` or `ERROR<TAB>missing chat`; `CHAT` and `TITLE` may append
  a queued `PROMPT`.
- `CMD` may attach `prompt` to its last result object.
- Paths returned by `LIST`, `list_recent_fib`, and `list_recent_src` use `/` as
  the canonical separator. Windows accepts these paths; compare them without
  converting back to `\`.

## Command examples and inventory

Wrap each example array below in the standard `CMD` request shown above. Replace
values in angle brackets with values returned by the corresponding discovery
command. Every command name and parameter remains a quoted JSON string. Blank
example cells mark source commands without a documented recommended example.
The final **Note** column gives the command purpose and the most important
parameter or workflow restriction.

### Main-window and Hub commands

| Command | Common example | Note |
|---|---|---|
| `list_recent_fib` | `["list_recent_fib"]` | List recently opened FIB (`.fz`) files from the main window. |
| `list_recent_src` | `["list_recent_src"]` | List recently opened SRC (`.sz`) files from the main window. |
| `hub repos` | `["hub","repos"]` | List Fiber Data Hub repositories. |
| `hub tags` | `["hub","tags","<repo from hub repos>"]` | List tags/releases for one repository. |
| `hub files` | `["hub","files","<repo>","<tag>","","0","20"]` | List files with `index`, `file`, `size`, and `downloaded`; supports filter, offset, and limit. |
| `hub open` | `["hub","open","<repo>","<tag>","0"]` | Download one returned file to the temporary cache and open it. |
| `hub download` | `["hub","download","<repo>","<tag>","0","C:/data"]` | Save one returned file to a persistent directory without opening it. |
| `hub help` |  | Show Hub subcommand syntax. |
| `open_image` | `["open_image","C:/data/T1w.nii.gz","C:/data/T2w.nii.gz"]` | Open multiple image paths together in one image window; target the main window. |
| `run_cli` | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` | Run a CLI command only when the user explicitly requests CLI execution. |

`hub files` returns `index`, `file`, `size`, and `downloaded`. For `hub open`
and `hub download`, pass either the exact filename or the numeric index from
that same `hub files` result. Indices apply only to the currently selected
repository and tag.

`hub open` downloads the selected file to DSI Studio's temporary cache and then
opens it. `hub download` saves the selected file to the supplied persistent
directory and does not open it. If the directory does not exist, DSI Studio
creates it and reports `directory_created`. A successful reply is returned only
after the downloaded data have been written to disk.

### Tracking-window file, workspace, and setting commands

| Command | Common example | Note |
|---|---|---|
| `open_fib` |  | Open another FIB in an existing tracking window; cannot create the first tracking window. |
| `correct_bias_field` |  | Correct bias field for the applicable image/index. |
| `save_fib_as` |  | Save the loaded FIB under another path. |
| `open_mapping` |  | Open a mapping file for the tracking window. |
| `save_workspace` |  | Save current tracking-window workspace state. |
| `load_workspace` |  | Load a previously saved workspace. |
| `save_setting` |  | Save combined settings. |
| `save_rendering_setting` |  | Save rendering settings only. |
| `save_tracking_setting` |  | Save tracking settings only. |
| `load_setting` |  | Load combined settings. |
| `load_rendering_setting` |  | Load rendering settings only. |
| `load_tracking_setting` |  | Load tracking settings only. |
| `restore_rendering` |  | Restore default rendering settings. |
| `restore_tracking` |  | Restore default tracking settings. |
| `presentation_mode` |  | Toggle presentation-oriented display state. |

### Slice and segmentation commands

| Command | Common example | Note |
|---|---|---|
| `list_slice` | `["list_slice"]` | List slice index, current flag, name, readiness, registration, download, and registered state. |
| `set_slice` | `["set_slice","7"]` | Select a slice by quoted numeric index; may trigger remote download/registration. |
| `set_slice_by_name` | `["set_slice_by_name","T1w"]` | Select a slice by exact name. |
| `move_slice` | `["move_slice","80 100 80"]` | Move the shared slice position to `x y z`. |
| `list_unet` | `["list_unet"]` | List available segmentation model names. |
| `segment_brain (current slice)` | `["segment_brain","<model from list_unet>"]` | Synchronously segment the current slice and create regions. |
| `segment_brain (by name)` | `["segment_brain","<model from list_unet>","T1w"]` | Select an exact slice name, wait for registration, then segment. |
| `segment_brain (by index)` | `["segment_brain","<model from list_unet>","7"]` | Select a quoted slice index, wait for registration, then segment. |
| `enable_slice` |  | Enable or disable slice display. |
| `set_slice_contrast` |  | Set slice contrast/display range. |
| `set_slice_dir_color` |  | Enable or disable directional coloring for the slice. |
| `set_slice_overlay` |  | Enable or disable overlay display. |
| `set_slice_stay` |  | Control whether the selected slice stays active. |
| `add_slice` |  | Add a native/custom slice image. |
| `add_mni_slice` |  | Add a slice interpreted in MNI space. |
| `skull_strip_slice` |  | Apply skull stripping to a slice. |
| `save_roi_screen` |  | Save a screen image centered on ROI display. |
| `save_slice_image` |  | Export the current slice image/index. |
| `save_slice_mni_image` |  | Export a slice image in MNI space. |
| `save_slice_mapping` |  | Save the current slice mapping. |
| `open_slice_mapping` |  | Open a saved slice mapping. |
| `save_slice_volume` |  | Save the slice volume. |
| `delete_slice` |  | Delete the selected custom slice. |

Use `list_unet` to obtain an available model and `list_slice` to obtain current
slice names and indices. When a `segment_brain` slice argument is supplied, DSI
Studio first tries an exact slice-name match and then treats the value as a
numeric index. Prefer the numeric index when slice names are duplicated.

The command internally selects the requested slice. If that slice references a
remote image, selection triggers download and loading. DSI Studio waits for
custom-slice registration before segmentation, so a separate
`set_slice_by_name` call is not required. To preload and inspect separately,
use `set_slice <index>`, poll `LIST` until the tracking window is idle, then
confirm detailed fields with `list_slice`.

`segment_brain` is synchronous. Its `CMD` reply arrives after inference and
region creation finish. While it runs, poll `LIST`; a delayed `LIST` reply alone
does not prove failure. Verify successful output with `list_region`.

### Region commands

`region_action_<operation>` uses `command[1]` for one region index or an
`&`-separated index list. `command[2]` supplies the extra value for threshold
and voxel-dilation operations. When `command[1]` is omitted, the current region
is used unless the action or GUI state selects checked regions.

| Command | Common example | Note |
|---|---|---|
| `list_region` | `["list_region"]` | List region index, visibility, name, type, color, dimensions, and resolution. |
| `list_atlas` | `["list_atlas"]` | List available templates/atlases for region creation. |
| `add_region_from_atlas` | `["add_region_from_atlas","<template atlas labels>"]` | Add one or more atlas labels; discover valid template, atlas, and label IDs first. |
| `set_region_name` | `["set_region_name","0","Left CST seed"]` | Rename a region by quoted index. |
| `set_region_type` | `["set_region_type","0","3"]` | Set region role: `0=ROI`, `1=ROA`, `2=End`, `3=Seed`, `4=Terminative`, `5=NotEnd`, `6=Limiting`. |
| `set_region_color` | `["set_region_color","0","4294901760"]` | Set packed Qt ARGB color for a region. |
| `show_only_regions` | `["show_only_regions","0&2&5"]` | Show only the listed `&`-separated region indices. |
| `new_region` |  | Create an empty region. |
| `new_region_whole_brain_seed` |  | Create a whole-brain seed from the current FA/Otsu threshold. |
| `new_region_from_threshold` |  | Create a region by thresholding the current slice. |
| `new_region_from_mni` |  | Create a spherical region from MNI coordinates and voxel radius. |
| `new_region_from_sphere` |  | Create a spherical region from image-space coordinates and voxel radius. |
| `open_region` |  | Open one or more region files in native space. |
| `open_mni_region` |  | Open one or more region files and map them from MNI space. |
| `save_region` |  | Save one region; optional index selects the target. |
| `save_4d_region` |  | Save checked regions as a 4D NIfTI. |
| `save_all_regions` |  | Save checked regions as one 3D label NIfTI. |
| `save_all_regions_to_folder` |  | Save each checked region as a separate file in a folder. |
| `save_region_info` |  | Save voxel coordinates, directions, and quantitative values for one region. |
| `load_region_color` |  | Load RGB/RGBA colors for regions from a text file. |
| `save_region_color` |  | Save region colors to a text file. |
| `delete_region` |  | Delete one region by index or current selection. |
| `delete_all_regions` |  | Delete all regions. |
| `copy_region` |  | Duplicate one region. |
| `merge_regions` |  | Merge supplied or checked region indices into the first region. |
| `check_region` |  | Set one region's checked/shown state. |
| `check_all_regions` |  | Check/show all regions. |
| `uncheck_all_regions` |  | Uncheck/hide all regions. |
| `move_up_region` |  | Move one region up in table order. |
| `move_down_region` |  | Move one region down in table order. |
| `move_region` |  | Move a region center to a specified location in region space. |
| `move_slice_to_region` |  | Move slice crosshairs to a region center. |
| `show_device_statistics` |  | Display device statistics in a dialog. |
| `save_device_statistics` |  | Save device statistics to a text file. |
| `show_region_statistics` |  | Display statistics for checked regions. |
| `save_region_statistics` |  | Save statistics for checked regions. |
| `show_t2r` |  | Display tract-to-region connectivity for checked tracts and regions. |
| `save_t2r` |  | Save tract-to-region connectivity to a text file. |
| `show_tract_statistics` |  | Display statistics for checked tracts. |
| `save_tract_statistics` |  | Save statistics for checked tracts. |
| `show_tract_recognition` |  | Display recognition scores for a tract. |
| `save_tract_recognition` |  | Save tract-recognition scores. |
| `region_action_shiftx` |  | Shift selected region(s) +1 voxel in X. |
| `region_action_shiftnx` |  | Shift selected region(s) −1 voxel in X. |
| `region_action_shifty` |  | Shift selected region(s) +1 voxel in Y. |
| `region_action_shiftny` |  | Shift selected region(s) −1 voxel in Y. |
| `region_action_shiftz` |  | Shift selected region(s) +1 voxel in Z. |
| `region_action_shiftnz` |  | Shift selected region(s) −1 voxel in Z. |
| `region_action_flipx` |  | Flip selected region(s) along X. |
| `region_action_flipy` |  | Flip selected region(s) along Y. |
| `region_action_flipz` |  | Flip selected region(s) along Z. |
| `region_action_smoothing` |  | Morphologically smooth selected region(s). |
| `region_action_erosion` |  | Erode selected region(s). |
| `region_action_dilation` |  | Dilate selected region(s). |
| `region_action_opening` |  | Apply morphological opening. |
| `region_action_closing` |  | Apply morphological closing. |
| `region_action_defragment` |  | Keep the principal connected component of selected region(s). |
| `region_action_negate` |  | Invert selected region mask(s). |
| `region_action_dilation_by_voxel` |  | Dilate by voxel radius; `command[2]` is the radius. |
| `region_action_threshold` |  | Replace selected region(s) with a thresholded current-slice mask; `command[2]` is threshold. |
| `region_action_threshold_current` |  | Threshold only voxels already inside selected region(s). |
| `region_action_dilation_by_threshold` |  | Grow selected region(s) using current-slice intensity threshold. |
| `region_action_erosion_by_threshold` |  | Shrink selected region(s) using current-slice intensity threshold. |
| `region_action_separate` |  | Split one region into connected components. |
| `region_action_sort_name` |  | Sort selected/checked regions by name; repeating reverses order. |
| `region_action_sort_x` |  | Sort selected/checked regions by X position. |
| `region_action_sort_y` |  | Sort selected/checked regions by Y position. |
| `region_action_sort_z` |  | Sort selected/checked regions by Z position. |
| `region_action_sort_size` |  | Sort selected/checked regions by volume. |
| `region_action_1st_ex_all` |  | Subtract every later region from the first. |
| `region_action_all_ex_1st` |  | Subtract the first region from every later region. |
| `region_action_all_inter_1st` |  | Intersect every later region with the first. |
| `region_action_all_to_1st` |  | Assign/fill later labels within the first region. |
| `region_action_refine_all` |  | Refine all supplied region labels using the current slice. |

### Tracking parameter commands

| Command | Common example | Note |
|---|---|---|
| `list_param (all IDs)` | `["list_param"]` | Discover all valid tracking/GUI parameter IDs; call once before mutation. |
| `list_param (one ID)` | `["list_param","step_size"]` | Read one discovered parameter value. |
| `set_param` | `["set_param","step_size","1.0"]` | Set one discovered parameter ID to a string value. |
| `set_params` | `["set_params","step_size=1.0&min_length=20"]` | Set multiple `id=value` pairs separated by `&`. |

Call parameterless `list_param` first and use only IDs it returns. Values remain
JSON strings even when they represent numbers.

### Fiber tracking and tract commands

| Command | Common example | Note |
|---|---|---|
| `list_tract` | `["list_tract"]` | List tract bundles and details. |
| `list_tract status` | `["list_tract","status"]` | Return only targeted tracking status (`running bundles`); prefer top-level `LIST` for routine polling. |
| `run_tracking` | `["run_tracking","Whole Brain"]` | Start asynchronous tracking with current parameters; second element is the new bundle label. |
| `run_tracking (with regions)` | `["run_tracking","Corticospinal Tract","0:3&1:0&2:1"]` | Optional third element is `region-index:type` items separated by `&`. |
| `list_auto_tract` | `["list_auto_tract"]` | List valid automatic tract names. |
| `run_auto_track` | `["run_auto_track","Corticospinal Tract"]` | Run automatic tracking for a discovered tract name. |
| `show_only_tracts` | `["show_only_tracts","0&2&5"]` | Show only listed tract indices. |
| `enable_auto_tract` |  | Enable or disable automatic-tract mode. |
| `open_tract` |  | Open one tract file. |
| `open_tracts` |  | Open multiple tract files. |
| `open_tract_dir` |  | Open tract files from a directory. |
| `save_tract` |  | Save the selected tract. |
| `save_mni_tract` |  | Save the selected tract in MNI coordinates. |
| `save_template_tract` |  | Save the selected tract in template space. |
| `save_slice_tract` |  | Save the selected tract in current slice space. |
| `save_tract_endpoint` |  | Save selected tract endpoints. |
| `save_mni_tract_endpoint` |  | Save endpoints in MNI coordinates. |
| `save_slice_tract_endpoint` |  | Save endpoints in current slice space. |
| `save_all_tracts` |  | Save all checked tracts together. |
| `save_all_tracts_to_folder` |  | Save checked tracts as separate files in a folder. |
| `save_all_tracts_to_dir` |  | Save checked tracts to a specified directory. |
| `save_tdi` |  | Save tract-density imaging output. |
| `save_tdi2` |  | Save alternate/high-resolution tract-density output. |
| `save_tract_values` |  | Save values sampled along tract(s). |
| `tract_to_region` |  | Convert tract trajectories to a region. |
| `endpoint_to_region` |  | Convert tract endpoints to region(s). |
| `update_tract` |  | Update/recalculate selected tract display/model. |
| `delete_tract` |  | Delete one tract bundle. |
| `delete_all_tracts` |  | Delete all tract bundles. |
| `copy_tract` |  | Duplicate selected tract bundle. |
| `rename_tract` |  | Rename selected tract bundle. |
| `merge_tract` |  | Merge selected tract bundles. |
| `merge_all_tracts` |  | Merge all tract bundles. |
| `merge_tract_by_name` |  | Merge tract bundles sharing a name. |
| `sort_tract_by_name` |  | Sort tract bundles by name. |
| `trim_tract` |  | Trim the selected tract. |
| `trim_all_tracts` |  | Trim all checked tracts. |
| `cut_tract` |  | Cut selected tract trajectories. |
| `cut_by_slice` |  | Cut tracts using the current slice plane. |
| `filter_tract` |  | Filter tracks using ROI/ROA/End regions. |
| `remove_repeated_tracts` |  | Remove duplicate/repeated trajectories. |
| `recognize_tract` |  | Recognize the selected tract against the tract atlas. |
| `cluster_tract` |  | Cluster selected tract trajectories. |
| `cluster_all_tracts` |  | Cluster all checked tracts. |
| `check_tract` |  | Set one tract's checked state. |
| `check_uncheck_all_tract` |  | Toggle checked state for all tracts. |
| `set_tract_color` |  | Set tract color. |
| `set_tract_color_style` |  | Set tract coloring style. |
| `set_tract_visible` |  | Set tract visibility. |

`run_tracking` has no tracking-method argument. DSI Studio uses the tracking
algorithm with directional information already stored in the loaded FIB and the
current tracking parameters. GQI, DTI, and Q-ball describe reconstruction of
the FIB's directional information; do not pass them to `run_tracking`.

The required bundle name is only the label assigned to the new tract bundle.
To change tracking settings, first use `list_param`, then `set_param` or
`set_params`. The optional region field contains `region-index:type` entries.
The fourth internal `run_tracking` parameter is reserved for automatic tracking
tolerance; agents should use `run_auto_track` instead.

Fiber tracking is asynchronous. A successful reply means tracking started.
Poll top-level `LIST`; the target row's `tracking-jobs` reaches zero when no
active tracking bundle remains. Request full `list_tract` afterward only when
bundle details are needed.

### Device commands

| Command | Common example | Note |
|---|---|---|
| `new_device` |  | Create a new device. |
| `move_device` |  | Move a device to a specified location. |
| `push_device` |  | Push the selected device along its axis. |
| `pull_device` |  | Pull the selected device along its axis. |
| `copy_device` |  | Duplicate the selected device. |
| `set_acpc` |  | Set AC, PC, and interhemispheric reference points. |
| `delete_device` |  | Delete one device. |
| `delete_all_devices` |  | Delete all devices. |
| `save_all_devices` |  | Save all devices. |

### Rendering, camera, and surface commands

| Command | Common example | Note |
|---|---|---|
| `rotate` | `["rotate","15 1 0 0"]` | Rotate the 3D view by degrees around axis `x y z`. |
| `save_hd_screen` | `["save_hd_screen","C:/output/tracts.png","1920 1080"]` | Save a high-resolution rendering to a specified size. |
| `set_view` |  | Set a predefined or explicit camera view. |
| `set_zoom` |  | Set camera zoom. |
| `set_camera` |  | Set camera parameters. |
| `get_camera` |  | Return current camera parameters. |
| `open_camera` |  | Load camera settings from a file. |
| `save_camera` |  | Save camera settings to a file. |
| `store_camera` |  | Store the current camera in the default slot. |
| `store_camera1` |  | Store camera in slot 1. |
| `store_camera2` |  | Store camera in slot 2. |
| `restore_camera` |  | Restore the default stored camera. |
| `restore_camera1` |  | Restore camera slot 1. |
| `restore_camera2` |  | Restore camera slot 2. |
| `save_screen` |  | Save the current 3D screen. |
| `add_surface` |  | Add/create a surface object; some variants are prefix-dispatched. |
| `delete_surface` |  | Delete a surface object. |
| `load_surface` |  | Load a surface file. |
| `save_surface` |  | Save a surface file. |
| `set_surface_color` |  | Set surface color. |
| `set_surface_alpha` |  | Set surface opacity. |
| `set_surface_visible` |  | Set surface visibility. |
| `set_device_color` |  | Set device color. |

### Image-window commands

DSI Studio handles these commands directly before delegating to TIPL:

| Command | Common example | Note |
|---|---|---|
| `change_type` |  | Change image voxel type. |
| `bias_field_correction` |  | Run image-window bias-field correction. |
| `brain_extraction` |  | Run image-window brain extraction. |
| `segmentation` |  | Run image-window segmentation. |
| `deface` |  | Deface the image. |
| `rotate_to_image` |  | Rotate/register the current image to another image. |
| `warp_to_image` |  | Warp the current image to another image. |
| `apply_to_image` |  | Apply an operation/transformation to another image. |

### TIPL generic image commands

TIPL `cmd.hpp` supplies these commands through the image-window handler:

| Command | Common example | Note |
|---|---|---|
| `morphology_defragment` |  | Keep the principal connected component. |
| `morphology_fill_holes` |  | Fill enclosed holes in 3D. |
| `morphology_fill_holes_by_slice` |  | Fill holes independently by slice. |
| `morphology_defragment_by_size` |  | Remove components below a size ratio; optional parameter defaults to `0.05`. |
| `morphology_dilation` |  | Dilate each label. |
| `morphology_erosion` |  | Erode each label. |
| `morphology_opening` |  | Apply opening to each label. |
| `morphology_closing` |  | Apply closing to each label. |
| `morphology_edge` |  | Extract 3D label edges. |
| `morphology_edge_xy` |  | Extract edges in XY planes. |
| `morphology_edge_xz` |  | Extract edges in XZ planes. |
| `morphology_smoothing` |  | Smooth binary or multi-label masks. |
| `sobel_filter` |  | Apply Sobel filtering. |
| `gaussian_filter` |  | Apply Gaussian smoothing. |
| `mean_filter` |  | Apply mean filtering. |
| `smoothing_filter` |  | Apply anisotropic diffusion smoothing. |
| `normalize` |  | Normalize image intensity. |
| `normalize_otsu_median` |  | Normalize using Otsu/median segmentation statistics. |
| `flip_x` |  | Flip voxel data along X. |
| `flip_y` |  | Flip voxel data along Y. |
| `flip_z` |  | Flip voxel data along Z. |
| `select_value` |  | Create a binary mask selecting one exact value. |
| `add_value` |  | Add a scalar constant. |
| `multiply_value` |  | Multiply by a scalar constant. |
| `lower_threshold` |  | Clamp values below the supplied threshold. |
| `upper_threshold` |  | Clamp values above the supplied threshold. |
| `threshold` |  | Binarize values greater than the supplied threshold. |
| `otsu_threshold` |  | Binarize using Otsu threshold multiplied by the supplied ratio. |
| `equation` |  | Apply a TIPL equation expression. |
| `set_transformation` |  | Replace the 4×4 image transformation; parameter contains 16 values. |
| `set_translocation` |  | Set transformation translation components. |
| `set_mni` |  | Set the MNI-space flag (`0` or `1`). |
| `upsampling` |  | Upsample image/labels and update voxel size/transformation. |
| `downsampling` |  | Downsample image/labels and update voxel size/transformation. |
| `header_flip_x` |  | Flip only the header transformation along X. |
| `header_flip_y` |  | Flip only the header transformation along Y. |
| `header_flip_z` |  | Flip only the header transformation along Z. |
| `header_swap_xy` |  | Swap X/Y axes in header metadata. |
| `header_swap_xz` |  | Swap X/Z axes in header metadata. |
| `header_swap_yz` |  | Swap Y/Z axes in header metadata. |
| `swap_xy` |  | Swap voxel X/Y axes and voxel sizes. |
| `swap_xz` |  | Swap voxel X/Z axes and voxel sizes. |
| `swap_yz` |  | Swap voxel Y/Z axes and voxel sizes. |
| `crop_to_fit` |  | Crop to nonzero content with optional margin. |
| `transform` |  | Resample/reorient to a supplied transformation matrix. |
| `translocate` |  | Shift image by voxel offsets and update transformation. |
| `resize` |  | Resize canvas to `width height depth`, anchored at origin. |
| `resize_at_center` |  | Resize canvas around image center. |
| `reshape` |  | Reshape data to new dimensions. |
| `regrid` |  | Resample to one or three supplied voxel sizes. |
| `concatenate_image` |  | Append another image along Z; width/height must match. |
| `refine_label` |  | Refine labels using a reference image file. |
| `load_image` |  | Replace data with another image mapped into current space. |
| `multiply_image` |  | Multiply voxelwise by another mapped image. |
| `add_image` |  | Add another mapped image voxelwise. |
| `minus_image` |  | Subtract another mapped image voxelwise. |
| `max_image` |  | Take voxelwise maximum with another image. |
| `min_image` |  | Take voxelwise minimum with another image. |
| `save` |  | Save image data and metadata. |
| `open` |  | Open/replace image data and metadata. |

## File and window workflows

To open one `.fz`, `.sz`, or image when only the main window exists, send its
absolute path as raw pipe text, then call `LIST` to obtain the new numeric
window ID. `open_fib` requires an existing tracking window and cannot create
the first one.

Use exactly one open mechanism for a file. Raw path transport and `hub open`
each open the file and create its window. After either request, call `LIST` and
use the newly added numeric ID. Do not repeat the open request or then call
`open_fib` for the same file; that intentionally creates another tracking
window. One open request should create one window, so duplicate windows usually
indicate that the file was opened more than once.

In DSI Studio, FIB means `.fz`; `.sz` is an SRC file.

To open multiple images in one image window, send one flat `open_image` command
to the numeric main-window ID. Do not send separate commands, target an image
window, split a path into fields, or substitute TIPL `add_image`.

After a window opens or closes, refresh `LIST`. Otherwise retain the IDs while
using later `LIST` replies for current busy state.

Most Hub FIB files contain an HTTP reference to their native T1w. Opening the
FIB alone does not download the T1w. First call `list_slice` on the new tracking
window. Then either pass the returned T1w name or index directly to
`segment_brain`, which selects and loads it automatically, or call
`set_slice <index>` to start download and registration separately. In the
second workflow, poll `LIST` until the tracking window is idle, then confirm
`downloaded=1`, `ready=1`, and `registering=0` with `list_slice`.

## Asynchronous and long-running work

Fiber tracking is asynchronous. A successful `run_tracking` reply means
tracking started. Poll `LIST`; the target row's `tracking-jobs` reaches zero
when no active tracking bundle remains. Call full `list_tract` afterward only
when bundle details are needed.

For a synchronous long-running command, the target window has `busy=1`, and the
global `level`/`status` describe available TIPL progress. Do not repeat the
operation because its original `CMD` connection is still pending. Use `LIST`
for status and process every returned `PROMPT`.

Slice loading and registration make the tracking-window row busy. When it
becomes idle, use `list_slice` if exact `downloaded`, `ready`, `registering`, or
`registered` fields must be verified.

### Wait etiquette

Before starting substantial work, inspect the first `LIST` line and all window
rows. Loading, registration, reconstruction, segmentation, batch processing,
fiber tracking, and similarly CPU/GPU-intensive work should not be stacked by
default.

- When `busy=0`, proceed.
- When the activity was started by this agent, do not start another substantial
  operation. Send one concise `CHAT` saying what is still running and that the
  user may interrupt or terminate it, then wait.
- When activity was already running before this agent's intended operation, or
  appears to belong to the user or another agent, send one concise `CHAT` such
  as: `DSI Studio is busy with <status>. I will wait by default. You may
  terminate the current work or tell me to proceed right away.` Do not start the
  substantial operation without an explicit instruction to proceed immediately.
- Poll only silent `LIST` after 4 seconds. While structural state remains
  unchanged, double the interval to 8, 16, 32, 64, 128, 256, 512, and 900
  seconds, then continue every 900 seconds.
- Reset to 4 seconds when global `busy` or `level`, the status phase, any
  window's `busy`, or any `tracking-jobs` value changes. Do not reset for
  changing numerical progress such as `(3/100)`.
- Use a local sleep or timer between checks. Waiting should perform no model
  reasoning and consume no task tokens. Silent `LIST` polls omit `chat` and are
  not stored in AI history or console. Do not send repeated `CHAT`, call `LOG`,
  request detailed lists, or narrate unchanged polling.
- Inspect every complete `LIST` reply for `PROMPT`. User instructions override
  waiting. When global `busy` becomes `0`, continue the pending operation
  without another status message unless the next phase itself warrants one.

Never automatically repeat a failed, timed-out, unavailable, or unexpected
operation.

## Token efficiency

- Process a new Codex task in its first run. Never create or wait for a bootstrap
  run.
- Use `CODEX_THREAD_ID` immediately instead of spending requests discovering or
  replacing the session ID.
- Retain window IDs until windows change; poll compact top-level `LIST` only
  while waiting for global or per-window activity to change.
- During waiting, use silent `LIST` with exponential backoff up to 900 seconds;
  do not use model reasoning, chat, `LOG`, or detailed discovery while state is
  unchanged.
- Use the `tracking-jobs` column instead of `list_tract status` when only
  completion is needed.
- Use parameterless `list_param` once, then query only needed IDs.
- Poll targeted detailed state rather than `LOG`.
- Batch only safe independent synchronous commands for one window.
- Attach progress `chat` to an existing request, but do not attach chat to every
  status poll.
- Send `cwd` once and omit it until it changes.
- Reuse discovered names and indices until relevant state changes.
- Stop after verification and one final `CHAT`.

## Safety and verification

Region types are `0=ROI`, `1=ROA`, `2=End`, `3=Seed`, `4=Terminative`,
`5=NotEnd`, and `6=Limiting`. Colors are unsigned packed Qt ARGB integers.

Do not use TumorSynth until its current model bug is fixed.

Obtain permission before overwriting files. Do not answer confirmation dialogs
remotely. Verify expected output paths, files, tract bundles, regions, and
renderings after completion.

If DSI Studio resumes an agent, reconnect using the exact same `agent` and
`session`, inspect every reply for `PROMPT`, and exit naturally when none
remains.
