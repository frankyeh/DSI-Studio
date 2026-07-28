# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` completely. Read the operating rules and examples
below, then search this manual only for commands needed by the request. Do not
reread the entire file for every action.

## Operating rules

- Each named-pipe connection sends exactly one request and then closes.
- Send JSON for every AI request. Raw non-JSON text is reserved for one existing
  local file path.
- Use a native client for `\\.\pipe\dsi-studio` first. Do not use
  `dsi_agent.ps1`, `dsi_studio.exe`, or another wrapper unless direct access is
  unavailable or fails and the user approves the fallback.
- Send separate non-empty `agent` and `session` fields and reuse the exact pair.
  `agent` must include `Codex` or `Claude` and must not contain `@`.
- For a new Codex chat launched by DSI Studio, process the initiating task in
  the first run and use `CODEX_THREAD_ID` as `session` immediately.
- For Claude Code, read `~/.claude/sessions/<pid>.json` and use `sessionId`, not
  `name`.
- Ollama is a model provider, not an agent identity.
- Send one concise `TITLE` after understanding the initiating prompt.
- Call top-level `LIST` first. It returns global activity plus every supported
  window's numeric ID and busy state; it does not use a window ID.
- Before loading, registration, reconstruction, segmentation, batch processing,
  fiber tracking, or other substantial work, inspect `LIST` and follow the wait
  etiquette when DSI Studio is busy.
- Every `CMD`, including every `list_*` command, requires a numeric `window`
  returned by `LIST`.
- Every command name and parameter inside `command` must be a JSON string.
  Write slice index `"7"`, not numeric `7`.
- Use GUI commands. Do not use `run_cli` unless the user explicitly requests
  CLI execution.
- Discover names, indices, and parameter IDs before mutation. Never guess them.
- `okay:true` means the handler accepted the command; asynchronous work may
  still be running.
- Poll top-level `LIST` for global and per-window activity. Use a targeted
  `list_*` command only when detailed verification is needed.
- Use `LOG` only for failures or states that `LIST` and targeted discovery
  cannot explain.
- User-facing text uses the top-level `chat` field. A standalone message uses
  top-level request `CHAT`, not a `CMD` command.
- If a required window disappears or returns `window not found`, assume the
  user closed it. Do not reopen or retry it without asking.
- Confirm destructive actions and overwrites. Verify every output file.
- Do not answer modal dialogs remotely; tell the user what action is required.
- Inspect every complete reply for a queued `PROMPT`.

## Top-level request examples

### LIST

```json
{"agent":"Codex","session":"<uuid>","cwd":"C:/work","request":"LIST"}
```

The reply is compact tab-separated text:

```text
OKAY<TAB>busy<TAB>level<TAB>status
type<TAB>id<TAB>busy<TAB>tracking-jobs<TAB>title
```

Example reply:

```text
OKAY	1	2	segment_brain (3/5)
main	1	0	0	DSI Studio
tracking	2	1	0	C:/data/subject.fz
tracking	3	1	2	C:/data/group.fz
```

- Global `busy=1` means a TIPL operation or a supported window is busy.
- `level` is the TIPL nesting depth excluding the persistent application root.
  It is `1` when only asynchronous work such as fiber tracking is active.
- Each window row reports `type`, numeric `id`, `busy`, `tracking-jobs`, and
  normalized title/path.
- Silent `LIST` requests without `chat` are not written to AI history or console
  and are preferred for polling.

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

`LOG` reads new DSI Studio console output. It returns at most 4096 new console
characters and advances the session cursor. It does not publish a message to
the user.

### CHAT

```json
{"agent":"Codex","session":"<uuid>","request":"CHAT","chat":"The requested fiber tracking has completed successfully."}
```

`CHAT` is a standalone top-level request. It uses no `window` or `command` field.
A non-empty message returns `OKAY`. Use it for the final answer, a required user
decision, or one blocked/waiting update.

### TITLE

```json
{"agent":"Codex","session":"<uuid>","request":"TITLE","title":"Segment T1w image"}
```

A later `TITLE` replaces the current title; send it only when renaming is useful.

### Open one local file

Send one existing absolute path as raw pipe text:

```text
C:\data\subject.fz
```

## CMD format

Use the numeric window ID returned by `LIST`:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["list_slice"]}
```

All command elements must be strings. For example, select slice index 7 with:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["set_slice","7"]}
```

Use `"7"`, not `7`.

A safe same-window batch:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":[["list_param"],["list_slice"]]}
```

Do not batch destructive, asynchronous, output-dependent, or modal-opening
commands. Do not send an empty command array.

## Command examples and inventory

Wrap each example array below in the standard `CMD` request shown above. Replace
values in angle brackets with values returned by the corresponding discovery
command. Every parameter remains a quoted JSON string. Blank example cells mark
source commands that are listed for discovery but do not yet have a documented
recommended example.

### Main-window and Hub commands

| Task or command | Common example |
|---|---|
| `list_recent_fib` | `["list_recent_fib"]` |
| `list_recent_src` | `["list_recent_src"]` |
| `hub repos` | `["hub","repos"]` |
| `hub tags` | `["hub","tags","<repo from hub repos>"]` |
| `hub files` | `["hub","files","<repo>","<tag>","","0","20"]` |
| `hub open` | `["hub","open","<repo>","<tag>","0"]` |
| `hub download` | `["hub","download","<repo>","<tag>","0","C:/data"]` |
| `hub help` |  |
| `open_image` | `["open_image","C:/data/T1w.nii.gz","C:/data/T2w.nii.gz"]` |
| `run_cli` | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` |

`hub files` returns `index`, `file`, `size`, and `downloaded`. Use a returned
filename or quoted index for `hub open` and `hub download`. If the download
directory does not exist, DSI Studio creates it and reports `directory_created`.
A successful reply is returned only after the data have been written to disk.

### Tracking-window file, workspace, and setting commands

| Command | Common example |
|---|---|
| `open_fib` |  |
| `correct_bias_field` |  |
| `save_fib_as` |  |
| `open_mapping` |  |
| `save_workspace` |  |
| `load_workspace` |  |
| `save_setting` |  |
| `save_rendering_setting` |  |
| `save_tracking_setting` |  |
| `load_setting` |  |
| `load_rendering_setting` |  |
| `load_tracking_setting` |  |
| `restore_rendering` |  |
| `restore_tracking` |  |
| `presentation_mode` |  |

### Slice and segmentation commands

| Task or command | Common example |
|---|---|
| `list_slice` | `["list_slice"]` |
| `set_slice` | `["set_slice","7"]` |
| `set_slice_by_name` | `["set_slice_by_name","T1w"]` |
| `move_slice` | `["move_slice","80 100 80"]` |
| `list_unet` | `["list_unet"]` |
| `segment_brain` current slice | `["segment_brain","<model from list_unet>"]` |
| `segment_brain` by name | `["segment_brain","<model from list_unet>","T1w"]` |
| `segment_brain` by index | `["segment_brain","<model from list_unet>","7"]` |
| `enable_slice` |  |
| `set_slice_contrast` |  |
| `set_slice_dir_color` |  |
| `set_slice_overlay` |  |
| `set_slice_stay` |  |
| `add_slice` |  |
| `add_mni_slice` |  |
| `skull_strip_slice` |  |
| `save_roi_screen` |  |
| `save_slice_image` |  |
| `save_slice_mni_image` |  |
| `save_slice_mapping` |  |
| `open_slice_mapping` |  |
| `save_slice_volume` |  |
| `delete_slice` |  |

`list_slice` returns:

```text
index current name ready registering downloaded registered
```

For custom volumes, `registering=1` means registration is still running.
`segment_brain` first tries an exact slice name and then a quoted numeric index.
It waits for custom-slice registration and returns after inference and region
creation finish. Verify the result with `list_region`.

### Region commands

| Task or command | Common example |
|---|---|
| `list_region` | `["list_region"]` |
| `list_atlas` | `["list_atlas"]` |
| `add_region_from_atlas` | `["add_region_from_atlas","<region returned by atlas selection>"]` |
| `set_region_name` | `["set_region_name","0","Left CST seed"]` |
| `set_region_type` | `["set_region_type","0","3"]` |
| `set_region_color` | `["set_region_color","0","4294901760"]` |
| `show_only_regions` | `["show_only_regions","0&2&5"]` |
| `new_region` |  |
| `open_region` |  |
| `open_regions` |  |
| `save_region` |  |
| `save_region_as` |  |
| `save_all_regions` |  |
| `save_all_regions_to_folder` |  |
| `delete_region` |  |
| `delete_all_regions` |  |
| `copy_region` |  |
| `merge_region` |  |
| `merge_all_regions` |  |
| `add_region_from_threshold` |  |
| `add_region_from_tract` |  |
| `add_region_from_endpoints` |  |
| `check_region` |  |
| `check_uncheck_all_region` |  |
| `move_region` |  |
| `shift_region` |  |
| `flip_region` |  |
| `sort_region` |  |
| `separate_region` |  |
| `smooth_region` |  |
| `erode_region` |  |
| `dilate_region` |  |
| `defragment_region` |  |
| `negate_region` |  |
| `threshold_region` |  |

Region types are `0=ROI`, `1=ROA`, `2=End`, `3=Seed`, `4=Terminative`,
`5=NotEnd`, and `6=Limiting`. Colors are unsigned packed Qt ARGB integers.

### Tracking parameter commands

| Task or command | Common example |
|---|---|
| `list_param` all IDs | `["list_param"]` |
| `list_param` one ID | `["list_param","step_size"]` |
| `set_param` | `["set_param","step_size","1.0"]` |
| `set_params` | `["set_params","step_size=1.0&min_length=20"]` |

Call parameterless `list_param` first and use only IDs it returns. Values are
strings even when they represent numbers.

### Fiber tracking and tract commands

| Task or command | Common example |
|---|---|
| `list_tract` | `["list_tract"]` |
| `list_tract status` | `["list_tract","status"]` |
| `run_tracking` | `["run_tracking","Whole Brain"]` |
| `run_tracking` with regions | `["run_tracking","Corticospinal Tract","0:3&1:0&2:1"]` |
| `list_auto_tract` | `["list_auto_tract"]` |
| `run_auto_track` | `["run_auto_track","Corticospinal Tract"]` |
| `show_only_tracts` | `["show_only_tracts","0&2&5"]` |
| `enable_auto_tract` |  |
| `open_tract` |  |
| `open_tracts` |  |
| `open_tract_dir` |  |
| `save_tract` |  |
| `save_mni_tract` |  |
| `save_template_tract` |  |
| `save_slice_tract` |  |
| `save_tract_endpoint` |  |
| `save_mni_tract_endpoint` |  |
| `save_slice_tract_endpoint` |  |
| `save_all_tracts` |  |
| `save_all_tracts_to_folder` |  |
| `save_all_tracts_to_dir` |  |
| `save_tdi` |  |
| `save_tdi2` |  |
| `save_tract_values` |  |
| `tract_to_region` |  |
| `endpoint_to_region` |  |
| `update_tract` |  |
| `delete_tract` |  |
| `delete_all_tracts` |  |
| `copy_tract` |  |
| `rename_tract` |  |
| `merge_tract` |  |
| `merge_all_tracts` |  |
| `merge_tract_by_name` |  |
| `sort_tract_by_name` |  |
| `trim_tract` |  |
| `trim_all_tracts` |  |
| `cut_tract` |  |
| `cut_by_slice` |  |
| `filter_tract` |  |
| `remove_repeated_tracts` |  |
| `recognize_tract` |  |
| `cluster_tract` |  |
| `cluster_all_tracts` |  |
| `check_tract` |  |
| `check_uncheck_all_tract` |  |
| `set_tract_color` |  |
| `set_tract_color_style` |  |
| `set_tract_visible` |  |

`run_tracking` uses the current tracking settings and the directional
information stored in the loaded FIB. Its required second element is the new
bundle label, not a reconstruction or tracking-method name. Each optional ROI
item is `region-index:type`.

Fiber tracking is asynchronous. A successful reply means tracking started.
Poll top-level `LIST`; the target window's `tracking-jobs` reaches zero when no
active tracking bundle remains. Request full `list_tract` afterward only when
bundle details are needed.

### Device commands

| Command | Common example |
|---|---|
| `new_device` |  |
| `move_device` |  |
| `push_device` |  |
| `pull_device` |  |
| `copy_device` |  |
| `set_acpc` |  |
| `delete_device` |  |
| `delete_all_devices` |  |
| `save_all_devices` |  |

### Rendering, camera, and surface commands

| Task or command | Common example |
|---|---|
| `rotate` | `["rotate","15 1 0 0"]` |
| `save_hd_screen` | `["save_hd_screen","C:/output/tracts.png","1920 1080"]` |
| `set_view` |  |
| `set_zoom` |  |
| `set_camera` |  |
| `get_camera` |  |
| `open_camera` |  |
| `save_camera` |  |
| `store_camera` |  |
| `store_camera1` |  |
| `store_camera2` |  |
| `restore_camera` |  |
| `restore_camera1` |  |
| `restore_camera2` |  |
| `save_screen` |  |
| `add_surface` |  |
| `delete_surface` |  |
| `load_surface` |  |
| `save_surface` |  |
| `set_surface_color` |  |
| `set_surface_alpha` |  |
| `set_surface_visible` |  |
| `set_device_color` |  |

### Image-window commands

| Command | Common example |
|---|---|
| `change_type` |  |
| `bias_field_correction` |  |
| `brain_extraction` |  |
| `segmentation` |  |
| `deface` |  |
| `rotate_to_image` |  |
| `warp_to_image` |  |
| `apply_to_image` |  |

Other generic image operations are delegated to TIPL's image-command handler
and are not enumerated by literal DSI Studio `cmd[0]` comparisons.

## Reply formats

- `LIST`, `LOG`, `CHAT`, `TITLE`, file-open replies, and validation errors are
  text.
- `CMD` returns a JSON array of `{index,okay,output,error?}` objects.
- List-command data remains tab-separated text inside `output`.
- `CHAT` returns `OKAY` or `ERROR<TAB>missing chat` and may append a queued
  `PROMPT`.
- `CMD` may attach `prompt` to its final result object.
- Paths returned by `LIST`, `list_recent_fib`, and `list_recent_src` use `/` as
  the canonical separator.

## File and window workflow

To open one `.fz`, `.sz`, or image when only the main window exists, send its
absolute path as raw pipe text, then call `LIST` for the new numeric window ID.
Use exactly one open mechanism for a file; repeating the open request creates a
duplicate window.

In DSI Studio, FIB means `.fz`; `.sz` is an SRC file.

Most Hub FIB files contain an HTTP reference to their native T1w. Opening the
FIB alone does not download it. Call `list_slice`, then pass the returned T1w
name/index directly to `segment_brain`, or use `set_slice` to start download and
registration separately. Poll `LIST` until the tracking window is idle, then
confirm `downloaded=1`, `ready=1`, and `registering=0` with `list_slice`.

## Wait etiquette

Before substantial work, inspect the global `LIST` line and every window row.
Loading, registration, reconstruction, segmentation, batch processing, and
fiber tracking should not be stacked by default.

- When `busy=0`, proceed.
- When activity was started by this agent, send one concise `CHAT` saying what
  is running and that the user may interrupt or terminate it, then wait.
- When activity predates the intended operation or appears to belong to the
  user or another agent, send one concise `CHAT`: `DSI Studio is busy with
  <status>. I will wait by default. You may terminate the current work or tell
  me to proceed right away.` Do not proceed without explicit instruction.
- Poll only silent `LIST` after 4 seconds. While structural state is unchanged,
  double the interval to 8, 16, 32, 64, 128, 256, 512, and 900 seconds, then
  continue every 900 seconds.
- Reset to 4 seconds when global `busy` or `level`, the status phase, any
  window's `busy`, or any `tracking-jobs` value changes. Do not reset for only
  numerical progress such as `(3/100)`.
- Use a local sleep or timer between checks. Waiting should perform no model
  reasoning and consume no task tokens. Do not send repeated `CHAT`, call
  `LOG`, request detailed lists, or narrate unchanged polling.
- Inspect every complete reply for `PROMPT`. User instructions override waiting.

Never automatically repeat a failed, timed-out, unavailable, or unexpected
operation.

## Token efficiency

- Retain window IDs until windows open or close.
- Use silent top-level `LIST` for status polling.
- Use each tracking row's `tracking-jobs` instead of targeted tract polling when
  only completion is needed.
- Call parameterless `list_param` once, then query only needed IDs.
- Use `LOG` only for diagnostics.
- Batch only safe independent synchronous commands for one window.
- Attach short progress `chat` to an already-needed request; do not attach it to
  every `LIST` poll.
- Send `cwd` once and omit it until it changes.
- Stop after verification and one final `CHAT`.

## Safety and verification

Do not use TumorSynth until its current model bug is fixed. Obtain permission
before overwriting files. Do not answer confirmation dialogs remotely. Verify
expected output paths, files, tract bundles, regions, and renderings after
completion.
