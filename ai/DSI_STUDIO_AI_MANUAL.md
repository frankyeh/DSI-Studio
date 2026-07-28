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
- Before substantial work, inspect `LIST` and follow the wait etiquette when
  DSI Studio is busy.
- Every `CMD`, including every `list_*` command, requires a numeric `window`
  returned by `LIST`.
- Every command name and parameter inside `command` must be a JSON string.
  Write slice index `"7"`, not numeric `7`.
- Use GUI commands. Do not use `run_cli` unless the user explicitly requests
  CLI execution.
- Discover names, indices, and parameter IDs before mutation. Never guess them.
- `okay:true` means the handler accepted the command; asynchronous work may
  still be running.
- Poll top-level `LIST` for activity. Use targeted `list_*` commands only for
  detailed verification.
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

Silent `LIST` requests without `chat` are not written to AI history or console
and are preferred for polling.

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

`LOG` reads new DSI Studio console output. It does not publish a message to the
user.

### CHAT

```json
{"agent":"Codex","session":"<uuid>","request":"CHAT","chat":"The requested fiber tracking has completed successfully."}
```

`CHAT` is a standalone top-level request. It uses no `window` or `command` field.

### TITLE

```json
{"agent":"Codex","session":"<uuid>","request":"TITLE","title":"Segment T1w image"}
```

### Open one local file

```text
C:\data\subject.fz
```

## CMD format

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["list_slice"]}
```

All command elements must be strings:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["set_slice","7"]}
```

Use `"7"`, not `7`.

Do not batch destructive, asynchronous, output-dependent, or modal-opening
commands. Do not send an empty command array.

## Command examples and inventory

Blank example cells mark source commands without a documented recommended
example. Every command name and parameter remains a quoted JSON string.

### Main-window and Hub commands

| Command | Common example |
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

| Command | Common example |
|---|---|
| `list_slice` | `["list_slice"]` |
| `set_slice` | `["set_slice","7"]` |
| `set_slice_by_name` | `["set_slice_by_name","T1w"]` |
| `move_slice` | `["move_slice","80 100 80"]` |
| `list_unet` | `["list_unet"]` |
| `segment_brain current slice` | `["segment_brain","<model from list_unet>"]` |
| `segment_brain by name` | `["segment_brain","<model from list_unet>","T1w"]` |
| `segment_brain by index` | `["segment_brain","<model from list_unet>","7"]` |
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

### Region commands

`region_action_<operation>` uses `command[1]` for one region index or an
`&`-separated index list. `command[2]` supplies the extra value for threshold
and voxel-dilation operations.

| Command | Common example |
|---|---|
| `list_region` | `["list_region"]` |
| `list_atlas` | `["list_atlas"]` |
| `add_region_from_atlas` | `["add_region_from_atlas","<region returned by atlas selection>"]` |
| `set_region_name` | `["set_region_name","0","Left CST seed"]` |
| `set_region_type` | `["set_region_type","0","3"]` |
| `set_region_color` | `["set_region_color","0","4294901760"]` |
| `show_only_regions` | `["show_only_regions","0&2&5"]` |
| `new_region` |  |
| `new_region_whole_brain_seed` |  |
| `new_region_from_threshold` |  |
| `new_region_from_mni` |  |
| `new_region_from_sphere` |  |
| `open_region` |  |
| `open_mni_region` |  |
| `save_region` |  |
| `save_4d_region` |  |
| `save_all_regions` |  |
| `save_all_regions_to_folder` |  |
| `save_region_info` |  |
| `load_region_color` |  |
| `save_region_color` |  |
| `delete_region` |  |
| `delete_all_regions` |  |
| `copy_region` |  |
| `merge_regions` |  |
| `check_region` |  |
| `check_all_regions` |  |
| `uncheck_all_regions` |  |
| `move_up_region` |  |
| `move_down_region` |  |
| `move_region` |  |
| `move_slice_to_region` |  |
| `show_device_statistics` |  |
| `save_device_statistics` |  |
| `show_region_statistics` |  |
| `save_region_statistics` |  |
| `show_t2r` |  |
| `save_t2r` |  |
| `show_tract_statistics` |  |
| `save_tract_statistics` |  |
| `show_tract_recognition` |  |
| `save_tract_recognition` |  |
| `region_action_shiftx` |  |
| `region_action_shiftnx` |  |
| `region_action_shifty` |  |
| `region_action_shiftny` |  |
| `region_action_shiftz` |  |
| `region_action_shiftnz` |  |
| `region_action_flipx` |  |
| `region_action_flipy` |  |
| `region_action_flipz` |  |
| `region_action_smoothing` |  |
| `region_action_erosion` |  |
| `region_action_dilation` |  |
| `region_action_opening` |  |
| `region_action_closing` |  |
| `region_action_defragment` |  |
| `region_action_negate` |  |
| `region_action_dilation_by_voxel` |  |
| `region_action_threshold` |  |
| `region_action_threshold_current` |  |
| `region_action_dilation_by_threshold` |  |
| `region_action_erosion_by_threshold` |  |
| `region_action_separate` |  |
| `region_action_sort_name` |  |
| `region_action_sort_x` |  |
| `region_action_sort_y` |  |
| `region_action_sort_z` |  |
| `region_action_sort_size` |  |
| `region_action_1st_ex_all` |  |
| `region_action_all_ex_1st` |  |
| `region_action_all_inter_1st` |  |
| `region_action_all_to_1st` |  |
| `region_action_refine_all` |  |

### Tracking parameter commands

| Command | Common example |
|---|---|
| `list_param all IDs` | `["list_param"]` |
| `list_param one ID` | `["list_param","step_size"]` |
| `set_param` | `["set_param","step_size","1.0"]` |
| `set_params` | `["set_params","step_size=1.0&min_length=20"]` |

### Fiber tracking and tract commands

| Command | Common example |
|---|---|
| `list_tract` | `["list_tract"]` |
| `list_tract status` | `["list_tract","status"]` |
| `run_tracking` | `["run_tracking","Whole Brain"]` |
| `run_tracking with regions` | `["run_tracking","Corticospinal Tract","0:3&1:0&2:1"]` |
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

| Command | Common example |
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

DSI Studio handles these commands directly before delegating to TIPL:

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

### TIPL generic image commands

TIPL `cmd.hpp` supplies these commands:

| Command | Common example |
|---|---|
| `morphology_defragment` |  |
| `morphology_fill_holes` |  |
| `morphology_fill_holes_by_slice` |  |
| `morphology_defragment_by_size` |  |
| `morphology_dilation` |  |
| `morphology_erosion` |  |
| `morphology_opening` |  |
| `morphology_closing` |  |
| `morphology_edge` |  |
| `morphology_edge_xy` |  |
| `morphology_edge_xz` |  |
| `morphology_smoothing` |  |
| `sobel_filter` |  |
| `gaussian_filter` |  |
| `mean_filter` |  |
| `smoothing_filter` |  |
| `normalize` |  |
| `normalize_otsu_median` |  |
| `flip_x` |  |
| `flip_y` |  |
| `flip_z` |  |
| `select_value` |  |
| `add_value` |  |
| `multiply_value` |  |
| `lower_threshold` |  |
| `upper_threshold` |  |
| `threshold` |  |
| `otsu_threshold` |  |
| `equation` |  |
| `set_transformation` |  |
| `set_translocation` |  |
| `set_mni` |  |
| `upsampling` |  |
| `downsampling` |  |
| `header_flip_x` |  |
| `header_flip_y` |  |
| `header_flip_z` |  |
| `header_swap_xy` |  |
| `header_swap_xz` |  |
| `header_swap_yz` |  |
| `swap_xy` |  |
| `swap_xz` |  |
| `swap_yz` |  |
| `crop_to_fit` |  |
| `transform` |  |
| `translocate` |  |
| `resize` |  |
| `resize_at_center` |  |
| `reshape` |  |
| `regrid` |  |
| `concatenate_image` |  |
| `refine_label` |  |
| `load_image` |  |
| `multiply_image` |  |
| `add_image` |  |
| `minus_image` |  |
| `max_image` |  |
| `min_image` |  |
| `save` |  |
| `open` |  |

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
- When activity was started by this agent, send one concise `CHAT`, then wait.
- When activity appears to belong to the user or another agent, send one concise
  `CHAT` saying DSI Studio is busy and that you will wait unless instructed to
  proceed immediately.
- Poll only silent `LIST` after 4 seconds. While structural state is unchanged,
  double the interval to 8, 16, 32, 64, 128, 256, 512, and 900 seconds, then
  continue every 900 seconds.
- Reset to 4 seconds only for structural state changes, not numerical progress.
- Waiting should perform no model reasoning and consume no task tokens.
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
- Attach short progress `chat` to an already-needed request.
- Send `cwd` once and omit it until it changes.
- Stop after verification and one final `CHAT`.

## Safety and verification

Do not use TumorSynth until its current model bug is fixed. Obtain permission
before overwriting files. Do not answer confirmation dialogs remotely. Verify
expected output paths, files, tract bundles, regions, and renderings after
completion.
