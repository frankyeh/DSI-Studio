# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` completely. Read the operating rules and common
syntax below, then search this manual only for commands needed by the request;
do not read the entire inventory.

## Operating rules

- Each named-pipe connection sends exactly one request and then closes.
- Use `LIST` to obtain fresh windows, then target only its numeric window ID.
  Never send `main`, `tracking`, `image`, a title, or a filename as `window`.
- In DSI Studio, FIB means `.fz`; never substitute `.sz`, which is an SRC file.
- Use GUI control by default. Use `run_cli` only when the user explicitly asks
  for CLI operation.
- Commands and every parameter are JSON strings. Never guess names or indices.
- Use list commands before mutation and after completion.
- `okay:true` means the handler accepted the command, not necessarily that
  asynchronous work finished.
- Confirm destructive actions and overwrites. Verify every output file.
- Do not use TumorSynth until its current model bug is fixed.
- Native structural images align reliably with native-space GQI `.fz` data.

## Discovery

| Need | Command |
|---|---|
| Windows | JSON `LIST` request |
| Main-window recent files | `list_recent` |
| Slices | `list_slice` |
| Regions and ROI/ROA type, color, resolution | `list_region` |
| Tracts and visible/deleted counts | `list_tract` |
| One GUI parameter | `list_param`, exact parameter ID |
| Atlases | `list_atlas` |
| Segmentation models and descriptions | `list_unet` |
| Automatic tract names | `list_auto_tract` |
| Console/errors | JSON `LOG` request |

## Common syntax

Parameters shown below are separate JSON array elements.

| Task | Command array |
|---|---|
| Hub repositories | `["hub","repos"]` |
| Hub tags | `["hub","tags",repo]` |
| Hub files | `["hub","files",repo,tag,filter,offset,limit]` |
| Open Hub file | `["hub","open",repo,tag,file]` |
| Download Hub file | `["hub","download",repo,tag,file,directory]` |
| Open local images together | Main only: one flat `["open_image",full-path1,full-path2,...]` |
| Run CLI action | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` |
| Select slice | `["set_slice",zero-based-index]` |
| Move slices | `["move_slice","x y z"]` |
| Segment a slice | `["segment_brain",exact-model,exact-slice]` |
| Add atlas region | `["add_region_from_atlas",exact-region]` |
| Set region metadata | `["set_region_name",index,name]`, `set_region_type`, `set_region_color` |
| Show only regions | `["show_only_regions","0&2&5"]` |
| Show only tracts | `["show_only_tracts","0&2&5"]` |
| Set parameter | `["set_param",parameter-id,value]` |
| Set parameters | `["set_params","id=value&id=value"]` |
| Start tracking | `["run_tracking",name,optional-settings-or-ROI,optional-tolerance]` |
| Automatic tracking | `["run_auto_track",exact-tract-name,optional-ROI]` |
| Rotate 3D view | `["rotate","degrees x y z"]` |
| Save rendering | `["save_hd_screen",path,"width height"]` |

`hub open` may download before opening; poll `LIST`. To open a local file when
only the main window exists, send its absolute filename directly through the
named pipe, then poll `LIST`. Do not send the path as a main-window `CMD`;
`open_fib` requires an existing tracking window.

For multiple images, send **exactly one flat `open_image` command** to the
**main window**. Each complete absolute filepath is one array element. The
first image is displayed and the remaining paths are retained for batch
processing:

```text
["open_image","C:\\data\\t1w1.nii.gz","C:\\data\\t1w2.nii.gz","C:\\data\\t1w3.nii.gz"]
```

Never send three separate `open_image` commands or a batch such as
`[["open_image",file1],["open_image",file2],["open_image",file3]]`. Never
target an `image` window, split `C:\data\` and `file.nii.gz` into separate
fields, or substitute `add_image`. `add_image` modifies the current image but
does not populate the retained file list, so the later save prompt will not
appear.

Refresh `LIST`, target the new `image` window, and send processing commands
such as `["smoothing_filter"]`. Saving over the first original with
`["save","C:\\data\\t1w1.nii.gz"]` opens the existing confirmation:
`Applying processing to other images and save them?`

This is a human-confirmed destructive workflow. Obtain overwrite permission
before `save`, do not answer the modal remotely, and do not batch `save` with
other commands. If the user selects **Yes**, DSI Studio replays smoothing and
saving for the remaining files, mapping each output to its original filename.
If the user selects **No**, only the first image is saved. Verify all expected
files after the dialog closes.

`run_cli` takes one complete DSI Studio argument string, requires `--action`,
and is not a shell. For an explicitly requested CLI batch, prefer one wildcard
loop over repeated commands:

```text
["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]
```

Use an absolute wildcard unless DSI Studio's current directory is the data
directory; there `--loop=*.sz` is sufficient. Wildcards in other CLI arguments
are expanded for each matched loop file. Verify all expected outputs.

With `segment_brain`, use exact values returned by `list_unet` and
`list_slice`. Use ampersand-joined indices for `show_only_regions` and
`show_only_tracts`. Region types are `0=ROI`, `1=ROA`, `2=End`, `3=Seed`,
`4=Terminative`, `5=NotEnd`, and `6=Limiting`; colors are unsigned packed Qt
ARGB integers.

Tracking is asynchronous. Poll `list_tract` until `running=0`. Segmentation is
complete when `list_region` shows the expected output. For rendering/export,
verify the created file and inspect it when possible.

Minimize round trips: use one initial `LIST`, batch independent synchronous
commands for one window, verify concisely, and send the final reply with
`LOG`. Diagnostic `LOG` is incremental and capped; final `LOG` returns no
console history but still advances the agent's cursor.
`[AI REQUEST] ... ⏱` reports synchronous DSI-side request handling time, not
agent runtime or asynchronous completion.

## Command inventory

`Parameters` is the number of fields after the command name. Dynamic commands
and GUI-dependent parameters should be discovered with the list commands
above. `Destructive` commands require confirmation unless the user's request
already grants it.

| Scope | Command | Parameters | Risk | Completion |
|---|---|---:|---|---|
| `atlas` | `add_region_from_atlas` | `1` | Computation | Synchronous extraction; verify `list_region`. |
| `atlas` | `list_atlas` | `0` | Read-only | Immediate list. |
| `auto` | `enable_auto_tract` | `0` | Computation | Synchronous atlas load. |
| `auto` | `list_auto_tract` | `0` | Read-only | Synchronous list. |
| `auto` | `run_auto_track` | `1-2` | Computation | Asynchronous; `"okay":true` means started only. |
| `auto` | `run_tracking` | `1-3` | Computation | Asynchronous; poll `list_tract` until `running=0` and inspect `LOG`. |
| `device` | `copy_device` | `0-1` | GUI-state change | Immediate. |
| `device` | `delete_all_devices` | `0` | Destructive | Immediate. |
| `device` | `delete_device` | `0-1` | Destructive | Immediate. |
| `device` | `move_device` | `1-2` | GUI-state change | Immediate. |
| `device` | `new_device` | `0-1` | GUI-state change | Immediate; anisotropic data may open a modal warning. |
| `device` | `pull_device` | `0-1` | GUI-state change | Immediate. |
| `device` | `push_device` | `0-1` | GUI-state change | Immediate. |
| `device` | `save_all_devices` | `1` | File creation | Synchronous; verify output. |
| `device` | `set_acpc` | `0` | Computation | Synchronous mapping; requires MNI mapping. |
| `image-core` | `brain_extraction` | `1` | Computation | Synchronous download/inference; likely to exceed timeout. |
| `image-core` | `change_type` | `1` | Computation | Synchronous. |
| `image-core` | `deface` | `1` | Computation | Synchronous download/inference; likely to exceed timeout. |
| `image-core` | `save` | `1` | File creation | Synchronous for one image; a multi-file session may open an apply-to-other-images modal. |
| `image-core` | `save_mini` | `1` | File creation | Synchronous; only meaningful for a MAT/FIB/SRC-backed image. |
| `image-core` | `segmentation` | `1` | Computation | Synchronous download/inference; likely to exceed timeout. |
| `image-mat` | `mat_add_float` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_add_int` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_add_int64` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_add_short` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_add_string` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_remove` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_resize` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_set_name` | `1` | Destructive | Synchronous. |
| `image-mat` | `mat_set_value` | `1` | Destructive | Synchronous. |
| `image-transform` | `add_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `add_value` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `apply_to_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `bias_field_correction` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `concatenate_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `crop_to_fit` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `downsampling` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `equation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `flip_x` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `flip_y` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `flip_z` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `gaussian_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `header_flip_x` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `header_flip_y` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `header_flip_z` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `header_swap_xy` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `header_swap_xz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `header_swap_yz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `histogram_sharpening` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `lower_threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `max_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `mean_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `min_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `minus_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_closing` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_defragment` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_defragment_by_size` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_dilation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_edge` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_edge_xy` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_edge_xz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_erosion` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_fill_holes` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_fill_holes_by_slice` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_negate` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_opening` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `morphology_smoothing` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `multiply_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `multiply_value` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `normalize` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `normalize_otsu_median` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `otsu_threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `refine_label` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `regrid` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `reshape` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `resize` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `resize_at_center` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `rotate_to_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `select_value` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `set_mni` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `set_transformation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `set_translocation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `smoothing_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `sobel_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `swap_xy` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `swap_xz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `swap_yz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `transform` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `translocate` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `upper_threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `upsampling` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `image-transform` | `warp_to_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. |
| `main` | `list_recent` | `0` | Read-only | Immediate list. |
| `main` | `open_image` | `1+` | GUI-state change | Main-window only; send one flat command containing all complete paths. |
| `main` | `run_cli` | `1` | Varies | Synchronous CLI action on GUI thread; verify outputs. |
| `main` | `hub download` | `4` | File creation | Deferred file write; verify path and stable size. |
| `main` | `hub files` | `2-5` | GUI-state change | Immediate filtered/paginated list; retry if Hub data is loading. |
| `main` | `hub help` | `0` | Read-only | Immediate. |
| `main` | `hub open` | `3` | File creation | Deferred: handler may schedule the open after a successful result; poll with JSON `LIST`. |
| `main` | `hub repos` | `0` | GUI-state change | Immediate unless Hub initialization itself is still loading. |
| `main` | `hub tags` | `1` | GUI-state change | Immediate list; retry if output says loading. |
| `parameters` | `list_param` | `1` | Read-only | Immediate. |
| `parameters` | `set_param` | `2` | GUI-state change | Immediate state mutation. |
| `parameters` | `set_params` | `1` | GUI-state change | Applies multiple values, then requests one redraw. |
| `region-action` | `region_action_1st_ex_all` | `1` | Destructive | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_all_ex_1st` | `1` | Destructive | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_all_inter_1st` | `1` | Destructive | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_all_to_1st` | `1` | Destructive | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_closing` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_defragment` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_dilation` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_dilation_by_threshold` | `2` | Destructive | Synchronous computation. |
| `region-action` | `region_action_dilation_by_voxel` | `2` | Destructive | Synchronous computation. |
| `region-action` | `region_action_erosion` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_erosion_by_threshold` | `2` | Destructive | Synchronous computation. |
| `region-action` | `region_action_flipx` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_flipy` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_flipz` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_negate` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_opening` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_refine_all` | `1` | Destructive | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_separate` | `1` | Destructive | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_shiftnx` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_shiftny` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_shiftnz` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_shiftx` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_shifty` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_shiftz` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_smoothing` | `0-1` | Destructive | Synchronous. |
| `region-action` | `region_action_sort_name` | `1` | GUI-state change | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_sort_size` | `1` | GUI-state change | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_sort_x` | `1` | GUI-state change | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_sort_y` | `1` | GUI-state change | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_sort_z` | `1` | GUI-state change | Synchronous; refresh `list_region`. |
| `region-action` | `region_action_threshold` | `2` | Destructive | Synchronous computation. |
| `region-action` | `region_action_threshold_current` | `2` | Destructive | Synchronous computation. |
| `region-create` | `list_region` | `0` | Read-only | Immediate. |
| `region-create` | `new_region` | `0` | GUI-state change | Immediate. |
| `region-create` | `new_region_from_mni` | `1` | Computation | Synchronous. |
| `region-create` | `new_region_from_sphere` | `1` | Computation | Synchronous. |
| `region-create` | `new_region_from_threshold` | `1` | Computation | Synchronous computation. |
| `region-create` | `new_region_whole_brain_seed` | `0-1` | Computation | Synchronous computation. |
| `region-io` | `load_region_color` | `1` | GUI-state change | Synchronous; verify save output. |
| `region-io` | `open_mni_region` | `1` | GUI-state change | Synchronous file load. |
| `region-io` | `open_region` | `1` | GUI-state change | Synchronous file load. |
| `region-io` | `save_4d_region` | `1` | File creation | Synchronous; verify outputs. |
| `region-io` | `save_all_regions` | `1` | File creation | Synchronous; verify outputs. |
| `region-io` | `save_all_regions_to_folder` | `1` | File creation | Synchronous; verify outputs. |
| `region-io` | `save_region` | `1-2` | File creation | Synchronous; verify output. |
| `region-io` | `save_region_color` | `1` | File creation | Synchronous; verify save output. |
| `region-io` | `save_region_info` | `1-2` | File creation | Synchronous; verify output. |
| `region-manage` | `check_all_regions` | `0` | GUI-state change | Immediate. |
| `region-manage` | `check_region` | `1-2` | GUI-state change | Immediate; refresh `list_region`. |
| `region-manage` | `show_only_regions` | `1` | GUI-state change | Immediate exact visibility selection; refresh `list_region`. |
| `region-manage` | `copy_region` | `0-1` | GUI-state change | Immediate; refresh list. |
| `region-manage` | `delete_all_regions` | `0` | Destructive | Immediate. |
| `region-manage` | `delete_region` | `0-1` | Destructive | Immediate. |
| `region-manage` | `merge_regions` | `0-1` | Destructive | Synchronous. |
| `region-manage` | `move_down_region` | `1` | GUI-state change | Immediate; refresh `list_region`. |
| `region-manage` | `move_region` | `1-2` | GUI-state change | Immediate. |
| `region-manage` | `move_slice_to_region` | `0-1` | GUI-state change | Immediate. |
| `region-manage` | `move_up_region` | `1` | GUI-state change | Immediate; refresh `list_region`. |
| `region-manage` | `set_region_color` | `2` | GUI-state change | Immediate indexed unsigned ARGB update; refresh `list_region`. |
| `region-manage` | `set_region_name` | `2` | GUI-state change | Immediate; refresh `list_region`. |
| `region-manage` | `set_region_type` | `2` | GUI-state change | Immediate; refresh `list_region`. |
| `region-manage` | `uncheck_all_regions` | `0` | GUI-state change | Immediate. |
| `region-stats` | `save_device_statistics` | `0-1` | File creation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `save_region_statistics` | `0-1` | File creation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `save_t2r` | `0-1` | File creation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `save_tract_recognition` | `0-2` | File creation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `save_tract_statistics` | `0-2` | File creation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `show_device_statistics` | `0-1` | Computation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `show_region_statistics` | `0-1` | Computation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `show_t2r` | `0-1` | Computation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `show_tract_recognition` | `0-2` | Computation | Synchronous computation; verify file when path supplied. |
| `region-stats` | `show_tract_statistics` | `0-2` | Computation | Synchronous computation; verify file when path supplied. |
| `render` | `open_camera` | `1` | GUI-state change | Immediate redraw. |
| `render` | `restore_camera1` | `0` | GUI-state change | Immediate redraw. |
| `render` | `restore_camera2` | `0` | GUI-state change | Immediate redraw. |
| `render` | `restore_camera3` | `0` | GUI-state change | Immediate redraw. |
| `render` | `restore_camera4` | `0` | GUI-state change | Immediate redraw. |
| `render` | `rotate` | `1` | GUI-state change | Immediate redraw. |
| `render` | `save_3view_screen` | `1` | File creation | Synchronous; verify image. |
| `render` | `save_camera` | `1` | File creation | Synchronous; verify output. |
| `render` | `save_h3view_screen` | `1` | File creation | Synchronous; verify image. |
| `render` | `save_hd_screen` | `2` | File creation | Synchronous; verify image dimensions. |
| `render` | `save_rotation_video` | `1` | File creation | Broken; never use as proof of file creation. |
| `render` | `save_screen` | `1` | File creation | Synchronous; verify image. |
| `render` | `save_v3view_screen` | `1` | File creation | Synchronous; verify image. |
| `render` | `set_camera` | `1` | GUI-state change | Immediate redraw. |
| `render` | `set_stereoscopic` | `0` | GUI-state change | Immediate redraw. |
| `render` | `set_view` | `1` | GUI-state change | Immediate redraw. |
| `render` | `set_zoom` | `1` | GUI-state change | Immediate redraw. |
| `render` | `store_camera1` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. |
| `render` | `store_camera2` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. |
| `render` | `store_camera3` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. |
| `render` | `store_camera4` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. |
| `slice` | `add_mni_slice` | `1` | Computation | Load may start asynchronous registration; poll `list_slice` and `LOG`. |
| `slice` | `add_slice` | `1` | Computation | Load may start asynchronous registration; poll `list_slice` and `LOG`. |
| `slice` | `delete_slice` | `0-1` | Destructive | Immediate. |
| `slice` | `enable_slice` | `0-1` | GUI-state change | Immediate redraw. |
| `slice` | `list_slice` | `0` | Read-only | Immediate. |
| `slice` | `move_slice` | `0-1` | GUI-state change | Immediate redraw. |
| `slice` | `open_slice_mapping` | `1-2` | GUI-state change | Synchronous; verify file for save commands. |
| `slice` | `save_roi_screen` | `1` | File creation | Synchronous; verify output. |
| `slice` | `save_slice_image` | `2` | File creation | Synchronous; verify output. |
| `slice` | `save_slice_mapping` | `1-2` | File creation | Synchronous; verify file for save commands. |
| `slice` | `save_slice_mni_image` | `2` | File creation | Synchronous; verify output. |
| `slice` | `save_slice_volume` | `1-2` | File creation | Synchronous; verify file for save commands. |
| `slice` | `set_roi_view` | `1` | GUI-state change | Immediate; an invalid integer silently changes nothing. |
| `slice` | `set_slice` | `0-1` | GUI-state change | Selection is immediate; derived data may remain asynchronous. |
| `slice` | `set_slice_by_name` | `1` | GUI-state change | Immediate. |
| `slice` | `set_slice_contrast` | `0-2` | GUI-state change | Immediate redraw. |
| `slice` | `set_slice_dir_color` | `0-2` | GUI-state change | Immediate redraw. |
| `slice` | `set_slice_overlay` | `0-2` | GUI-state change | Immediate redraw. |
| `slice` | `set_slice_stay` | `0-2` | GUI-state change | Immediate redraw. |
| `slice` | `skull_strip_slice` | `0-1` | Computation | Synchronous computation; may time out. |
| `surface` | `add_surface` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_anterior` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_anterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_left` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_left_anterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_left_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_left_posterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_posterior` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_posterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_right` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_right_anterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_right_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `surface` | `add_surface_upper` | `0-2` | Computation | Synchronous computation; may exceed client timeout. |
| `tracking-files` | `correct_bias_field` | `0` | Computation | Synchronous computation; may exceed client timeout. |
| `tracking-files` | `open_fib` | `1` | GUI-state change | Synchronous load; then refresh `LIST`. |
| `tracking-files` | `open_mapping` | `1` | GUI-state change | Synchronous file load. |
| `tracking-files` | `save_fib_as` | `1` | File creation | Synchronous; verify the output file. |
| `tracking-files2` | `load_rendering_setting` | `1` | GUI-state change | Synchronous. |
| `tracking-files2` | `load_setting` | `1` | GUI-state change | Synchronous. |
| `tracking-files2` | `load_tracking_setting` | `1` | GUI-state change | Synchronous. |
| `tracking-files2` | `load_workspace` | `1` | Destructive | Synchronous file load. |
| `tracking-files2` | `presentation_mode` | `0` | GUI-state change | Immediate. |
| `tracking-files2` | `restore_rendering` | `0` | GUI-state change | Immediate. |
| `tracking-files2` | `restore_tracking` | `0` | GUI-state change | Immediate. |
| `tracking-files2` | `save_rendering_setting` | `1` | File creation | Synchronous; verify file. |
| `tracking-files2` | `save_setting` | `1` | File creation | Synchronous; verify file. |
| `tracking-files2` | `save_tracking_setting` | `1` | File creation | Synchronous; verify file. |
| `tracking-files2` | `save_workspace` | `1` | File creation | Synchronous and potentially large; verify directory contents. |
| `tract-color` | `color_all_cluster` | `0` | GUI-state change | Immediate redraw. |
| `tract-color` | `load_cluster_color` | `1` | GUI-state change | Synchronous. |
| `tract-color` | `load_cluster_values` | `1` | GUI-state change | Synchronous. |
| `tract-color` | `load_tract_color` | `1-2` | GUI-state change | Synchronous. |
| `tract-color` | `load_tract_values` | `1-2` | GUI-state change | Synchronous. |
| `tract-color` | `save_cluster_color` | `1` | File creation | Synchronous. |
| `tract-color` | `save_tract_color` | `1-2` | File creation | Synchronous. |
| `tract-color` | `select_cluster_color` | `1-2` | GUI-state change | Immediate redraw. |
| `tract-discovery` | `list_tract` | `0` | Read-only | Immediate snapshot. |
| `tract-discovery` | `load_tract_atlas` | `0-1` | Computation | Synchronous mapping/computation; may time out. |
| `tract-discovery` | `open_mni_tract` | `1-2` | GUI-state change | Synchronous file load. |
| `tract-discovery` | `open_tract` | `1-2` | GUI-state change | Synchronous file load. |
| `tract-discovery` | `open_tract_name` | `1` | GUI-state change | Immediate. |
| `tract-discovery` | `set_dt_index` | `2` | GUI-state change | Immediate. |
| `tract-edit` | `cut_tract_by_x` | `0-1` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `cut_tract_by_x2` | `0-1` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `cut_tract_by_y` | `0-1` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `cut_tract_by_y2` | `0-1` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `cut_tract_by_z` | `0-1` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `cut_tract_by_z2` | `0-1` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `cut_tract_end_portion` | `0-1` | Destructive | Synchronous. |
| `tract-edit` | `cut_tract_lps_end` | `0-1` | Destructive | Synchronous. |
| `tract-edit` | `cut_tract_rai_end` | `0-1` | Destructive | Synchronous. |
| `tract-edit` | `delete_branch` | `0` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `flip_tract_x` | `0-1` | Destructive | Synchronous. |
| `tract-edit` | `flip_tract_y` | `0-1` | Destructive | Synchronous. |
| `tract-edit` | `flip_tract_z` | `0-1` | Destructive | Synchronous. |
| `tract-edit` | `redo_tract` | `0` | GUI-state change | Synchronous parallel edit. |
| `tract-edit` | `trim_tract` | `0` | Destructive | Synchronous parallel edit. |
| `tract-edit` | `undo_tract` | `0` | GUI-state change | Synchronous parallel edit. |
| `tract-io` | `endpoint_to_region` | `0-1` | Computation | Synchronous; refresh `list_region`. |
| `tract-io` | `save_all_tracts` | `1` | File creation | Synchronous; verify output(s). |
| `tract-io` | `save_all_tracts_to_folder` | `1` | File creation | Synchronous; verify output(s). |
| `tract-io` | `save_mni_tract` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_mni_tract_endpoint` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_slice_tract` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_slice_tract_endpoint` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_tdi` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_tdi2` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_template_tract` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_tract` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_tract_endpoint` | `1-2` | File creation | Synchronous; verify output. |
| `tract-io` | `save_tract_values` | `2-3` | File creation | Synchronous; verify output. |
| `tract-io` | `tract_to_region` | `0-1` | Computation | Synchronous; refresh `list_region`. |
| `tract-manage` | `check_tract` | `2` | GUI-state change | Immediate. |
| `tract-manage` | `show_only_tracts` | `1` | GUI-state change | Immediate exact visibility selection. |
| `tract-manage` | `check_uncheck_all_tract` | `0-1` | GUI-state change | Immediate. |
| `tract-manage` | `copy_tract` | `0-1` | GUI-state change | Synchronous. |
| `tract-manage` | `delete_all_tracts` | `0` | Destructive | Immediate. |
| `tract-manage` | `delete_tract` | `0-1` | Destructive | Synchronous. |
| `tract-manage` | `filter_tract` | `0-1` | Destructive | Synchronous. |
| `tract-manage` | `update_tract` | `0-1` | GUI-state change | Immediate. |
| `tract-process` | `cluster_tract_by_em` | `1-2` | Destructive | Synchronous computation; refresh list. |
| `tract-process` | `cluster_tract_by_hy` | `1-2` | Destructive | Synchronous computation; refresh list. |
| `tract-process` | `cluster_tract_by_km` | `1-2` | Destructive | Synchronous computation; refresh list. |
| `tract-process` | `cluster_tract_by_label` | `1-2` | Destructive | Synchronous computation; refresh list. |
| `tract-process` | `delete_repeated_tract` | `0-1` | Destructive | Synchronous computation. |
| `tract-process` | `delete_tract_by_length` | `0-1` | Destructive | Synchronous computation. |
| `tract-process` | `merge_all_tracts` | `0` | Destructive | Synchronous; refresh list. |
| `tract-process` | `merge_tract_by_name` | `0` | Destructive | Synchronous; refresh list. |
| `tract-process` | `recognize_and_cluster_tract` | `1-2` | Destructive | Synchronous computation; refresh list. |
| `tract-process` | `recognize_and_rename_tract` | `0` | Destructive | Synchronous; refresh list. |
| `tract-process` | `reconnect_tract` | `1-2` | Destructive | Synchronous computation. |
| `tract-process` | `resample_tract` | `0-1` | Destructive | Synchronous computation. |
| `tract-process` | `separate_deleted_tract` | `1` | Destructive | Synchronous computation. |
| `tract-process` | `sort_tract_by_name` | `0` | GUI-state change | Synchronous; refresh list. |
| `unet` | `list_unet` | `0` | Read-only | Immediate after model-menu refresh. |
| `unet` | `segment_brain` | `0 or 2` | Computation | With an explicit model, exact slice name is required; waits for slice registration/readiness, then runs synchronously. Verify with `list_region`. |
