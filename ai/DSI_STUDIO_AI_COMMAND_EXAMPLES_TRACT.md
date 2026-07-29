# DSI Studio AI Tract Command Examples and Inventory

Use these with the standard top-level `CMD` request. Command names and text,
path, or composite parameters are strings. Send standalone numeric parameters
as JSON numbers.

This file contains tract and automatic-tracking commands confirmed in the current source. Earlier generic inventory names that had no command handler were removed only after checking the GL, tract, region, device, and tracking dispatch chain.

| Command | Common example | Important behavior |
|---|---|---|
| `list_tract` | `["list_tract"]` | List every tract bundle with `index`, readable `status`, shown state, name, tract count, deleted count, and seeds. |
| `list_tract status` | `["list_tract","status"]` | Return compact `status` and total bundle count. `status=done` means no tracking thread remains active. |
| `run_tracking` | `["run_tracking","Whole Brain"]` | Start asynchronous tracking with the current tracking parameters and checked region settings; `command[1]` is the mandatory new bundle name. |
| `run_tracking` | `["run_tracking","CST","0:3&1:0"]` | Start tracking with explicit region settings: region 0 as Seed and region 1 as ROI. The third element uses `index:type` entries separated by `&`. See **ROI settings syntax** and footnote 2. |
| `list_auto_tract` | `["list_auto_tract"]` | List valid automatic tract names. |
| `run_auto_track` | `["run_auto_track","ProjectionBrainstem_CorticospinalTractL"]` | Use an exact name from `list_auto_tract`; never guess atlas labels. For a clean display bundle, generate about 50,000 tracts with `tip_iteration=0`, run `["trim_tract"]` 4–5 times, run `["delete_repeated_tract",1]`, then repeat `["trim_tract"]` until about 10,000 remain. |
| `run_auto_track` | `["run_auto_track","ProjectionBrainstem_CorticospinalTractL","0:0&1:1"]` | Run automatic tracking while also applying explicit region 0 as ROI and region 1 as ROA. Use an exact name from `list_auto_tract`; see **ROI settings syntax** and footnote 2. |
| `show_only_tracts` | `["show_only_tracts","0&2&5"]` | Show only listed `&`-separated tract indices and hide all others. |
| `enable_auto_tract` | `["enable_auto_tract"]` | Load the symmetric tract atlas and enable automatic-tract controls. |
| `open_tract` | `["open_tract","C:/output/cst.tt.gz"]` | Open one native-space tract file and show each loaded bundle. Open multiple files by sending one command per path. |
| `open_tract` | `["open_tract","C:/output/all_bundles.tt.gz",0]` | Open the tract file with newly loaded bundles unchecked/hidden. The source tests only whether the third element is empty; any supplied value has this effect. |
| `open_mni_tract` | `["open_mni_tract","C:/data/cst_mni.tt.gz"]` | Open an MNI-space tract and map it into the current subject. |
| `open_tract_name` | `["open_tract_name","C:/data/tract_names.txt"]` | Load whitespace-separated names and apply them in reverse order to the most recently listed tract rows. |
| `load_tract_atlas` | `["load_tract_atlas","Corticospinal_Tract"]` | Load one named population tract-atlas bundle. |
| `load_tract_atlas` | `["load_tract_atlas"]` | Load every tract name from the asymmetric tract atlas; this may create many bundles. |
| `save_tract` | `["save_tract","C:/output/cst.tt.gz",0]` | Save one completed tract bundle by index. |
| `save_mni_tract` | `["save_mni_tract","C:/output/cst_mni.tt.gz",0]` | Save one tract in MNI coordinates. |
| `save_template_tract` | `["save_template_tract","C:/output/cst_template.tt.gz",0]` | Save one tract in loaded template space. |
| `save_slice_tract` | `["save_slice_tract","C:/output/cst_T1w.tt.gz",0]` | Save one tract in current slice space. |
| `save_tract_endpoint` | `["save_tract_endpoint","C:/output/cst_endpoints.txt",0]` | Save native-space endpoints for one tract bundle index. |
| `save_mni_tract_endpoint` | `["save_mni_tract_endpoint","C:/output/cst_mni_endpoints.txt",0]` | Intended to save endpoints in MNI coordinates, but the current implementation is unreliable. See footnote 1. |
| `save_slice_tract_endpoint` | `["save_slice_tract_endpoint","C:/output/cst_T1w_endpoints.txt",0]` | Intended to save endpoints in current slice space, but the current implementation is unreliable. See footnote 1. |
| `save_all_tracts` | `["save_all_tracts","C:/output/checked_tracts.tt.gz"]` | Save all checked tracts together. |
| `save_all_tracts_to_folder` | `["save_all_tracts_to_folder","C:/output/tracts"]` | Save checked tracts as separate files in a folder. |
| `save_tdi` | `["save_tdi","C:/output/cst_tdi.nii.gz",0]` | Save tract-density imaging output in current slice space. |
| `save_tdi2` | `["save_tdi2","C:/output/cst_tdi_2x.nii.gz",0]` | Save the alternate two-times-resolution tract-density output. |
| `save_tract_values` | `["save_tract_values","C:/output/cst_qa.txt",0,"qa"]` | Save the named metric along one tract bundle; arguments are filename, tract index, and metric name. |
| `tract_to_region` | `["tract_to_region",0]` | Convert tract trajectories to a region. |
| `endpoint_to_region` | `["endpoint_to_region",0]` | Convert tract endpoints to region(s). |
| `update_tract` | `["update_tract"]` | Refresh counts and rendering for tract bundles. |
| `delete_tract` | `["delete_tract",0]` | Delete one tract bundle. |
| `delete_all_tracts` | `["delete_all_tracts"]` | Delete all tract bundles. |
| `copy_tract` | `["copy_tract",0]` | Duplicate one tract bundle. |
| `merge_all_tracts` | `["merge_all_tracts"]` | Merge all checked tract bundles into the first checked row. |
| `merge_tract_by_name` | `["merge_tract_by_name"]` | Merge tract bundles sharing an identical name. |
| `sort_tract_by_name` | `["sort_tract_by_name"]` | Sort tract bundles by name. |
| `delete_branch` | `["delete_branch","0&2"]` | Delete branch-like portions from tract bundles 0 and 2. Omit the index list to edit every checked bundle. |
| `undo_tract` | `["undo_tract","0&2"]` | Undo the latest supported tract edit in tract bundles 0 and 2. Omit the index list to use checked bundles. |
| `redo_tract` | `["redo_tract","0&2"]` | Redo the latest supported tract edit in tract bundles 0 and 2. Omit the index list to use checked bundles. |
| `trim_tract` | `["trim_tract",0]` | Apply one TIP iteration to tract bundle 0. Omit the index to use every checked bundle; bundles below 1,000 tracts are generally unsuitable. Start near 50,000, trim 4–5 times, run `["delete_repeated_tract",1]`, then repeat trimming until about 10,000 remain. |
| `cut_tract_end_portion` | `["cut_tract_end_portion",0]` | Apply `cut_end_portion(0.25,0.75)` to tract bundle 0. |
| `cut_tract_lps_end` | `["cut_tract_lps_end",0]` | Apply `cut_end_portion(0.25,1.0)` to tract bundle 0. |
| `cut_tract_rai_end` | `["cut_tract_rai_end",0]` | Apply `cut_end_portion(0.0,0.75)` to tract bundle 0. |
| `flip_tract_x` | `["flip_tract_x",0]` | Flip tract bundle 0 along X. |
| `flip_tract_y` | `["flip_tract_y",0]` | Flip tract bundle 0 along Y. |
| `flip_tract_z` | `["flip_tract_z",0]` | Flip tract bundle 0 along Z. |
| `cut_tract_by_x` | `["cut_tract_by_x",80]` | Cut every checked bundle at X slice 80 and retain the default side. |
| `cut_tract_by_x2` | `["cut_tract_by_x2",80]` | Cut every checked bundle at X slice 80 and retain the opposite side. |
| `cut_tract_by_y` | `["cut_tract_by_y",100]` | Cut every checked bundle at Y slice 100 and retain the default side. |
| `cut_tract_by_y2` | `["cut_tract_by_y2",100]` | Cut every checked bundle at Y slice 100 and retain the opposite side. |
| `cut_tract_by_z` | `["cut_tract_by_z",80]` | Cut every checked bundle at Z slice 80 and retain the default side. |
| `cut_tract_by_z2` | `["cut_tract_by_z2",80]` | Cut every checked bundle at Z slice 80 and retain the opposite side. |
| `set_dt_index` | `["set_dt_index","qa&iso",0]` | Set differential metrics `m1&m2` and calculation type; creates the `dT_metrics` slice the first time. |
| `filter_tract` | `["filter_tract","0:3&1:0"]` | Filter every checked tract using region 0 as Seed and region 1 as ROI. The argument uses the same `index:type` encoding as tracking. |
| `check_tract` | `["check_tract",0,1]` | Set one tract's checked state. |
| `check_uncheck_all_tract` | `["check_uncheck_all_tract",1]` | Check/uncheck all tracts; explicit `1` or `0` is preferred. |
| `select_cluster_color` | `["select_cluster_color",0,4294901760]` | Set one bundle to a packed Qt ARGB color and switch to assigned coloring. |
| `show_tract_statistics` | `["show_tract_statistics"]` | Display statistics for checked tracts in a modal dialog. |
| `save_tract_statistics` | `["save_tract_statistics","C:/output/tract_stat.txt"]` | Save statistics for checked tracts. |
| `show_tract_recognition` | `["show_tract_recognition","",0]` | Recognize tract index 0 and display ranked atlas matches in a modal dialog; at least one tract must be checked. |
| `save_tract_recognition` | `["save_tract_recognition","C:/output/tract_names.txt",0]` | Save tract-recognition scores. |
| `save_tract_color` | `["save_tract_color","C:/output/cst_color.txt",0]` | Save per-trajectory colors for one tract bundle. |
| `load_tract_color` | `["load_tract_color","C:/output/cst_color.txt",0]` | Load per-trajectory colors and switch to manual tract coloring. |
| `load_tract_values` | `["load_tract_values","C:/output/cst_values.txt",0]` | Load one value per visible trajectory; counts must match. |
| `save_cluster_color` | `["save_cluster_color","C:/output/bundle_colors.txt"]` | Save one RGB line per checked bundle. |
| `load_cluster_color` | `["load_cluster_color","C:/output/bundle_colors.txt"]` | Load one RGB line per checked bundle. |
| `load_cluster_values` | `["load_cluster_values","C:/output/bundle_values.txt"]` | Load one value per checked bundle; counts must match. |
| `color_all_cluster` | `["color_all_cluster"]` | Assign a generated distinct color to every bundle. |
| `cluster_tract_by_label` | `["cluster_tract_by_label",0,"C:/data/cluster_labels.txt"]` | Replace one bundle with clusters defined by one integer label per visible trajectory. |
| `recognize_and_cluster_tract` | `["recognize_and_cluster_tract",0]` | Replace one bundle with tract-atlas-recognized bundles. |
| `cluster_tract_by_km` | `["cluster_tract_by_km",0,"10 0"]` | Replace one bundle with k-means clusters. |
| `cluster_tract_by_em` | `["cluster_tract_by_em",0,"10 0"]` | Replace one bundle with expectation-maximization clusters. |
| `cluster_tract_by_hy` | `["cluster_tract_by_hy",0,"50 1.0"]` | Replace one bundle with hierarchical clusters and create an `others` bundle. |
| `delete_repeated_tract` | `["delete_repeated_tract",1]` | Delete repeated trajectories in every checked bundle; the default distance threshold is `1` voxel. |
| `resample_tract` | `["resample_tract",0.5]` | Resample checked bundles using a step size in voxels. |
| `delete_tract_by_length` | `["delete_tract_by_length",20]` | Delete trajectories shorter than the supplied millimeter threshold from checked bundles. |
| `separate_deleted_tract` | `["separate_deleted_tract",0]` | Move deleted trajectories into a new bundle. |
| `reconnect_tract` | `["reconnect_tract",0,"4 30"]` | Reconnect trajectories using a maximum distance and angle. |
| `recognize_and_rename_tract` | `["recognize_and_rename_tract"]` | Recognize each checked bundle and rename it to the top atlas match. |

## `list_tract` output

The full reply columns are:

```text
index    status    shown    name    tracts    deleted    seeds
```

Each row's `status` is:

- `running` — that bundle still has an active tracking thread.
- `done` — no tracking thread remains attached to that bundle.

The compact form returns:

```text
status    bundles
running   3
```

or:

```text
status    bundles
done      3
```

Here, `bundles` is the total number of tract rows, not the number of running jobs. Poll `["list_tract","status"]` until `status` is `done` before starting a dependent operation. The separate `shown` column remains a `1`/`0` visibility state and should not be confused with tracking status.

## Tract-index selection for edit commands

Call `["list_tract"]` immediately before an indexed edit and use values from its `index` column.

These four commands accept one numeric tract index or one string containing multiple `&`-separated indices:

```json
["delete_branch","0&2&5"]
["undo_tract","0&2&5"]
["redo_tract","0&2&5"]
["trim_tract","0&2&5"]
```

When an index or index list is present, the command edits those bundle indices directly, regardless of their checked state. When it is omitted or empty, the command edits every checked bundle, preserving the original GUI behavior. Duplicate indices are applied only once. A nonnumeric or out-of-range value fails with `invalid tract index: <value>`.

Do not assume every checked-bundle command accepts this index-list argument. `cut_tract_by_*`, `filter_tract`, `delete_repeated_tract`, `resample_tract`, and `delete_tract_by_length` currently use their second element for another parameter and still operate on checked bundles.

## ROI settings syntax

Tracking and filtering accept an `&`-separated list of `region-index:role` entries:

```text
0:3&1:0&2:1
```

This means region 0 is a Seed, region 1 is an ROI, and region 2 is an ROA. Role values are:

- `0` = ROI
- `1` = ROA
- `2` = End
- `3` = Seed
- `4` = Terminative
- `5` = NotEnd
- `6` = Limiting

Use `list_region` immediately before constructing the string. Explicit settings use the supplied rows and roles directly; they do not require those rows to be checked in the table.

## Tracking workflow notes

- `run_tracking` requires a nonempty bundle name in `command[1]`.
- The two-element `run_tracking` form uses the current tracking parameters and the region settings currently checked in the table.
- The three-element `run_tracking` form is recognized as the convenient explicit-ROI form when its third string is empty or contains `:`; DSI Studio inserts the current tracking parameter code internally.
- Fiber tracking is asynchronous. A successful reply means tracking started; poll top-level `LIST` for general activity and `["list_tract","status"]` for definitive tract completion.
- `list_tract` takes no required parameter. The optional literal `"status"` returns compact `status` and total bundle count.
- `trim_tract`, `delete_branch`, `undo_tract`, and `redo_tract` accept an optional tract index or `&`-separated tract-index list; without it they operate on checked bundles.
- `cut_tract_by_*`, `filter_tract`, `delete_repeated_tract`, `resample_tract`, and `delete_tract_by_length` operate on checked bundles.
- `cut_tract_end_portion`, `cut_tract_lps_end`, `cut_tract_rai_end`, and `flip_tract_*` operate on one selected tract index.
- Clustering commands delete the original bundle and replace it with newly created cluster bundles.
- Confirm destructive operations such as deleting, trimming, cutting, clustering, reconnecting, and merging.
- The removed generic names (`open_tracts`, `open_tract_dir`, `save_all_tracts_to_dir`, `rename_tract`, `merge_tract`, `trim_all_tracts`, `cut_tract`, `cut_by_slice`, `remove_repeated_tracts`, `recognize_tract`, `cluster_tract`, `cluster_all_tracts`, `set_tract_color`, `set_tract_color_style`, and `set_tract_visible`) had no command handler in the current dispatch chain. Use the exact commands documented above.

## Tracking parameter reference

These parameters are the `Tracking`, `Tracking_dT`, and `Tracking_adv` groups from the embedded `:/data/options.txt` resource. Use:

```json
["list_param","tracking"]
["list_param","fa_threshold"]
["set_param","fa_threshold",0.08]
["set_params","fa_threshold=0.08&min_length=20&turning_angle=60"]
```

`["list_param","tracking"]` returns every current tracking parameter and value from all three tracking groups. Use a single parameter ID only when one value is needed. Send numeric values as JSON numbers with `set_param`; `set_params` keeps its combined assignment expression as one string. Enum values are zero-based indices. The metric lists shown for `tracking_index`, `dt_index1`, and `dt_index2` are resource defaults and may be replaced by metrics available in the loaded FIB.

### Basic tracking

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `tracking_index` | Tracking Index | `0`=fa; `1`=adc | `0` (fa) |
| `fa_threshold` | Tracking Threshold (0=random) | float `0–2`; step `0.01` | `0.0` |
| `turning_angle` | Angular Threshold (0=random) | integer `0–90`; step `5` | `0` |
| `step_size` | Step Size(mm)(0=random) | float `0.00–10`; step `0.1` | `0` |
| `min_length` | Min Length(mm) | float `0–800`; step `10` | `30` |
| `max_length` | Max Length(mm) | float `0–10000`; step `10` | `300` |
| `max_seed_count` | Max Seeds(0=default) | integer `0–100000000`; step `1000` | `0` |
| `max_tract_count` | Max Tracts(0=default) | integer `0–100000000`; step `1000` | `0` |
| `track_voxel_ratio` | Tract-to-Voxel Ratio | float `0–2`; step `0.005` | `1.0` |
| `tip_iteration` | Topology-Informed Pruning (iteration) | integer `0–100`; step `2` | `4` |
| `tolerance` | Autotrack tolerance (mm) | float `0–100`; step `10` | `22` |

### Differential tracking

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `dt_index1` | Metrics1(m1) | `0`=none; `1`=adc | `0` (none) |
| `dt_index2` | Metrics2(m2) | `0`=none; `1`=adc | `0` (none) |
| `dt_threshold_type` | Type | `0`=(m1-m2)÷m1; `1`=(m1-m2)÷m2; `2`=m1-m2; `3`=(m2-m1)÷m1; `4`=(m2-m1)÷m2; `5`=m2-m1; `6`=m1÷max(m1); `7`=m2÷max(m2) | `0` ((m1-m2)÷m1) |
| `dt_threshold` | Threshold | float `0.0–2.0`; step `0.05` | `0.2` |

### Advanced tracking

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `tracking_method` | Tracking Algorithm | `0`=Euler; `1`=RK4; `2`=Voxel tracking | `0` (Euler) |
| `smoothing` | Smoothing (1=random) | float `-1.5–1`; step `0.1` | `0` |
| `check_ending` | Check Ending | `0`=Off; `1`=On | `0` (Off) |
| `otsu_threshold` | Default Otsu | float `0.1–1`; step `0.1` | `0.6` |
| `track_format` | Output Format | `0`=tt.gz; `1`=trk.gz; `2`=txt | `0` (tt.gz) |

## Footnotes

1. The current transformed-endpoint implementation should not be relied on. `save_slice_tract_endpoint` first writes transformed endpoints and then falls through to native `save_end_points()` on the same path. `save_mni_tract_endpoint` calls `sub2mni()` on a temporary point but appends the original native `points1` coordinates to the output buffer, then also falls through to native `save_end_points()`. The examples document the accepted argument syntax only; verify or avoid the output until these branches are fixed.
2. `run_tracking` creates and appends the new tract bundle and assigns its thread object before validating explicit ROI settings. If an `index:type` entry is invalid, the command returns failure without removing that newly appended bundle/thread entry. Validate every region index and role with `list_region` before sending the command.
