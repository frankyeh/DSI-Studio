# DSI Studio AI Tract Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

This file contains the complete tract and automatic-tracking inventory preserved from the previous manual, followed by later source-verified tract commands. Blank example cells mean that the previous manual listed the command but did not provide source-verified argument syntax. Similar legacy and newer names are retained separately; do not assume that they are aliases.

| Command | Common example | Important behavior |
|---|---|---|
| `list_tract` | `["list_tract"]` | List all tract bundles and full details. |
| `list_tract status` | `["list_tract","status"]` | Return only targeted tracking status (`running bundles`); use top-level `LIST` for routine polling. |
| `run_tracking` | `["run_tracking","Whole Brain"]` | Start asynchronous tracking; `command[1]` is the mandatory new bundle name. |
| `list_auto_tract` | `["list_auto_tract"]` | List valid automatic tract names. |
| `run_auto_track` | `["run_auto_track","Corticospinal Tract"]` | Run automatic tracking for a discovered tract name. |
| `show_only_tracts` | `["show_only_tracts","0&2&5"]` | Show only listed `&`-separated tract indices and hide all others. |
| `enable_auto_tract` | `["enable_auto_tract"]` | Load the symmetric tract atlas and enable automatic-tract controls. |
| `open_tract` | `["open_tract","C:/output/cst.tt.gz"]` | Open one native-space tract file. |
| `open_tracts` |  | Open multiple tract files. |
| `open_tract_dir` |  | Open tract files from a directory. |
| `open_mni_tract` | `["open_mni_tract","C:/data/cst_mni.tt.gz"]` | Open an MNI-space tract and map it into the current subject. |
| `save_tract` | `["save_tract","C:/output/cst.tt.gz","0"]` | Save one completed tract bundle by index. |
| `save_mni_tract` | `["save_mni_tract","C:/output/cst_mni.tt.gz","0"]` | Save one tract in MNI coordinates. |
| `save_template_tract` | `["save_template_tract","C:/output/cst_template.tt.gz","0"]` | Save one tract in loaded template space. |
| `save_slice_tract` | `["save_slice_tract","C:/output/cst_T1w.tt.gz","0"]` | Save one tract in current slice space. |
| `save_tract_endpoint` |  | Save selected tract endpoints. |
| `save_mni_tract_endpoint` |  | Save endpoints in MNI coordinates. |
| `save_slice_tract_endpoint` |  | Save endpoints in current slice space. |
| `save_all_tracts` | `["save_all_tracts","C:/output/checked_tracts.tt.gz"]` | Save all checked tracts together. |
| `save_all_tracts_to_folder` | `["save_all_tracts_to_folder","C:/output/tracts"]` | Save checked tracts as separate files in a folder. |
| `save_all_tracts_to_dir` |  | Save checked tracts to a specified directory. |
| `save_tdi` | `["save_tdi","C:/output/cst_tdi.nii.gz","0"]` | Save tract-density imaging output in current slice space. |
| `save_tdi2` | `["save_tdi2","C:/output/cst_tdi_2x.nii.gz","0"]` | Save the alternate two-times-resolution tract-density output. |
| `save_tract_values` |  | Save values sampled along tract(s). |
| `tract_to_region` | `["tract_to_region","0"]` | Convert tract trajectories to a region. |
| `endpoint_to_region` | `["endpoint_to_region","0"]` | Convert tract endpoints to region(s). |
| `update_tract` | `["update_tract"]` | Refresh counts and rendering for tract bundles. |
| `delete_tract` | `["delete_tract","0"]` | Delete one tract bundle. |
| `delete_all_tracts` | `["delete_all_tracts"]` | Delete all tract bundles. |
| `copy_tract` | `["copy_tract","0"]` | Duplicate one tract bundle. |
| `rename_tract` |  | Rename a selected tract bundle. |
| `merge_tract` |  | Merge selected tract bundles. |
| `merge_all_tracts` | `["merge_all_tracts"]` | Merge all checked tract bundles into the first checked row. |
| `merge_tract_by_name` | `["merge_tract_by_name"]` | Merge tract bundles sharing an identical name. |
| `sort_tract_by_name` | `["sort_tract_by_name"]` | Sort tract bundles by name. |
| `trim_tract` |  | Trim the selected tract. |
| `trim_all_tracts` |  | Trim all checked tracts. |
| `cut_tract` |  | Cut selected tract trajectories. |
| `cut_by_slice` |  | Cut tracts using the current slice plane. |
| `filter_tract` | `["filter_tract","0:3&1:0"]` | Filter tracks using ROI/ROA/End settings. |
| `remove_repeated_tracts` |  | Remove duplicate/repeated trajectories. |
| `recognize_tract` |  | Recognize the selected tract against the tract atlas. |
| `cluster_tract` |  | Cluster selected tract trajectories. |
| `cluster_all_tracts` |  | Cluster all checked tracts. |
| `check_tract` | `["check_tract","0","1"]` | Set one tract's checked state. |
| `check_uncheck_all_tract` | `["check_uncheck_all_tract","1"]` | Check/uncheck all tracts; explicit `1` or `0` is preferred. |
| `set_tract_color` |  | Set tract color. |
| `set_tract_color_style` |  | Set tract coloring style. |
| `set_tract_visible` |  | Set tract visibility. |
| `show_tract_statistics` |  | Display statistics for checked tracts in a modal dialog. |
| `save_tract_statistics` | `["save_tract_statistics","C:/output/tract_stat.txt"]` | Save statistics for checked tracts. |
| `show_tract_recognition` |  | Display recognition scores for a tract in a modal dialog. |
| `save_tract_recognition` | `["save_tract_recognition","C:/output/tract_names.txt","0"]` | Save tract-recognition scores. |
| `save_tract_color` | `["save_tract_color","C:/output/cst_color.txt","0"]` | Save per-trajectory colors for one tract bundle. |
| `load_tract_color` | `["load_tract_color","C:/output/cst_color.txt","0"]` | Load per-trajectory colors and switch to manual tract coloring. |
| `load_tract_values` | `["load_tract_values","C:/output/cst_values.txt","0"]` | Load one value per visible trajectory; counts must match. |
| `save_cluster_color` | `["save_cluster_color","C:/output/bundle_colors.txt"]` | Save one RGB line per checked bundle. |
| `load_cluster_color` | `["load_cluster_color","C:/output/bundle_colors.txt"]` | Load one RGB line per checked bundle. |
| `load_cluster_values` | `["load_cluster_values","C:/output/bundle_values.txt"]` | Load one value per checked bundle; counts must match. |
| `color_all_cluster` | `["color_all_cluster"]` | Assign a generated distinct color to every bundle. |
| `cluster_tract_by_km` | `["cluster_tract_by_km","0","10 0"]` | Replace one bundle with k-means clusters. |
| `cluster_tract_by_em` | `["cluster_tract_by_em","0","10 0"]` | Replace one bundle with expectation-maximization clusters. |
| `cluster_tract_by_hy` | `["cluster_tract_by_hy","0","50 1.0"]` | Replace one bundle with hierarchical clusters and create an `others` bundle. |
| `delete_repeated_tract` | `["delete_repeated_tract","1.0"]` | Delete repeated trajectories using a voxel-distance threshold. |
| `resample_tract` | `["resample_tract","0.5"]` | Resample trajectories using a step size in voxels. |
| `delete_tract_by_length` | `["delete_tract_by_length","20"]` | Delete trajectories shorter than the supplied millimeter threshold. |
| `separate_deleted_tract` | `["separate_deleted_tract","0"]` | Move deleted trajectories into a new bundle. |
| `reconnect_tract` | `["reconnect_tract","0","4 30"]` | Reconnect trajectories using a maximum distance and angle. |
| `recognize_and_rename_tract` | `["recognize_and_rename_tract"]` | Recognize each checked bundle and rename it to the top atlas match. |

## Tracking workflow notes

- `run_tracking` requires a nonempty bundle name in `command[1]`.
- Fiber tracking is asynchronous. A successful reply means tracking started; poll top-level `LIST`.
- `list_tract` takes no required parameter. The optional literal `"status"` returns compact status.
- Confirm destructive operations such as deleting, trimming, cutting, clustering, reconnecting, and merging.
- Blank examples are inventory preservation only; inspect source before constructing their parameters.
