# DSI Studio AI Tract Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `open_tract` | `["open_tract","C:/output/cst.tt.gz"]` | Loads native-space tract data and shows the imported bundle. |
| `open_mni_tract` | `["open_mni_tract","C:/data/cst_mni.tt.gz"]` | Loads an MNI-space tract and maps it into the current subject. MNI mapping is required. |
| `save_tract` | `["save_tract","C:/output/cst.tt.gz","0"]` | Saves one bundle in native space. Omit the index to use the current bundle. |
| `save_mni_tract` | `["save_mni_tract","C:/output/cst_mni.tt.gz","0"]` | Saves one bundle in MNI space. |
| `save_template_tract` | `["save_template_tract","C:/output/cst_template.tt.gz","0"]` | Saves one bundle in loaded template space. |
| `save_slice_tract` | `["save_slice_tract","C:/output/cst_T1w.tt.gz","0"]` | Saves one bundle in the current slice coordinate space. |
| `save_all_tracts` | `["save_all_tracts","C:/output/checked_tracts.tt.gz"]` | Saves all checked bundles together in one tract file. |
| `save_all_tracts_to_folder` | `["save_all_tracts_to_folder","C:/output/tracts"]` | Saves checked bundles separately using bundle names and the current tract format. |
| `copy_tract` | `["copy_tract","0"]` | Duplicates one bundle as `<name>_copy`. |
| `delete_tract` | `["delete_tract","0"]` | Permanently removes one bundle. Confirm destructive actions first. |
| `delete_all_tracts` | `["delete_all_tracts"]` | Permanently removes every tract bundle. Confirm destructive actions first. |
| `check_tract` | `["check_tract","0","1"]` | Checks or unchecks one bundle. Use `1` or `0`. |
| `check_uncheck_all_tract` | `["check_uncheck_all_tract","1"]` | Checks all bundles with `1` or unchecks all with `0`; omitting the argument toggles. |
| `filter_tract` | `["filter_tract","0:3&1:0"]` | Filters the applicable bundle(s) using explicit `region-index:type` settings. |
| `update_tract` | `["update_tract"]` | Refreshes visible/deleted counts and redraws tract rendering. |
| `save_tract_color` | `["save_tract_color","C:/output/cst_color.txt","0"]` | Saves per-trajectory colors for one bundle. |
| `load_tract_color` | `["load_tract_color","C:/output/cst_color.txt","0"]` | Loads per-trajectory colors and switches to manual tract coloring. |
| `load_tract_values` | `["load_tract_values","C:/output/cst_values.txt","0"]` | Loads one value per visible trajectory; counts must match exactly. |
| `save_cluster_color` | `["save_cluster_color","C:/output/bundle_colors.txt"]` | Writes one RGB line per checked bundle in table order. |
| `load_cluster_color` | `["load_cluster_color","C:/output/bundle_colors.txt"]` | Applies one RGB line per checked bundle in table order. |
| `load_cluster_values` | `["load_cluster_values","C:/output/bundle_values.txt"]` | Loads one value per checked bundle; counts must match exactly. |
| `color_all_cluster` | `["color_all_cluster"]` | Assigns a generated distinct color to every bundle. |
| `cluster_tract_by_km` | `["cluster_tract_by_km","0","10 0"]` | Replaces tract `0` with k-means clusters. |
| `cluster_tract_by_em` | `["cluster_tract_by_em","0","10 0"]` | Replaces tract `0` with expectation-maximization clusters. |
| `cluster_tract_by_hy` | `["cluster_tract_by_hy","0","50 1.0"]` | Replaces tract `0` with hierarchical clusters and creates an `others` bundle. |
| `delete_repeated_tract` | `["delete_repeated_tract","1.0"]` | Deletes repeated trajectories using a voxel-distance threshold. |
| `resample_tract` | `["resample_tract","0.5"]` | Resamples trajectories using a step size in voxels. |
| `delete_tract_by_length` | `["delete_tract_by_length","20"]` | Deletes trajectories shorter than the supplied millimeter threshold. |
| `separate_deleted_tract` | `["separate_deleted_tract","0"]` | Moves deleted trajectories into a new bundle and clears them from the original. |
| `reconnect_tract` | `["reconnect_tract","0","4 30"]` | Reconnects using maximum bridge distance in voxels and angle in degrees. |
| `recognize_and_rename_tract` | `["recognize_and_rename_tract"]` | Recognizes each checked nonempty bundle and renames it to the top atlas match. |
| `merge_all_tracts` | `["merge_all_tracts"]` | Merges checked bundles into the first checked row and deletes the others. |
| `merge_tract_by_name` | `["merge_tract_by_name"]` | Merges bundles with identical names, keeping the earlier row. |
| `sort_tract_by_name` | `["sort_tract_by_name"]` | Sorts tract rows alphabetically while preserving associated models and states. |
| `save_tdi` | `["save_tdi","C:/output/cst_tdi.nii.gz","0"]` | Saves TDI for bundle `0` in current slice space. |
| `save_tdi2` | `["save_tdi2","C:/output/cst_tdi_2x.nii.gz","0"]` | Saves TDI at twice native FIB resolution. |
| `save_tract_statistics` | `["save_tract_statistics","C:/output/tract_stat.txt"]` | Saves statistics for all checked bundles. |
| `save_tract_recognition` | `["save_tract_recognition","C:/output/tract_names.txt","0"]` | Saves atlas-recognition percentages and names for one bundle. |
