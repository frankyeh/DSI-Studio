# DSI Studio AI Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

## Additional source-verified examples

| Command | Common example | Important behavior |
|---|---|---|
| `save_workspace` | `["save_workspace","C:/work/session"]` | Creates the workspace directory and saves available tracts, regions, devices, the current custom slice and mapping, settings, camera, and basic view commands. Existing output may be replaced; obtain overwrite permission first. |
| `load_workspace` | `["load_workspace","C:/work/session"]` | Restores saved tracts, slices, devices, regions, settings, camera, and view commands. When corresponding workspace folders exist, currently loaded tracts, devices, or regions are deleted before replacement. |
| `save_tracking_setting` | `["save_tracking_setting","C:/work/tracking.ini"]` | Saves only parameters in the `Tracking`, `Tracking_dT`, and `Tracking_adv` groups. It does not save rendering or region-display settings. |
| `load_tracking_setting` | `["load_tracking_setting","C:/work/tracking.ini"]` | Loads only recognized tracking-group keys that are present in the INI file. The file must already exist. |
| `new_region_from_mni` | `["new_region_from_mni","0 -10 21 5"]` | Creates a spherical region from `x y z radius`; coordinates are MNI millimeters and radius is in voxels. MNI mapping must be available. |
| `save_region` | `["save_region","C:/output/seed.nii.gz","0"]` | Saves one region by index; omit the index to use the current region. If the path lacks `.mat`, `.txt`, `.nii`, or `.nii.gz`, DSI Studio appends `.nii.gz`. |
| `copy_region` | `["copy_region","0"]` | Duplicates the selected region immediately after the source row, preserving its mask, properties, and name. Omit the index to copy the current region. |
| `open_tract` | `["open_tract","C:/output/cst.tt.gz"]` | Loads tract data in the current FIB/native space and shows the imported bundle. Use `open_mni_tract` instead when the file is in MNI space. |
| `save_tract` | `["save_tract","C:/output/cst.tt.gz","0"]` | Saves one tract bundle in its current/native space. Use the bundle index returned by `list_tract`; omit it to use the current bundle. |
| `save_all_tracts_to_folder` | `["save_all_tracts_to_folder","C:/output/tracts"]` | Saves only checked tract bundles as separate files named from each bundle plus the current tract output extension. Verify that all expected files were created. |

## Twenty more source-verified examples

| Command | Common example | Important behavior |
|---|---|---|
| `open_fib` | `["open_fib","C:/data/subject.fz"]` | Loads the FIB and creates another tracking window. It must target an existing tracking window and should not be used after the same file was already opened by raw path or `hub open`. |
| `save_fib_as` | `["save_fib_as","C:/output/subject.fz"]` | Saves the current FIB to the supplied path. Supplying the path avoids an interactive save dialog. |
| `open_mapping` | `["open_mapping","C:/data/subject.mz"]` | Loads the template and the specified MNI mapping into the current tracking window. Mapping may be long-running and can fail when template data are unavailable. |
| `save_setting` | `["save_setting","C:/work/all_settings.ini"]` | Saves all parameters returned by the rendering parameter model, including tracking and display settings, to one INI file. |
| `load_setting` | `["load_setting","C:/work/all_settings.ini"]` | Loads recognized keys found in the INI file and updates the 3D view. The file must already exist. |
| `restore_tracking` | `["restore_tracking"]` | Restores the `Tracking`, `Tracking_dT`, and `Tracking_adv` groups and recalculates default length, voxel-ratio, and tolerance values for the loaded FIB. |
| `enable_slice` | `["enable_slice","1 1 0"]` | Sets sagittal, coronal, and axial slice visibility in that order. Each value is interpreted as a Boolean state. |
| `set_slice_contrast` | `["set_slice_contrast","0 1"]` | Sets the current slice minimum and maximum display values. An optional third command element may provide packed minimum and maximum colors; omitting it keeps the current colors. |
| `set_slice_overlay` | `["set_slice_overlay","7","1"]` | Enables or disables overlay mode for the quoted slice index. Use `list_slice` first; setting the existing state returns a canceled result rather than changing anything. |
| `new_region` | `["new_region"]` | Creates an empty region named `new region` in the current slice space and makes it the current region. |
| `new_region_whole_brain_seed` | `["new_region_whole_brain_seed"]` | Creates a Seed region from the FA map using the current Otsu-threshold ratio. An optional second element overrides that ratio. |
| `new_region_from_threshold` | `["new_region_from_threshold","<threshold>"]` | Creates a new region by thresholding the current slice with the supplied value. The threshold is passed to `region_action_threshold`. |
| `new_region_from_sphere` | `["new_region_from_sphere","80 100 80 5"]` | Creates a spherical region from current-slice voxel coordinates `x y z` and a radius in voxels. Use `new_region_from_mni` for MNI-millimeter coordinates. |
| `check_region` | `["check_region","0","1"]` | Sets one region's shown/checked state. Use `1` to show/check and `0` to hide/uncheck; omit the index to target the current region. |
| `merge_regions` | `["merge_regions","0&1&2"]` | Unions the listed regions into the first listed region and removes the later rows. Use valid ascending region indices discovered with `list_region`. |
| `save_all_regions` | `["save_all_regions","C:/output/regions.nii.gz"]` | Saves all checked regions into one 3D label image. At least one region must be checked; where regions overlap, the later checked region receives the higher label. |
| `check_all_regions` | `["check_all_regions"]` | Checks and displays every region in the current tracking window. |
| `uncheck_all_regions` | `["uncheck_all_regions"]` | Unchecks and hides every region in the current tracking window. |
| `tract_to_region` | `["tract_to_region","0"]` | Converts all trajectories of one tract bundle into a voxel region in the current slice space. Omit the index to use the current tract bundle. |
| `endpoint_to_region` | `["endpoint_to_region","0"]` | Converts one tract bundle's two endpoint sets into two new regions named with `endpoints1` and `endpoints2`. |

## Twenty-two more source-verified examples

| Command | Common example | Important behavior |
|---|---|---|
| `correct_bias_field` | `["correct_bias_field"]` | Runs the FIB-side bias-field correction. It fails with `cannot find iso` when the required isotropic image is unavailable. |
| `save_rendering_setting` | `["save_rendering_setting","C:/work/rendering.ini"]` | Saves the ROI, rendering, slice, tract, region, surface, device, label, and ODF parameter groups, but not tracking parameter groups. |
| `load_rendering_setting` | `["load_rendering_setting","C:/work/rendering.ini"]` | Loads only recognized rendering-related keys present in the INI file. The file must already exist. |
| `restore_rendering` | `["restore_rendering"]` | Restores rendering, visibility, color, region-graph, and ODF defaults, then refreshes tract and region color maps. |
| `presentation_mode` | `["presentation_mode"]` | Hides the ROI dock and also hides the region dock when no regions exist. It does not toggle back to the prior layout. |
| `set_slice_dir_color` | `["set_slice_dir_color","7","1"]` | Sets directional coloring for one slice index. Use `list_slice` first; requesting the existing state returns a canceled result. |
| `set_slice_stay` | `["set_slice_stay","7","1"]` | Adds or removes one slice from the persistent/stay slice list. Use `1` to keep it displayed and `0` to remove it. |
| `add_slice` | `["add_slice","C:/data/T1w.nii.gz"]` | Adds a custom image slice. One command element may contain comma-separated source files for a multi-file image; loading can start registration asynchronously. |
| `add_mni_slice` | `["add_mni_slice","C:/data/atlas.nii.gz"]` | Adds a custom slice interpreted in MNI space. DSI Studio first requires a working subject-to-MNI mapping. |
| `skull_strip_slice` | `["skull_strip_slice","7"]` | Applies the template mask to a custom slice after Gaussian smoothing. Built-in slices are rejected/canceled; discover the custom index with `list_slice`. |
| `save_slice_mapping` | `["save_slice_mapping","C:/output/T1w.linear_reg.txt","7"]` | Saves the registration mapping for a custom slice. The file path is the second element and the slice index is the third. |
| `open_slice_mapping` | `["open_slice_mapping","C:/output/T1w.linear_reg.txt","7"]` | Stops any running registration for the selected custom slice, then loads the supplied mapping file. Built-in slices are rejected. |
| `save_slice_volume` | `["save_slice_volume","C:/output/T1w.nii.gz","7"]` | Saves the bound volume of one custom slice as NIfTI. The slice index is the optional third element and defaults to the current slice. |
| `delete_slice` | `["delete_slice","7"]` | Deletes one custom slice and its texture/table entry. Built-in slices cannot be deleted; use only an index returned by `list_slice`. |
| `enable_auto_tract` | `["enable_auto_tract"]` | Loads the symmetric tract atlas, populates the automatic-tract selectors, and exposes the tracking controls. It can fail when the tract atlas cannot be loaded. |
| `open_mni_tract` | `["open_mni_tract","C:/data/cst_mni.tt.gz"]` | Loads a tract file interpreted in MNI space and maps it into the current subject. MNI mapping must be available. |
| `save_mni_tract` | `["save_mni_tract","C:/output/cst_mni.tt.gz","0"]` | Saves one bundle in MNI coordinates. The optional third element is the tract index; MNI mapping must be available. |
| `save_template_tract` | `["save_template_tract","C:/output/cst_template.tt.gz","0"]` | Saves one bundle in the loaded template space. The optional third element selects the tract index. |
| `save_slice_tract` | `["save_slice_tract","C:/output/cst_T1w.tt.gz","0"]` | Transforms and saves one bundle in the current slice's coordinate space and dimensions. |
| `save_all_tracts` | `["save_all_tracts","C:/output/checked_tracts.tt.gz"]` | Saves all checked bundles together in one tract file. Unchecked bundles are excluded. |
| `filter_tract` | `["filter_tract","0:3&1:0"]` | Filters the current or checked tract bundle(s) using explicit `region-index:type` settings. Discover region indices and types with `list_region` first. |
| `update_tract` | `["update_tract"]` | Refreshes visible/deleted counts for every tract bundle, marks every tract rendering for update, and redraws the tract display. |

## Brief `chat` with `CMD`

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","SynthSeg V2","7"],"chat":"I found the T1w slice. I’m starting SynthSeg now."}
```

The top-level `chat` field is shown to the user and does not alter the command array or its execution.
