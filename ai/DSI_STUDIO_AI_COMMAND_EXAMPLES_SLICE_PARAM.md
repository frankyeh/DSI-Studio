# DSI Studio AI Slice and Parameter Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `list_slice` | `["list_slice"]` | Lists slice indices, names, readiness, registration, and download state. |
| `set_slice` | `["set_slice","7"]` | Selects a slice by index, loading or registering it when needed. |
| `set_slice_by_name` | `["set_slice_by_name","T1w"]` | Selects a slice by exact displayed name. |
| `move_slice` | `["move_slice","80 100 80"]` | Moves the crosshair to voxel coordinates in current slice space. |
| `enable_slice` | `["enable_slice","1 1 0"]` | Sets sagittal, coronal, and axial visibility in that order. |
| `set_slice_contrast` | `["set_slice_contrast","0 1"]` | Sets current-slice minimum and maximum display values. |
| `set_slice_overlay` | `["set_slice_overlay","7","1"]` | Enables or disables overlay mode for one slice index. |
| `set_slice_dir_color` | `["set_slice_dir_color","7","1"]` | Enables or disables directional coloring for one slice index. |
| `set_slice_stay` | `["set_slice_stay","7","1"]` | Adds or removes one slice from the persistent display list. |
| `set_roi_view` | `["set_roi_view","2"]` | Selects ROI editing view: `0` sagittal, `1` coronal, `2` axial. |
| `add_slice` | `["add_slice","C:/data/T1w.nii.gz"]` | Adds a custom image slice. Comma-separated files may define one multi-file image. |
| `add_mni_slice` | `["add_mni_slice","C:/data/atlas.nii.gz"]` | Adds a custom slice interpreted in MNI space; mapping is required. |
| `skull_strip_slice` | `["skull_strip_slice","7"]` | Applies the template mask to a custom slice. Built-in slices are rejected. |
| `save_slice_mapping` | `["save_slice_mapping","C:/output/T1w.linear_reg.txt","7"]` | Saves registration mapping for a custom slice. |
| `open_slice_mapping` | `["open_slice_mapping","C:/output/T1w.linear_reg.txt","7"]` | Stops registration and loads a mapping for a custom slice. |
| `save_slice_volume` | `["save_slice_volume","C:/output/T1w.nii.gz","7"]` | Saves the bound custom-slice volume as NIfTI. |
| `delete_slice` | `["delete_slice","7"]` | Deletes one custom slice. Built-in slices cannot be deleted. |
| `save_roi_screen` | `["save_roi_screen","C:/output/roi_view.png"]` | Saves the current 2D ROI/slice scene as an image. |
| `show_only_regions` | `["show_only_regions","0&3&5"]` | Shows only the listed region rows and unchecks all others. |
| `show_only_tracts` | `["show_only_tracts","1&4"]` | Shows only the listed tract rows and unchecks all others. |
| `list_unet` | `["list_unet"]` | Lists segmentation model index, availability, identifier, name, and description. |
| `segment_brain` | `["segment_brain","SynthSeg V2","7"]` | Runs the named model on a slice index or exact slice name and creates label regions. |
| `enable_auto_tract` | `["enable_auto_tract"]` | Loads the symmetric tract atlas and enables automatic-tract controls. |
| `list_auto_tract` | `["list_auto_tract"]` | Lists exact tract names accepted by `run_auto_track`. |
| `run_auto_track` | `["run_auto_track","Left Corticospinal Tract"]` | Runs tracking for an exact atlas tract name using current tracking settings. |
| `list_param` | `["list_param","tract_style"]` | Returns one parameter value; omit the ID to list every valid parameter ID. |
| `set_param` | `["set_param","tract_style","1"]` | Sets one parameter and refreshes rendering. Use IDs returned by `list_param`. |
| `set_params` | `["set_params","tract_style=1&tract_alpha=0.8"]` | Sets multiple `id=value` pairs separated by `&`, then refreshes once. |

## Source-confirmed cautions

- `show_only_regions` and `show_only_tracts` replace the current checked selection rather than adding to it.
- `segment_brain` is synchronous; a client timeout does not prove inference stopped.
- `set_param` and `set_params` call `set_data` directly, so use valid IDs and appropriate values.
