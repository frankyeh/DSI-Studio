# DSI Studio AI Slice and Parameter Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `list_slice` | `["list_slice"]` | Lists each slice index, current state, name, readiness, registration activity, download state, and registration state. Use its numeric index before other slice commands. |
| `set_slice` | `["set_slice","7"]` | Selects slice index `7`. It validates the index, loads the image if needed, may start registration for a custom slice, preserves the prior physical slice position across spaces, and refreshes contrast and available UNet models. |
| `set_slice_by_name` | `["set_slice_by_name","T1w"]` | Selects the slice whose displayed name exactly matches `T1w` and returns its index and name. It fails when no exact text match exists; prefer `set_slice` after `list_slice` when names may vary. |
| `move_slice` | `["move_slice","80 100 80"]` | Moves the current slice crosshair to voxel coordinates `x y z` in the current slice space. Supplying all three values avoids merely recording the current position. |
| `set_roi_view` | `["set_roi_view","2"]` | Selects the axial ROI editing view. Values are `0` sagittal, `1` coronal, and `2` axial. Other values produce no view change but still return success. |
| `save_roi_screen` | `["save_roi_screen","C:/output/roi_view.png"]` | Renders the current 2D ROI/slice scene and saves it as an image. Supplying the filename avoids the save dialog. It disables simple drawing for the render pass. |
| `show_only_regions` | `["show_only_regions","0&3&5"]` | Checks and displays only region rows `0`, `3`, and `5`, while unchecking every other region. The indices are joined with `&` and all must be valid. |
| `show_only_tracts` | `["show_only_tracts","1&4"]` | Checks and displays only tract rows `1` and `4`, while unchecking every other tract. This changes tract-table checked states directly. |
| `list_unet` | `["list_unet"]` | Lists each segmentation action index, availability, model identifier, displayed name, and description after refreshing the models available for the current slice. |
| `segment_brain` | `["segment_brain","SynthSeg V2","7"]` | Runs the named segmentation model on slice index `7`. The third element may also be an exact slice name. The command waits for custom-slice registration, performs inference synchronously, and creates one region per nonzero label. |
| `list_auto_tract` | `["list_auto_tract"]` | Lists tract names accepted by `run_auto_track`. It first loads the tractography name list and fails if that list is unavailable. |
| `run_auto_track` | `["run_auto_track","Left Corticospinal Tract"]` | Starts tracking for the exact tract name. It loads the symmetric tract atlas and forwards the current tracking parameters and tolerance to `run_tracking`. Use a name returned by `list_auto_tract`. |
| `list_param` | `["list_param","tract_style"]` | Returns the current value of one exact rendering/tracking parameter ID. Omit the ID to list every valid parameter ID. |
| `set_param` | `["set_param","tract_style","1"]` | Sets one parameter using `parameter-id` and `value`, then refreshes the 3D view and marks slice rendering for update. Discover the exact ID with `list_param`; the command does not independently validate the ID. |
| `set_params` | `["set_params","tract_style=1&tract_alpha=0.8"]` | Sets multiple parameters from one `id=value&id=value` string, then performs one view refresh. Entries without `=` are ignored. |
| `set_region_name` | `["set_region_name","0","Tumor Core"]` | Renames region row `0`. The index must be valid and the new name cannot be empty. |
| `set_region_type` | `["set_region_type","0","3"]` | Sets region row `0` to type `3`, which is `Seed`. Valid numeric types are `0` ROI, `1` ROA, `2` End, `3` Seed, `4` Terminative, `5` NotEnd, and `6` Limiting. |

## Source-confirmed cautions

- `set_slice_by_name` uses an exact GUI text match; `set_slice` with an index from `list_slice` is more robust.
- `show_only_regions` and `show_only_tracts` uncheck all rows not named in the command; they are not additive selections.
- `segment_brain` is synchronous. A client-side pipe timeout does not prove the inference stopped, so do not resend automatically.
- `set_param` and `set_params` call `set_data` directly. Use only IDs returned by `list_param` and values appropriate for those parameters.
