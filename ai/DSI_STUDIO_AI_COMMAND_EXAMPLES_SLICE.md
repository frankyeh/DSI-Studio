# DSI Studio AI Slice Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

This file contains the complete slice and segmentation inventory preserved from the previous manual. Blank example cells mean that the prior manual did not provide source-verified argument syntax.

| Command | Common example | Important behavior |
|---|---|---|
| `list_slice` | `["list_slice"]` | List slice indices, names, readiness, registration, and download state. |
| `set_slice` | `["set_slice","7"]` | Select a slice by numeric index, loading/registering it when needed. |
| `set_slice_by_name` | `["set_slice_by_name","T1w"]` | Select a slice by exact displayed name. |
| `move_slice` | `["move_slice","80 100 80"]` | Move the shared crosshair to voxel coordinates in current slice space. |
| `enable_slice` | `["enable_slice","1 1 0"]` | Set sagittal, coronal, and axial visibility in that order. |
| `set_slice_contrast` | `["set_slice_contrast","0 1"]` | Set the current slice minimum and maximum display values. An optional third string sets packed Qt minimum and maximum colors. |
| `set_slice_dir_color` | `["set_slice_dir_color","7","1"]` | Enable or disable directional coloring for one slice index. |
| `set_slice_overlay` | `["set_slice_overlay","7","1"]` | Enable or disable overlay mode for one slice index. |
| `set_slice_stay` | `["set_slice_stay","7","1"]` | Add or remove one slice from the persistent display list. |
| `set_roi_view` | `["set_roi_view","2"]` | Select ROI editing view: `0` sagittal, `1` coronal, `2` axial. |
| `add_slice` | `["add_slice","C:/data/T1w.nii.gz"]` | Add a native/custom slice; comma-separated files may define one multi-file image. |
| `add_mni_slice` | `["add_mni_slice","C:/data/atlas.nii.gz"]` | Add a custom slice interpreted in MNI space; mapping is required. |
| `skull_strip_slice` | `["skull_strip_slice","7"]` | Apply the template mask to a custom slice; built-in slices are rejected. |
| `save_roi_screen` | `["save_roi_screen","C:/output/roi_view.png"]` | Save the current 2D ROI/slice scene. |
| `save_slice_image` | `["save_slice_image","C:/output/qa.nii.gz","qa"]` | Export the named metric/data map in current subject space; arguments are output path then data-map name. |
| `save_slice_mni_image` | `["save_slice_mni_image","C:/output/qa_mni.nii.gz","qa"]` | Export the named metric/data map in template/MNI space; a valid subject-to-template mapping is required. |
| `save_slice_mapping` | `["save_slice_mapping","C:/output/T1w.linear_reg.txt","7"]` | Save registration mapping for a custom slice. |
| `open_slice_mapping` | `["open_slice_mapping","C:/output/T1w.linear_reg.txt","7"]` | Stop registration and load a mapping for a custom slice. |
| `save_slice_volume` | `["save_slice_volume","C:/output/T1w.nii.gz","7"]` | Save the bound custom-slice volume as NIfTI. |
| `delete_slice` | `["delete_slice","7"]` | Delete one custom slice; built-in slices cannot be deleted. |
| `list_unet` | `["list_unet"]` | List segmentation model index, availability, internal model ID, display name, and description. |
| `segment_brain` | `["segment_brain","<model-ID-from-list_unet>","7"]` | Run an available model using the exact `model` column value, on a slice index or exact slice name, and create label regions. See footnote 1. |

## Source-confirmed cautions

- `segment_brain` is synchronous; a client timeout does not prove inference stopped.
- Use `list_slice` to discover the exact data-map name before export.
- `save_slice_image` and `save_slice_mni_image` use `command[1]` as the output filename and `command[2]` as the metric/data-map name, not a slice-row index.
- The export source also supports special data names such as `fiber`, `dirs`, `dir0` through the available fiber count, `odfs`, and `color`; use these only when the loaded data supports them.

## Footnotes

1. The earlier example used the display name `SynthSeg V2`. The source passes `command[1]` directly to `download_unet_model()`, which matches the `.nz` filename stem. Therefore the correct argument is the internal value in the `model` column returned by `list_unet`, not the human-readable `name` column.