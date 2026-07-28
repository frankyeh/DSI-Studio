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

## Brief `chat` with `CMD`

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","SynthSeg V2","7"],"chat":"I found the T1w slice. I’m starting SynthSeg now."}
```

The top-level `chat` field is shown to the user and does not alter the command array or its execution.
