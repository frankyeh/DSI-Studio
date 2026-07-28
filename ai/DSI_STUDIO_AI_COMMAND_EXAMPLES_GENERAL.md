# DSI Studio AI General Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `save_workspace` | `["save_workspace","C:/work/session"]` | Creates the workspace directory and saves available tracts, regions, devices, the current custom slice and mapping, settings, camera, and basic view commands. Existing output may be replaced; obtain overwrite permission first. |
| `load_workspace` | `["load_workspace","C:/work/session"]` | Restores saved tracts, slices, devices, regions, settings, camera, and view commands. When corresponding workspace folders exist, currently loaded tracts, devices, or regions are deleted before replacement. |
| `open_fib` | `["open_fib","C:/data/subject.fz"]` | Loads the FIB and creates another tracking window. It must target an existing tracking window and should not be used after the same file was already opened by raw path or `hub open`. |
| `save_fib_as` | `["save_fib_as","C:/output/subject.fz"]` | Saves the current FIB to the supplied path. Supplying the path avoids an interactive save dialog. |
| `open_mapping` | `["open_mapping","C:/data/subject.mz"]` | Loads the template and the specified MNI mapping into the current tracking window. Mapping may be long-running and can fail when template data are unavailable. |
| `correct_bias_field` | `["correct_bias_field"]` | Runs the FIB-side bias-field correction. It fails with `cannot find iso` when the required isotropic image is unavailable. |
| `save_setting` | `["save_setting","C:/work/all_settings.ini"]` | Saves all parameters returned by the rendering parameter model, including tracking and display settings, to one INI file. |
| `load_setting` | `["load_setting","C:/work/all_settings.ini"]` | Loads recognized keys found in the INI file and updates the 3D view. The file must already exist. |
| `save_tracking_setting` | `["save_tracking_setting","C:/work/tracking.ini"]` | Saves only parameters in the `Tracking`, `Tracking_dT`, and `Tracking_adv` groups. |
| `load_tracking_setting` | `["load_tracking_setting","C:/work/tracking.ini"]` | Loads only recognized tracking-group keys present in the INI file. The file must already exist. |
| `restore_tracking` | `["restore_tracking"]` | Restores tracking defaults and recalculates default length, voxel-ratio, and tolerance values for the loaded FIB. |
| `save_rendering_setting` | `["save_rendering_setting","C:/work/rendering.ini"]` | Saves ROI, rendering, slice, tract, region, surface, device, label, and ODF parameters, but not tracking parameters. |
| `load_rendering_setting` | `["load_rendering_setting","C:/work/rendering.ini"]` | Loads only recognized rendering-related keys present in the INI file. |
| `restore_rendering` | `["restore_rendering"]` | Restores rendering, visibility, color, region-graph, and ODF defaults, then refreshes tract and region rendering. |
| `presentation_mode` | `["presentation_mode"]` | Hides the ROI dock and also hides the region dock when no regions exist. It does not toggle back to the prior layout. |

## Brief `chat` with `CMD`

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["save_workspace","C:/work/session"],"chat":"I verified the current tracts, regions, slices, and devices. I’m saving the complete workspace now."}
```
