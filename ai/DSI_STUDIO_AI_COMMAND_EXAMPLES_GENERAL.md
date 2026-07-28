# DSI Studio AI General Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

This file contains the complete main-window, Hub, tracking-file, workspace, setting, and parameter inventory preserved from the previous manual. Blank example cells mean that the prior manual listed the command but did not provide source-verified argument syntax.

| Command | Common example | Important behavior |
|---|---|---|
| `list_recent_fib` | `["list_recent_fib"]` | List recently opened FIB/FZ files from the main window. |
| `list_recent_src` | `["list_recent_src"]` | List recently opened SRC/SZ files from the main window. |
| `hub_repo` | `["hub_repo"]` | List Fiber Data Hub repository indices and exact `owner/repository` identifiers. |
| `hub_tags` | `["hub_tags","<repo>"]` | Select an exact repository returned by `hub_repo` and list its release tags. Repository metadata may still be loading; retry the same command when instructed. |
| `hub_files` | `["hub_files","<repo>","<tag>","","0","20"]` | List file row indices, names, sizes, and temporary-cache status. Optional filter, offset, and limit follow the tag. |
| `hub_open` | `["hub_open","<repo>","<tag>","0"]` | Select an exact filename or row index returned by `hub_files`, download it to temporary cache when needed, and open it using the Hub-selected file mode. |
| `hub_download` | `["hub_download","<repo>","<tag>","0","C:/data"]` | Download an exact filename or `hub_files` row index to a persistent directory. The directory is created when needed, and existing files are skipped. |
| `open_image` | `["open_image","C:/data/subject.fz"]` | Main-window file router; can open FIB/FZ, SRC/SZ, and ordinary images according to extension. |
| `run_cli` | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` | Run CLI only when the user explicitly requests CLI execution. |
| `open_fib` | `["open_fib","C:/data/subject.fz"]` | Open another FIB from an existing tracking window; cannot create the first tracking window. |
| `correct_bias_field` | `["correct_bias_field"]` | Run FIB-side bias-field correction; fails when the required isotropic image is unavailable. |
| `save_fib_as` | `["save_fib_as","C:/output/subject.fz"]` | Save the current FIB to the supplied path. |
| `open_mapping` | `["open_mapping","C:/data/subject.mz"]` | Load a mapping file into the current tracking window. |
| `save_workspace` | `["save_workspace","C:/work/session"]` | Save tracts, regions, devices, slices, settings, camera, and view state. |
| `load_workspace` | `["load_workspace","C:/work/session"]` | Restore a saved workspace; existing table contents may be replaced. |
| `save_setting` | `["save_setting","C:/work/all_settings.ini"]` | Save all recognized rendering and tracking parameters. |
| `save_rendering_setting` | `["save_rendering_setting","C:/work/rendering.ini"]` | Save rendering-related parameter groups only. |
| `save_tracking_setting` | `["save_tracking_setting","C:/work/tracking.ini"]` | Save tracking parameter groups only. |
| `load_setting` | `["load_setting","C:/work/all_settings.ini"]` | Load recognized keys from a combined INI file. |
| `load_rendering_setting` | `["load_rendering_setting","C:/work/rendering.ini"]` | Load recognized rendering-related keys. |
| `load_tracking_setting` | `["load_tracking_setting","C:/work/tracking.ini"]` | Load recognized tracking-related keys. |
| `restore_rendering` | `["restore_rendering"]` | Restore rendering, visibility, and color defaults. |
| `restore_tracking` | `["restore_tracking"]` | Restore tracking defaults and data-dependent length/tolerance values. |
| `presentation_mode` | `["presentation_mode"]` | Hide editing docks for presentation-oriented display. |
| `list_param` | `["list_param","tracking"]` | List the current values for one parameter domain; omit the argument or use `all` for every domain, or provide one parameter ID. |
| `set_param` | `["set_param","step_size","1.0"]` | Set one discovered parameter ID to a string value. |
| `set_params` | `["set_params","step_size=1.0&min_length=20"]` | Set multiple `id=value` entries separated by `&`. |

## Fiber Data Hub workflow

All Hub commands are separate top-level command names. The old `["hub","files",...]` form is no longer accepted.

```json
["hub_repo"]
["hub_tags","<repo>"]
["hub_files","<repo>","<tag>",".fz","0","20"]
["hub_open","<repo>","<tag>","<exact-filename-or-returned-index>"]
```

- Use the exact `owner/repository` string returned by `hub_repo`.
- Use the exact tag returned by `hub_tags`.
- `hub_files` performs a case-insensitive substring filter, then applies offset and limit. Its first column remains the actual file-table row index; do not replace it with the filtered result's ordinal position.
- `hub_open` and `hub_download` accept either the exact filename or the numeric row index returned by `hub_files`.
- `hub_download` requires exactly five elements, including the destination directory.
- Hub opening and downloading use GUI-backed network routines. Verify the new window or destination file rather than relying only on `okay:true`.

## `list_param` domains

Use a domain to retrieve only the relevant current values:

- `tracking`
- `region_window`
- `background_rendering`
- `slice_rendering`
- `tract_rendering`
- `region_rendering`
- `surface_rendering`
- `device_rendering`
- `label_rendering`
- `odf_rendering`

For example, `["list_param","tracking"]` returns all parameters from `Tracking`, `Tracking_dT`, and `Tracking_adv`. Domain names are case-insensitive, and `-` is normalized to `_`.

## Important routing notes

- All `hub_*` commands target the **main** window.
- `open_image` targets the **main** window and routes files by extension.
- `open_fib` targets an existing **tracking** window and opens another FIB/FZ.
- Use top-level `LIST` to obtain the correct numeric window ID before every `CMD`.
- Use the appropriate `list_param` domain before `set_param` or `set_params`; do not guess parameter IDs.