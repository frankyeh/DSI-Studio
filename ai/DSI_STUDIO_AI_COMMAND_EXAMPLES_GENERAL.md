# DSI Studio AI General Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

This file contains the complete main-window, Hub, tracking-file, workspace, setting, and parameter inventory preserved from the previous manual. Blank example cells mean that the prior manual listed the command but did not provide source-verified argument syntax.

| Command | Common example | Important behavior |
|---|---|---|
| `list_recent_fib` | `["list_recent_fib"]` | List recently opened FIB/FZ files from the main window. |
| `list_recent_src` | `["list_recent_src"]` | List recently opened SRC/SZ files from the main window. |
| `hub repos` | `["hub","repos"]` | List Fiber Data Hub repositories. |
| `hub tags` | `["hub","tags","<repo>"]` | List tags/releases for one repository. |
| `hub files` | `["hub","files","<repo>","<tag>","","0","20"]` | List files with index, name, size, and download state. |
| `hub open` | `["hub","open","<repo>","<tag>","0"]` | Download one Hub file to temporary cache and open it. |
| `hub download` | `["hub","download","<repo>","<tag>","0","C:/data"]` | Download one Hub file to a persistent directory without opening it. |
| `hub help` |  | Show Hub subcommand syntax. |
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

- `open_image` targets the **main** window and routes files by extension.
- `open_fib` targets an existing **tracking** window and opens another FIB/FZ.
- Use top-level `LIST` to obtain the correct numeric window ID before every `CMD`.
- Use the appropriate `list_param` domain before `set_param` or `set_params`; do not guess parameter IDs.