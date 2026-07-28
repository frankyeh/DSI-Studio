# DSI Studio AI General Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

The same command name can have a different argument contract on different window types. In particular, main-window `open_fib` takes no arguments and opens a file dialog, whereas tracking-window `open_fib` requires a file path.

## Main-window commands

| Command | Common example | Important behavior |
|---|---|---|
| `list_recent_fib` | `["list_recent_fib"]` | List recently opened FIB/FZ files. Takes no arguments. |
| `list_recent_src` | `["list_recent_src"]` | List recently opened SRC/SZ files. Takes no arguments. |
| `set_work_dir` | `["set_work_dir"]` | Open a directory-selection dialog and add the selected directory to the work-directory list. It does not accept a path argument. See footnote 1. |
| `open_src` | `["open_src"]` | Open a file-selection dialog for `.sz`, `*src.gz`, `.jpg`, or `.tif` inputs and create a reconstruction window. Takes no arguments. See footnote 1. |
| `open_fib` | `["open_fib"]` | **Main window:** open a file-selection dialog for `.fz`, `*fib.gz`, or `.dz`, then create a tracking window. Takes no arguments. See footnotes 1 and 2. |
| `open_structural_tracking` | `["open_structural_tracking"]` | Open a file-selection dialog for `.nii.gz`, `.nii`, or `2dseq`, then pass the selected structural image to `loadFib`. Takes no arguments. See footnote 1. |
| `open_template` | `["open_template"]` | Open the template currently selected in the main-window template list. Fails when no list item is selected. See footnote 3. |
| `open_db` | `["open_db"]` | Open a database picker and create a database window for the selected `.dz`, `*db.fz`, or `*db?fib.gz` file. Takes no arguments. See footnote 1. |
| `open_connectometry` | `["open_connectometry"]` | Open the same database picker, load the database, and create a group-connectometry window. Takes no arguments. See footnote 1. |
| `open_auto_track` | `["open_auto_track"]` | Open the main AutoTrack window. Takes no arguments. |
| `open_nonlinear_registration` | `["open_nonlinear_registration"]` | Open a nonlinear-registration toolbox window. Takes no arguments. |
| `open_xnat` | `["open_xnat"]` | Open the XNAT dialog. Takes no arguments. |
| `open_console` | `["open_console"]` | Show the application console, creating its singleton window on first use. Takes no arguments. |
| `clear_recent_src` | `["clear_recent_src"]` | Immediately clear the recent SRC/SZ table and saved `recentSrcFileList`. No confirmation is requested. |
| `clear_recent_fib` | `["clear_recent_fib"]` | Immediately clear the recent FIB/FZ table and saved `recentFibFileList`. No confirmation is requested. |
| `run_cli` | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` | Parse and run one CLI command line. The string must contain a valid `--action`. |
| `open_image` | `["open_image"]` | With no path, open an image-selection dialog and create an image window. See footnote 1. |
| `open_image` | `["open_image","C:/data/T1w.nii.gz"]` | With one or more explicit paths, open those files in a `view_image` window. This is the image-viewing/editing route, not the FIB tracking route. |
| `open_hub` | `["open_hub"]` | Show, raise, and activate the Fiber Data Hub window without running a Hub query. Takes no arguments. |
| `hub_repo` | `["hub_repo"]` | Show the Hub window and list repository indices and exact `owner/repository` identifiers. |
| `hub_tags` | `["hub_tags","<repo>"]` | Show the Hub window, select an exact repository returned by `hub_repo`, and list its release tags. Repository metadata may still be loading; retry when instructed. |
| `hub_files` | `["hub_files","<repo>","<tag>","","0","20"]` | Show the Hub window and list file row indices, names, sizes, and temporary-cache status. Optional filter, offset, and limit follow the tag. |
| `hub_open` | `["hub_open","<repo>","<tag>","0"]` | Show the Hub window, select an exact filename or row index returned by `hub_files`, download it to temporary cache when needed, and open it. |
| `hub_download` | `["hub_download","<repo>","<tag>","0","C:/data"]` | Show the Hub window and download an exact filename or `hub_files` row index to a persistent directory. The directory is created when needed, and existing files are skipped. |

## Tracking-window general commands

| Command | Common example | Important behavior |
|---|---|---|
| `open_fib` | `["open_fib","C:/data/subject.fz"]` | **Tracking window:** load an explicit FIB/FZ path and create another tracking window. This argument contract differs from main-window `open_fib`. |
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
| `list_param` | `["list_param","tracking"]` | List current values for one parameter domain; omit the argument or use `all` for every domain, or provide one parameter ID. |
| `set_param` | `["set_param","step_size","1.0"]` | Set one discovered parameter ID to a string value. |
| `set_params` | `["set_params","step_size=1.0&min_length=20"]` | Set multiple `id=value` entries separated by `&`. |

## Fiber Data Hub workflow

All Hub queries are separate top-level command names. The old `["hub","files",...]` form is no longer accepted. `open_hub` only opens the Hub window.

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
- Hub opening and downloading use GUI-backed network routines. Verify the new window or destination file rather than relying only on a response without an `error` field.

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

- The GUI-opening commands target the **main** window.
- Main-window `open_fib` takes no arguments and opens a picker; tracking-window `open_fib` takes a file path.
- Main-window `open_image` opens ordinary image data in an image window. Use the documented main-window opening command when a first tracking window is needed.
- Use top-level `LIST` to obtain the correct numeric window ID before every `CMD`.
- Use the appropriate `list_param` domain before `set_param` or `set_params`; do not guess parameter IDs.

## Footnotes

1. `set_work_dir`, `open_src`, main-window `open_fib`, `open_structural_tracking`, `open_db`, `open_connectometry`, and parameterless `open_image` require a local GUI file/directory dialog. The current source may report command completion when these dialogs are canceled; therefore the response alone does not prove that a directory, file, or window was created. Verify the resulting work directory or window with the GUI or top-level `LIST`.
2. `open_fib` is intentionally overloaded by window type. Supplying a path to the main-window command fails because it requires exactly one command-array element; omitting the path on a tracking-window target invokes that target's separate contract and may open its own dialog.
3. Main-window `open_template` verifies that a template-list item is selected, but the helper it calls returns `void`. The command then reports success even when no template stem matches or `loadFib()` fails. Verify that a new tracking window appears.
