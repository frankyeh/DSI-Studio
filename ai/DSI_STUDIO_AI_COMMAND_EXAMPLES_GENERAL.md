# DSI Studio AI General Command Examples and Inventory

Use these commands with the standard top-level `CMD` request and the numeric
**main** window ID returned by `LIST`. Every command name and parameter must be a
quoted JSON string. The main-window router accepts no more than one parameter;
commands that operate on multiple files encode them in one string separated by
`&`.

Do not send a filesystem path by itself as a named-pipe request. Supply paths
only as parameters of the documented commands below.

## Main-window commands

| Command | Common example | Exact behavior |
|---|---|---|
| `list_recent_fib` | `["list_recent_fib"]` | List saved recent FIB/FZ paths using forward slashes. Takes no arguments. |
| `list_recent_src` | `["list_recent_src"]` | List saved recent SRC/SZ paths using forward slashes. Takes no arguments. |
| `reset_settings` | `["reset_settings"]` | Clear all application settings, synchronize them, and show a confirmation message. Takes no arguments. |
| `set_work_dir` | `["set_work_dir","C:/work"]` | Add the supplied directory to the work-directory list. Without a parameter, open a directory picker. |
| `rename_dicom` | `["rename_dicom","C:/dicom/a.dcm&C:/dicom/b.dcm"]` | Rename one or more DICOM files in their current parent directories. Multiple files use one `&`-separated parameter. Without a parameter, open a file picker. |
| `rename_dicom_dir` | `["rename_dicom_dir","C:/dicom"]` | Rename DICOM files recursively at the supplied directory and show a completion message. Without a parameter, open a directory picker. |
| `convert_dicom_dir` | `["convert_dicom_dir","C:/dicom"]` | Recursively convert DICOM series in the supplied directory to SRC/SZ or NIfTI output without overwriting existing output. Without a parameter, open a directory picker. |
| `bids_to_src` | `["bids_to_src","C:/bids"]` | Search the supplied BIDS folder for diffusion NIfTI data, then ask the local user to choose an output folder and create SRC/SZ files. Without an input parameter, first open a BIDS-folder picker. |
| `nifti_dir_to_src` | `["nifti_dir_to_src","C:/nifti"]` | Find diffusion NIfTI data in the supplied directory and create SRC/SZ files there. Existing outputs may prompt for overwrite decisions. Without a parameter, open a directory picker. |
| `collect_network_measures` | `["collect_network_measures","C:/net/a.txt&C:/net/b.txt"]` | Collect network-measure text files into `<first-file>.collected.txt`. Multiple files use one `&`-separated parameter. |
| `open_src` | `["open_src","C:/data/a.sz&C:/data/b.sz"]` | Open one or more SRC/SZ or histology inputs and create reconstruction windows. Multiple files use one `&`-separated parameter. Without a parameter, open a file picker. |
| `open_dwi_nifti` | `["open_dwi_nifti","C:/data/dwi.nii.gz"]` | Open diffusion NIfTI input through `open_DWI`. Without a parameter, open a NIfTI picker. |
| `open_dwi_dicom` | `["open_dwi_dicom","C:/dicom/a.dcm&C:/dicom/b.dcm"]` | Open one or more DICOM inputs through `open_DWI`. Multiple files use one `&`-separated parameter. Without a parameter, open a DICOM picker. |
| `open_dwi_2dseq` | `["open_dwi_2dseq","C:/scan/2dseq"]` | Open 2dseq, FDF, or NRRD diffusion input through `open_DWI`. Multiple files may use one `&`-separated parameter. Without a parameter, open a picker. |
| `open_src_dir` | `["open_src_dir","C:/src"]` | Search the supplied directory for `*src.gz` and `.sz` files and load them. |
| `open_fib` | `["open_fib","C:/data/subject.fz"]` | Open the supplied `.fz`, `*fib.gz`, or `.dz` file and create a tracking window. Without a parameter, open a FIB picker. |
| `open_structural_tracking` | `["open_structural_tracking","C:/data/T1w.nii.gz"]` | Pass the supplied NIfTI or 2dseq structural image to `loadFib`. Without a parameter, open a structural-image picker. |
| `open_template` | `["open_template","<template-name>"]` | Open the supplied built-in template name. Without a parameter, open the template currently selected in the main-window list. |
| `create_db` | `["create_db"]` | Open the connectometry database-creation dialog. Takes no arguments. |
| `create_average` | `["create_average"]` | Open the average-database creation dialog. Takes no arguments. |
| `open_db` | `["open_db","C:/data/group.db.fz"]` | Load the supplied connectometry database and create a database window. Without a parameter, open a database picker. |
| `open_connectometry` | `["open_connectometry","C:/data/group.db.fz"]` | Load the supplied connectometry database and create a group-connectometry window. Without a parameter, open a database picker. |
| `open_auto_track` | `["open_auto_track"]` | Create and show the main AutoTrack window. Takes no arguments. |
| `open_nonlinear_registration` | `["open_nonlinear_registration"]` | Create and show the nonlinear-registration toolbox. Takes no arguments. |
| `open_xnat` | `["open_xnat"]` | Create and show the XNAT dialog. Takes no arguments. |
| `open_console` | `["open_console"]` | Show the singleton application console. Takes no arguments. |
| `clear_recent_src` | `["clear_recent_src"]` | Immediately clear the recent SRC/SZ table and saved `recentSrcFileList`. Takes no arguments and asks for no confirmation. |
| `clear_recent_fib` | `["clear_recent_fib"]` | Immediately clear the recent FIB/FZ table and saved `recentFibFileList`. Takes no arguments and asks for no confirmation. |
| `qc_nii` | `["qc_nii","C:/data/a.nii.gz&C:/data/b.nii.gz"]` | Run NIfTI quality checks and display a report. Multiple files use one `&`-separated parameter. Without a parameter, open a file picker. |
| `qc_src` | `["qc_src","C:/data/a.sz&C:/data/b.sz"]` | Run SRC/SZ quality checks and display a report. Multiple files use one `&`-separated parameter. Without a parameter, open a file picker. |
| `qc_fib` | `["qc_fib","C:/data/a.fz&C:/data/b.fz"]` | Run FIB/FZ quality checks and display a report. Multiple files use one `&`-separated parameter. Without a parameter, open a file picker. |
| `run_cli` | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` | Parse and execute one DSI Studio CLI command line. Exactly one parameter is required and it must include a valid `--action`. |
| `open_image` | `["open_image","C:/data/T1w.nii.gz&C:/data/T2w.nii.gz"]` | Open one or more ordinary image paths in a `view_image` window. Multiple files use one `&`-separated parameter. Without a parameter, open an image picker. Do not use this command for the FIB tracking interface. |
| `open_ai` | `["open_ai"]` | Show, raise, and activate the AI Agent window. Takes no arguments. |
| `open_hub` | `["open_hub"]` | Show, raise, and activate the Fiber Data Hub without running a query. Takes no arguments. |
| `hub_repo` | `["hub_repo"]` | Show the Fiber Data Hub and delegate this one-element command to its command router. |
| `hub_*` with parameters | — | The current main-window router rejects command arrays with more than two elements before Hub delegation. Parameterized Hub commands therefore require a router revision before they can be used through `CMD`. |

## File-list parameter format

Commands that accept multiple files still receive only one command parameter.
Join the paths with `&`:

```json
["open_src","C:/data/a.sz&C:/data/b.sz"]
["qc_fib","C:/data/a.fz&C:/data/b.fz"]
["rename_dicom","C:/dicom/a.dcm&C:/dicom/b.dcm"]
```

Do not submit each path as a separate command-array element because the
main-window router rejects more than one parameter.

## Fiber Data Hub routing limitation

The source delegates `open_hub` and names beginning with `hub_` to the Fiber Data
Hub, but the main-window router first rejects arrays containing more than one
parameter. Consequently, `hub_repo` works because it has one element, while
forms such as these are currently blocked before Hub delegation:

```json
["hub_tags","<repo>"]
["hub_files","<repo>","<tag>",".fz","0","20"]
["hub_open","<repo>","<tag>","<filename-or-index>"]
```

Do not claim that a parameterized Hub command is available through `CMD` until
the main-window argument limit is revised.

## Important routing and response notes

- Call top-level `LIST` and target its quoted numeric main-window ID.
- Do not invent aliases. Use `list_recent_fib` and `list_recent_src` exactly.
- Supplying a path as a documented command parameter is supported. Never send a
  path alone as the complete named-pipe request.
- Commands without a parameter may open a local picker. Cancellation can return
  without an immediate command error, so verify the resulting window, file, or
  application state.
- A successful `CMD` result contains `output`; no captured text is represented as
  `"command completed"`. A failed result contains `error`.
- Confirm `reset_settings`, `clear_recent_src`, and `clear_recent_fib` before use
  because they immediately modify saved application state.
- Use the appropriate `list_param` command in a tracking window before changing
  tracking or rendering parameters.

## Known source limitations

- `bids_to_src` always asks the local user to select an output directory, even
  when the input BIDS path is supplied.
- The no-parameter branches of `collect_network_measures` and `open_src_dir`
  currently shadow their outer variables, so their picker-selected values are
  not propagated correctly. Use an explicit command parameter for these two
  commands until the source is corrected.
- Picker-based commands require local GUI interaction and their cancellation may
  return without an immediate error.
- Verify opened windows and generated files rather than relying only on the lack
  of an `error` field.
