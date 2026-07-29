# DSI Studio AI General Command Examples and Inventory

Use these commands with the standard top-level `CMD` request and the **main**
window ID returned by `LIST`. Command names and text or path parameters are
strings. Send standalone numeric parameters as JSON numbers.

Do not send a filesystem path by itself as a named-pipe request. Supply paths
only as parameters of the documented commands below.

## Main-window commands

| Command | Common example | Exact behavior |
|---|---|---|
| `list_recent_fib` | `["list_recent_fib"]` | List saved recent FIB/FZ paths using forward slashes. Takes no arguments. |
| `list_recent_src` | `["list_recent_src"]` | List saved recent SRC/SZ paths using forward slashes. Takes no arguments. |
| `reset_settings` | `["reset_settings"]` | Clear all application settings, synchronize them, and show a confirmation message. Takes no arguments. |
| `set_work_dir` | `["set_work_dir","C:/work"]` | Add the supplied directory to the work-directory list. Without a parameter, open a directory picker. |
| `rename_dicom` | `["rename_dicom","C:/dicom/a.dcm","C:/dicom/b.dcm"]` | Rename one or more DICOM files in their current parent directories. Each file is a separate command element. Without file parameters, open a file picker. |
| `rename_dicom_dir` | `["rename_dicom_dir","C:/dicom"]` | Rename DICOM files recursively at the supplied directory. Without a parameter, open a directory picker. |
| `convert_dicom_dir` | `["convert_dicom_dir","C:/dicom"]` | Recursively convert DICOM series in the supplied directory to SRC/SZ or NIfTI output without overwriting existing output. Without a parameter, open a directory picker. |
| `bids_to_src` | `["bids_to_src","C:/bids"]` | Search the supplied BIDS folder for diffusion NIfTI data, ask the local user to choose an output folder, and create SRC/SZ files. Without an input parameter, first open a BIDS-folder picker. |
| `nifti_dir_to_src` | `["nifti_dir_to_src","C:/nifti"]` | Find diffusion NIfTI data in the supplied directory and create SRC/SZ files there. Existing outputs may prompt for overwrite decisions. Without a parameter, open a directory picker. |
| `collect_network_measures` | `["collect_network_measures","C:/net/a.txt","C:/net/b.txt"]` | Collect one or more network-measure text files into `<first-file>.collected.txt`. Each file is a separate command element. Without file parameters, open a file picker. |
| `open_src` | `["open_src","C:/data/a.sz","C:/data/b.sz"]` | Open one or more SRC/SZ or histology inputs and create reconstruction windows. Each file is a separate command element. Without file parameters, open a file picker. |
| `open_dwi_nifti` | `["open_dwi_nifti","C:/data/dwi.nii.gz"]` | Open one or more diffusion NIfTI inputs through `open_DWI`. Without file parameters, open a NIfTI picker. |
| `open_dwi_dicom` | `["open_dwi_dicom","C:/dicom/a.dcm","C:/dicom/b.dcm"]` | Open one or more DICOM inputs through `open_DWI`. Each file is a separate command element. Without file parameters, open a DICOM picker. |
| `open_dwi_2dseq` | `["open_dwi_2dseq","C:/scan/2dseq"]` | Open one or more 2dseq, FDF, or NRRD diffusion inputs through `open_DWI`. Without file parameters, open a picker. |
| `open_src_dir` | `["open_src_dir","C:/src"]` | Search the supplied directory for `*src.gz` and `.sz` files and load them. Without a parameter, open a directory picker. |
| `open_fib` | `["open_fib","C:/data/subject.fz"]` | Open the supplied `.fz`, `*fib.gz`, or `.dz` file and create a tracking window. Without a parameter, open a FIB picker. |
| `open_structural_tracking` | `["open_structural_tracking","C:/data/T1w.nii.gz"]` | Pass the supplied NIfTI or 2dseq structural image to `loadFib`. Without a parameter, open a structural-image picker. |
| `open_template` | `["open_template","<template-name>"]` | Open the exact built-in template name. An invalid name returns an error. Without a parameter, open the template currently selected in the main-window list. |
| `create_db` | `["create_db"]` | Open the connectometry database-creation dialog. Takes no arguments. |
| `create_average` | `["create_average"]` | Open the average-database creation dialog. Takes no arguments. |
| `open_db` | `["open_db","C:/data/group.db.fz"]` | Load the supplied connectometry database and create a database window. Database-loading failures are returned through `error`. Without a parameter, open a database picker. |
| `open_connectometry` | `["open_connectometry","C:/data/group.db.fz"]` | Load the supplied connectometry database and create a group-connectometry window. Database-loading failures are returned through `error`. Without a parameter, open a database picker. |
| `open_auto_track` | `["open_auto_track"]` | Create and show the main AutoTrack window. Takes no arguments. |
| `open_nonlinear_registration` | `["open_nonlinear_registration"]` | Create and show the nonlinear-registration toolbox. Takes no arguments. |
| `open_xnat` | `["open_xnat"]` | Create and show the XNAT dialog. Takes no arguments. |
| `open_console` | `["open_console"]` | Show the singleton application console. Takes no arguments. |
| `clear_recent_src` | `["clear_recent_src"]` | Immediately clear the recent SRC/SZ table and saved `recentSrcFileList`. Takes no arguments and asks for no confirmation. |
| `clear_recent_fib` | `["clear_recent_fib"]` | Immediately clear the recent FIB/FZ table and saved `recentFibFileList`. Takes no arguments and asks for no confirmation. |
| `qc_nii` | `["qc_nii","C:/data/a.nii.gz","C:/data/b.nii.gz"]` | Run NIfTI quality checks and display a report. Each file is a separate command element. Without file parameters, open a file picker. |
| `qc_src` | `["qc_src","C:/data/a.sz","C:/data/b.sz"]` | Run SRC/SZ quality checks and display a report. Each file is a separate command element. Without file parameters, open a file picker. |
| `qc_fib` | `["qc_fib","C:/data/a.fz","C:/data/b.fz"]` | Run FIB/FZ quality checks and display a report. Each file is a separate command element. Without file parameters, open a file picker. |
| `run_cli` | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` | Parse and execute one DSI Studio CLI command line. Exactly one parameter is required and it must include a valid `--action`. |
| `open_image` | `["open_image","C:/data/T1w.nii.gz","C:/data/T2w.nii.gz"]` | Open one or more ordinary image paths in a `view_image` window. Each file is a separate command element. Image-opening failures are returned through `error`. Without file parameters, open an image picker. Do not use this command for the FIB tracking interface. |
| `open_ai` | `["open_ai"]` | Show, raise, and activate the AI Agent window. Takes no arguments. |
| `open_hub` | `["open_hub"]` | Show, raise, and activate the Fiber Data Hub without running a query. Takes no arguments. |
| `hub_repo` | `["hub_repo"]` | Show the Fiber Data Hub and list available repositories. |
| `hub_tags` | `["hub_tags","<repo>"]` | List release tags for the exact repository returned by `hub_repo`. |
| `hub_files` | `["hub_files","<repo>","<tag>",".fz",0,20]` | List files using the Hub router's filter, offset, and limit parameters. |
| `hub_open` | `["hub_open","<repo>","<tag>",12]` | Download to temporary cache when needed and open the selected Hub file by returned row index. An exact filename may be used instead. |
| `hub_download` | `["hub_download","<repo>","<tag>",12,"C:/data"]` | Download the selected Hub file by returned row index to the supplied persistent directory. An exact filename may be used instead. |

## Multiple-file parameter format

Commands that accept multiple files use one command-array element per file:

```json
["open_src","C:/data/a.sz","C:/data/b.sz"]
["qc_fib","C:/data/a.fz","C:/data/b.fz"]
["rename_dicom","C:/dicom/a.dcm","C:/dicom/b.dcm"]
```

Do not combine multiple paths into one `&`-separated string. The current router
collects every command element after the command name as a separate file path.

## Fiber Data Hub workflow

Hub commands are routed before the regular main-window command handling, so they
may use their full documented argument lists:

```json
["hub_repo"]
["hub_tags","<repo>"]
["hub_files","<repo>","<tag>",".fz",0,20]
["hub_open","<repo>","<tag>",12]
```

- Use the exact `owner/repository` string returned by `hub_repo`.
- Use the exact tag returned by `hub_tags`.
- `hub_files` filters before applying offset and limit. Its first column remains
  the actual file-table row index.
- `hub_open` and `hub_download` accept the exact filename or returned row index.
- Send offset, limit, and returned row indices as JSON numbers.
- `hub_download` requires its documented destination-directory parameter.
- Verify the created window or destination file after GUI-backed network work.

## Important routing and response notes

- Call top-level `LIST` and target `main`.
- Do not invent aliases. Use `list_recent_fib` and `list_recent_src` exactly.
- Supplying paths as documented command parameters is supported. Never send a
  path alone as the complete named-pipe request.
- Commands without parameters may open a local picker. Cancellation can return
  without an immediate command error, so verify the resulting window, file, or
  application state.
- A successful `CMD` result contains `output`; no captured text is represented as
  `"command completed"`. A failed result contains `error`.
- Invalid template names, database-loading failures, and image-opening failures
  now propagate through the `error` field.
- Confirm `reset_settings`, `clear_recent_src`, and `clear_recent_fib` before use
  because they immediately modify saved application state.
- Use the appropriate `list_param` command in a tracking window before changing
  tracking or rendering parameters.

## GUI interaction notes

- `bids_to_src` always asks the local user to select an output directory, even
  when the input BIDS path is supplied.
- Picker-based commands require local GUI interaction and their cancellation may
  return without an immediate error.
- Verify opened windows and generated files rather than relying only on the lack
  of an `error` field.
