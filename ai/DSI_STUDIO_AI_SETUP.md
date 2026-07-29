# DSI Studio AI Setup

Read this file once, then use `DSI_STUDIO_AI_MANUAL.md` and the topic-specific example files only as needed.

## Identity

Reuse the exact session UUID assigned to the current agent task.

- Codex uses `$env:CODEX_THREAD_ID`; never search for, guess, or generate it.
- Claude uses the exact session supplied by DSI Studio.

Send the same session with every request.

## Agent wrapper

Use one `dsi_agent.ps1` invocation per request:

```powershell
./dsi_agent.ps1 -Agent <Codex|Claude> -Session <SESSION> -Target <TITLE|LIST|LOG|CHAT|window-id> [command/values...]
```

The wrapper creates a new named-pipe connection, sends one request, reads the complete reply, and closes it. Do not access or reuse the pipe directly, inspect the wrapper, launch another DSI Studio instance, or modify GitHub Actions to operate these instructions.

## Basic requests

### Name the chat

After understanding the task, send one concise `TITLE` derived from it:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target TITLE "Corticospinal tract analysis"
```

Send another `TITLE` whenever the active task changes substantially. Repeated `TITLE` requests update the displayed chat name while keeping the same session.

### Window and command routing

Use `main` directly for main-window commands. Call top-level `LIST` only when a tracking or image window ID is needed.

| Window ID | Use it for | Important opening command |
|---|---|---|
| `main` | Recent files, Fiber Data Hub, opening the first FIB/FZ, reconstruction, templates, and main tools | Use `open_fib` with a path to open a known FIB/FZ, or without a path to open the FIB picker. |
| `image<hex-address>` | General image viewing and image-window operations | Created when ordinary image formats are opened with `open_image`. |
| `tracking<hex-address>` | FIB/FZ slices, regions, tracts, tracking, devices, settings | Use `open_fib` with an explicit path to open an additional FIB/FZ from an existing tracking window. |

`main` is fixed. Tracking and image IDs append the window pointer address in lowercase hexadecimal without `0x`. Do not construct or guess these IDs. A `CMD` targeting a tracking or image window must use the exact key from the latest `LIST`. The ID is valid only while that window remains open.

To discover recent files, target `main` and use these exact commands:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target main list_recent_fib
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target main list_recent_src
```

Use `list_recent_fib` for recent FIB/FZ files and `list_recent_src` for recent SRC/SZ files. Do not invent alternatives such as `recent_list`.

### Discover tracking and image windows

Call `LIST` only when a tracking or image window ID is needed:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target LIST
```

Example reply:

```json
{
  "status":"success",
  "application":{"status":"busy"},
  "windows":{
    "main":{"status":"idle","title":"DSI Studio"},
    "tracking7ff6ab123410":{"status":"busy","title":"subject.fz"},
    "image7ff6ab456780":{"status":"idle","title":"T1w.nii.gz"}
  }
}
```

Use the exact tracking or image key returned by `LIST`, such as `tracking7ff6ab123410`, as the command target.

### Command field format

The wrapper treats the first value after `Target` as the command name and later values as parameters. It converts standalone numbers to JSON numbers and preserves text or path parameters as strings.

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target main hub_repo
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target main hub_tags data-hcp/lifespan
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target main hub_files data-hcp/lifespan tag 0 20
```

### Send a command

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target tracking7ff6ab123410 list_region -Chat "Checking the available regions before making changes."
```

A useful `-Chat` message may accompany a meaningful command. Silent polling may omit it.

Every reply has `status`; `CMD` puts one result per executed command in `result`. Each result has its own `status`, and `cmd` identifies the command.

A command that produces text returns:

```json
{"status":"success","result":[{"cmd":"list_region","status":"success","output":"<command output>"}]}
```

A successful command with no captured text returns:

```json
{"status":"success","result":[{"cmd":"set_slice","status":"success","output":"completed"}]}
```

An executed command that fails includes `error`:

```json
{"status":"error","result":[{"cmd":"set_slice","status":"error","error":"<reason>"}]}
```

A request rejected before execution returns `status:"error"` with an `error` field. Status is `success`, `error`, or `busy`. `success` means the handler returned without an immediate error; asynchronous or GUI-backed work still requires verification with the relevant discovery command or expected window, object, or file.

### Send a standalone message

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target CHAT "The requested operation completed and the output was verified."
```

## Opening FIB/FZ files

Use `open_fib`; do not send a filesystem path by itself.

### Open the first FIB/FZ

Target `main`:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target main open_fib C:/data/subject.fz -Chat "Opening the FIB file."
```

This opens the supplied `.fz`, `*fib.gz`, or `.dz` file and creates a tracking window. Omit the path to open the local FIB picker. Call `LIST` afterward only when the new tracking-window ID is needed.

### Open an additional FIB/FZ

Target an existing tracking window using its exact current ID:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target tracking7ff6ab123410 open_fib C:/data/second_subject.fz -Chat "Opening an additional FIB file."
```

Do not use `open_image` for FIB/FZ files; reserve it for ordinary image files and image-window workflows.

## Slice and tract status

### `list_slice`

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target tracking7ff6ab123410 list_slice
```

The reply columns are:

```text
index    current    name    status
```

Use `status` directly:

- `available` — listed but not loaded locally.
- `registering` — registration is running.
- `ready` — ready for a dependent operation.

The `current` column is only a `1`/`0` selected-state flag. After `set_slice`, poll until the selected row reports `ready`.

### `list_tract`

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target tracking7ff6ab123410 list_tract
```

The full reply columns are:

```text
index    status    shown    name    tracts    deleted    seeds
```

Each bundle's `status` is `running` or `done`. Compact status uses:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target tracking7ff6ab123410 list_tract status
```

`status=done` means tracking is complete. `bundles` is the total number of tract rows, not the number of running jobs.

### `run_tracking`

A new bundle name is mandatory:

```powershell
./dsi_agent.ps1 -Agent <AGENT> -Session <SESSION> -Target tracking7ff6ab123410 run_tracking CST
```

The name becomes the new tract-bundle name. An empty name fails. This form uses the current tracking parameters and checked regions.

## Polling and progress

Use targeted commands for definitive state:

- Poll `list_slice` until the selected slice reports `status=ready`.
- Poll `list_tract status` until it reports `status=done`.

Call `LIST` only when a tracking or image window ID must be obtained or refreshed. Do not repeatedly resend a long-running command after a client timeout.

## Where to find commands

The concise protocol and critical syntax are in `DSI_STUDIO_AI_MANUAL.md`.

Source-verified command examples are separated by topic:

- `DSI_STUDIO_AI_COMMAND_EXAMPLES_GENERAL.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_SLICE.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_REGION.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_TRACT.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_DEVICE.md`

Read only the topic needed for the current task. Do not retrieve the entire command inventory at once.
