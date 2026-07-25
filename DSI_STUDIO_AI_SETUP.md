# DSI Studio AI-Agent Setup

Use this file to connect a local AI agent to an AI-control-enabled DSI Studio. Read `dsi_studio_manual.md` completely before issuing domain commands; it is the authoritative command reference.

## Requirements

- The agent must be able to execute local Windows processes on the same computer as DSI Studio. A remote or browser-only agent cannot control the local application without a separate local execution bridge.
- Obtain the exact path to `dsi_studio.exe` and do not substitute another installation without the user's approval.
- Obtain access only to the executable, requested input data, manual, and output directories needed for the task.
- DSI Studio should already be running unless the user explicitly asks the agent to start it.

## Connect

Use a task-specific PowerShell variable:

```powershell
$dsiExe = 'C:\DSI-Studio\dsi_studio.exe'
& $dsiExe 'LIST'
```

A successful response resembles:

```text
OKAY
main    1    DSI Studio ...
tracking    2    C:\data\subject.fz
```

`LIST` assigns and returns remote window IDs. Always call it before the first command and again after opening or closing a window. If it returns `NO_INSTANCE`, ask the user to start the AI-control-enabled DSI Studio.

## Send a command

Build the request with actual tab characters:

```powershell
$request = [string]::Join([char]9, @('CMD', '2', 'list_slice'))
& $dsiExe $request
Start-Sleep -Seconds 2
```

Protocol:

```text
CMD<TAB>window_id<TAB>command<TAB>parameter_1<TAB>parameter_2...
```

Each array element is one command field. A parameter containing spaces must remain one element. Never split or reinterpret compound parameters described by `dsi_studio_manual.md`.

Target commands according to the window type returned by `LIST`:

- `main`: Fiber Data Hub and main-window operations
- `tracking`: slices, regions, tracts, atlases, segmentation, rendering, and tracking
- `image`: image-window operations

## Open a local file

Send an existing filename as the executable's single argument:

```powershell
& $dsiExe 'C:\data\subject.fz'
Start-Sleep -Seconds 2
& $dsiExe 'LIST'
```

The client forwards the filename to the running DSI Studio. Refresh `LIST` to discover the new window ID.

## Inspect before acting

Never guess names, indices, or parameter values. Use the available introspection commands first, including:

```text
list_slice
list_region
list_tract
list_param
list_atlas
list_unet
list_auto_tract
```

Use only commands and parameters documented in `dsi_studio_manual.md`.

## Console output

Retrieve DSI Studio's available console history with:

```powershell
& $dsiExe 'LOG'
```

Use the console to diagnose loading, registration, downloading, segmentation, tracking, and export failures. Do not treat the absence of an error message as proof of success.

## Completion and verification

- `OKAY` means the command was accepted; it may not mean an asynchronous operation has finished.
- Poll the relevant list or status command until completion.
- Refresh `LIST` when a command opens or closes a window.
- Verify created regions or tracts using their list commands.
- Verify every exported file exists and is readable before reporting success.
- If possible, inspect and show exported images to the user.
- Report errors and partial results accurately; do not silently retry destructive operations.

## Safety

Obtain confirmation before:

- Overwriting or deleting files
- Deleting or replacing regions or tracts
- Closing windows with unsaved work
- Downloading unexpectedly large datasets
- Starting expensive batch processing outside the user's stated scope

Avoid commands that require modal dialogs during unattended operation. Prefer explicit, fully parameterized commands. Use native-space GQI `.fz` data when alignment with native structural images is required.

## Agent behavior

Translate the user's scientific goal into documented DSI Studio commands; do not require the user to know command names. State which dataset and window will be affected, execute the smallest safe sequence, monitor it to completion, verify the result, and summarize what DSI Studio produced.
