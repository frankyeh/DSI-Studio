# DSI Studio AI Setup

Read this file once, then use `DSI_STUDIO_AI_MANUAL.md` and the topic-specific example files only as needed.

## Identity

Choose one exact nonempty `agent` name. Reuse unchanged for the conversation.

Native agents:

```text
Codex
Claude
```

Obtain the resumable session UUID assigned by the current agent runtime. The
discovery method differs by agent:

### Codex

Use the exact UUID of the current Codex task/thread exposed by its injected
runtime context or task-specific runtime path. When DSI Studio launches Codex
with JSON output, this is the same `thread_id` reported by `thread.started` and
captured by DSI Studio for later resume. Codex Desktop may expose it as the UUID
component of an injected task path such as
`...\visualizations\YYYY\MM\DD\<uuid>`. Use only an ID explicitly associated
with the current task; do not scan for, guess, or generate one.

### Claude Code

Read the current Claude process's file:

```text
~/.claude/sessions/<pid>.json
```

Use its `sessionId` value, not its friendly `name` or the process ID.

Reuse the exact agent and session values in every request.

## Direct named-pipe connection

Use the local named pipe directly first:

```text
\\.\pipe\dsi-studio
```

Each connection sends one complete request, reads until DSI Studio closes the
server side, then closes. Do not launch another DSI Studio instance.

PowerShell direct client:

```powershell
function Invoke-Dsi($request)
{
    $pipe = $writer = $reader = $null
    try
    {
        $pipe = [IO.Pipes.NamedPipeClientStream]::new('.','dsi-studio')
        $pipe.Connect(5000)
        $utf8 = [Text.UTF8Encoding]::new($false)
        $writer = [IO.StreamWriter]::new($pipe,$utf8,1024,$true)
        $writer.AutoFlush = $true
        $data = if($request -is [string]) {
            $request
        } else {
            $request | ConvertTo-Json -Compress -Depth 6
        }
        $writer.Write($data)
        $reader = [IO.StreamReader]::new($pipe,$utf8,$false,1024,$true)
        $reader.ReadToEnd()
    }
    finally
    {
        foreach($stream in @($reader,$writer,$pipe))
        {
            try { if($stream) { $stream.Dispose() } }
            catch [IO.IOException] {}
        }
    }
}

$DsiAgent = '<agent-name>'
# Obtain this using the matching Codex or Claude instructions above.
$DsiSession = '<resumable-session-uuid>'
```

Use the wrapper or executable fallback only when direct pipe access cannot run
or connect and the user approves the fallback. Do not create or modify GitHub
Actions to edit or operate these instructions.

## Basic requests

### Name the chat

After understanding the user's initial prompt, send one concise `TITLE` before
the first `LIST` or `CMD`:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='TITLE'
    title='Corticospinal tract analysis'
}
```

Use the required `title` field only with `TITLE`, not with `CMD`, `CHAT`, `LIST`,
or `LOG`, and do not put the title in `chat` or `text`. Keep the same exact
`agent` and `session`; `TITLE` changes only the displayed chat name. Send another
`TITLE` later only when the user permits renaming.


## Window and command routing — read this first

Call top-level `LIST` before any `CMD`. It returns the application status and the
current ID for each open window.

| Window ID | Use it for | Important opening command |
|---|---|---|
| `main` | Recent files, Fiber Data Hub, opening the first FIB/FZ, reconstruction, templates, and main tools | Use `open_fib` with a path to open a known FIB/FZ, or without a path to open the FIB picker. |
| `image<hex-address>` | General image viewing and image-window operations | Created when ordinary image formats are opened with `open_image`. |
| `tracking<hex-address>` | FIB/FZ slices, regions, tracts, tracking, devices, settings | Use `open_fib` with an explicit path to open an additional FIB/FZ from an existing tracking window. |

`main` is fixed. Tracking and image IDs append the window pointer address in
lowercase hexadecimal without `0x`. Do not construct or guess an ID. A `CMD`
must use the exact quoted `window` key from the latest `LIST`. The ID is valid
only while that window remains open; reopening a window or restarting DSI Studio
may produce a different ID.

Do not invent command names. To discover recent files, target the **main** window
and use these exact commands:

```json
["list_recent_fib"]
["list_recent_src"]
```

Use `list_recent_fib` for recent FIB/FZ files and `list_recent_src` for recent
SRC/SZ files. Do not substitute guessed names such as `recent_list`.



### Discover windows

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='LIST'
}
```

Example reply:

```json
{
  "application":{"status":"busy"},
  "windows":{
    "main":{"status":"idle","title":"DSI Studio"},
    "tracking7ff6ab123410":{"status":"busy","title":"subject.fz"},
    "image7ff6ab456780":{"status":"idle","title":"T1w.nii.gz"}
  }
}
```

`status` is `idle`, `busy`, or `waiting`. Use the exact window key returned by
`LIST`, such as `main` or `tracking7ff6ab123410`, as the `CMD` target.

### Send a command

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='tracking7ff6ab123410'
    command=@('list_region')
    chat='Checking the available regions before making changes.'
}
```

Every command element must be a string. Use `'7'`, not numeric `7`.
An optional `chat` may accompany any request. Keep it on `CMD` when reporting
the command already being sent instead of making a separate `CHAT` request.

Every `CMD` returns a JSON array with one result object per command.

A command that produces text returns:

```json
[{"index":0,"output":"<command output>"}]
```

A successful command with no captured text returns:

```json
[{"index":0,"output":"command completed"}]
```

A failed command returns an `error` field:

```json
[{"index":0,"error":"<reason>"}]
```

The presence of `error` means that command failed. A batch stops after the first
error. `command completed` means the command handler returned without an
immediate error; asynchronous or GUI-backed work may still require verification
with `LIST`, the relevant `list_*` command, or the expected window, object, or
file.

### Send a final or standalone message

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CHAT'
    chat='The requested operation completed and the output was verified.'
}
```

## Opening FIB/FZ files

Use the documented `open_fib` command workflow. Do not send a filesystem path by
itself as the file-opening request.

### Open the first FIB/FZ

Target `main`, then send `open_fib` with the known path:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='main'
    command=@('open_fib','C:/data/subject.fz')
    chat='Opening the FIB file.'
}
```

This opens the supplied `.fz`, `*fib.gz`, or `.dz` file and creates a tracking
window. Omit the path to open the local FIB picker instead. Afterward, call
`LIST` and use the new tracking-window ID.

### Open an additional FIB/FZ

When a tracking window already exists, target its exact current ID and supply
the explicit path as the command parameter:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='tracking7ff6ab123410'
    command=@('open_fib','C:/data/second_subject.fz')
    chat='Opening an additional FIB file.'
}
```

Both main- and tracking-window `open_fib` accept a path, but they are separate
command implementations. Always target the exact window ID returned by `LIST`.
Do not use `open_image` to open FIB/FZ files; use it for ordinary image files
and image-window workflows.

## Slice and tract status that commonly cause confusion

### `list_slice`

```powershell
command=@('list_slice')
```

The reply columns are:

```text
index    current    name    status
```

Use the `status` word directly:

- `available` — the URL-backed slice is listed but has not yet been loaded locally.
- `registering` — registration is still running; poll again.
- `ready` — the slice is ready for a dependent operation.

The `current` column is only a `1`/`0` selected-state flag. After `set_slice`,
poll until the selected row reports `ready`.

### `list_tract`

Full details require no parameter:

```powershell
command=@('list_tract')
```

The full reply uses these columns:

```text
index    status    shown    name    tracts    deleted    seeds
```

Each bundle's `status` is `running` or `done`. The `shown` field is a separate
`1`/`0` visibility flag.

Compact status uses the literal string `status`:

```powershell
command=@('list_tract','status')
```

The compact reply uses:

```text
status    bundles
```

`status=running` means at least one tracking thread is active. `status=done`
means tracking is complete. `bundles` is the total number of tract rows, not the
number of running jobs.

A numeric tract index is not required. If `["list_tract"]` produces
`need-param1`, the command was likely sent through a malformed or incompatible
wrapper rather than the standard JSON `CMD` interface.

### `run_tracking`

A new bundle name is mandatory:

```powershell
command=@('run_tracking','CST')
```

The second element becomes the new bundle name. An empty name fails. The simple
form uses current tracking parameters and checked regions.

## Polling and progress

Use top-level `LIST` for routine application status. Use targeted commands for
definitive state:

- Poll `list_slice` until the selected slice reports `status=ready`.
- Poll `list_tract status` until it reports `status=done`.

Do not repeatedly resend a long-running command after a client timeout.

Attach a useful top-level `chat` message to meaningful commands:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='tracking7ff6ab123410'
    command=@('run_tracking','CST')
    chat='The seed and tracking parameters are ready. I am starting the CST bundle now.'
}
```

Silent repetitive polling may omit `chat`.

## Where to find commands

The concise protocol and critical syntax are in:

- `DSI_STUDIO_AI_MANUAL.md`

Source-verified command examples are separated to avoid truncation:

- `DSI_STUDIO_AI_COMMAND_EXAMPLES_GENERAL.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_SLICE.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_REGION.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_TRACT.md`
- `DSI_STUDIO_AI_COMMAND_EXAMPLES_DEVICE.md`

Read only the topic needed for the current task. Do not print or retrieve the
entire command inventory at once.
