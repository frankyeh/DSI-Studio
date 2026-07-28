# DSI Studio AI Setup

This setup is intentionally concise so an agent can read it without losing the
end of the file to output truncation. Read this file once, then use
`DSI_STUDIO_AI_MANUAL.md` and the topic-specific example files only as needed.

## Window and command routing — read this first

Call top-level `LIST` before any `CMD`. It returns a numeric ID and type for each
open window.

| Window type | Use it for | Important opening command |
|---|---|---|
| **main** | Recent files, Fiber Data Hub, main file routing, CLI fallback | `open_image` opens paths through the main-window file router. It also opens `.fz`/`*fib.gz` as tracking data despite the command name. |
| **image** | General image viewing and image-window operations | Created when ordinary image formats are opened. |
| **tracking** | FIB/FZ slices, regions, tracts, tracking, devices, settings | `open_fib` opens another FIB from an existing tracking window; it cannot create the first tracking window. |

A `CMD` must use the quoted numeric `window` returned by `LIST`. Never target a
window using its type, title, filename, or a guessed ID.

## Identity

Choose one exact nonempty `agent` name and one exact resumable `session` ID.
Reuse both unchanged for the conversation.

Native agents:

```text
Codex
Claude
```

Ollama-backed examples:

```text
Codex/Ollama(192.168.1.14)
Claude/Ollama(192.168.1.14)
```

The parenthesized host is part of the agent name. Do not shorten it.

For Codex launched by DSI Studio, use `CODEX_THREAD_ID` immediately as the
session. DSI Studio also reads `thread.started.thread_id` for later resume.

For Claude Code, use the current process's `sessionId` from:

```text
~/.claude/sessions/<pid>.json
```

Do not use the friendly `name` field.

The optional wrapper identity is:

```text
<agent>@<session>
```

Because the Ollama host uses parentheses, the agent name contains no `@`.

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

$DsiAgent = 'Codex'
# Ollama example:
# $DsiAgent = 'Codex/Ollama(192.168.1.14)'
$DsiSession = $env:CODEX_THREAD_ID
```

Use the wrapper or executable fallback only when direct pipe access cannot run
or connect and the user approves the fallback. Do not create or modify GitHub
Actions to edit or operate these instructions.

## Required first request

As soon as identity and the pipe client are ready, send this before continuing:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    cwd=(Get-Location).Path
    request='CHAT'
    chat='I am reading the DSI Studio instructions and identifying the commands needed for this task.'
}
```

Read the complete reply and process any returned `PROMPT`.

## Basic requests

### Discover windows

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    cwd=(Get-Location).Path
    request='LIST'
}
```

Example reply shape:

```text
OKAY    busy    level    status
main    1       0       0    DSI Studio
tracking 2      0       0    C:/data/subject.fz
image   3       0       0    C:/data/T1w.nii.gz
```

### Send a command

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='2'
    command=@('list_region')
    chat='Checking the available regions before making changes.'
}
```

Every command element must be a string. Use `'7'`, not numeric `7`.

### Send a final or standalone message

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CHAT'
    chat='The requested operation completed and the output was verified.'
}
```

## Opening files

### One existing local file: raw path

Send one absolute path directly as raw pipe text:

```powershell
Invoke-Dsi 'C:\data\subject.fz'
```

DSI Studio routes by extension. This is the simplest way to open one existing
local file.

### JSON route: main-window `open_image`

First obtain the main-window ID with `LIST`, then:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='1'
    command=@('open_image','C:/data/subject.fz')
    chat='Opening the FZ file through the main-window file router.'
}
```

`open_image` is the main-window routing command. It can open `.fz` and
`*fib.gz` into a tracking window as well as ordinary images into an image
window. Its name does not restrict it to ordinary image files.

### Existing tracking window: `open_fib`

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='2'
    command=@('open_fib','C:/data/second_subject.fz')
    chat='Opening a second FIB from the existing tracking window.'
}
```

Do not use `open_fib` to create the first tracking window.

## Tract commands that commonly cause confusion

### `list_tract`

Full details require no parameter:

```powershell
command=@('list_tract')
```

Compact status uses the literal string `status`:

```powershell
command=@('list_tract','status')
```

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

Use top-level `LIST` for routine status polling. Do not repeatedly resend a
long-running command after a client timeout.

Attach a useful top-level `chat` message to meaningful commands:

```powershell
Invoke-Dsi @{
    agent=$DsiAgent
    session=$DsiSession
    request='CMD'
    window='2'
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
