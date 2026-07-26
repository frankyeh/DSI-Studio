# DSI Studio AI Setup

Read this file completely. Search `DSI_STUDIO_AI_MANUAL.md` only for commands
needed by the request; do not read its entire inventory.

## Identity

Generate one UUID when the AI conversation starts and reuse the exact same
`agent` and `session` in every request. Never generate a UUID inside an
individual command.

Provider prefixes are `@C` for Codex, `@A` for Claude Code, and another
two-character prefix for other agents. The agent ID is the prefix followed by
the first 12 characters of the session UUID.

```powershell
$DsiSession = '<one UUID retained for this conversation>'
$DsiAgent = '<provider prefix>'+$DsiSession.Substring(0,12)
```

## Connect

Run on the same Windows computer as DSI Studio. Define this helper once per
PowerShell process:

```powershell
function Send-Dsi($request)
{
    if($request -is [string])
    {
        $payload = (Resolve-Path -LiteralPath $request).Path
    }
    else
    {
        $request = @{}+$request
        $request.agent = $DsiAgent
        $request.session = $DsiSession
        $request.cwd = (Get-Location).Path
        $payload = $request | ConvertTo-Json -Compress -Depth 8
    }
    $pipe = [IO.Pipes.NamedPipeClientStream]::new(
        '.', 'dsi-studio', [IO.Pipes.PipeDirection]::InOut)
    try
    {
        $pipe.Connect(2000)
        $utf8 = [Text.UTF8Encoding]::new($false)
        $bytes = $utf8.GetBytes($payload)
        $pipe.Write($bytes,0,$bytes.Length)
        $pipe.Flush()
        $reader = [IO.StreamReader]::new($pipe,$utf8)
        $reader.ReadToEnd()
    }
    finally
    {
        $pipe.Dispose()
    }
}
```

Each call opens one connection, sends exactly one request, reads its complete
reply, and closes. Never combine requests on one connection or send incomplete
JSON.

## Requests

```powershell
# Discover windows.
$list = Send-Dsi @{request='LIST'}

# Use the numeric ID returned by LIST.
$reply = Send-Dsi @{
    request='CMD'; window='2'; command=@('list_region')
}

# Ordered same-window batch.
$reply = Send-Dsi @{
    request='CMD'; window='2'
    command=@(@('list_slice'),@('list_region'),@('list_tract'))
}

# Incremental diagnostics and final user-facing reply.
$log = Send-Dsi @{request='LOG'}
$log = Send-Dsi @{request='LOG'; chat='Task completed.'}
```

Always use the numeric window ID returned by the latest `LIST`; never use a
window type, title, filename, guessed ID, or stale ID as `window`.

JSON fields are `agent`, `session`, `cwd`, `request`, `window`, `command`, and
optional `chat`. The helper supplies `agent`, `session`, and `cwd`. Requests
are `LIST`, `CMD`, or `LOG`. A command is an array of strings; a batch is an
array of command arrays. Keep parameters containing spaces as one element.
Batches run in order and stop at the first error. Do not batch asynchronous
work with commands that depend on its completion.

`LIST` and `LOG` replies begin with `OKAY`. Diagnostic `LOG` returns at most
4096 new console characters since the prior `LOG` or first request. Every
`LOG` advances the cursor. The console is global, so concurrent agents may see
each other's new DSI output. Final `LOG` with `chat` returns no console history.
`[AI AGENT]` trace lines are omitted. `[AI REQUEST]` groups and closing `⏱`
lines report synchronous DSI-side request handling, not agent runtime or
asynchronous completion.

A queued user prompt may follow a text reply as `PROMPT<TAB><JSON>`. `CMD`
returns `{index,okay,output,error?}` objects; its last result may contain a
`prompt` property. Treat returned prompts as new user input.

## Opening local files

When only the main window exists, send one absolute filename:

```powershell
Send-Dsi 'C:\data\subject.fz'
$list = Send-Dsi @{request='LIST'}
```

Poll `LIST` for the new numeric `tracking` or `image` window ID. `open_fib`
requires an existing tracking window and cannot create the first one.

In DSI Studio, **FIB means `.fz`**. Never substitute `.sz`; `.sz` is an SRC
file. `Send-Dsi` can open one `.fz`, `.sz`, or image file.

To open multiple images in one O1 window, send one flat command to the numeric
main-window ID:

```powershell
Send-Dsi @{
    request='CMD'; window='1'
    command=@('open_image','C:\data\a.nii.gz','C:\data\b.nii.gz')
}
```

Do not send separate `open_image` commands, target an image window, split a
path into fields, or substitute `add_image`. Refresh `LIST` afterward.

## Required behavior

1. Prefer GUI commands; use `run_cli` only when explicitly requested.
2. Call `LIST` first and after windows open or close.
3. Use only numeric IDs returned by the latest `LIST`.
4. Discover values with `list_slice`, `list_region`, `list_tract`,
   `list_param`, `list_atlas`, `list_unet`, and `list_auto_tract`.
5. Treat `okay:true` as acceptance. Poll the relevant list command for
   asynchronous completion; use `LOG` only when diagnostics are needed.
6. On `window not found`, refresh `LIST` once and do not repeat the stale ID.
7. Verify outputs. Ask before destructive operations or overwrites.
8. Do not answer modal dialogs remotely; tell the user what is required.
9. Put only new user-facing text in `chat`; never include reasoning/tool output.
10. Send the final answer once with the final `LOG`.
11. Minimize round trips: one initial `LIST`, a safe same-window batch, concise
    verification, and final `LOG`.

If DSI Studio resumes an agent, reconnect with the same provider, agent ID,
and session UUID. Process every returned `PROMPT` and exit naturally when none
remains.
