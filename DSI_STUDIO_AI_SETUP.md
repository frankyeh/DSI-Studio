# DSI Studio AI Setup

Use this file for transport and session rules. Use
`DSI_STUDIO_AI_MANUAL.md` for commands.

## Requirements

- Run on the same Windows computer as an AI-enabled DSI Studio.
- Keep one stable agent ID for the task. Codex uses `@C` plus a prefix of
  `CODEX_THREAD_ID`, and sends the full ID as `session`.
- Send JSON only. An existing filename is the sole legacy non-JSON request.
- Use the `dsi-studio` named pipe. Do not launch `dsi_studio.exe` per command.

## PowerShell client

```powershell
function Invoke-DsiPipe([string]$json)
{
    $pipe = [IO.Pipes.NamedPipeClientStream]::new(
        '.', 'dsi-studio', [IO.Pipes.PipeDirection]::InOut)
    try {
        $pipe.Connect(2000)
        $utf8 = [Text.UTF8Encoding]::new($false)
        $bytes = $utf8.GetBytes($json)
        $pipe.Write($bytes,0,$bytes.Length)
        $pipe.Flush()
        $buf = [byte[]]::new(8192)
        $reply = [Text.StringBuilder]::new()
        while(($n = $pipe.Read($buf,0,$buf.Length)) -gt 0) {
            [void]$reply.Append($utf8.GetString($buf,0,$n))
        }
        $reply.ToString()
    } finally { $pipe.Dispose() }
}

$DsiSession = $env:CODEX_THREAD_ID
if(-not $DsiSession) { throw 'CODEX_THREAD_ID is required' }
$DsiAgent = '@C' + $DsiSession.Substring(0,12)

function Invoke-Dsi([hashtable]$request)
{
    $request = @{} + $request
    $request.agent = $DsiAgent
    $request.session = $DsiSession
    $request.cwd = (Get-Location).Path
    Invoke-DsiPipe ($request | ConvertTo-Json -Compress -Depth 8)
}
```

## Protocol

```powershell
# Discover current windows.
$list = Invoke-Dsi @{request='LIST'}

# One command.
$reply = Invoke-Dsi @{
    request='CMD'; window='2'; command=@('list_region')
}

# Ordered same-window batch.
$reply = Invoke-Dsi @{
    request='CMD'; window='2'
    command=@(@('list_slice'),@('list_region'),@('list_tract'))
}

# Console and final user-facing reply.
$log = Invoke-Dsi @{request='LOG'}
$log = Invoke-Dsi @{request='LOG'; chat='Task completed.'}
```

JSON fields are `agent`, `session`, `cwd`, `request`, `window`, `command`, and
optional `chat`. Requests are `LIST`, `CMD`, or `LOG`. A command is an array of
strings; a batch is an array of command arrays. Keep a parameter containing
spaces as one element.

`LIST` and `LOG` are text replies beginning with `OKAY`. A queued user prompt
may follow as `PROMPT<TAB><JSON>`. `CMD` returns an array of
`{index,okay,output,error?}`; a queued prompt is the last result's optional
`prompt` property. Process prompts as new user input.

Commands in a batch run in order and stop at the first error. Do not batch an
asynchronous command with work that depends on its completion.

## Required behavior

1. Call `LIST` before the first command and after windows open or close.
2. Target `main`, `tracking`, or `image` windows by the returned ID.
3. Discover names and values with `list_slice`, `list_region`, `list_tract`,
   `list_param`, `list_atlas`, `list_unet`, and `list_auto_tract`.
4. Treat `okay:true` as acceptance. Poll the relevant list/status command for
   asynchronous work and use `LOG` for errors.
5. On `window not found`, refresh `LIST` once; never repeat the stale ID.
6. Verify exported files before reporting success.
7. Ask before overwriting/deleting files or replacing unsaved regions/tracts.
8. Put only new user-facing text in `chat`; never send reasoning or tool output.
9. Send the final answer once on the final `LOG`.

## DSI-initiated Codex turns

DSI Studio may resume the saved task with
`codex exec resume <session> <prompt>`. Contact the named pipe, do the work,
send progress through `chat` only when useful, send the final answer on `LOG`,
then exit. DSI Studio displays a waiting indicator, ignores CLI diagnostic
output as chat, and stops a process that remains after the final reply.
