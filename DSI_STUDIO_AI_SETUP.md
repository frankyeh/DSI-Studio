# DSI Studio AI Setup

Use this file for transport and session rules. Use
`DSI_STUDIO_AI_MANUAL.md` for commands.

## Requirements

- Run on the same Windows computer as an AI-enabled DSI Studio.
- Keep one stable agent ID for the task. Codex uses `@C` plus a prefix of
  `CODEX_THREAD_ID`, and sends the full ID as `session`.
- Use GUI control by default. Use `run_cli` only when the user explicitly asks
  for CLI operation.
- Send control requests as JSON. To open an existing local file in the GUI,
  send its absolute filename directly; this is the only non-JSON request.
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

## Open a local file in the GUI

When only the main window exists, open an `.fz`, `.sz`, or image by sending its
absolute filename directly—not as `CMD`, `run_cli`, or `open_fib`:

```powershell
$path = (Resolve-Path 'C:\data\subject.fz').Path
Invoke-DsiPipe $path
$list = Invoke-Dsi @{request='LIST'}
```

Poll `LIST` for the new `tracking` or `image` window. `open_fib` targets an
already-open tracking window; it cannot create the first tracking window from
the main window.

To open multiple local images together in one O1 image window, send exactly
one flat `open_image` command to the `main` window. Put every complete absolute
filepath in that same command:

```powershell
Invoke-Dsi @{
    request='CMD'; window='1'
    command=@('open_image','C:\data\a.nii.gz','C:\data\b.nii.gz')
}
```

Do not send a batch of separate `open_image` commands, target an existing
`image` window, split a directory and filename into separate parameters, or
use `add_image`; none of these creates the retained batch-file list. Then
refresh `LIST`. The direct non-JSON filename transport opens only one file.

## Required behavior

1. Prefer GUI windows and commands unless the user explicitly requests CLI.
2. Call `LIST` before the first command and after windows open or close.
3. Target `main`, `tracking`, or `image` windows by the returned ID.
4. Discover names and values with `list_slice`, `list_region`, `list_tract`,
   `list_param`, `list_atlas`, `list_unet`, and `list_auto_tract`.
5. Treat `okay:true` as acceptance. Poll the relevant list/status command for
   asynchronous work and use `LOG` for errors.
6. On `window not found`, refresh `LIST` once; never repeat the stale ID.
7. Verify exported files before reporting success.
8. Ask before overwriting/deleting files or replacing unsaved regions/tracts.
9. Do not answer modal dialogs remotely. Tell the user what confirmation is
   expected and wait for the human response.
10. Put only new user-facing text in `chat`; never send reasoning or tool output.
11. Send the final answer once on the final `LOG`.

## DSI-initiated Codex turns

DSI Studio may resume the saved task with
`codex exec resume <session> <prompt>`. Contact the named pipe, do the work,
send progress through `chat` only when useful, send the final answer on `LOG`,
then continue any returned `PROMPT` and exit naturally when none remains. DSI
Studio displays a waiting indicator and ignores CLI diagnostic output as chat.
