# DSI Studio AI Setup

Read this file completely. Search `DSI_STUDIO_AI_MANUAL.md` only for commands
needed by the request; do not read its entire inventory.

## Identity

Choose one non-empty agent name and one non-empty session name when the AI
conversation starts. Reuse both strings exactly in every request. The agent
name must not contain `@`, which DSI Studio reserves as the separator.

DSI Studio identifies the conversation as `agent@session`. The request still
sends `agent` and `session` as separate JSON fields. Both may be arbitrary
names; no provider-specific prefix is required.

```powershell
$DsiAgent = '<agent name>'
$DsiSession = '<session name>'
```

## Universal client

Run on the same Windows computer as DSI Studio. If `dsi.ps1` does not exist
beside `dsi_studio.exe`, create it there once with this exact content:

```powershell
param(
    [Parameter(Mandatory,Position=0)]
    [string]$Identity,

    [Parameter(Mandatory,Position=1)]
    [string]$Target,

    [Parameter(Position=2,ValueFromRemainingArguments)]
    [string[]]$Command
)

$separator = $Identity.IndexOf('@')
if($separator -lt 1 -or $separator -eq $Identity.Length-1)
{
    Write-Error 'Identity must be agent@session.'
    exit 2
}
$agent = $Identity.Substring(0,$separator)
$session = $Identity.Substring($separator+1)
$exe = @(
    (Join-Path $PSScriptRoot 'dsi_studio.exe')
    $env:DSI_STUDIO_EXE
    (Join-Path (Split-Path $PSScriptRoot -Parent) 'DSI-Studio-CMAKE\dsi_studio.exe')
) | Where-Object {$_ -and (Test-Path -LiteralPath $_)} |
    Select-Object -First 1
if(!$exe)
{
    Write-Error 'Cannot find dsi_studio.exe. Place dsi.ps1 beside it or set DSI_STUDIO_EXE.'
    exit 2
}

function Invoke-DsiStudio([string]$Argument)
{
    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $exe
    $start.UseShellExecute = $false
    $start.CreateNoWindow = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.Arguments = '"'+$Argument.Replace('"','\"')+'"'
    $process = [Diagnostics.Process]::Start($start)
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    [Console]::Out.Write($stdout.Result)
    [Console]::Error.Write($stderr.Result)
    return $process.ExitCode
}

if($Target -eq 'OPEN')
{
    if($Command.Count -ne 1)
    {
        Write-Error 'Usage: .\dsi.ps1 agent@session OPEN <file>'
        exit 2
    }
    exit (Invoke-DsiStudio (Resolve-Path -LiteralPath $Command[0]).Path)
}

$request = [ordered]@{
    agent = $agent
    session = $session
    cwd = (Get-Location).Path
    request = $Target.ToUpper()
}

if($request.request -eq 'LOG')
{
    if($Command.Count)
    {
        $request.chat = $Command -join ' '
    }
}
elseif($request.request -ne 'LIST')
{
    if($Target -notmatch '^\d+$' -or !$Command.Count)
    {
        Write-Error 'Usage: .\dsi.ps1 agent@session <window-id> <command> [parameters...]'
        exit 2
    }
    $request.request = 'CMD'
    $request.window = $Target
    $request.command = $Command
}

$json = $request | ConvertTo-Json -Compress -Depth 8
exit (Invoke-DsiStudio $json)
```

Use the same `agent@session` identity for the entire conversation. Each script
invocation sends exactly one request through `dsi_studio.exe` and exits. Use
this script directly; do not create another PowerShell, Python, batch, or
temporary client.

If Windows blocks direct script execution, do not change the user's execution
policy. Invoke the same file with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\dsi.ps1 `
    myagent@session1 LIST
```

## Requests

```powershell
# Discover windows.
.\dsi.ps1 myagent@session1 LIST

# Use a numeric ID returned by LIST.
.\dsi.ps1 myagent@session1 2 list_region

# Parameters containing spaces remain one quoted argument.
.\dsi.ps1 myagent@session1 2 set_region_name 0 "Tumor Core"

# Incremental diagnostics and final user-facing reply.
.\dsi.ps1 myagent@session1 LOG
.\dsi.ps1 myagent@session1 LOG "Task completed."
```

Always use the numeric window ID returned by the latest `LIST`; never use a
window type, title, filename, guessed ID, or stale ID as `window`.

JSON fields are `agent`, `session`, `cwd`, `request`, `window`, `command`, and
optional `chat`. The script supplies them from its arguments. Requests are
`LIST`, `CMD`, or `LOG`; `OPEN` sends one filename through the existing
filename transport. Keep parameters containing spaces as one quoted argument.

`LIST` and `LOG` replies begin with `OKAY`. Diagnostic `LOG` returns at most
4096 new console characters since the prior `LOG` or first request. Every
`LOG` advances the cursor. The console is global, so concurrent agents may see
each other's new DSI output. Final `LOG` with `chat` returns no console history.
`[AI AGENT]` trace lines are omitted. `[AI REQUEST]` groups and closing `⏱`
lines report synchronous DSI-side request handling, not agent runtime or
asynchronous completion.

Filename-open replies and validation errors are also text. A `CMD` reply
beginning with `[` is a JSON array of `{index,okay,output,error?}` objects.
List-command data remains text inside each result's `output`; do not invent
properties such as `.windows` or `.tracks`.

Every `CMD`, including every `list_*` command, requires the numeric `window`
from the latest `LIST`. Use the `main` window ID for `list_recent_fib` and
`list_recent_src`; use a `tracking` window ID for `list_tract`.

A queued user prompt may follow a text reply as `PROMPT<TAB><JSON>` or appear
in the last command result's `prompt` property. Treat it as new user input.

## Opening local files

When only the main window exists, send one absolute filename:

```powershell
.\dsi.ps1 myagent@session1 OPEN 'C:\data\subject.fz'
.\dsi.ps1 myagent@session1 LIST
```

Poll `LIST` for the new numeric `tracking` or `image` window ID. `open_fib`
requires an existing tracking window and cannot create the first one.

In DSI Studio, **FIB means `.fz`**. Never substitute `.sz`; `.sz` is an SRC
file. `OPEN` can open one `.fz`, `.sz`, or image file.

To open multiple images in one O1 window, send one flat command to the numeric
main-window ID:

```powershell
.\dsi.ps1 myagent@session1 1 open_image `
    'C:\data\a.nii.gz' 'C:\data\b.nii.gz'
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
11. Minimize round trips: one initial `LIST`, only necessary commands, concise
    verification, and final `LOG`.
12. When asked to operate DSI Studio, execute the requests. Do not return a
    script or tutorial unless the user asks for one.

If DSI Studio resumes an agent, reconnect with the exact same agent and
session strings. Process every returned `PROMPT` and exit naturally when none
remains.
