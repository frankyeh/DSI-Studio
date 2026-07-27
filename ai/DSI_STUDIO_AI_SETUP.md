# DSI Studio AI Setup

Read this file completely. Search `DSI_STUDIO_AI_MANUAL.md` only for commands
needed by the request; do not read its entire inventory.

## Identity

Choose one non-empty agent name and the exact non-empty thread/session ID of
the initiating chat when the AI conversation starts. Reuse both exactly in
every request. The agent name must include `Codex` or `Claude` and
must not contain `@`.

Ollama is a model provider for Claude Code, not an agent. Selecting an Ollama
model runs Claude with the configured endpoint, `ANTHROPIC_AUTH_TOKEN=ollama`,
an empty `ANTHROPIC_API_KEY`, and Claude's resumable session ID.

DSI Studio uses `session` as the unique conversation key and stores the agent
name separately. Requests send both as separate JSON fields; `agent@session`
is only the combined display/wrapper form. The session must be the agent's
exact resumable thread ID (for Codex, the `thread_id` from
`thread.started`) in canonical UUID form, never a friendly label or a
request-local GUID. An agent initiating a DSI connection must provide it in
its first request; otherwise DSI Studio cannot later resume the correct chat.

### Codex launched by DSI Studio

DSI Studio creates the thread, then resumes it with the exact thread ID in the
task prompt. Use that supplied value as `session` in every request; never
replace it with a generated ID.

### Codex Desktop fallback

Codex Desktop may expose a task UUID indirectly in its injected runtime context
or a task-specific path, for example `...\visualizations\...\019f...`. If a
UUID is explicitly present there, use it as the Codex session ID. Do not guess,
generate, or scan for an ID. This is a best-effort fallback; the reliable route
is DSI Studio launching Codex with `exec --json` and capturing `thread.started`.

### Claude Code

Claude Code stores its session ID in `~/.claude/sessions/<pid>.json`. Always
read the current Claude process's file and send its `sessionId` field as the
DSI Studio `session` value; DSI Studio uses it for
`claude -p --resume <sessionId>`. Do not use the JSON `name` field. For example, if
`C:\Users\YEHFC\.claude\sessions\42232.json` contains
`"sessionId":"c24d222a-7e8e-4aed-a7ca-18624978eaf9"`, use
`dsi-claude@c24d222a-7e8e-4aed-a7ca-18624978eaf9`.

```powershell
$DsiAgent = 'Codex'
$DsiSession = '<initiating-chat-session-id>'
```

## Direct named-pipe connection

**Use the named pipe directly first.** Connect to `\\\\.\\pipe\\dsi-studio`, send
one complete request, read until DSI Studio closes the connection, then close
the client. Do **not** run `dsi_agent.ps1`, `dsi_studio.exe`, or another
existing wrapper merely because it is present. They are fallbacks only after
direct pipe access is unavailable or a direct connection fails and the user
approves using one.

If the runtime needs PowerShell, this is still a direct pipe client; it does
not launch DSI Studio:

```powershell
function Invoke-Dsi($request)
{
    $pipe = [IO.Pipes.NamedPipeClientStream]::new(
        '.','dsi-studio',[IO.Pipes.PipeDirection]::InOut)
    $pipe.Connect(5000)
    $utf8 = [Text.UTF8Encoding]::new($false)
    $writer = [IO.StreamWriter]::new($pipe,$utf8,1024,$true)
    $writer.AutoFlush = $true
    if($request -is [string]) { $data = $request }
    else { $data = $request | ConvertTo-Json -Compress -Depth 8 }
    $writer.Write($data)
    $reader = [IO.StreamReader]::new($pipe,$utf8,$false,1024,$true)
    $reply = $reader.ReadToEnd()
    $reader.Dispose()
    $writer.Dispose()
    $pipe.Dispose()
    $reply
}
$DsiAgent = 'Codex'
$DsiSession = '<initiating-chat-session-id>'
```

Reuse this client and the same identity throughout the conversation. Do not
regenerate it for every request.

## Optional executable fallback

Use this only when a direct named-pipe client cannot run or cannot connect. Run
on the same Windows computer as DSI Studio. Create one client
for the current AI conversation, for example `dsi_agent.ps1`, using this exact
template. Save it in the agent's working directory and reuse it for every
request in that conversation.

```powershell
param(
    [Parameter(Mandatory,Position=0)]
    [string]$Identity,

    [Parameter(Mandatory,Position=1)]
    [string]$Target,

    [Parameter(Position=2,ValueFromRemainingArguments)]
    [string[]]$Command,

    [Parameter()]
    [string]$Chat
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
        Write-Error 'Usage: <client.ps1> agent@session OPEN <file>'
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
if($Chat)
{
    $request.chat = $Chat
}

if($request.request -in @('LOG','CHAT'))
{
    if(!$Chat -and $Command.Count)
    {
        $request.chat = $Command -join ' '
    }
}
elseif($request.request -ne 'LIST')
{
    if($Target -notmatch '^\d+$' -or !$Command.Count)
    {
        Write-Error 'Usage: <client.ps1> agent@session <window-id> <command> [parameters...]'
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
the same generated script for all requests; do not regenerate it per request
or create additional PowerShell, Python, or batch clients.

If Windows blocks direct script execution, do not change the user's execution
policy. Invoke the same file with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\dsi_agent.ps1 `
    Codex@your-session-id LIST
```

## Requests

```powershell
# Discover windows.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='LIST';chat='Connecting and checking open windows.'}

# Use a numeric ID returned by LIST.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CMD';window='2';command=@('list_region')}

# Command parameters stay separate array elements.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CMD';window='2';command=@('set_region_name','0','Tumor Core')}

# Incremental diagnostics and final user-facing reply.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;request='LOG'}
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CHAT';chat='Task completed.'}
```

Always use the numeric window ID returned by the latest `LIST`; never use a
window type, title, filename, guessed ID, or stale ID as `window`.

JSON fields are `agent`, `session`, `cwd`, `request`, `window`, `command`, and
optional `chat`. Requests are `LIST`, `CMD`, `LOG`, or `CHAT`; send one absolute path
as raw pipe text to open a file. Keep every command parameter as one array
element.

`LIST`, `LOG`, and `CHAT` replies begin with `OKAY`. `CHAT` returns no console
history. Diagnostic `LOG` returns at most
4096 new console characters since the prior `LOG` or first request. Every
`LOG` advances the cursor. The console is global, so concurrent agents may see
each other's new DSI output.
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

## Progress chat

DSI Studio's activity history shows which commands ran but cannot explain why
the agent ran them. Attach a short `chat` update to the next necessary JSON
request so the user can follow the agent's intent in the AI Agents tab.

Send one concise sentence at task start, before each meaningful phase, when
waiting for a long operation, when blocked, and at completion. Do not expose
reasoning, logs, or tool details. Do not repeat unchanged status, attach chat to
every polling request, or create a separate request only to report status.
`OPEN` uses filename transport, so attach its intent to the nearest necessary
`LIST` or `CMD`.

## Opening local files

When only the main window exists, send one absolute filename:

```powershell
Invoke-Dsi (Resolve-Path -LiteralPath 'C:\data\subject.fz').Path
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;request='LIST'}
```

Poll `LIST` for the new numeric `tracking` or `image` window ID. `open_fib`
requires an existing tracking window and cannot create the first one.

In DSI Studio, **FIB means `.fz`**. Never substitute `.sz`; `.sz` is an SRC
file. `OPEN` can open one `.fz`, `.sz`, or image file.

To open multiple images in one O1 window, send one flat command to the numeric
main-window ID:

```powershell
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CMD';window='1';
             command=@('open_image','C:\data\a.nii.gz','C:\data\b.nii.gz')}
```

Do not send separate `open_image` commands, target an image window, split a
path into fields, or substitute `add_image`. Refresh `LIST` afterward.

## Required behavior

1. Use a direct local named-pipe client. If it is unavailable or fails, ask
   before using an executable wrapper. Use GUI commands. **Do not use `run_cli` unless the user explicitly says to
   run the CLI.** Never infer CLI permission from a requested outcome.
2. Call `LIST` first and after windows open or close.
3. Use only numeric IDs returned by the latest `LIST`.
4. Discover values with `list_slice`, `list_region`, `list_tract`,
   `list_param`, `list_atlas`, `list_unet`, and `list_auto_tract`.
5. Treat `okay:true` as acceptance. Poll the relevant list command for
   asynchronous completion; use `LOG` only when diagnostics are needed.
   If a response says it is loading, first send `CHAT` stating that waiting
   will continue and the user can interrupt, then check every 3 seconds unless
   interrupted; process every reply.
   Never automatically repeat a failed, timed-out, unavailable, or unexpected
   request.
6. If a required window disappears or returns `window not found`, assume the
   user closed it. Do not recheck, reopen, or retry it. Stop and send one
   `CHAT` asking whether to continue or stop; resume only after the reply.
7. Verify outputs. Ask before destructive operations or overwrites.
8. Do not answer modal dialogs remotely; tell the user what is required.
9. Keep the user informed with brief intent-focused `chat` updates at meaningful
   phase changes, attached to requests already needed for the task.
10. Send the final answer once with `CHAT`.
11. Minimize round trips: one initial `LIST`, only necessary commands, concise
    verification, and final `CHAT`.
12. When asked to operate DSI Studio, execute the requests. Do not return a
    script or tutorial unless the user asks for one.

If DSI Studio resumes an agent, reconnect with the exact same agent and
session strings. Process every returned `PROMPT` and exit naturally when none
remains.
