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

DSI Studio starts a new Codex chat once with `codex exec --json` and the actual
user task. There is no initialization/bootstrap run and DSI Studio does not
launch the same task again. Process the initiating prompt immediately.

For this first run, read `CODEX_THREAD_ID` from the Codex environment and use
that exact UUID as `session` in the first and every subsequent local-server
request. DSI Studio also captures the same value from
`thread.started.thread_id` so it can resume later with `codex exec resume`.
Never send an empty, guessed, or generated session ID, and do not wait for a
second launch before contacting DSI Studio.

### Codex Desktop fallback

Codex Desktop may expose a task UUID indirectly in its injected runtime context
or a task-specific path, for example `...\visualizations\...\019f...`. If a
UUID is explicitly present there, use it as the Codex session ID. Do not guess,
generate, or scan for an ID. This is a best-effort fallback; the reliable route
is DSI Studio launching Codex with `exec --json`, using `CODEX_THREAD_ID`, and
capturing `thread.started`.

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
$DsiSession = $env:CODEX_THREAD_ID
```

For Claude or an externally initiated/resumed agent, set `$DsiSession` to the
exact initiating-chat session ID obtained as described above.

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
    $pipe = $writer = $reader = $null
    try
    {
        $pipe = [IO.Pipes.NamedPipeClientStream]::new('.','dsi-studio')
        $pipe.Connect(5000)
        $utf8 = [Text.UTF8Encoding]::new($false)
        $writer = [IO.StreamWriter]::new($pipe,$utf8,1024,$true)
        $writer.AutoFlush = $true
        if($request -is [string]) { $data = $request }
        else { $data = $request | ConvertTo-Json -Compress -Depth 4 }
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
$DsiSession = $env:CODEX_THREAD_ID
```

The two-argument constructor works in Windows PowerShell 5.1 and defaults to a
duplex (`InOut`) pipe. DSI Studio closes the server side after each reply, so
an `IOException` raised only while disposing completed streams is benign. The
helper suppresses only cleanup errors; connection, write, and read failures
still surface.

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
elseif($request.request -eq 'TITLE')
{
    if(!$Command.Count)
    {
        Write-Error 'Usage: <client.ps1> agent@session TITLE <title>'
        exit 2
    }
    $request.title = $Command -join ' '
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

$json = $request | ConvertTo-Json -Compress -Depth 4
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
# Discover windows and obtain global/per-window status.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='LIST';chat='Connecting and checking open windows.'}

# Use a numeric ID returned by LIST.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CMD';window='2';command=@('list_region')}

# Discover parameter IDs, then query one value.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;request='CMD';window='2';
             command=@('list_param')}
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;request='CMD';window='2';
             command=@('list_param','fa_threshold')}

# Poll global/per-window activity, including active tracking jobs.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;request='LIST'}

# Request targeted tract details only when needed.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;request='CMD';window='2';
             command=@('list_tract')}

# Command parameters stay separate array elements.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CMD';window='2';command=@('set_region_name','0','Tumor Core')}

# Use LOG only when LIST and targeted commands cannot diagnose the issue.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;request='LOG'}

# Send the final user-facing reply once.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='CHAT';chat='Task completed.'}

# Name the chat after understanding the initiating prompt.
Invoke-Dsi @{agent=$DsiAgent;session=$DsiSession;cwd=(Get-Location).Path;
             request='TITLE';title='Concise task name'}
```

`LIST` is a top-level request and does not use a `window` field. Its first line
is:

```text
OKAY<TAB>busy<TAB>level<TAB>status
```

Each following window line is:

```text
type<TAB>numeric-id<TAB>busy<TAB>tracking-jobs<TAB>title
```

The global `busy` value is true when TIPL progress or any supported window is
busy. `level` is the TIPL nesting depth excluding the persistent application
root; it is `1` when only asynchronous work such as fiber tracking is active.
Each window row independently reports whether that window is busy and how many
tracking bundles are active there. Global busy may be true while all window
rows are idle when TIPL cannot associate an operation with one window.

Every `CMD`, including every `list_*` command, requires a numeric window ID
returned by `LIST`; never use a window type, title, filename, guessed ID, or
stale ID as `window`.

JSON fields are `agent`, `session`, `cwd`, `request`, `window`, `command`,
`chat`, and `title`. Requests are `LIST`, `CMD`, `LOG`, `CHAT`, or
`TITLE`; send one absolute path as raw pipe text to open a file. Keep every
command parameter as one array element and never send an empty command array.

`LIST`, `LOG`, `CHAT`, and successful `TITLE` replies begin with `OKAY`.
`CHAT` and `TITLE` return no console history, but either may append a queued
`PROMPT`. Diagnostic `LOG` returns at most 4096 new console characters since
the prior `LOG` or first request. Every `LOG` advances the cursor. The console
is global, so concurrent agents may see each other's new DSI output.
`[AI AGENT]` trace lines are omitted. `[AI REQUEST]` groups and closing `⏱`
lines report synchronous DSI-side request handling, not agent runtime or
asynchronous completion.

Filename-open replies and validation errors are also text. A `CMD` reply
beginning with `[` is a JSON array of `{index,okay,output,error?}` objects.
List-command data remains text inside each result's `output`; do not invent
properties such as `.windows` or `.tracks`.

Always inspect the entire reply. A queued user prompt may follow any text reply,
including `LIST`, `CHAT`, or `TITLE`, as `PROMPT<TAB><JSON>`, or appear in the
last command result's `prompt` property. An initial `OKAY` does not mean no
`PROMPT` follows. Treat every returned prompt as new user input.

## Wait etiquette

Before loading, registration, reconstruction, segmentation, batch processing,
fiber tracking, or another substantial CPU/GPU operation, call `LIST`.

- If global `busy=0`, proceed.
- If the activity was started by this agent, do not start another substantial
  task. Send one concise `CHAT` saying what is still running and that the user
  may interrupt or terminate it, then wait.
- If the activity predates the intended operation or may belong to the user or
  another agent, send one concise `CHAT`: `DSI Studio is busy with <status>. I
  will wait by default. You may terminate the current work or tell me to proceed
  right away.` Without a direct instruction to proceed immediately, wait until
  global `busy=0`.
- If the user explicitly says to proceed right away, issue the requested work.
  If DSI Studio rejects it because another synchronous `CMD` is active, do not
  retry repeatedly; continue waiting.
- Poll only `LIST` after 4 seconds, then 8, 16, and 32 seconds. Continue every 32
  seconds while state is unchanged. Reset to 4 seconds whenever `status`,
  `level`, any window's `busy`, or any `tracking-jobs` value changes.
- Use a local sleep or timer between checks. Waiting should perform no model
  reasoning and consume no task tokens. Do not send repeated `CHAT`, call
  `LOG`, request detailed lists, or narrate unchanged polling.
- Inspect every complete `LIST` reply for `PROMPT`. User instructions always
  take priority. When global `busy` becomes `0`, continue without another
  waiting message unless a meaningful phase change needs one.

## Token efficiency

- Retain numeric window IDs until a window opens or closes. Poll compact `LIST`
  only while waiting for global or per-window activity to change.
- During waiting, use the 4/8/16/32-second schedule and sleep between checks;
  do not use model reasoning, chat, `LOG`, or detailed discovery while state is
  unchanged.
- Use each tracking row's `tracking-jobs` count to detect tracking completion.
  Request full `list_tract` only after completion when details are needed.
- Call `list_param` without an ID once to discover valid IDs, then query only
  the needed parameter values.
- Use `LOG` only after failure, unexpected state, or when `LIST` and a targeted
  list command cannot verify completion. Each call may return up to 4096
  characters.
- Batch independent synchronous commands for the same window into one `CMD`.
  Do not batch destructive, asynchronous, output-dependent, or modal-opening
  commands.
- Attach short progress `chat` to an already-needed request. Do not attach chat
  to every `LIST` poll. Use a standalone `CHAT` only for the final reply, a
  required user decision, or the one initial waiting/blocked update.
- Send `cwd` in the first request after connecting or restarting DSI Studio,
  then omit it until the working directory changes.
- Reuse discovered names and indices until the relevant state changes. Search
  the manual only for needed commands and never reread the command inventory.
- Keep requests compact. PowerShell `-Compress -Depth 4` covers the supported
  request shape, including batched command arrays.
- Stop after the requested result is verified and the final `CHAT` is sent; do
  not add redundant summaries, `LIST`, or `LOG` calls.

## Progress chat

DSI Studio's activity history shows which commands ran but cannot explain why
the agent ran them. Attach a short `chat` update to the next necessary JSON
request so the user can follow the agent's intent in the AI Agents tab.

Send one concise sentence at task start, before each meaningful phase, when
first entering a wait, when blocked, and at completion. For pre-existing busy
work, state that waiting is the default and that the user may terminate the
current work or explicitly request immediate execution. Do not expose reasoning,
logs, or tool details. Do not repeat unchanged status, attach chat to every
polling request, or create a separate request only to report unchanged status.
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

1. For a new Codex chat launched by DSI Studio, process the initiating task in
   the first run, use `CODEX_THREAD_ID` as `session`, and do not wait for or
   trigger a bootstrap/resume run.
2. Use a direct local named-pipe client. If it is unavailable or fails, ask
   before using an executable wrapper. Use GUI commands. **Do not use `run_cli`
   unless the user explicitly says to run the CLI.** Never infer CLI permission
   from a requested outcome.
3. Call top-level `LIST` first, after windows open or close, before substantial
   work, and while waiting for global or per-window activity to change. It does
   not use a `window` field.
4. Use only numeric IDs returned by `LIST` for `CMD` requests.
5. Use `LIST` for compact busy/progress and tracking-job polling. Discover
   detailed values with `list_slice`, `list_region`, `list_tract`, `list_param`,
   `list_atlas`, `list_unet`, and `list_auto_tract` only when needed.
6. Do not stack substantial work on an already busy DSI Studio by default. Send
   one waiting `CHAT`. For pre-existing activity, state that the user may
   terminate it or explicitly request immediate execution. Without that direct
   instruction, wait for global `busy=0` using `LIST` after 4, 8, 16, and 32
   seconds, then every 32 seconds. Reset to 4 seconds when state changes. Sleep
   without model reasoning between checks and inspect every reply for `PROMPT`.
7. Treat `okay:true` as acceptance. Poll `LIST` for long-running and
   asynchronous activity, then use a targeted list command to verify detailed
   output. Use `LOG` only when diagnostics are needed. Never automatically
   repeat a failed, timed-out, unavailable, or unexpected request.
8. If a required window disappears or returns `window not found`, assume the
   user closed it. Do not recheck, reopen, or retry it. Stop and send one
   `CHAT` asking whether to continue or stop; resume only after the reply.
9. Verify outputs. Ask before destructive operations or overwrites.
10. Do not answer modal dialogs remotely; tell the user what is required.
11. Keep the user informed with brief intent-focused `chat` updates at meaningful
    phase changes, attached to requests already needed for the task.
12. After understanding the initiating prompt, send one concise `TITLE`.
    Send another `TITLE` later only with user permission to rename.
13. Send the final answer once with `CHAT`.
14. Minimize round trips: retain window IDs, use compact `LIST` status, batch
    safe same-window commands, and avoid unnecessary detailed or `LOG` calls.
15. When asked to operate DSI Studio, execute the requests. Do not return a
    script or tutorial unless the user asks for one.

If DSI Studio resumes an agent, reconnect with the exact same agent and
session strings. Inspect every complete reply, process every returned `PROMPT`,
and exit naturally when none remains.