# DSI Studio AI-Agent Setup

Use this file to connect a local AI agent to an AI-control-enabled DSI Studio. Read `DSI_STUDIO_AI_MANUAL.md` completely before issuing domain commands; it is the authoritative command reference. On Windows, connect directly to the `dsi-studio` named pipe. Launching `dsi_studio.exe` for every request is only a fallback.

## Requirements

- The agent must be able to execute local Windows processes on the same computer as DSI Studio. A remote or browser-only agent cannot control the local application without a separate local execution bridge.
- Obtain the exact path to `dsi_studio.exe` only when DSI Studio must be started or the executable fallback is needed.
- Obtain access only to the executable (if needed), requested input data, manual, and output directories required for the task.
- DSI Studio should already be running unless the user explicitly asks the agent to start it.
- Generate one stable, unique ID for each agent session. Keep the leading `@`, reuse the exact same case-sensitive ID for every request in that session, and never share it with another simultaneous agent.

## Connect directly

`QLocalServer("dsi-studio")` is the Windows named pipe `\\.\pipe\dsi-studio`. The server handles one request per connection and then disconnects. A request may contain one command or a same-window command batch.

```powershell
function Invoke-DsiRequest([string]$Request)
{
    $pipe = [System.IO.Pipes.NamedPipeClientStream]::new(
        '.', 'dsi-studio',
        [System.IO.Pipes.PipeDirection]::InOut)
    try
    {
        $pipe.Connect(2000)
        $utf8 = [System.Text.UTF8Encoding]::new($false)
        $bytes = $utf8.GetBytes($Request)
        $pipe.Write($bytes, 0, $bytes.Length)
        $pipe.Flush()

        $buffer = New-Object byte[] 8192
        $reply = [System.Text.StringBuilder]::new()
        while(($count = $pipe.Read($buffer, 0, $buffer.Length)) -gt 0)
        {
            [void]$reply.Append($utf8.GetString($buffer, 0, $count))
        }
        $reply.ToString()
    }
    finally
    {
        $pipe.Dispose()
    }
}

$DsiAgentId = '@' + [guid]::NewGuid().ToString('N').Substring(0,12)

function Invoke-Dsi([string[]]$Fields)
{
    if($Fields.Count -eq 0) { throw 'Empty DSI Studio request' }
    $wire = @($Fields[0], $DsiAgentId)
    if($Fields.Count -gt 1) { $wire += $Fields[1..($Fields.Count-1)] }
    Invoke-DsiRequest ([string]::Join([char]9, $wire))
}

function Read-DsiTextReply([string]$Reply)
{
    $lines = @($Reply -split '\r?\n')
    $prompts = @()
    if($lines.Count -gt 1 -and $lines[1].StartsWith("PROMPT`t"))
    {
        $prompts = @($lines[1].Substring(7) | ConvertFrom-Json)
        $lines = @($lines[0]) + @($lines | Select-Object -Skip 2)
    }
    [pscustomobject]@{ Text=($lines -join "`n"); Prompts=$prompts }
}

$listReply = Read-DsiTextReply (Invoke-Dsi @('LIST'))
if(-not $listReply.Text.StartsWith('OKAY')) { throw $listReply.Text }
```

A successful response resembles:

```text
OKAY
main    1    DSI Studio ...
tracking    2    C:\data\subject.fz
```

`LIST` assigns and returns remote window IDs. Always call it before the first command and again after opening or closing a window. A pipe connection timeout means no compatible DSI Studio server is available; ask the user to start it.

The preferred wire forms are:

```text
LIST<TAB>@C7f2a
CMD<TAB>@C7f2a<TAB>2<TAB>list_region
LOG<TAB>@C7f2a
```

`@C7f2a` is an example only. Generate a new ID when the agent session begins, then keep it stable for that session.

## Send a command

Build the request with actual tab characters:

```powershell
$reply = Read-DsiTextReply (Invoke-Dsi @('CMD', '2', 'list_slice'))
```

Protocol:

```text
CMD<TAB>@agent_id<TAB>window_id<TAB>command<TAB>parameter_1<TAB>parameter_2...
```

Each array element is one command field. A parameter containing spaces must remain one element. Never split or reinterpret compound parameters described by `DSI_STUDIO_AI_MANUAL.md`.

## Send a command batch

Use the same `CMD` route with a JSON array as its third field. There is no separate `BATCH` request:

```powershell
$json = '[["list_slice"],["list_region"],["list_tract"]]'
$reply = Invoke-Dsi @('CMD', '2', $json)
$results = $reply | ConvertFrom-Json
if($results | Where-Object { -not $_.okay }) { throw $reply }
$prompts = @($results[-1].prompt)
```

Protocol:

```text
CMD<TAB>@agent_id<TAB>window_id<TAB>[["command","parameter"],["command",...]]
```

All JSON values must be strings, all commands target the same window, and commands run in order. DSI Studio suppresses intermediate widget repaint and redraws once afterward. The reply is a compact JSON array containing `index`, `okay`, `output`, and, on failure, `error`. When DSI Studio has queued prompts for this agent, the last returned result also contains a `prompt` JSON-array property. Read and act on that property before continuing. Execution stops at the first failure; a batch is not transactional and has no rollback.

Use batching for short synchronous commands whose inputs are already ready. Do not send an empty batch or batch a command that depends on an earlier asynchronous download, registration, segmentation, tracking, or window-opening result.

Target commands according to the window type returned by `LIST`:

- `main`: Fiber Data Hub and main-window operations
- `tracking`: slices, regions, tracts, atlases, segmentation, rendering, and tracking
- `image`: image-window operations

## Open a local file

Send an existing filename as the executable's single argument:

```powershell
Invoke-DsiRequest 'C:\data\subject.fz'
$listReply = Read-DsiTextReply (Invoke-Dsi @('LIST'))
```

The client forwards the filename to the running DSI Studio. Refresh `LIST` to discover the new window ID.

## Inspect before acting

Never guess names, indices, or parameter values. Use the available introspection commands first, including:

```text
list_recent
list_slice
list_region
list_tract
list_param
list_atlas
list_unet
list_auto_tract
```

Use only commands and parameters documented in `DSI_STUDIO_AI_MANUAL.md`.

`list_recent` targets the main window and lists recent `.sz` and `.fz` paths. The main-window command `run_cli` accepts one complete DSI Studio command line as a single parameter field:

```powershell
$reply = Read-DsiTextReply (
    Invoke-Dsi @('CMD', '1', 'run_cli', '--action=qc --source=C:\data\subject.fz'))
```

`--action` is required. DSI Studio parses this string internally and runs the CLI action synchronously on the GUI thread, including wildcard or `--loop` processing. Inspect the full command line and obtain any required confirmation before running it; do not use `run_cli` as a shell.

## Console output

Retrieve DSI Studio's available console history with:

```powershell
$logReply = Read-DsiTextReply (Invoke-Dsi @('LOG'))
```

Use the console to diagnose loading, registration, downloading, segmentation, tracking, and export failures. Do not treat the absence of an error message as proof of success.

## Prompts from DSI Studio

For `LIST`, `LOG`, and single-command text replies, DSI Studio inserts this immediately after the first status line when prompts are pending for the current agent:

```text
PROMPT<TAB><JSON>
```

Pass every non-batch `LIST`, `CMD`, and `LOG` reply through `Read-DsiTextReply`. Process its `Prompts` values as agent input before continuing, and interpret only its cleaned `Text` as command output. For a batch reply, parse the JSON array and read the optional `prompt` property from its last result instead; no `PROMPT` text line is added to a batch.

DSI Studio keeps prompt queues separate by agent ID and clears only the matching queue after the complete reply is written. This protection works only when the same stable, unique session ID is included in `LIST`, `CMD`, and `LOG`.

Legacy requests without an ID (`LIST`, `CMD<TAB>window_id...`, and `LOG`) remain accepted, but they all share the empty legacy identity. They cannot safely separate prompts when multiple agents are active and must not be used for simultaneous-agent sessions.

## Executable fallback

If the agent cannot use Windows named pipes, invoke the executable with one complete request:

```powershell
$dsiExe = 'C:\DSI-Studio\dsi_studio.exe'
$request = [string]::Join([char]9, @('CMD', $DsiAgentId, '2', 'list_slice'))
& $dsiExe $request
```

This starts a new client process for every request and has a fixed five-second reply timeout, so it is slower and can print `TIMEOUT` while the GUI operation continues. Do not retry an unknown operation blindly.

## Completion and verification

- `OKAY` means a single command was accepted; it may not mean an asynchronous operation has finished.
- A batch succeeds only when every returned object has `"okay":true`; later commands are absent after the first failure.
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
