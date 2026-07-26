# DSI Studio AI-Agent Setup

Use this file to connect a local AI agent to an AI-control-enabled DSI Studio.
Read `DSI_STUDIO_AI_MANUAL.md` completely before issuing domain commands; it is
the authoritative command reference. On Windows, connect directly to the
`dsi-studio` named pipe. Launching `dsi_studio.exe` for every request is only a
fallback.

This setup was reviewed against the current DSI Studio source on 2026-07-25,
including the JSON request envelope and AI chat-history integration added after
`bdfd98b5647f2a68b9bb8bd0691240f34c1c9a4b`.

## Mandatory agent rules

- Generate one stable, unique agent ID at the start of the session. It must
  begin with `@`. Reuse the exact same case-sensitive ID for every request in
  that session.
- Send every control request as one JSON object. `LIST`, `CMD`, `LOG`, and `WAIT` are
  values of the JSON `request` property; never send any of them as a standalone
  text request. The only non-JSON request is an existing filename that DSI
  Studio should open.
- Include `chat` only when the agent has new user-facing text that DSI Studio
  should display. Do not attach internal reasoning, tool output, or text that
  has already been attached.
- Never resend previously attached `chat`.
- When DSI Studio should display the final answer, attach that answer once as
  `chat` on the final JSON `LOG` request.

These rules preserve one continuous agent identity and prevent duplicate chat
messages in DSI Studio.

## Requirements

- The agent must be able to execute local Windows processes on the same
  computer as DSI Studio. A remote or browser-only agent requires a separate
  local execution bridge.
- Obtain the exact path to `dsi_studio.exe` only when DSI Studio must be started
  or the executable fallback is needed.
- Obtain access only to the executable, requested input data, manuals, and
  output directories required for the task.
- DSI Studio should already be running unless the user explicitly asks the
  agent to start it.

## Connect directly

`QLocalServer("dsi-studio")` is the Windows named pipe
`\\.\pipe\dsi-studio`. The server handles one request per connection, writes
one reply, and disconnects. `WAIT` is the exception: its connection remains
open until DSI Studio returns a user prompt.

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

function Invoke-Dsi([hashtable]$Request)
{
    if($Request.ContainsKey('agent')) { throw 'Invoke-Dsi supplies the agent ID' }
    $Request = @{} + $Request
    $Request.agent = $DsiAgentId
    Invoke-DsiRequest ($Request | ConvertTo-Json -Compress -Depth 8)
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

$listReply = Read-DsiTextReply (Invoke-Dsi @{request='LIST'})
if(-not $listReply.Text.StartsWith('OKAY')) { throw $listReply.Text }
```

A successful `LIST` response resembles:

```text
OKAY
main<TAB>1<TAB>DSI Studio ...
tracking<TAB>2<TAB>C:\data\subject.fz
```

Always request `LIST` before the first command and again after opening or
closing a window. A pipe connection timeout means no compatible DSI Studio
server is available; ask the user to start it.

## JSON request forms

Use the same agent ID in every object:

```json
{"agent":"@C7f2a","request":"LIST"}
{"agent":"@C7f2a","request":"CMD","window":"2","command":["list_region"]}
{"agent":"@C7f2a","request":"LOG"}
{"agent":"@C7f2a","request":"WAIT"}
```

`@C7f2a` is only an example. Generate a new ID once when the agent session
begins.

### Send one command

`command` is an array of strings. Its first element is the command name and the
remaining elements are parameters:

```powershell
$reply = Invoke-Dsi @{
    request='CMD'
    window='2'
    command=@('list_slice')
}
$results = $reply | ConvertFrom-Json
if($results | Where-Object { -not $_.okay }) { throw $reply }
$prompts = @($results[-1].prompt)
```

Each parameter remains one array element. A parameter containing spaces must
not be split. Empty strings are valid only where the command reference
explicitly documents them.

### Send a command batch

For a same-window batch, `command` is an array of command arrays:

```powershell
$reply = Invoke-Dsi @{
    request='CMD'
    window='2'
    command=@(
        @('list_slice'),
        @('list_region'),
        @('list_tract')
    )
}
$results = $reply | ConvertFrom-Json
if($results | Where-Object { -not $_.okay }) { throw $reply }
$prompts = @($results[-1].prompt)
```

All command and parameter values must be strings. Commands run in order and
stop at the first failure. The result is a compact JSON array containing
`index`, `okay`, `output`, and, on failure, `error`. A queued DSI Studio prompt
appears in the optional `prompt` property of the last result.

Batch only short synchronous commands whose inputs are ready. Do not send an
empty batch or place a command after an asynchronous operation on which it
depends. A batch is not transactional and does not roll back earlier commands.

Target commands according to the window type returned by `LIST`:

- `main`: Fiber Data Hub and main-window operations
- `tracking`: slices, regions, tracts, atlases, segmentation, rendering, and
  tracking
- `image`: image-window operations

### Wait for a user prompt

After completing a turn, keep the agent available without polling:

```powershell
$waitReply = Invoke-Dsi @{request='WAIT'} | ConvertFrom-Json
$prompts = @($waitReply.prompt)
```

`WAIT` keeps its pipe connection open. When the user sends text from DSI
Studio, the reply is a JSON object whose `prompt` property is an array. Process
the prompt, complete the requested work, and issue another `WAIT`.

If a prompt was already queued, `WAIT` returns it immediately. If no `WAIT`
connection exists, DSI Studio retains the prompt for the agent's next `LIST`,
`LOG`, `CMD`, or `WAIT` request. Do not repeatedly poll `LIST`. DSI Studio
cannot use `WAIT` to restart an agent process that has exited.

## Attach chat without duplication

`chat` is optional and independent of command execution:

```powershell
$reply = Invoke-Dsi @{
    request='CMD'
    window='2'
    command=@('list_region')
    chat='I am checking the current regions before making changes.'
}
```

Attach a message only on the first request made after that user-facing text is
created. Omit `chat` from later polling, verification, and retry requests.

At the end of the task, if DSI Studio should display the agent's final answer,
send one final log request:

```powershell
$final = 'The requested tract was created and saved to C:\data\CST_L.tt.gz.'
$logReply = Read-DsiTextReply (Invoke-Dsi @{request='LOG';chat=$final})
```

Chat is stored in `ai_chat_history`, not console history, so the `LOG` reply
does not echo the attached text. Do not send the same final text again.

## Open a local file

An existing filename is the only non-JSON request:

```powershell
Invoke-DsiRequest 'C:\data\subject.fz'
$listReply = Read-DsiTextReply (Invoke-Dsi @{request='LIST'})
```

The server forwards the filename to the running DSI Studio. Poll `LIST` to
discover the new window ID.

## Inspect before acting

Never guess names, indices, or parameter values. Use the available
introspection commands first, including:

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

Use only commands and parameters documented in
`DSI_STUDIO_AI_MANUAL.md`.

`list_recent` targets the main window. `run_cli` accepts one complete DSI
Studio command line as one string:

```powershell
$reply = Invoke-Dsi @{
    request='CMD'
    window='1'
    command=@('run_cli','--action=qc --source=C:\data\subject.fz')
}
```

`--action` is required. Inspect the full command line and obtain any required
confirmation before running it; `run_cli` is not a shell.

For large Fiber Data Hub listings, `hub files` accepts optional `text`,
`offset`, and `limit` strings after repository and tag:

```powershell
$reply = Invoke-Dsi @{
    request='CMD'
    window='1'
    command=@('hub','files','owner/repository','tag','','100','50')
}
```

For segmentation, first select the intended slice, run `list_unet`, and then
send both the exact model and exact slice name:

```powershell
$reply = Invoke-Dsi @{
    request='CMD'
    window='2'
    command=@('segment_brain','<model-from-list_unet>','<exact-slice-name>')
}
```

An explicit model without a slice name is rejected. For deterministic
visibility changes, use `show_only_regions` or `show_only_tracts` with
ampersand-joined indices. Region metadata commands take an index and value:
`set_region_name`, `set_region_type` (`0..6`), and `set_region_color`
(unsigned packed Qt ARGB).

## Console output and prompts

Retrieve console history with:

```powershell
$logReply = Read-DsiTextReply (Invoke-Dsi @{request='LOG'})
```

For `LIST` and `LOG`, pending prompts are inserted after the first status line:

```text
PROMPT<TAB><JSON>
```

Pass `LIST` and `LOG` replies through `Read-DsiTextReply`, process `Prompts` as
agent input, and interpret only `Text` as normal output. Every JSON `CMD`
request returns a JSON result array; read the optional `prompt` property from
its last result.

Prompt queues are keyed by the exact agent ID and cleared only after the
matching reply is written. Different agents may interleave requests when their
IDs differ, although GUI commands still execute serially on Qt's main thread.

## Executable fallback

If the agent cannot use Windows named pipes, invoke the executable with one
complete JSON request:

```powershell
$dsiExe = 'C:\DSI-Studio\dsi_studio.exe'
$request = @{agent=$DsiAgentId;request='CMD';window='2';
             command=@('list_slice')} | ConvertTo-Json -Compress
& $dsiExe $request
```

This starts a new client process for every request and has a fixed five-second
reply timeout. It can print `TIMEOUT` while a GUI operation continues. Do not
retry an operation whose state is unknown.

## Completion and verification

- A `CMD` request succeeds only when every returned object has `"okay":true`.
- Later batch results are absent after the first failure.
- Poll the relevant list or status command until completion.
- Refresh `LIST` when a command opens or closes a window.
- Verify created regions or tracts using their list commands.
- Verify every exported file exists and is readable before reporting success.
- If possible, inspect and show exported images to the user.
- Report errors and partial results accurately; do not silently retry
  destructive operations.

## Safety

Obtain confirmation before:

- overwriting or deleting files;
- deleting or replacing regions or tracts;
- closing windows with unsaved work;
- downloading unexpectedly large datasets;
- starting expensive batch processing outside the user's stated scope.

Avoid commands that require modal dialogs during unattended operation. Prefer
explicit, fully parameterized commands. Use native-space GQI `.fz` data when
alignment with native structural images is required.

## Agent behavior

Translate the user's scientific goal into documented DSI Studio commands; do
not require the user to know command names. State which dataset and window will
be affected, execute the smallest safe sequence, monitor it to completion,
verify the result, and summarize what DSI Studio produced. Keep the same agent
identity throughout, attach only newly generated user-facing text, and attach
the final answer once to the final `LOG` request when DSI Studio should display
it.
