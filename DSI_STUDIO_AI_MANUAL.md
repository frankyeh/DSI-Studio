# DSI Studio AI-Agent Control Manual

**Source review:** The comprehensive command audit is pinned to
[`9e00c9c23f49df581a78bc1c9928134d262092ad`](https://github.com/frankyeh/DSI-Studio/commit/9e00c9c23f49df581a78bc1c9928134d262092ad).
Protocol, batching, agent-session routing, and prompt delivery were source-reviewed through
[`ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58`](https://github.com/frankyeh/DSI-Studio/commit/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58)
on 2026-07-25. Existing command-audit links remain pinned to the base commit;
revised entries link to their implementing commits. No GitHub Actions or status
checks were present on this HEAD, so the protocol description is source-verified
but compilation was not independently confirmed.

## Purpose and scope

Use this manual to control an already-running DSI Studio GUI from a local
Windows AI-agent session. Treat the source as authoritative: the public command surface is
the local-server routing in [`main.cpp`](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/main.cpp#L571-L615) and
[`mainwindow.cpp`](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/mainwindow.cpp#L45-L245), then the target window's
handler and its delegated handlers. The current surface exposes `main`,
`tracking`, and `image` windows. Reconstruction-window and `src_data` command
methods exist in source but are not remotely targetable through `LIST`.

The command-interface work was reviewed, including the twelve commits
from [`4c44366a`](https://github.com/frankyeh/DSI-Studio/commit/4c44366aba7ce5696f1ac3df9dccb6822637395b)
through [`9e00c9c23f49df581a78bc1c9928134d262092ad`](https://github.com/frankyeh/DSI-Studio/commit/9e00c9c23f49df581a78bc1c9928134d262092ad). Those
commits added and refined the IPC list/command routes and the tracking, tract,
and region commands described here. The later review through
[`8ad955a3`](https://github.com/frankyeh/DSI-Studio/commit/8ad955a333cdfcb8d6433a6a88137c0ae76f6cad)
adds JSON batching inside `CMD`, `list_recent`, `run_cli`, richer readiness
lists, multiple-parameter updates, and shorter `run_tracking` forms. Review
through [`ecacbd04`](https://github.com/frankyeh/DSI-Studio/commit/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58)
adds stable agent/session identities and per-agent prompt delivery while
retaining the legacy wire forms.

## Critical safety rules

Classify every operation before sending it:

| Class | Meaning | AI-agent rule |
|---|---|---|
| **Read-only** | Lists state or returns logs without changing data or GUI state. | Run directly. |
| **GUI-state change** | Changes selection, visibility, camera, slice position, or display parameters. | Run directly when reversible; record prior state when practical. |
| **Computation** | Registration, segmentation, tracking, clustering, filtering, or other expensive work. | State the intended inputs first; verify readiness; never infer completion from `OKAY`. |
| **File creation** | Writes images, tracts, regions, settings, workspaces, or downloads. | Resolve an absolute destination; confirm before overwrite; verify the file afterward. |
| **Destructive** | Deletes/replaces/merges regions or tracts, loads a workspace over current objects, or overwrites source data. | Obtain explicit user confirmation immediately before execution. |

Always obtain confirmation before:

- overwriting any existing path;
- deleting, merging, trimming, cutting, filtering, or replacing regions/tracts;
- loading a workspace when it will replace current regions, tracts, or devices;
- closing a window with unsaved state (no remote close command currently exists);
- downloading a large file or model;
- starting expensive batch work, processing many subjects, or starting tracking/segmentation whose scope is not already explicit.

Never run **TumorSynth**. It is temporarily unavailable for operational use,
even if `list_unet` happens to show a TumorSynth model. Do not download it,
select it, or invoke `segment_brain` with it.

Do not send an empty command parameter when the source falls back to a modal
file/input dialog. An unattended modal dialog can block the GUI and prevent all
subsequent IPC commands.

## Connection protocol

### Server and client lifecycle

The server name is exactly `dsi-studio`. On Windows, `QLocalServer` exposes it
as the named pipe `\\.\pipe\dsi-studio`. Connect to this pipe directly for
normal AI operation. The server processes one request per connection, writes
one reply, and disconnects; the client should therefore make a fresh lightweight
pipe connection for every request.

Direct pipe access was operationally verified on 2026-07-25 with the legacy
forms `LIST` and `CMD<TAB>1<TAB>hub<TAB>repos`. In a warm PowerShell process, measured local
round trips were approximately 1 ms (the first .NET call took approximately
381 ms for initialization). The identity-aware forms documented below were
source-reviewed through `ecacbd04`; they do not change the one-connection-per-request lifecycle.

Invoking `dsi_studio.exe` with one argument remains a fallback. That client
tries to connect for 5,000 ms, writes the argument as the complete request,
waits up to 5,000 ms for a reply, prints `TIMEOUT` if no bytes arrive, and
exits `0` when a single reply starts with `OKAY`, or when a JSON batch reply
contains only results whose `okay` value is `true`; otherwise it exits `1`
([`main()`](https://github.com/frankyeh/DSI-Studio/blob/8ad955a333cdfcb8d6433a6a88137c0ae76f6cad/main.cpp#L474-L502)).

The running GUI creates a world-access local server named `dsi-studio`, waits
500 ms for each request, and recognizes identity-aware or legacy `LIST`, `LOG`,
`CMD<TAB>...`, or a raw filename
([server dispatch](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/main.cpp#L571-L615)).
`main.cpp` only dispatches these requests; agent parsing, command execution,
prompt attachment, and reply writing are handled in `mainwindow.cpp`.

**Precondition:** start DSI Studio normally and wait for its main window. If no
instance exists, `LIST` returns `NO_INSTANCE`; other one-argument strings may
instead start a new GUI and be interpreted as filenames.

### Preferred PowerShell named-pipe helper

Use `[string]::Join([char]9, ...)` because the protocol delimiter is a literal
tab. It preserves empty fields and avoids editors, shells, or copied text
silently converting a tab to spaces.

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

function Invoke-DsiBatch([string]$WindowId, [string]$Json)
{
    Invoke-Dsi @('CMD', $WindowId, $Json)
}

$listReply = Read-DsiTextReply (Invoke-Dsi @('LIST'))
if(-not $listReply.Text.StartsWith('OKAY')) { throw $listReply.Text }
```

Do not manually embed tabs. Do not join a parameter's internal words with
tabs: `move_slice` needs one parameter field containing `80 90 60`, whereas
`save_slice_image` needs two tab-delimited parameter fields.

The single-command form has no escaping layer. Spaces are safe inside a field;
a field cannot contain a literal tab. Paths with spaces are safe when they are
one PowerShell array element. The batch form uses JSON escaping. UTF-8
conversion occurs in
[`ai_request_command()`](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/mainwindow.cpp#L101-L233).

### Request forms

| Request | Exact wire form | Meaning |
|---|---|---|
| Discover | `LIST<TAB>@agent_id` | Return targetable windows and deliver prompts queued for this agent. |
| Console | `LOG<TAB>@agent_id` | Return the rolling console history and deliver prompts queued for this agent. |
| Single command | `CMD<TAB>@agent_id<TAB>window_id<TAB>command<TAB>parameter...` | Route one command to one target. |
| Command batch | `CMD<TAB>@agent_id<TAB>window_id<TAB>[["command","parameter"],["command",...]]` | Run several commands sequentially on one target. |
| Raw open | one absolute filename | Ask the main window to open a file. |

There is no standalone `BATCH` request: both forms use `CMD`. Tracking and main
windows accept any number of fields after the command.
Image windows reject more than one parameter field with `ERROR` `"too many
parameters"` ([image route](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L111-L120)); put all image
command parameters in one space-delimited field.

### Agent session IDs and prompt delivery

Generate one stable, unique ID when an agent session begins. The ID must start
with `@`, contain no tab or newline, and be reused with identical case in every
`LIST`, `CMD`, and `LOG` request from that session. Do not generate a new ID for
each connection and do not share one ID between simultaneous agents.

```text
LIST<TAB>@C7f2a
CMD<TAB>@C7f2a<TAB>2<TAB>list_region
LOG<TAB>@C7f2a
```

For a text reply, pending prompts are inserted immediately after the first
status line:

```text
OKAY
PROMPT<TAB><JSON>
<normal command output...>
```

Pass every non-batch reply through `Read-DsiTextReply`. Process its `Prompts`
values as agent input before continuing, and interpret only its cleaned `Text`
as normal command output. For a batch, no text metadata line is added; the last
result object instead receives an optional `prompt` JSON-array property. Read
that property before continuing.

Prompt queues are keyed by the exact agent ID. DSI Studio clears only the
matching agent's queue, and only after the complete reply is written. Multiple
agents may interleave pipe requests safely when their IDs differ, although all
GUI command handlers still run serially on Qt's main thread.

Legacy `LIST`, `LOG`, and `CMD<TAB>window_id...` requests remain compatible.
They use the empty legacy identity, so simultaneous legacy agents share one
prompt queue and cannot safely separate their prompts. Never use legacy forms
when more than one agent may be active.

### Command batches

In batch form, the entire payload after the window ID is a JSON outer array.
Every outer element must be an array, and every command or parameter value must
be a JSON string. All commands use the same target window.

```powershell
$json = '[["list_slice"],["list_region"],["list_tract"]]'
$reply = Invoke-DsiBatch $trackingId $json
$results = $reply | ConvertFrom-Json
if($results | Where-Object { -not $_.okay }) { throw $reply }
$prompts = @($results[-1].prompt)
```

The compact reply is a JSON array in execution order:

```json
[
  {"index":0,"okay":true,"output":"..."},
  {"index":1,"okay":false,"output":"...","error":"canceled","prompt":[...]}
]
```

DSI Studio stops at the first failed command, so later commands are absent.
Each command gets its own captured `output`. During the batch, widget updates
are disabled and the target is redrawn once afterward. This reduces connection
and repaint overhead but does not skip command computation or side effects.
The optional `prompt` property appears only on the last returned object and
only when this agent has queued prompts. The batch is not a transaction and does not roll back earlier commands
([batch and prompt implementation](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/mainwindow.cpp#L45-L233)).
An invalid window ID or invalid outer JSON document returns a plain
`ERROR<TAB>message` instead of a JSON result array.

Batch only short synchronous commands whose inputs are already ready. A command
that starts asynchronous loading, registration, segmentation, tracking, a Hub
download, or a new window can return before that work completes; do not place a
dependent command after it in the same batch. Never send an empty batch.

Raw filename forwarding checks the global progress flag and path existence. It
returns `BUSY`, `OKAY`, or `ERROR`; `OKAY` means only that the path existed and
`openFile()` was called, not that the correct new window became ready
([raw open route](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/main.cpp#L583-L603)). After `OKAY`, poll `LIST` for
the expected window.

### Replies, fallback exit status, and completion

| Reply form | Executable fallback exit | Interpretation |
|---|---:|---|
| `OKAY` | 0 | Handler returned `true`; optional `PROMPT<TAB><JSON>` metadata and captured console text may follow. |
| `ERROR<TAB>message` | 1 | Handler returned `false`, target was invalid, or an exception was caught; optional prompt metadata may follow. |
| JSON result array | 0 only when every returned `okay` is `true` | Batch result; inspect every object, the final object's optional `prompt`, and the expected command count. |
| `BUSY` | 1 | Raw filename forwarding was refused because global progress was active. |
| `TIMEOUT` | 1 | Executable fallback only: no bytes arrived within five seconds. The GUI command may still be running. |
| `NO_INSTANCE` | 1 | Executable fallback only: `LIST` could not connect to a running server. A direct pipe client instead gets a connection timeout/exception. |

The server executes command handlers synchronously on the GUI thread. Long
synchronous work can outlive the executable fallback's five-second wait; the
direct pipe helper waits for server disconnect. Some handlers return after
starting a background task, and Hub handlers defer their final GUI/file action
to a zero-delay timer. Therefore:

1. Treat `ERROR` as failure unless the documented caveat says the command
   changed state before returning an error.
2. Treat `TIMEOUT` as **unknown**, never as permission to retry blindly.
3. Treat `OKAY` as dispatch acknowledgement unless the command is documented
   as synchronous.
4. Verify completion through `LIST`, `LOG`, a list/status command, and/or
   `Test-Path` plus a stable file size.

## Discovering and targeting windows

`LIST` returns:

```text
OKAY
main<TAB>1<TAB>DSI Studio ...
tracking<TAB>2<TAB>subject.fib.gz
image<TAB>3<TAB>image.nii.gz
```

The columns are `type`, `window_id`, and current window title. IDs are assigned
monotonically the first time a widget appears in `LIST`, persist for that
widget's lifetime, and are not intentionally reused
([`ai_request_list()`](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/mainwindow.cpp#L73-L99)). Refresh `LIST` after
opening or closing anything. Never cache an ID across application restarts.

```powershell
$listReply = Read-DsiTextReply (Invoke-Dsi @('LIST'))
$rows = $listReply.Text -split '\r?\n' | Select-Object -Skip 1 |
    ForEach-Object {
        $c = $_ -split "`t", 3
        [pscustomobject]@{ Type=$c[0]; Id=$c[1]; Title=$c[2] }
    }
$trackingId = ($rows | Where-Object Type -eq 'tracking' | Select-Object -First 1).Id
if (-not $trackingId) { throw 'No tracking window is ready' }
```

If several windows have the same type, match both type and an expected title.
If the title is insufficient, stop and ask; the protocol exposes no file path,
dirty-state flag, or creation timestamp.

## Console output and verification

During a command, `ai_request_command()` points `console.capture` at a temporary
buffer and appends captured text to the reply
([capture and reply](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/mainwindow.cpp#L127-L233)). `LOG` returns the
entire rolling history, not only text since the last call
([`ai_request_log()`](https://github.com/frankyeh/DSI-Studio/blob/ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58/mainwindow.cpp#L236-L245)). The history is
capped at 4 MiB by dropping its oldest half
([console history](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/console.cpp#L81-L114)).

Capture a baseline before asynchronous work, then compare later text:

```powershell
$beforeReply = Read-DsiTextReply (Invoke-Dsi @('LOG'))
$before = $beforeReply.Text
$ackReply = Read-DsiTextReply (
    Invoke-Dsi -Fields @('CMD',$trackingId,'run_auto_track','CST_L',''))
$ack = $ackReply.Text
Start-Sleep -Seconds 2
$afterReply = Read-DsiTextReply (Invoke-Dsi @('LOG'))
$after = $afterReply.Text
$delta = if ($after.StartsWith($before)) { $after.Substring($before.Length) } else { $after }
```

This comparison is best effort: history truncation and unrelated concurrent
output can break a simple prefix comparison. Console text emitted after a
handler returns is absent from that handler's immediate reply.

## Main-window commands

The main window accepts `list_recent`, `run_cli`, and the `hub` command family
([`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1901-L2013)).
Repository and tag arguments are exact strings returned by the preceding list
command. File filtering is case-insensitive substring matching.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_recent` | No parameters. | Recent `.sz` and `.fz` paths, one per line; no header. | Reads the stored source/FIB recent-file lists. Paths may no longer exist. **Completion:** Immediate. | Read-only | `Invoke-Dsi -Fields @("CMD",$mainId,"list_recent")` | `MainWindow::command`; [current implementation](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1914-L1922) |
| `run_cli` | One parameter containing the complete DSI Studio command line. `--action` is required. | CLI progress, warnings, and errors captured in the reply while the handler runs. | Parses internally and calls `run_action_with_wildcard()` synchronously on the GUI thread. Supports `rec`, `trk`, `src`, `ana`, `exp`, `atl`, `db`, `tmp`, `cnt`, `cnt_cl`, `vis`, `ren`, `qc`, `reg`, `atk`, `xnat`, and `img`. Explicit `--loop`, or a wildcard `--source` for supported actions, may process many files. **Completion:** Synchronous handler result; still verify outputs. | Varies; inspect the full CLI action, sources, wildcards, and outputs. Confirm destructive, overwrite, download, or unexpectedly large batch scope. | `Invoke-Dsi -Fields @("CMD",$mainId,"run_cli","--action=qc --source=E:\data\subject.fz")` | `MainWindow::command`; [`run_cli`](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1924-L1933), [action dispatch](https://github.com/frankyeh/DSI-Studio/blob/8ad955a333cdfcb8d6433a6a88137c0ae76f6cad/main.cpp#L347-L465) |
| `hub help` | Send command field `hub`, parameter `help`. | Usage line. | None. **Completion:** Immediate. | Read-only | `Invoke-Dsi -Fields @("CMD",$mainId,"hub","help")` | `MainWindow::command`; [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `hub repos` | Command `hub`; first parameter `repos`. | `index<TAB>repository` rows. | Initializes/selects the Hub tab. **Completion:** Immediate unless Hub initialization itself is still loading. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$mainId,"hub","repos")` | `MainWindow::command`; [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `hub tags` | `hub`, `tags`, exact repository. | `index<TAB>tag` rows; may print loading warning. | Selects repository. **Completion:** Immediate list; retry if output says loading. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$mainId,"hub","tags","owner/repository")` | `MainWindow::command`; [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `hub files` | `hub`, `files`, repository, tag, optional text filter. | `row<TAB>filename<TAB>display-size<TAB>cached`; `cached` is `0`/`1`. | Selects repository/tag. The cache flag tests the Hub temporary cache path. **Completion:** Immediate list; retry if Hub data is loading. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$mainId,"hub","files","owner/repository","tag","CST")` | `MainWindow::command`; [current implementation](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1980-L1991) |
| `hub open` | `hub`, `open`, repository, tag, exact filename. | Console messages only. | May download/cache, then open the file. **Completion:** Deferred: handler may schedule the open after `OKAY`; poll `LIST`. | File creation | `Invoke-Dsi -Fields @("CMD",$mainId,"hub","open","owner/repository","tag","file.fz")` | `MainWindow::command`; [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `hub download` | `hub`, `download`, repository, tag, exact filename, absolute directory. | Console messages only. | Downloads with overwrite disabled. **Completion:** Deferred file write; verify path and stable size. | File creation | `Invoke-Dsi -Fields @("CMD",$mainId,"hub","download","owner/repository","tag","file.fz","E:\data")` | `MainWindow::command`; [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |

`run_cli` is not a shell: DSI Studio parses the one parameter string itself.
Unknown or unused options can be reported as warnings only after the action
runs, so inspect the captured output as well as the reply status.

Hub readiness is not exposed as a state bit. An empty repository list causes
`ERROR` “Fiber Data Hub is not ready; retry.” Retry with bounded backoff. `hub
open` downloads synchronously when uncached but defers `git_open()`; `hub
download` defers the final `QFile::write()` to a zero-delay timer
([open path](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L2297-L2405), [download path](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L2193-L2260)). Never treat the first `OKAY` as file or
window readiness.

## Tracking-window commands

`tracking_window::command()` dispatches in this order: rendering/camera
(`GLWidget`), tract, region, device, then tracking-window commands. A handler
returning `not_processed` allows the next handler to try; another error stops
dispatch ([dispatch](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L118-L149)). All examples assume `$trackingId` was freshly resolved.

### Files, mapping, workspace, and settings

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `open_fib` | Absolute `.fib.gz`/`.fz` path. | Console messages. | Loads a FIB and opens another tracking window. **Completion:** Synchronous load; then refresh `LIST`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_fib","E:\data\subject.fz")` | `tracking_window::command`; [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `correct_bias_field` | No parameters. | Console/progress output. | Corrects bias in the loaded diffusion data. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"correct_bias_field")` | `tracking_window::command`; [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `save_fib_as` | Absolute output path; omission opens a dialog. | Console/error output. | Writes the current FIB. **Completion:** Synchronous; verify the output file. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_fib_as","E:\out\subject.fz")` | `tracking_window::command`; [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `open_mapping` | Absolute mapping path; omission opens a dialog. | Console/error output. | Loads template mapping. **Completion:** Synchronous file load. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_mapping","E:\data\subject.map.gz")` | `tracking_window::command`; [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `presentation_mode` | No parameters. | None. | Hides docking panels. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"presentation_mode")` | `tracking_window::command`; [workspace commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618) |
| `save_workspace` | Absolute directory. | Progress/error output. | Creates tract, region, device, slice, settings, camera, and command-history files. **Completion:** Synchronous and potentially large; verify directory contents. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_workspace","E:\workspaces\case01")` | `tracking_window::command`; [workspace commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618) |
| `load_workspace` | Absolute workspace directory. | Progress/error output. | Replaces current tract/device/region sets when matching workspace subdirectories exist. **Completion:** Synchronous file load. **Caveat:** Confirm immediately before running. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_workspace","E:\workspaces\case01")` | `tracking_window::command`; [workspace commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618) |
| `save_setting` | Absolute `.ini` output; omission opens a dialog. | Console/error output. | Writes all GUI settings. **Completion:** Synchronous; verify file. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_setting","E:\settings\save_setting.ini")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `save_rendering_setting` | Absolute `.ini` output; omission opens a dialog. | Console/error output. | Writes rendering settings. **Completion:** Synchronous; verify file. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_rendering_setting","E:\settings\save_rendering_setting.ini")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `save_tracking_setting` | Absolute `.ini` output; omission opens a dialog. | Console/error output. | Writes tracking settings. **Completion:** Synchronous; verify file. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tracking_setting","E:\settings\save_tracking_setting.ini")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `load_setting` | Absolute `.ini` input; omission opens a dialog. | Console/error output. | Applies all GUI settings. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_setting","E:\settings\load_setting.ini")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `load_rendering_setting` | Absolute `.ini` input; omission opens a dialog. | Console/error output. | Applies rendering settings. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_rendering_setting","E:\settings\load_rendering_setting.ini")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `load_tracking_setting` | Absolute `.ini` input; omission opens a dialog. | Console/error output. | Applies tracking settings. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_tracking_setting","E:\settings\load_tracking_setting.ini")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `restore_rendering` | No parameters. | None. | Restores default rendering settings. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"restore_rendering")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `restore_tracking` | No parameters. | None. | Restores default tracking settings. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"restore_tracking")` | `tracking_window::command`; [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |

### Slice commands

Slice indices come from `list_slice`; they are zero-based. Refresh the list
after adding or deleting custom slices. Several custom-slice operations can
start registration or lazy data loading, so verify the list and logs again
before downstream use.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_slice` | No parameters. | Header `index<TAB>current<TAB>name<TAB>ready<TAB>running<TAB>downloaded<TAB>registered`; flags are `0`/`1`. | Reports image readiness, active custom-slice work, local download presence, and custom-slice registration readiness. Built-in slices report downloaded/registered as true. **Completion:** Immediate snapshot. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_slice")` | `tracking_window::command`; [current implementation](https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4/tracking/tracking_window_action.cpp#L198-L221) |
| `set_slice` | Optional zero-based slice index; default current. | Console/error output. | Selects/loads the slice and may start registration for a custom slice. **Completion:** Selection is immediate; derived data may remain asynchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_slice","2")` | `tracking_window::command`; [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `set_slice_by_name` | Exact slice name. | Error if not found. | Selects the named slice. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_slice_by_name","qa")` | `tracking_window::command`; [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `enable_slice` | One field: `sagittal coronal axial`, each `0`/`1`; default current values. | None. | Changes slice-plane visibility. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"enable_slice","1 0 1")` | `tracking_window::command`; [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `move_slice` | One field: `x y z` voxel indices; default current positions. | None. | Moves all three slice positions. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"move_slice","80 90 60")` | `tracking_window::command`; [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `set_roi_view` | `0` sagittal, `1` coronal, or `2` axial. | None. | Changes the 2-D ROI view. **Completion:** Immediate; an invalid integer silently changes nothing. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_roi_view","2")` | `tracking_window::command`; [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `set_slice_contrast` | Field 1: `minimum maximum`; field 2: `minColor maxColor` as packed Qt color integers. | None. | Updates slice intensity range/colors. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_slice_contrast","0 1","4278190080 4294967295")` | `tracking_window::command`; [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `set_slice_dir_color` | Field 1: slice index (default current); field 2: `0`/`1` (default current setting). | Error `canceled` when no valid change occurs. | Changes directional-color display. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_slice_dir_color","2","1")` | `tracking_window::command`; [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `set_slice_overlay` | Field 1: slice index (default current); field 2: `0`/`1` (default current setting). | Error `canceled` when no valid change occurs. | Changes slice overlay. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_slice_overlay","2","1")` | `tracking_window::command`; [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `set_slice_stay` | Field 1: slice index (default current); field 2: `0`/`1` (default current setting). | Error `canceled` when no valid change occurs. | Changes slice persistence. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_slice_stay","2","1")` | `tracking_window::command`; [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `save_slice_image` | Field 1: absolute output; field 2: exact slice/metric name. | Console/error output. | Writes native-space image data. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_slice_image","E:\out\save_slice_image.nii.gz","qa")` | `tracking_window::command`; [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `save_slice_mni_image` | Field 1: absolute output; field 2: exact slice/metric name. | Console/error output. | Writes MNI-space image data. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_slice_mni_image","E:\out\save_slice_mni_image.nii.gz","qa")` | `tracking_window::command`; [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `save_roi_screen` | Absolute image output; omission opens a dialog. | Console/error output. | Writes the current 2-D ROI view. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_roi_screen","E:\out\roi.png")` | `tracking_window::command`; [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `add_slice` | Absolute path, or comma-joined paths accepted by the custom-slice loader. | Console/error output. | Adds a native/custom slice. **Completion:** Load may start asynchronous registration; poll `list_slice` and `LOG`. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_slice","E:\data\t1.nii.gz")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `add_mni_slice` | Absolute path, or comma-joined paths accepted by the custom-slice loader. | Console/error output. | Adds a MNI-space slice. **Completion:** Load may start asynchronous registration; poll `list_slice` and `LOG`. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_mni_slice","E:\data\t1.nii.gz")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `skull_strip_slice` | Optional custom-slice index; default current. | Progress/error output. | Runs skull stripping on a custom slice. **Completion:** Synchronous computation; may time out. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"skull_strip_slice","2")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `save_slice_mapping` | Field 1: absolute path; field 2: optional custom-slice index (default current). | Console/error output. | Writes a custom slice mapping. **Completion:** Synchronous; verify file for save commands. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_slice_mapping","E:\out\save_slice_mapping.nii.gz","2")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `open_slice_mapping` | Field 1: absolute path; field 2: optional custom-slice index (default current). | Console/error output. | Loads a custom slice mapping. **Completion:** Synchronous; verify file for save commands. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_slice_mapping","E:\out\open_slice_mapping.nii.gz","2")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `save_slice_volume` | Field 1: absolute path; field 2: optional custom-slice index (default current). | Console/error output. | Writes the custom slice volume. **Completion:** Synchronous; verify file for save commands. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_slice_volume","E:\out\save_slice_volume.nii.gz","2")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `delete_slice` | Optional custom-slice index; default current. | None/error. | Deletes one custom slice. **Completion:** Immediate. **Caveat:** The handler does not bounds-check before indexing; use a fresh valid index from `list_slice`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_slice","2")` | `tracking_window::command`; [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |

### UNet segmentation

`list_unet` is the required discovery step. Its columns are:

- `index`: current zero-based action order;
- `available`: `1` when the action is enabled for the current slice and `0`
  otherwise;
- `model`: the exact model identifier accepted by `segment_brain`;
- `name`: display name;
- `description`: model description.

Availability is computed from current-slice and model-name tokens (`t1`/`mpr`,
`t2`/`tse`, and `flair`/`t2f`) in
[UNet eligibility](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1740-L1806). A listed-but-disabled model must not be invoked. The model is cached under
the application data directory and may be downloaded before inference
([model download](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/SliceModel.cpp#L94-L107)). **TumorSynth remains temporarily unavailable regardless of this list.**

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_unet` | No parameters. | Header `index<TAB>available<TAB>model<TAB>name<TAB>description`. | Refreshes model availability. **Completion:** Immediate after model-menu refresh. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_unet")` | `tracking_window::command`; [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `segment_brain` | Exact available `model` or `name` from `list_unet`; omission opens a dialog. | Progress, label, and error output. | May download a model, run inference, and create one region per non-background label. **Completion:** Synchronous computation and download; likely to exceed five seconds. Verify with `list_region`. **Caveat:** Do not use TumorSynth. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"segment_brain","<model-from-list_unet>")` | `tracking_window::command`; [`segment_brain`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L315-L437) |

### Atlas commands

`list_atlas` returns `template`, `atlas`, `name`, and number of `regions`.
`add_region_from_atlas` requires numeric label IDs, but the current remote
surface does **not** list the atlas label IDs or names. Do not guess them.
Obtain them through an existing trusted mapping or wait for the recommended
`list_atlas_label` command.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_atlas` | No parameters. | Header `template<TAB>atlas<TAB>name<TAB>regions`. | May lazily retrieve atlas objects. **Completion:** Immediate list. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_atlas")` | `tracking_window::command`; [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `add_region_from_atlas` | One field: `template_id atlas_id label_id&label_id...`; labels are optional and default to all. | Console/errors; created rows appear in `list_region`. | Switches template and creates regions from atlas labels. **Completion:** Synchronous extraction; verify `list_region`. **Caveat:** There is no current label-discovery command; never invent label IDs. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_region_from_atlas","0 1 18&19")` | `RegionTableWidget::command`; [`add_region_from_atlas`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L785-L846) |

### Automatic fiber tracking

`list_auto_tract` is the authoritative source of tract names. `run_auto_track`
returns after starting `ThreadData`; it does not report final success. Use
`list_tract` immediately to identify the new row, then poll its counts and
`running` flag and `LOG`. A transition from `running=1` to `running=0` shows
that the thread ended, but does not distinguish success, failure, or
cancellation.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `enable_auto_tract` | No parameters. | Console/error output. | Loads the symmetric tract atlas and enables auto-track UI. **Completion:** Synchronous atlas load. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"enable_auto_tract")` | `tracking_window::command`; [automatic tracking](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776) |
| `list_auto_tract` | No parameters. | Header `name`, then exact accepted tract names. | Loads tract atlas if necessary. **Completion:** Synchronous list. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_auto_tract")` | `tracking_window::command`; [automatic tracking](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776) |
| `run_auto_track` | Field 1: exact tract name from `list_auto_tract`; field 2: optional ROI setting appended to the current tracking parameter. | Immediate start/error output; progress later appears in `LOG`/`list_tract`. | Creates a tract row and starts background tracking with tolerance. **Completion:** Asynchronous; `OKAY` means started only. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"run_auto_track","CST_L","18:0")` | `tracking_window::command`; [automatic tracking](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776) |
| `run_tracking` | Field 1: new tract name. If field 2 is absent, current GUI tracking settings are used. If field 2 contains `:`, it is ROI grammar appended to current settings; otherwise it is an explicit opaque parameter ID optionally followed by a space and ROI grammar. Field 3 is optional auto-track tolerance. | Immediate start/error output. | Creates a tract row and starts background tracking. **Completion:** Asynchronous; poll `list_tract` until `running=0` and inspect `LOG`. **Caveat:** Prefer the short current-settings form; do not hand-edit opaque parameter IDs. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"run_tracking","thalamic_fibers","18:0&21:1")` | `TractTableWidget::command`; [current-settings expansion](https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L451-L460), [tracking start](https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L575-L605) |

The ROI grammar used by tracking is:

```text
<region-index>:<type>&<region-index>:<type>...
```

Both numbers are zero-based integers. Region type values are defined by
`ROIRegion::RegionType`: `0=ROI`, `1=ROA`, `2=End`, `3=Seed`,
`4=Terminative`, `5=NotEnd`, `6=Limiting`
([ROI type enum](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.h#L14-L21)). The parser rejects region indices outside the current region table and
types greater than `6` ([ROI grammar parser](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1239-L1272)). Therefore `18:0` means exactly: **use region row
18 as an inclusion ROI**.

### Surfaces

Surface commands take an optional slice index (field 1, default current) and an
optional threshold (field 2). If threshold is absent, a modal dialog opens.
Always supply it for unattended control. Directional suffixes crop the
generated surface to the named half-space combination.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `add_surface` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_right` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_right","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_left` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_left","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_upper` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_upper","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_anterior` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_anterior","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_posterior` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_posterior","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_right_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_right_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_left_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_left_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_right_anterior_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_right_anterior_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_left_anterior_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_left_anterior_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_left_posterior_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_left_posterior_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_anterior_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_anterior_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `add_surface_posterior_lower` | Field 1: optional slice index; field 2: threshold (supply it to avoid a dialog). | Progress/error output. | Extracts and adds a rendered isosurface, with cropping encoded by the suffix. **Completion:** Synchronous computation; may exceed client timeout. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"add_surface_posterior_lower","0","0.5")` | `tracking_window::command`; [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |

### Generic tracking and rendering parameters

`list_param` is misleadingly named: it does not enumerate parameters. It
requires one exact parameter ID and prints `name: value`. `set_param` accepts
the ID and a textual value. `set_params` accepts one
`name=value&name=value...` field and applies all entries before requesting one
GL and slice redraw ([parameter commands](https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L902-L922)). Enum parameters use their **zero-based numeric choice
index**, not choice text. Checkbox values are Qt states (`0` unchecked, `2`
checked). Colors are packed Qt ARGB integers.

The UI normally constrains values, but remote `set_param` does not reproduce all
widget-side validation. Stay within the table's range. A redraw is requested
for every parameter; tract/color changes that rely on cached geometry may also
require `update_tract`. Read back with `list_param`, take a screen if visual
state matters, and restore the old value when the result is wrong.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_param` | Exact parameter ID. | One line: `name: value`. | None. **Completion:** Immediate. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_param","tract_style")` | `tracking_window::command`; [parameter commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L889-L901) |
| `set_param` | Field 1: exact parameter ID; field 2: value. | None/error. | Changes one rendering/tracking value and requests redraw. **Completion:** Immediate state mutation. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_param","tract_style","1")` | `tracking_window::command`; [current implementation](https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L908-L922) |
| `set_params` | One field: `name=value&name=value...`. | None/error. Malformed fragments without `=` are ignored. | Changes several rendering/tracking values and requests one redraw. **Completion:** Immediate state mutation. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_params","tract_style=1&tract_alpha=0.8")` | `tracking_window::command`; [current implementation](https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L908-L922) |

#### Complete parameter schema

| Parameter | Type | Valid UI range / choices | Default | Meaning | Redraw | Source |
|---|---|---|---|---|---|---|
| `show_slice` | checkbox | `0` or `2` | UI-dependent | Show slice planes. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `show_tract` | checkbox | `0` or `2` | UI-dependent | Show tracts. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `show_region` | checkbox | `0` or `2` | UI-dependent | Show regions. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `show_surface` | checkbox | `0` or `2` | UI-dependent | Show surfaces. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `show_device` | checkbox | `0` or `2` | UI-dependent | Show devices. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `show_label` | checkbox | `0` or `2` | UI-dependent | Show labels. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `show_odf` | checkbox | `0` or `2` | UI-dependent | Show ODFs. | Automatic GL + slice request. | [top-level visibility parameters](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/renderingtablewidget.cpp#L262-L268) |
| `orientation_convention` | enum index | `0`=Radiology; `1`=Neurology | `0` | Radiology views from foot, whereas Neurology views from top | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:1`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L1-L1) |
| `roi_zoom` | float | `0.2..40`, UI step `0.5` | `5.0` | Zoom in or zoom out | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:2`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L2-L2) |
| `roi_draw_edge` | enum index | `0`=Off; `1`=On | `0` | Draw edge of the region | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:3`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L3-L3) |
| `roi_composition` | enum index | `0`=SourceAtop; `1`=DestinationAtop; `2`=Xor; `3`=Plus; `4`=Multiply; `5`=Screen; `6`=Overlay; `7`=Darken; `8`=Lighten; `9`=ColorDodge; `10`=ColorBun; `11`=HardLight; `12`=SoftLight; `13`=Difference; `14`=Exclusion | `0` | The composition mode for drawing region | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:4`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L4-L4) |
| `roi_opacity` | float | `0..1`, UI step `0.1` | `1` | Opacity | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:5`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L5-L5) |
| `roi_edge_width` | int | `1..5`, UI step `1` | `1` | Line width for edge | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:6`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L6-L6) |
| `roi_track` | enum index | `0`=Off; `1`=On | `1` | Show tracts | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:7`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L7-L7) |
| `roi_track_count` | int | `1000..500000`, UI step `1000` | `5000` | Visible track count | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:8`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L8-L8) |
| `roi_fiber` | enum index | `0`=Off; `1`=On; `2`=1st; `3`=2nd | `1` | Fiber Direction | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:9`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L9-L9) |
| `roi_fiber_color` | enum index | `0`=RGB; `1`=red; `2`=green; `3`=blue | `0` | Fiber Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:10`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L10-L10) |
| `roi_fiber_width` | float | `0.1..1`, UI step `0.1` | `0.2` | Fiber Width | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:11`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L11-L11) |
| `roi_fiber_length` | float | `0.1..4`, UI step `0.1` | `2.0` | Fiber Length | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:12`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L12-L12) |
| `roi_fiber_antialiasing` | enum index | `0`=Off; `1`=On | `0` | Antialiasing | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:13`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L13-L13) |
| `roi_label` | enum index | `0`=Off; `1`=On | `1` | "R" label | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:14`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L14-L14) |
| `roi_position` | enum index | `0`=Off; `1`=On | `1` | Position Line | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:15`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L15-L15) |
| `roi_ruler` | enum index | `0`=Off; `1`=On | `1` | show ruler | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:16`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L16-L16) |
| `roi_tic` | int | `1..8`, UI step `1` | `2` | Tic distance on ruler | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:17`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L17-L17) |
| `roi_layout` | enum index | `0`=Single Slice; `1`=3 Slices; `2`=Mosaic; `3`=Mosaic 2; `4`=Mosaic 3; `5`=Mosaic 4; `6`=Mosaic 5; `7`=Mosaic 6; `8`=Mosaic 7; `9`=Mosaic 8; `10`=Mosaic 9; `11`=Mosaic 10 | `0` | Slice Layout | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:18`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L18-L18) |
| `roi_mosaic_column` | int | `0..30`, UI step `5` | `0` | Column count for the mosaic view (0 for default square number) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:19`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L19-L19) |
| `roi_mosaic_skip_row` | int | `0..10`, UI step `1` | `1` | Remove first and last row from the mosaic view | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:20`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L20-L20) |
| `roi_format` | enum index | `0`=nii.gz; `1`=mat; `2`=txt | `0` | Default Output Format | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:21`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L21-L21) |
| `tracking_index` | enum index | `0`=fa; `1`=adc | `0` | The anisotropy metrics that will be used as the termination criterion | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:22`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L22-L22) |
| `fa_threshold` | float | `0..2`, UI step `0.01` | `0.0` | The anisotropy threshold to terminate tracking | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:23`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L23-L23) |
| `turning_angle` | int | `0..90`, UI step `5` | `0` | The angular threshold to terminate tracking | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:24`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L24-L24) |
| `step_size` | float | `0.00..10`, UI step `0.1` | `0` | The propagation distance for each tracking iteration | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:25`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L25-L25) |
| `min_length` | float | `0..800`, UI step `10` | `30` | Remove tracks with length shorter than this threshold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:26`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L26-L26) |
| `max_length` | float | `0..10000`, UI step `10` | `300` | Remove tracks with length longer than this threshold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:27`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L27-L27) |
| `max_seed_count` | int | `0..100000000`, UI step `1000` | `0` | Specify the maximum number of seeds | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:28`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L28-L28) |
| `max_tract_count` | int | `0..100000000`, UI step `1000` | `0` | Specify the maximum number of tracks | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:29`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L29-L29) |
| `track_voxel_ratio` | float | `0..2`, UI step `0.005` | `1.0` | Specify the maximum tracks to voxel ratio | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:30`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L30-L30) |
| `tip_iteration` | int | `0..100`, UI step `2` | `4` | The number of pruning iterations used to remove noisy tracks | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:31`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L31-L31) |
| `tolerance` | float | `0..100`, UI step `10` | `22` | The inclusion distance for automated fiber tracking | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:32`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L32-L32) |
| `dt_index1` | enum index | `0`=none; `1`=adc | `0` | The baseline metrics | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:34`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L34-L34) |
| `dt_index2` | enum index | `0`=none; `1`=adc | `0` | The followup metrics | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:35`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L35-L35) |
| `dt_threshold_type` | enum index | `0`=(m1-m2)÷m1; `1`=(m1-m2)÷m2; `2`=m1-m2; `3`=(m2-m1)÷m1; `4`=(m2-m1)÷m2; `5`=m2-m1; `6`=m1÷max(m1); `7`=m2÷max(m2) | `0` | Type | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:36`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L36-L36) |
| `dt_threshold` | float | `0.0..2.0`, UI step `0.05` | `0.2` | 0.05 means tracking differences > 5% | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:37`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L37-L37) |
| `tracking_method` | enum index | `0`=Euler; `1`=RK4; `2`=Voxel tracking | `0` | Tracking Algorithm | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:39`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L39-L39) |
| `smoothing` | float | `-1.5..1`, UI step `0.1` | `0` | Smoothing (1=random) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:40`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L40-L40) |
| `check_ending` | enum index | `0`=Off; `1`=On | `0` | Check Ending | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:41`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L41-L41) |
| `otsu_threshold` | float | `0.1..1`, UI step `0.1` | `0.6` | Default Otsu | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:42`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L42-L42) |
| `track_format` | enum index | `0`=tt.gz; `1`=trk.gz; `2`=txt | `0` | Output Format | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:43`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L43-L43) |
| `scale_voxel` | enum index | `0`=Off; `1`=On | `1` | Scale with voxel size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:44`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L44-L44) |
| `perspective` | integer slider | `0..10` | `5` | Perspective | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:45`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L45-L45) |
| `3d_perspective` | float | `0.5..3`, UI step `0.5` | `1.0` | 3D Perspective | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:46`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L46-L46) |
| `bkg_color` | color | packed Qt ARGB integer | `-1` | Background Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:47`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L47-L47) |
| `anti_aliasing` | enum index | `0`=Off; `1`=On | `1` | Anti-aliasing | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:48`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L48-L48) |
| `line_smooth` | enum index | `0`=Off; `1`=On | `0` | Line Smooth | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:49`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L49-L49) |
| `point_smooth` | enum index | `0`=Off; `1`=On | `0` | Point Smooth | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:50`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L50-L50) |
| `poly_smooth` | enum index | `0`=Off; `1`=On | `0` | Polygon Smooth | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:51`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L51-L51) |
| `stereoscopy_angle` | float | `0.0..5.0`, UI step `0.2` | `1` | Stereoview Angle | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:52`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L52-L52) |
| `slice_alpha` | float | `0..1`, UI step `0.1` | `1` | Opacity | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:53`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L53-L53) |
| `slice_mag_filter` | enum index | `0`=NEAREST; `1`=LINEAR | `1` | Mag Filter | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:54`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L54-L54) |
| `slice_smoothing` | enum index | `0`=Off; `1`=On | `0` | Smoothing | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:55`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L55-L55) |
| `slice_match_bkcolor` | enum index | `0`=Off; `1`=On | `0` | Modify slice background color and match it with that of the 3D background | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:56`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L56-L56) |
| `slice_bend1` | enum index | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `2` | Blend Func1 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:57`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L57-L57) |
| `slice_bend2` | enum index | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` | Blend Func2 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:58`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L58-L58) |
| `tract_alpha` | float | `0..1`, UI step `0.1` | `1` | Opacity | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:60`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L60-L60) |
| `tract_color_saturation` | float | `0..1`, UI step `0.1` | `0.7` | Saturation | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:61`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L61-L61) |
| `tract_color_brightness` | float | `0..1`, UI step `0.1` | `0.5` | Brightness | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:62`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L62-L62) |
| `tract_color_style` | enum index | `0`=Directional; `1`=Assigned; `2`=Local Metrics; `3`=Averaged Metrics; `4`=Max Metrics; `5`=Loaded Value | `0` | Style | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:63`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L63-L63) |
| `tract_color_metrics` | enum index | `0`=qa; `1`=iso | `0` | Metrics | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:64`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L64-L64) |
| `tract_color_max_value` | float | `0..1`, UI step `0.1` | `1.0` | Max Value | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:65`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L65-L65) |
| `tract_color_min_value` | float | `0..1`, UI step `0.1` | `0.0` | Min Value | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:66`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L66-L66) |
| `tract_color_map` | enum index | `0`=assigned; `1`=files | `0` | Map | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:67`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L67-L67) |
| `tract_color_max` | color | packed Qt ARGB integer | `12079178` | Max Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:68`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L68-L68) |
| `tract_color_min` | color | packed Qt ARGB integer | `14465098` | Min Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:69`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L69-L69) |
| `tract_show_color_bar` | enum index | `0`=Off; `1`=On | `1` | Show Color Bar | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:70`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L70-L70) |
| `tract_style` | enum index | `0`=Line; `1`=Tube; `2`=End; `3`=End1; `4`=End2 | `1` | Style | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:71`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L71-L71) |
| `tract_line_width` | float | `1.0..10`, UI step `0.5` | `3` | Line Width | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:72`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L72-L72) |
| `tract_visible_tract` | int | `5000..1000000`, UI step `5000` | `25000` | Visible Tracts | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:73`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L73-L73) |
| `tract_shader` | int | `0..20`, UI step `1` | `4` | Shade | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:74`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L74-L74) |
| `tract_tube_detail` | enum index | `0`=Coarse; `1`=Fine; `2`=Finer; `3`=Finest | `1` | Tube Detail | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:75`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L75-L75) |
| `tube_diameter` | float | `0.01..5`, UI step `0.1` | `0.2` | Tube Diameter (voxel) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:76`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L76-L76) |
| `end_point_shift` | int | `0..10`, UI step `1` | `0` | Endpoint Shift (voxel) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:77`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L77-L77) |
| `tract_light_option` | enum index | `0`=One source; `1`=Two sources; `2`=Off | `1` | Light | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:78`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L78-L78) |
| `tract_light_dir` | integer slider | `0..10` | `2` | Light Direction | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:79`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L79-L79) |
| `tract_light_shading` | integer slider | `0..10` | `10` | Light Shading | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:80`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L80-L80) |
| `tract_light_diffuse` | integer slider | `0..10` | `10` | Light Diffuse | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:81`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L81-L81) |
| `tract_light_ambient` | integer slider | `0..10` | `0` | Light Ambient | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:82`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L82-L82) |
| `tract_light_specular` | integer slider | `0..10` | `0` | Light Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:83`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L83-L83) |
| `tract_specular` | integer slider | `0..10` | `0` | Material Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:84`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L84-L84) |
| `tract_emission` | integer slider | `0..10` | `0` | Material Emission | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:85`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L85-L85) |
| `tract_shininess` | integer slider | `0..10` | `0` | Material Shininess | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:86`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L86-L86) |
| `tract_sel_angle` | int | `0..90`, UI step `5` | `45` | Tract Selection Angle | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:87`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L87-L87) |
| `tract_bend1` | enum index | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `6` | Blend Func1 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:88`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L88-L88) |
| `tract_bend2` | enum index | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `2` | Blend Func2 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:89`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L89-L89) |
| `region_alpha` | float | `0..1`, UI step `0.1` | `0.8` | Opacity | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:91`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L91-L91) |
| `region_color_style` | enum index | `0`=Assigned; `1`=Metrics | `0` | Style | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:92`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L92-L92) |
| `region_color_metrics` | enum index | `0`=qa; `1`=iso | `0` | Metrics | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:93`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L93-L93) |
| `region_color_max_value` | float | `0..1`, UI step `0.1` | `1.0` | Max Value | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:94`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L94-L94) |
| `region_color_min_value` | float | `0..1`, UI step `0.1` | `0.0` | Min Value | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:95`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L95-L95) |
| `region_color_map` | enum index | `0`=assigned; `1`=files | `0` | Map | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:96`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L96-L96) |
| `region_color_max` | color | packed Qt ARGB integer | `12079178` | Max Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:97`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L97-L97) |
| `region_color_min` | color | packed Qt ARGB integer | `14465098` | Min Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:98`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L98-L98) |
| `region_show_color_bar` | enum index | `0`=Off; `1`=On | `1` | Show Color Bar | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:99`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L99-L99) |
| `region_graph` | enum index | `0`=Off; `1`=On | `0` | Graph | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:101`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L101-L101) |
| `region_node_size` | integer slider | `0..10` | `4` | Node Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:102`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L102-L102) |
| `region_constant_node_size` | enum index | `0`=Off; `1`=On | `0` | Constant Node Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:103`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L103-L103) |
| `region_hide_unconnected_node` | enum index | `0`=Off; `1`=On | `1` | Hide Unconnected Node | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:104`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L104-L104) |
| `region_edge_size` | integer slider | `0..10` | `4` | Edge Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:105`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L105-L105) |
| `region_constant_edge_size` | enum index | `0`=Off; `1`=On | `0` | Constant Edge Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:106`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L106-L106) |
| `region_pos_edge_color1` | color | packed Qt ARGB integer | `-1` | Edge Min Color(Positive) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:107`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L107-L107) |
| `region_pos_edge_color2` | color | packed Qt ARGB integer | `8224255` | Edge Max Color(Positive) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:108`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L108-L108) |
| `region_neg_edge_color1` | color | packed Qt ARGB integer | `-1` | Edge Min Color(Negative) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:109`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L109-L109) |
| `region_neg_edge_color2` | color | packed Qt ARGB integer | `16743293` | Edge Max Color(Negative) | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:110`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L110-L110) |
| `region_edge_threshold` | float | `0.00..1`, UI step `0.1` | `0.1` | Binary Graph Threshold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:111`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L111-L111) |
| `region_mesh_smoothed` | enum index | `0`=Original; `1`=Smoothed; `2`=Smoothed2 | `1` | Mesh Rendering | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:112`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L112-L112) |
| `region_light_option` | enum index | `0`=One source; `1`=Two sources; `2`=Three sources | `0` | Light | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:113`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L113-L113) |
| `region_light_dir` | integer slider | `0..10` | `2` | Light Direction | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:114`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L114-L114) |
| `region_light_shading` | integer slider | `0..10` | `2` | Light Shading | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:115`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L115-L115) |
| `region_light_diffuse` | integer slider | `0..10` | `10` | Light Diffuse | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:116`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L116-L116) |
| `region_light_ambient` | integer slider | `0..10` | `0` | Light Ambient | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:117`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L117-L117) |
| `region_light_specular` | integer slider | `0..10` | `0` | Light Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:118`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L118-L118) |
| `region_specular` | integer slider | `0..10` | `0` | Material Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:119`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L119-L119) |
| `region_emission` | integer slider | `0..10` | `1` | Material Emission | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:120`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L120-L120) |
| `region_shininess` | integer slider | `0..10` | `0` | Material Shininess | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:121`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L121-L121) |
| `region_bend1` | enum index | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `4` | Blend Func1 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:122`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L122-L122) |
| `region_bend2` | enum index | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` | Blend Func2 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:123`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L123-L123) |
| `surface_color` | color | packed Qt ARGB integer | `11184810` | Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:124`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L124-L124) |
| `surface_alpha` | float | `0..1`, UI step `0.05` | `0.2` | Opacity | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:125`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L125-L125) |
| `surface_mesh_smoothed` | enum index | `0`=Original; `1`=Smoothed; `2`=Smoothed2 | `2` | Mesh Rendering | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:126`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L126-L126) |
| `surface_light_option` | enum index | `0`=One source; `1`=Two sources; `2`=Three sources | `2` | Light | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:127`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L127-L127) |
| `surface_light_dir` | integer slider | `0..10` | `5` | Light Direction | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:128`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L128-L128) |
| `surface_light_shading` | integer slider | `0..10` | `4` | Light Shading | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:129`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L129-L129) |
| `surface_light_diffuse` | integer slider | `0..10` | `2` | Light Diffuse | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:130`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L130-L130) |
| `surface_light_ambient` | integer slider | `0..10` | `0` | Light Ambient | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:131`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L131-L131) |
| `surface_light_specular` | integer slider | `0..10` | `0` | Light Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:132`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L132-L132) |
| `surface_specular` | integer slider | `0..10` | `0` | Material Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:133`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L133-L133) |
| `surface_emission` | integer slider | `0..10` | `0` | Material Emission | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:134`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L134-L134) |
| `surface_shininess` | integer slider | `0..10` | `0` | Material Shininess | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:135`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L135-L135) |
| `surface_bend1` | enum index | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `2` | Blend Func1 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:136`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L136-L136) |
| `surface_bend2` | enum index | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` | Blend Func2 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:137`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L137-L137) |
| `device_light_option` | enum index | `0`=One source; `1`=Two sources; `2`=Three sources | `2` | Light | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:138`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L138-L138) |
| `device_light_dir` | integer slider | `0..10` | `5` | Light Direction | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:139`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L139-L139) |
| `device_light_shading` | integer slider | `0..10` | `4` | Light Shading | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:140`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L140-L140) |
| `device_light_diffuse` | integer slider | `0..10` | `6` | Light Diffuse | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:141`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L141-L141) |
| `device_light_ambient` | integer slider | `0..10` | `0` | Light Ambient | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:142`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L142-L142) |
| `device_light_specular` | integer slider | `0..10` | `0` | Light Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:143`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L143-L143) |
| `device_specular` | integer slider | `0..10` | `0` | Material Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:144`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L144-L144) |
| `device_emission` | integer slider | `0..10` | `0` | Material Emission | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:145`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L145-L145) |
| `device_shininess` | integer slider | `0..10` | `0` | Material Shininess | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:146`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L146-L146) |
| `device_bend1` | enum index | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `4` | Blend Func1 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:147`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L147-L147) |
| `device_bend2` | enum index | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` | Blend Func2 | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:148`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L148-L148) |
| `show_track_label` | enum index | `0`=Off; `1`=On | `1` | Show Track Label | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:149`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L149-L149) |
| `show_track_label_location` | enum index | `0`=With Track; `1`=Middle; `2`=Middle Bottom | `0` | Track Label Location | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:150`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L150-L150) |
| `track_label_color` | color | packed Qt ARGB integer | `9868955` | Track Label Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:151`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L151-L151) |
| `track_label_bold` | enum index | `0`=Off; `1`=On | `1` | Track Label Bold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:152`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L152-L152) |
| `track_label_size` | int | `2..100`, UI step `2` | `12` | Tract Label Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:153`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L153-L153) |
| `show_region_label` | enum index | `0`=Off; `1`=On | `1` | Show Region Label | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:154`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L154-L154) |
| `region_label_color` | color | packed Qt ARGB integer | `9868955` | Region Label Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:155`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L155-L155) |
| `region_label_bold` | enum index | `0`=Off; `1`=On | `1` | Region Label Bold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:156`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L156-L156) |
| `region_label_size` | int | `2..100`, UI step `2` | `12` | Region Label Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:157`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L157-L157) |
| `show_device_label` | enum index | `0`=Off; `1`=On | `1` | Show Device Label | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:158`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L158-L158) |
| `device_label_color` | color | packed Qt ARGB integer | `9868955` | Device Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:159`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L159-L159) |
| `device_label_bold` | enum index | `0`=Off; `1`=On | `1` | Device Bold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:160`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L160-L160) |
| `device_label_size` | int | `2..100`, UI step `2` | `12` | Device Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:161`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L161-L161) |
| `show_directional_axis` | enum index | `0`=Off; `1`=On | `0` | Axis | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:162`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L162-L162) |
| `axis_line_thickness` | float | `1.0..20.0`, UI step `0.5` | `10` | Axis Line Thickness | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:163`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L163-L163) |
| `axis_line_length` | float | `1.0..10.0`, UI step `0.5` | `5` | Axis Line Length | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:164`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L164-L164) |
| `show_axis_label` | enum index | `0`=Off; `1`=On | `1` | Axis Label | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:165`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L165-L165) |
| `axis_label_size` | int | `2..48`, UI step `2` | `26` | Axis Label Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:166`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L166-L166) |
| `axis_label_bold` | enum index | `0`=Off; `1`=On | `1` | Axis Label Bold | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:167`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L167-L167) |
| `odf_position` | enum index | `0`=Along Slide; `1`=Slide Intersection; `2`=All | `0` | Position | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:168`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L168-L168) |
| `odf_scale` | float | `0.1..32`, UI step `1` | `2` | Size | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:169`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L169-L169) |
| `odf_color` | enum index | `0`=Dir; `1`=Blue; `2`=Red | `0` | Color | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:170`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L170-L170) |
| `odf_skip` | enum index | `0`=none; `1`=2; `2`=4 | `0` | Interleaved | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:171`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L171-L171) |
| `odf_smoothing` | enum index | `0`=off; `1`=on | `0` | Smoothing | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:172`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L172-L172) |
| `odf_shape` | enum index | `0`=original; `1`=1st; `2`=2nd | `0` | Shape | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:173`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L173-L173) |
| `odf_min_max` | enum index | `0`=off; `1`=on | `1` | Min-Max Normalization | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:174`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L174-L174) |
| `odf_light_option` | enum index | `0`=One source; `1`=Two sources; `2`=Three sources | `0` | Light | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:175`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L175-L175) |
| `odf_light_dir` | integer slider | `0..10` | `2` | Light Direction | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:176`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L176-L176) |
| `odf_light_shading` | integer slider | `0..10` | `2` | Light Shading | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:177`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L177-L177) |
| `odf_light_diffuse` | integer slider | `0..10` | `10` | Light Diffuse | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:178`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L178-L178) |
| `odf_light_ambient` | integer slider | `0..10` | `0` | Light Ambient | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:179`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L179-L179) |
| `odf_light_specular` | integer slider | `0..10` | `0` | Light Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:180`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L180-L180) |
| `odf_specular` | integer slider | `0..10` | `0` | Material Specular | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:181`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L181-L181) |
| `odf_emission` | integer slider | `0..10` | `1` | Material Emission | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:182`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L182-L182) |
| `odf_shininess` | integer slider | `0..10` | `0` | Material Shininess | Automatic GL + slice request; cached tract geometry may need `update_tract`. | [`options.txt:183`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/options.txt#L183-L183) |

## Region commands

`list_region` is the authoritative index map. Its columns are `index`, `shown`,
`name`, `type`, `color`, `dimension`, and `resolution`. `shown` is the checkbox
state (`0`/`1`); color is `#AARRGGBB`. Refresh it after every creation,
deletion, merge, separation, sort, or workspace load.

Commands whose index field is empty generally use the current row. Commands
whose semantics say “checked” operate on checked rows. Never assume an old
index still identifies the same region.

### Creation and discovery

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_region` | No parameters. | Header and rows: `index shown name type color dimension resolution` (tab-separated). | None. **Completion:** Immediate. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_region")` | `RegionTableWidget::command`; [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `new_region` | No parameters. | New row appears. | Creates an empty region. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"new_region")` | `RegionTableWidget::command`; [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `new_region_whole_brain_seed` | Optional Otsu ratio; default current tracking Otsu value. | New row/progress. | Creates a whole-brain seed region from anisotropy thresholding. **Completion:** Synchronous computation. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"new_region_whole_brain_seed","0.6")` | `RegionTableWidget::command`; [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `new_region_from_threshold` | Threshold; negative requests low-pass behavior. | New row/progress. | Creates an empty row, then fills it from current-slice thresholding. **Completion:** Synchronous computation. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"new_region_from_threshold","0.5")` | `RegionTableWidget::command`; [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `new_region_from_mni` | One field: `x y z radius` in MNI coordinates. | New row/error. | Creates a spherical region. **Completion:** Synchronous. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"new_region_from_mni","30 -20 50 8")` | `RegionTableWidget::command`; [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `new_region_from_sphere` | One field: `x y z radius` in current region-space coordinates. | New row/error. | Creates a spherical region. **Completion:** Synchronous. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"new_region_from_sphere","30 -20 50 8")` | `RegionTableWidget::command`; [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |

### Selection and table management

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `check_region` | Field 1: region index; field 2: `1` to check, other value to uncheck. | None/error. | Sets one region's shown checkbox. **Completion:** Immediate; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"check_region","3","1")` | `RegionTableWidget::command`; [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `move_up_region` | Region index. | None/error. | Moves one row up. **Completion:** Immediate; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"move_up_region","3")` | `RegionTableWidget::command`; [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `move_down_region` | Region index. | None/error. | Moves one row down. **Completion:** Immediate; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"move_down_region","3")` | `RegionTableWidget::command`; [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `move_region` | Field 1: target `x y z`; field 2: optional region index (default current). | None/error. | Moves region center to target voxel coordinates. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"move_region","40 50 30","3")` | `RegionTableWidget::command`; [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `set_region_color` | Packed Qt ARGB integer. | Error `canceled` if no regions exist. | Changes only the last region's rendering color. **Completion:** Immediate redraw. **Caveat:** It cannot target an arbitrary region index; use only immediately after creating the intended last row. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_region_color","4294901760")` | `RegionTableWidget::command`; [`set_region_color`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L902-L910) |
| `check_all_regions` | No parameters. | None. | Checks all regions. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"check_all_regions")` | `RegionTableWidget::command`; [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `uncheck_all_regions` | No parameters. | None. | Unchecks all regions. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"uncheck_all_regions")` | `RegionTableWidget::command`; [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `copy_region` | Optional region index; default current. | New row. | Duplicates a region. **Completion:** Immediate; refresh list. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"copy_region","3")` | `RegionTableWidget::command`; [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `merge_regions` | Ampersand-joined region indices; default checked rows. | Rows removed/updated. | First region absorbs the others; other rows are deleted. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"merge_regions","2&3&4")` | `RegionTableWidget::command`; [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `delete_region` | Optional region index; default current. | Row removed. | Deletes one region. **Completion:** Immediate. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_region","3")` | `RegionTableWidget::command`; [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `delete_all_regions` | No parameters. | All rows removed. | Deletes every region. **Completion:** Immediate. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_all_regions")` | `RegionTableWidget::command`; [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `move_slice_to_region` | Optional region index; default current. | None/error. | Moves slice crosshairs to the region center. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"move_slice_to_region","3")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |

### File input and output

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `save_region` | Field 1: absolute output; field 2: optional region index (default current). | Console/error output. | Writes one region. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_region","E:\out\save_region.nii.gz","3")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `save_region_info` | Field 1: absolute output; field 2: optional region index (default current). | Console/error output. | Writes one region's information. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_region_info","E:\out\save_region_info.nii.gz","3")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `save_4d_region` | Absolute output path/directory. | Progress/error output. | Writes checked regions as 4-D data. **Completion:** Synchronous; verify outputs. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_4d_region","E:\out\save_4d_region")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `save_all_regions` | Absolute output path/directory. | Progress/error output. | Writes checked regions as one label map. **Completion:** Synchronous; verify outputs. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_all_regions","E:\out\save_all_regions")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `save_all_regions_to_folder` | Absolute output path/directory. | Progress/error output. | Writes checked regions to a directory. **Completion:** Synchronous; verify outputs. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_all_regions_to_folder","E:\out\save_all_regions_to_folder")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `open_region` | Absolute region file; omission opens a dialog. | New row/error. | Loads a native-space region. **Completion:** Synchronous file load. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_region","E:\data\roi.nii.gz")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `open_mni_region` | Absolute region file; omission opens a dialog. | New row/error. | Loads a MNI-space region. **Completion:** Synchronous file load. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_mni_region","E:\data\roi.nii.gz")` | `RegionTableWidget::command`; [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `load_region_color` | Absolute text file; omission opens a dialog. | Console/error output. | Loads colors for existing rows. **Completion:** Synchronous; verify save output. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_region_color","E:\out\load_region_color.txt")` | `RegionTableWidget::command`; [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `save_region_color` | Absolute text file; omission opens a dialog. | Console/error output. | Writes checked-region colors. **Completion:** Synchronous; verify save output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_region_color","E:\out\save_region_color.txt")` | `RegionTableWidget::command`; [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |

The color file reader accepts numeric RGB or RGBA rows. The writer emits
`B G R A` component order, so do not assume its saved order is RGB
([region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768)).

### Statistics

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `show_region_statistics` | Field 1: absolute output (supply it to avoid a modal result dialog);  | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"show_region_statistics","E:\out\show_region_statistics.txt")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `save_region_statistics` | Field 1: absolute output (supply it to avoid a modal result dialog);  | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_region_statistics","E:\out\save_region_statistics.txt")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `show_device_statistics` | Field 1: absolute output (supply it to avoid a modal result dialog);  | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"show_device_statistics","E:\out\show_device_statistics.txt")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `save_device_statistics` | Field 1: absolute output (supply it to avoid a modal result dialog);  | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_device_statistics","E:\out\save_device_statistics.txt")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `show_t2r` | Field 1: absolute output (supply it to avoid a modal result dialog);  | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"show_t2r","E:\out\show_t2r.txt")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `save_t2r` | Field 1: absolute output (supply it to avoid a modal result dialog);  | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_t2r","E:\out\save_t2r.txt")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `show_tract_statistics` | Field 1: absolute output (supply it to avoid a modal result dialog); field 2: optional tract index. | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"show_tract_statistics","E:\out\show_tract_statistics.txt","0")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `save_tract_statistics` | Field 1: absolute output (supply it to avoid a modal result dialog); field 2: optional tract index. | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tract_statistics","E:\out\save_tract_statistics.txt","0")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `show_tract_recognition` | Field 1: absolute output (supply it to avoid a modal result dialog); field 2: optional tract index. | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"show_tract_recognition","E:\out\show_tract_recognition.txt","0")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `save_tract_recognition` | Field 1: absolute output (supply it to avoid a modal result dialog); field 2: optional tract index. | Tabular statistics or error output. | Computes statistics for checked regions; tract variants also use a tract row. **Completion:** Synchronous computation; verify file when path supplied. **Caveat:** A `show_*` command without an output path opens a modal dialog. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tract_recognition","E:\out\save_tract_recognition.txt","0")` | `RegionTableWidget::command`; [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |

### Region transformations

Every `region_action_*` command uses field 1 as region indices joined by `&`
(default current row) and field 2 as an optional action-specific value.
Actions containing `all` require at least two regions. The source routes any
`region_action_` prefix, and unknown suffixes can return success without doing
anything because `ROIRegion::perform()`'s result is ignored. Use only the
documented suffixes below.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `region_action_flipx` | Optional `index&index...`; default current. | Progress/error output. | Mirror in x on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_flipx","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_flipy` | Optional `index&index...`; default current. | Progress/error output. | Mirror in y on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_flipy","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_flipz` | Optional `index&index...`; default current. | Progress/error output. | Mirror in z on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_flipz","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_shiftx` | Optional `index&index...`; default current. | Progress/error output. | Shift +1 voxel in x on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_shiftx","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_shiftnx` | Optional `index&index...`; default current. | Progress/error output. | Shift -1 voxel in x on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_shiftnx","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_shifty` | Optional `index&index...`; default current. | Progress/error output. | Shift +1 voxel in y on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_shifty","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_shiftny` | Optional `index&index...`; default current. | Progress/error output. | Shift -1 voxel in y on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_shiftny","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_shiftz` | Optional `index&index...`; default current. | Progress/error output. | Shift +1 voxel in z on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_shiftz","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_shiftnz` | Optional `index&index...`; default current. | Progress/error output. | Shift -1 voxel in z on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_shiftnz","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_smoothing` | Optional `index&index...`; default current. | Progress/error output. | Morphological smoothing on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_smoothing","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_erosion` | Optional `index&index...`; default current. | Progress/error output. | One erosion on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_erosion","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_dilation` | Optional `index&index...`; default current. | Progress/error output. | One dilation on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_dilation","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_opening` | Optional `index&index...`; default current. | Progress/error output. | Morphological opening on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_opening","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_closing` | Optional `index&index...`; default current. | Progress/error output. | Morphological closing on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_closing","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_defragment` | Optional `index&index...`; default current. | Progress/error output. | Keep main component on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_defragment","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_negate` | Optional `index&index...`; default current. | Progress/error output. | Invert the mask on selected regions. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_negate","2&3")` | `RegionTableWidget::do_action`; [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region_action_1st_ex_all` | `index&index...`. | Progress/error output. | Subtract every later region from the first. **Completion:** Synchronous; refresh `list_region`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_1st_ex_all","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_all_ex_1st` | `index&index...`. | Progress/error output. | Subtract the first from every later region. **Completion:** Synchronous; refresh `list_region`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_all_ex_1st","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_all_inter_1st` | `index&index...`. | Progress/error output. | Intersect every later region with the first. **Completion:** Synchronous; refresh `list_region`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_all_inter_1st","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_all_to_1st` | `index&index...`. | Progress/error output. | Constrain all later labels to the first region. **Completion:** Synchronous; refresh `list_region`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_all_to_1st","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_refine_all` | `index&index...`. | Progress/error output. | Refine all labels using current-slice intensities. **Completion:** Synchronous; refresh `list_region`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_refine_all","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_sort_name` | `index&index...`. | Progress/error output. | Sort selected rows by name; repeating toggles direction. **Completion:** Synchronous; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_sort_name","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_sort_x` | `index&index...`. | Progress/error output. | Sort selected rows by x position; repeating toggles direction. **Completion:** Synchronous; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_sort_x","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_sort_y` | `index&index...`. | Progress/error output. | Sort selected rows by y position; repeating toggles direction. **Completion:** Synchronous; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_sort_y","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_sort_z` | `index&index...`. | Progress/error output. | Sort selected rows by z position; repeating toggles direction. **Completion:** Synchronous; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_sort_z","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_sort_size` | `index&index...`. | Progress/error output. | Sort selected rows by size; repeating toggles direction. **Completion:** Synchronous; refresh `list_region`. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_sort_size","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_separate` | `index&index...`. | Progress/error output. | Create up to 256 component regions from the first selected region. **Completion:** Synchronous; refresh `list_region`. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_separate","2&3&4")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_dilation_by_voxel` | Field 1: `index&index...`; field 2: radius in voxels. | Progress/error output. | Dilate by Euclidean radius. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_dilation_by_voxel","2&3","2")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_threshold` | Field 1: `index&index...`; field 2: threshold; negative means low-pass. | Progress/error output. | Replace each selected mask from current-slice thresholding. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_threshold","2&3","2")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_threshold_current` | Field 1: `index&index...`; field 2: threshold; negative means low-pass. | Progress/error output. | Threshold only currently included voxels. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_threshold_current","2&3","2")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_dilation_by_threshold` | Field 1: `index&index...`; field 2: threshold; negative means low-pass. | Progress/error output. | Grow through threshold-qualified voxels. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_dilation_by_threshold","2&3","2")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region_action_erosion_by_threshold` | Field 1: `index&index...`; field 2: threshold; negative means low-pass. | Progress/error output. | Erode using threshold-qualified voxels. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"region_action_erosion_by_threshold","2&3","2")` | `RegionTableWidget::do_action`; [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |

## Tract commands

`list_tract` returns `index`, `shown`, `name`, `tracts`, `deleted`, and `seeds`.
The table initially may show `initiating`; later it shows numeric counts.
Numeric counts alone do not distinguish “still running” from “finished.”
Refresh indices after any load, delete, cluster, merge, sort, or recognition
operation.

Commands documented as acting on “checked bundles” use the checkbox set, not an
unselected current row ([`for_each_bundle()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.h#L61-L90)). Commands using a single row call
`for_current_bundle`, which silently does nothing if that row is unchecked
([`for_current_bundle()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.h#L47-L59)). Explicitly check the intended rows first.

### Discovery and input

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `list_tract` | No parameters. | Header `index running shown name tracts deleted seeds` (tab-separated); flags are `0`/`1`. | Fetches newly generated tracts into table models and reports whether each tracking thread is active. **Completion:** Immediate snapshot; `running=0` does not itself prove success. | Read-only | `Invoke-Dsi -Fields @("CMD",$trackingId,"list_tract")` | `TractTableWidget::command`; [current implementation](https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L487-L500) |
| `set_dt_index` | Field 1: `metric1&metric2`; field 2: threshold-type integer. | None/error. | Sets differential-tracking metric indices. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_dt_index","qa&inc_qa","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `open_tract` | Field 1: absolute tract path; field 2: any nonempty value suppresses showing the loaded row. | New row/error. | Loads a native-space tract. **Completion:** Synchronous file load. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_tract","E:\data\tract.tt.gz","0")` | `TractTableWidget::command`; [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `open_mni_tract` | Field 1: absolute tract path; field 2: any nonempty value suppresses showing the loaded row. | New row/error. | Loads a MNI-space tract. **Completion:** Synchronous file load. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_mni_tract","E:\data\tract.tt.gz","0")` | `TractTableWidget::command`; [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `open_tract_name` | Absolute text file containing replacement names. | Console/error output. | Renames loaded tract rows from file tokens. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_tract_name","E:\data\names.txt")` | `TractTableWidget::command`; [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `load_tract_atlas` | Optional exact atlas tract name; empty loads all. | New rows/progress. | Loads template tract atlas data into subject space. **Completion:** Synchronous mapping/computation; may time out. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_tract_atlas","CST_L")` | `TractTableWidget::command`; [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |


The GUI tooltip spells `cut_tract_RAI_end` with uppercase `RAI`, but the handler
accepts lowercase `cut_tract_rai_end`; commands are case-sensitive
([handler](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L509-L518), [GUI tooltip mismatch](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window.ui#L5099-L5105)). Use the lowercase handler spelling.

### Editing

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `delete_branch` | No parameters; acts on checked bundles. | Counts update. | Deletes branches on checked bundles. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_branch")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `undo_tract` | No parameters; acts on checked bundles. | Counts update. | Undoes the last model edit on checked bundles. **Completion:** Synchronous parallel edit. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"undo_tract")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `redo_tract` | No parameters; acts on checked bundles. | Counts update. | Redoes the last model edit on checked bundles. **Completion:** Synchronous parallel edit. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"redo_tract")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `trim_tract` | No parameters; acts on checked bundles. | Counts update. | Trims checked bundles. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"trim_tract")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_end_portion` | Optional tract index; default current; row must be checked. | Counts update. | Keeps the middle 50% of streamline points. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_end_portion","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_lps_end` | Optional tract index; default current; row must be checked. | Counts update. | Cuts the L/P/S-directed end portion. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_lps_end","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_rai_end` | Optional tract index; default current; row must be checked. | Counts update. | Cuts the R/A/I-directed end portion. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_rai_end","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `flip_tract_x` | Optional tract index; default current; row must be checked. | None/count update. | Mirrors every streamline coordinate across image axis `x`. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"flip_tract_x","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `flip_tract_y` | Optional tract index; default current; row must be checked. | None/count update. | Mirrors every streamline coordinate across image axis `y`. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"flip_tract_y","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `flip_tract_z` | Optional tract index; default current; row must be checked. | None/count update. | Mirrors every streamline coordinate across image axis `z`. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"flip_tract_z","0")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_by_x` | Optional slice coordinate; default current position on named axis; acts on checked bundles. | Counts update. | Cuts on the named side; suffix `2` selects the opposite side. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_by_x","40")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_by_x2` | Optional slice coordinate; default current position on named axis; acts on checked bundles. | Counts update. | Cuts on the named side; suffix `2` selects the opposite side. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_by_x2","40")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_by_y` | Optional slice coordinate; default current position on named axis; acts on checked bundles. | Counts update. | Cuts on the named side; suffix `2` selects the opposite side. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_by_y","40")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_by_y2` | Optional slice coordinate; default current position on named axis; acts on checked bundles. | Counts update. | Cuts on the named side; suffix `2` selects the opposite side. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_by_y2","40")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_by_z` | Optional slice coordinate; default current position on named axis; acts on checked bundles. | Counts update. | Cuts on the named side; suffix `2` selects the opposite side. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_by_z","40")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `cut_tract_by_z2` | Optional slice coordinate; default current position on named axis; acts on checked bundles. | Counts update. | Cuts on the named side; suffix `2` selects the opposite side. **Completion:** Synchronous parallel edit. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cut_tract_by_z2","40")` | `TractTableWidget::command`; [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |

### Selection and management

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `filter_tract` | Optional ROI grammar; empty derives settings from checked regions; acts on checked tracts. | Counts/rows update. | Filters streamlines through ROI settings. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"filter_tract","18:0&21:1")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `copy_tract` | Optional tract index; default current. | Counts/rows update. | Duplicates one tract row. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"copy_tract","0")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `delete_tract` | Optional tract index; default current. | Counts/rows update. | Deletes one tract row. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_tract","0")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `delete_all_tracts` | No parameters. | All rows removed. | Deletes all tracts. **Completion:** Immediate. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_all_tracts")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `update_tract` | Optional tract index; default current. | Counts/render update. | Refreshes tract counts and cached rendering. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"update_tract","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `check_tract` | Field 1: tract index; field 2: `1` to check, other value to uncheck. | None. | Sets one tract checkbox. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"check_tract","0","1")` | `TractTableWidget::command`; [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `check_uncheck_all_tract` | Optional `1`/`0`; empty toggles based on the first row. | None. | Checks or unchecks all tracts. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"check_uncheck_all_tract","0")` | `TractTableWidget::command`; [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |

### Output and tract-to-region conversion

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `save_tract` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes native tract. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tract","E:\out\save_tract.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_mni_tract` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes MNI-space tract. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_mni_tract","E:\out\save_mni_tract.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_template_tract` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes template-space tract. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_template_tract","E:\out\save_template_tract.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_slice_tract` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes current-slice-space tract. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_slice_tract","E:\out\save_slice_tract.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_tract_endpoint` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes native endpoints. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tract_endpoint","E:\out\save_tract_endpoint.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_mni_tract_endpoint` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes MNI endpoints. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_mni_tract_endpoint","E:\out\save_mni_tract_endpoint.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_slice_tract_endpoint` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes slice-space endpoints. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_slice_tract_endpoint","E:\out\save_slice_tract_endpoint.nii.gz","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_tdi` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes tract-density image. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tdi","E:\out\save_tdi.nii.gz","0")` | `TractTableWidget::command`; [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `save_tdi2` | Field 1: absolute output; field 2: optional tract index (default current). | Console/error output. | Writes 2×-resolution tract-density image. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tdi2","E:\out\save_tdi2.nii.gz","0")` | `TractTableWidget::command`; [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `save_tract_values` | Field 1: absolute output; field 2: tract index; field 3: exact metric name (supply it to avoid a dialog). | Console/error output. | Writes per-tract values. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tract_values","E:\out\values.txt","0","qa")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_all_tracts_to_folder` | Absolute directory/output. | Progress/error output. | Writes checked tracts separately to a directory. **Completion:** Synchronous; verify output(s). | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_all_tracts_to_folder","E:\out\save_all_tracts_to_folder")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `save_all_tracts` | Absolute directory/output. | Progress/error output. | Writes checked tracts into one file. **Completion:** Synchronous; verify output(s). | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_all_tracts","E:\out\save_all_tracts")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract_to_region` | Optional tract index; default current. | New region row(s). | Converts one tract to one region. **Completion:** Synchronous; refresh `list_region`. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"tract_to_region","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `endpoint_to_region` | Optional tract index; default current. | New region row(s). | Converts one tract's endpoints to two regions. **Completion:** Synchronous; refresh `list_region`. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"endpoint_to_region","0")` | `TractTableWidget::command`; [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |

### Colors and values

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `load_tract_color` | Field 1: absolute file; field 2: optional tract index (default current). | Console/error output. | Loads per-point/per-tract colors. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_tract_color","E:\out\load_tract_color.txt","0")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `load_tract_values` | Field 1: absolute file; field 2: optional tract index (default current). | Console/error output. | Loads tract values. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_tract_values","E:\out\load_tract_values.txt","0")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `save_tract_color` | Field 1: absolute file; field 2: optional tract index (default current). | Console/error output. | Writes tract colors. **Completion:** Synchronous. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_tract_color","E:\out\save_tract_color.txt","0")` | `TractTableWidget::command`; [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `load_cluster_color` | Absolute file. | Console/error output. | Loads cluster colors into checked tracts. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_cluster_color","E:\out\load_cluster_color.txt")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `load_cluster_values` | Absolute file. | Console/error output. | Loads cluster values into checked tracts. **Completion:** Synchronous. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"load_cluster_values","E:\out\load_cluster_values.txt")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `save_cluster_color` | Absolute file. | Console/error output. | Writes checked-tract cluster colors. **Completion:** Synchronous. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_cluster_color","E:\out\save_cluster_color.txt")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `select_cluster_color` | Field 1: tract index; field 2: packed Qt ARGB color (supply it to avoid a dialog). | None/error. | Colors one selected cluster. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"select_cluster_color","0","4294901760")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `color_all_cluster` | No parameters; acts on checked tracts. | None. | Assigns colors to all clusters. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"color_all_cluster")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |

### Clustering and processing

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `cluster_tract_by_label` | Field 1: tract index; field 2: absolute label file. | Rows/counts update. | Clusters from labels and replaces/splits source tract rows. **Completion:** Synchronous computation; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cluster_tract_by_label","0","10")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `recognize_and_cluster_tract` | Field 1: tract index. | Rows/counts update. | Recognizes and clusters and replaces/splits source tract rows. **Completion:** Synchronous computation; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"recognize_and_cluster_tract","0","10")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `cluster_tract_by_hy` | Field 1: tract index; field 2: `cluster_count detail`. | Rows/counts update. | Hierarchical clustering and replaces/splits source tract rows. **Completion:** Synchronous computation; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cluster_tract_by_hy","0","10 5")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `cluster_tract_by_km` | Field 1: tract index; field 2: cluster count. | Rows/counts update. | K-means clustering and replaces/splits source tract rows. **Completion:** Synchronous computation; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cluster_tract_by_km","0","10")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `cluster_tract_by_em` | Field 1: tract index; field 2: cluster count. | Rows/counts update. | Expectation-maximization clustering and replaces/splits source tract rows. **Completion:** Synchronous computation; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"cluster_tract_by_em","0","10")` | `TractTableWidget::command`; [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `delete_repeated_tract` | distance in voxels (default `1`). | Counts/rows update. | Deletes near-duplicate streamlines. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_repeated_tract","1")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `resample_tract` | step in voxels (default `0.5`). | Counts/rows update. | Resamples streamlines. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"resample_tract","1")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `delete_tract_by_length` | minimum length in mm (default `0.5`). | Counts/rows update. | Deletes short streamlines. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_tract_by_length","1")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `separate_deleted_tract` | tract index. | Counts/rows update. | Moves deleted streamlines into a new row. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"separate_deleted_tract","0")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `reconnect_tract` | field 1 tract index; field 2 `distance_voxels angle_degrees` (defaults `4 30`). | Counts/rows update. | Reconnects streamline fragments. **Completion:** Synchronous computation. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"reconnect_tract","0","4 30")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `recognize_and_rename_tract` | No parameters. | Rows update. | Recognizes and renames checked tracts. **Completion:** Synchronous; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"recognize_and_rename_tract")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `merge_all_tracts` | No parameters. | Rows update. | Merges checked tracts into one. **Completion:** Synchronous; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"merge_all_tracts")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `merge_tract_by_name` | No parameters. | Rows update. | Merges rows sharing a name. **Completion:** Synchronous; refresh list. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"merge_tract_by_name")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `sort_tract_by_name` | No parameters. | Rows update. | Sorts rows by name. **Completion:** Synchronous; refresh list. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"sort_tract_by_name")` | `TractTableWidget::command`; [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |

## Device commands

There is no `list_device` command. Device indices are therefore unsafe to
guess; use a known current table state or add the recommended command before
automating indexed device work. Coordinates are diffusion-space voxel
coordinates.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `new_device` | Optional one field `i j k`; empty chooses a random location near center. | New table row. | Creates a device with default type/orientation. **Completion:** Immediate; anisotropic data may open a modal warning. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"new_device","40 50 30")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `move_device` | Field 1: target `i j k`; field 2: optional device index (default current). | None/error. | Moves device center. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"move_device","40 50 30","0")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `push_device` | Optional device index; default current. | None/error. | Moves 0.5 mm opposite its direction. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"push_device","0")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `pull_device` | Optional device index; default current. | None/error. | Moves 0.5 mm along its direction. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"pull_device","0")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `copy_device` | Optional device index; default current. | New row. | Duplicates one device. **Completion:** Immediate. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"copy_device","0")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `set_acpc` | No parameters. | Rows/error. | Creates or replaces AC, PC, and interhemispheric locators from fixed MNI positions. **Completion:** Synchronous mapping; requires MNI mapping. | Computation | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_acpc")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `delete_device` | Optional device index; default current. | Row removed. | Deletes one device. **Completion:** Immediate. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_device","0")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `delete_all_devices` | No parameters. | All rows removed. | Deletes all devices. **Completion:** Immediate. | Destructive | `Invoke-Dsi -Fields @("CMD",$trackingId,"delete_all_devices")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `save_all_devices` | Absolute `.dv.csv` output; saves checked devices. | Console/error output. | Writes device CSV. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_all_devices","E:\out\devices.dv.csv")` | `DeviceTableWidget::command`; [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |

## Rendering and camera commands

Camera matrices contain 16 whitespace-separated floats. `set_view` accepts
`0=sagittal`, `1=coronal`, and `2=axial`; those indices are tied to the tracking
window's `cur_dim` assignments ([view index assignments](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window.cpp#L447-L453)) and the matrix setup in
[`GLWidget::set_view()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L137-L195). `set_view()` flips the orientation flag on each call, so repeated calls to
the same view alternate opposite faces. For reproducibility, prefer a saved
16-value camera matrix.

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `set_zoom` | Nonzero floating-point absolute zoom. | None/error. | Scales the current camera to requested zoom. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_zoom","1.5")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `set_view` | `0` sagittal, `1` coronal, `2` axial. | None/error. | Resets camera to canonical view; repeated calls alternate face orientation. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_view","2")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `rotate` | One field: `angle_degrees x y z`; omitted axis components default to `0 1 0`. | None. | Rotates camera around axis. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"rotate","15 0 1 0")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `set_stereoscopic` | No parameters. | None. | Enables stereo view mode. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_stereoscopic")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `open_camera` | Absolute text file containing at least 16 floats. | Error if unreadable/short. | Loads camera matrix. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"open_camera","E:\settings\camera.txt")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_camera` | Absolute output path. | Error on write failure. | Writes 16 camera floats. **Completion:** Synchronous; verify output. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_camera","E:\settings\camera.txt")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `set_camera` | One field containing at least 16 floats. | Error `canceled` when empty/short. | Loads camera matrix from the request. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"set_camera","1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `store_camera1` | No parameters. | Shows a modal information box, then returns `ERROR` `canceled`. | Stores current camera in QSettings slot `1` before reporting error. **Completion:** State is stored immediately, but the modal must be dismissed. **Caveat:** Not safe for an unattended AI agent despite changing state. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"store_camera1")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `restore_camera1` | No parameters. | Error if slot is empty. | Restores QSettings camera slot `1`. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"restore_camera1")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `store_camera2` | No parameters. | Shows a modal information box, then returns `ERROR` `canceled`. | Stores current camera in QSettings slot `2` before reporting error. **Completion:** State is stored immediately, but the modal must be dismissed. **Caveat:** Not safe for an unattended AI agent despite changing state. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"store_camera2")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `restore_camera2` | No parameters. | Error if slot is empty. | Restores QSettings camera slot `2`. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"restore_camera2")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `store_camera3` | No parameters. | Shows a modal information box, then returns `ERROR` `canceled`. | Stores current camera in QSettings slot `3` before reporting error. **Completion:** State is stored immediately, but the modal must be dismissed. **Caveat:** Not safe for an unattended AI agent despite changing state. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"store_camera3")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `restore_camera3` | No parameters. | Error if slot is empty. | Restores QSettings camera slot `3`. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"restore_camera3")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `store_camera4` | No parameters. | Shows a modal information box, then returns `ERROR` `canceled`. | Stores current camera in QSettings slot `4` before reporting error. **Completion:** State is stored immediately, but the modal must be dismissed. **Caveat:** Not safe for an unattended AI agent despite changing state. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"store_camera4")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `restore_camera4` | No parameters. | Error if slot is empty. | Restores QSettings camera slot `4`. **Completion:** Immediate redraw. | GUI-state change | `Invoke-Dsi -Fields @("CMD",$trackingId,"restore_camera4")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_screen` | Absolute image output; omission opens a dialog. | Error on image-save failure. | Writes current 3-D view. **Completion:** Synchronous; verify image. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_screen","E:\out\save_screen.png")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_3view_screen` | Absolute image output; omission opens a dialog. | Error on image-save failure. | Writes 2×2 slice/3-D composite. **Completion:** Synchronous; verify image. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_3view_screen","E:\out\save_3view_screen.png")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_h3view_screen` | Absolute image output; omission opens a dialog. | Error on image-save failure. | Writes horizontal three-view composite. **Completion:** Synchronous; verify image. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_h3view_screen","E:\out\save_h3view_screen.png")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_v3view_screen` | Absolute image output; omission opens a dialog. | Error on image-save failure. | Writes vertical three-view composite. **Completion:** Synchronous; verify image. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_v3view_screen","E:\out\save_v3view_screen.png")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_hd_screen` | Field 1: absolute image output; field 2: one field `width height`. | Error on image-save failure. | Temporarily resizes GL widget, saves, then restores size. **Completion:** Synchronous; verify image dimensions. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_hd_screen","E:\out\hd.png","1920 1080")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `save_rotation_video` | Absolute `.avi` output. | Returns `OKAY` when a path is supplied. | None: the actual video-writing block is unreachable. **Completion:** Broken; never use as proof of file creation. **Caveat:** The unconditional return at lines 2407-2410 bypasses all encoding code. | File creation | `Invoke-Dsi -Fields @("CMD",$trackingId,"save_rotation_video","E:\out\rotation.avi")` | `GLWidget::command`; [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |

`save_rotation_video` is currently unusable: the handler returns before its AVI
code ([unreachable video block](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2405-L2451)). Do not run it. Recommend fixing or replacing it with a job-based
capture command.

## View-image commands

Image windows accept `CMD`, window ID, command, and **at most one** parameter
field. Pack all numbers or path-plus-value syntax into that single field.
Commands mutate the in-memory image and are recorded in the image command
history. Save explicitly when persistence is intended.

The following display controls are not remotely exposed in this commit:
orientation, current slice/4-D volume, zoom, contrast range/colors, overlay,
axis grid, apply-to-all checkbox, undo, and redo. Their GUI slots are separate
from `view_image::command()` ([image display slots](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L1114-L1203)). Do not claim they were changed remotely;
use the recommended image-state commands below.

On a 4-D image, `normalize`, `normalize_otsu_median`, and `change_type` have
special all-volume behavior; other transforms apply to all volumes only when
the GUI's **Apply to all** checkbox is already checked ([4-D command behavior](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L151-L220)). On a MAT/FIB/SRC
container, commands run through `modify_fib()` and apply to every matrix that
matches the image dimensions; changes remain in memory until `save` or
`save_mini` ([MAT command backend](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L942-L1109)).

### Core, save, and UNet

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `change_type` | `0=uint8`, `1=uint16`, `2=uint32`, `3=float32`. | None/error. | Converts in-memory pixel type; all 4-D volumes are converted. **Completion:** Synchronous. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"change_type","3")` | `view_image::command → variant_image::command`; [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `save` | Absolute output path. | Console/error output. | Writes current image/container. **Completion:** Synchronous for one image; a multi-file session may open an apply-to-other-images modal. **Caveat:** Confirm overwrite and avoid unattended multi-file saves. | File creation | `Invoke-Dsi -Fields @("CMD",$imageId,"save","E:\out\image.nii.gz")` | `view_image::command → modify_fib/save path`; [`view_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L77-L323) |
| `save_mini` | Absolute `.fz`/MAT-compatible output path. | Console/error output. | Writes a reduced MAT/FIB container. **Completion:** Synchronous; only meaningful for a MAT/FIB/SRC-backed image. | File creation | `Invoke-Dsi -Fields @("CMD",$imageId,"save_mini","E:\out\mini.fz")` | `view_image::command → modify_fib/save path`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `brain_extraction` | Exact UNet model stem. | Download/progress/error output. | Masks the image with UNet foreground probability. **Completion:** Synchronous download/inference; likely to exceed timeout. **Caveat:** Image windows have no model-list command. Use only a trusted available stem; never TumorSynth. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"brain_extraction","model_stem")` | `view_image::command → variant_image::command`; [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `segmentation` | Exact UNet model stem. | Download/progress/error output. | Replaces image with an 8-bit label image. **Completion:** Synchronous download/inference; likely to exceed timeout. **Caveat:** Image windows have no model-list command. Use only a trusted available stem; never TumorSynth. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"segmentation","model_stem")` | `view_image::command → variant_image::command`; [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `deface` | Exact UNet model stem. | Download/progress/error output. | Masks facial tissue using UNet output. **Completion:** Synchronous download/inference; likely to exceed timeout. **Caveat:** Image windows have no model-list command. Use only a trusted available stem; never TumorSynth. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"deface","model_stem")` | `view_image::command → variant_image::command`; [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |

### Image transforms

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `regrid` | Single field: isotropic voxel size; UI default `1.0`. | Console/error output. | Resample to isotropic resolution in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"regrid","1.0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:584-594`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L584-L594) |
| `multiply_image` | Single field: absolute NIfTI path. | Console/error output. | Multiply by another image in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"multiply_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:595-609`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L595-L609) |
| `resize` | Single field: optional new `x y z` shape. | Console/error output. | Change image size in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"resize","128 128 64")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:610-617`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L610-L617) |
| `translocate` | Single field: `x y z`; UI default `0 0 0`. | Console/error output. | Translate image data in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"translocate","0 0 0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:618-628`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L618-L628) |
| `crop_to_fit` | Single field: padding `x y z`; UI default `0 0 0`. | Console/error output. | Crop image to nonzero extent in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"crop_to_fit","0 0 0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:629-639`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L629-L639) |
| `set_translocation` | Single field: `x y z` transformation translation. | Console/error output. | Change header translation in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"set_translocation","0 0 0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:640-647`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L640-L647) |
| `lower_threshold` | Single field: lower bound; UI default `0`. | Console/error output. | Clamp below a lower intensity in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"lower_threshold","0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:648-658`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L648-L658) |
| `set_transformation` | Single field: 16 matrix floats. | Console/error output. | Replace image transformation matrix in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"set_transformation","1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:659-666`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L659-L666) |
| `add_value` | Single field: scalar; UI default `0`. | Console/error output. | Add scalar intensity in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"add_value","1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:667-677`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L667-L677) |
| `multiply_value` | Single field: scalar; UI default `1.0`. | Console/error output. | Multiply scalar intensity in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"multiply_value","2")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:678-688`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L678-L688) |
| `upper_threshold` | Single field: upper bound; UI default `0`. | Console/error output. | Clamp above an upper intensity in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"upper_threshold","100")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:689-699`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L689-L699) |
| `normalize` | Single field: empty. | Console/error output. | Scale intensities to 0..1 in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"normalize")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:700-707`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L700-L707) |
| `morphology_edge` | Single field: empty. | Console/error output. | Extract label edges in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_edge")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:708-715`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L708-L715) |
| `morphology_edge_xy` | Single field: empty. | Console/error output. | Extract label edges in x/y in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_edge_xy")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:716-723`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L716-L723) |
| `morphology_edge_xz` | Single field: empty. | Console/error output. | Extract label edges in x/z in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_edge_xz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:724-731`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L724-L731) |
| `add_image` | Single field: absolute NIfTI path. | Console/error output. | Add another image in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"add_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:744-758`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L744-L758) |
| `morphology_smoothing` | Single field: empty. | Console/error output. | Smooth region labels in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_smoothing")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:759-766`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L759-L766) |
| `downsampling` | Single field: empty. | Console/error output. | Downsample by two in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"downsampling")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:767-774`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L767-L774) |
| `upsampling` | Single field: empty. | Console/error output. | Upsample by two in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"upsampling")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:775-782`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L775-L782) |
| `flip_x` | Single field: empty. | Console/error output. | Flip data in x in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"flip_x")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:783-790`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L783-L790) |
| `flip_y` | Single field: empty. | Console/error output. | Flip data in y in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"flip_y")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:791-798`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L791-L798) |
| `flip_z` | Single field: empty. | Console/error output. | Flip data in z in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"flip_z")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:799-806`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L799-L806) |
| `swap_xy` | Single field: empty. | Console/error output. | Swap x/y axes in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"swap_xy")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:807-814`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L807-L814) |
| `swap_xz` | Single field: empty. | Console/error output. | Swap x/z axes in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"swap_xz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:815-822`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L815-L822) |
| `swap_yz` | Single field: empty. | Console/error output. | Swap y/z axes in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"swap_yz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:823-830`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L823-L830) |
| `minus_image` | Single field: absolute NIfTI path. | Console/error output. | Subtract another image in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"minus_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:831-845`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L831-L845) |
| `morphology_dilation` | Single field: empty. | Console/error output. | Dilate region labels in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_dilation")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:846-853`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L846-L853) |
| `threshold` | Single field: threshold accepted by TIPL; UI supplies empty. | Console/error output. | Binarize by threshold in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"threshold","0.5")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:854-861`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L854-L861) |
| `morphology_defragment` | Single field: empty. | Console/error output. | Remove label fragments in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_defragment")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:862-869`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L862-L869) |
| `morphology_erosion` | Single field: empty. | Console/error output. | Erode region labels in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_erosion")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:870-877`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L870-L877) |
| `mean_filter` | Single field: empty. | Console/error output. | Mean-filter intensities in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"mean_filter")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:878-885`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L878-L885) |
| `gaussian_filter` | Single field: optional TIPL filter parameter. | Console/error output. | Gaussian-filter intensities in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"gaussian_filter","1.0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:886-896`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L886-L896) |
| `sobel_filter` | Single field: empty. | Console/error output. | Calculate Sobel gradient in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"sobel_filter")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:897-904`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L897-L904) |
| `smoothing_filter` | Single field: optional TIPL filter parameter. | Console/error output. | Edge-preserving smoothing in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"smoothing_filter","1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:905-915`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L905-L915) |
| `transform` | Single field: optional TIPL transform parameter. | Console/error output. | Transform to new coordinates in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"transform","1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:916-923`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L916-L923) |
| `header_flip_x` | Single field: empty. | Console/error output. | Flip x in header only in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"header_flip_x")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:933-937`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L933-L937) |
| `header_flip_y` | Single field: empty. | Console/error output. | Flip y in header only in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"header_flip_y")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:938-942`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L938-L942) |
| `header_flip_z` | Single field: empty. | Console/error output. | Flip z in header only in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"header_flip_z")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:943-947`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L943-L947) |
| `header_swap_xy` | Single field: empty. | Console/error output. | Swap x/y in header only in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"header_swap_xy")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:948-952`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L948-L952) |
| `header_swap_xz` | Single field: empty. | Console/error output. | Swap x/z in header only in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"header_swap_xz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:953-957`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L953-L957) |
| `header_swap_yz` | Single field: empty. | Console/error output. | Swap y/z in header only in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"header_swap_yz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:958-962`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L958-L962) |
| `select_value` | Single field: label value accepted by TIPL; UI supplies empty. | Console/error output. | Select one value as ROI in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"select_value","1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:979-986`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L979-L986) |
| `concatenate_image` | Single field: absolute NIfTI path. | Console/error output. | Concatenate along z, or append a 4-D volume in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"concatenate_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:987-1001`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L987-L1001) |
| `reshape` | Single field: `x y z [dim4]`. | Console/error output. | Reshape data without resampling in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"reshape","128 128 64")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1002-1009`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1002-L1009) |
| `max_image` | Single field: absolute NIfTI path. | Console/error output. | Voxelwise maximum in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"max_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1010-1024`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1010-L1024) |
| `min_image` | Single field: absolute NIfTI path. | Console/error output. | Voxelwise minimum in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"min_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1025-1039`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1025-L1039) |
| `morphology_defragment_by_size` | Single field: size ratio; UI default `0.1`. | Console/error output. | Remove components below size in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_defragment_by_size","0.1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1040-1050`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1040-L1050) |
| `equation` | Single field: expression using `x`; UI default `(x+1)*(x>0)`. | Console/error output. | Apply voxelwise equation in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"equation","(x+1)*(x>0)")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1051-1061`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1051-L1061) |
| `set_mni` | Single field: `1` yes or `0` no. | Console/error output. | Set MNI-space flag in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"set_mni","1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1062-1072`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1062-L1072) |
| `normalize_otsu_median` | Single field: empty. | Console/error output. | Scale above-Otsu median to 0.5 in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"normalize_otsu_median")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1073-1080`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1073-L1080) |
| `otsu_threshold` | Single field: ratio; UI default `1.0`. | Console/error output. | Binarize using Otsu threshold in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"otsu_threshold","1.0")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1081-1091`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1081-L1091) |
| `resize_at_center` | Single field: optional new `x y z` shape. | Console/error output. | Resize around center in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"resize_at_center","128 128 64")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1092-1099`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1092-L1099) |
| `histogram_sharpening` | Single field: optional TIPL parameter. | Console/error output. | Sharpen histogram in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"histogram_sharpening","1")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1100-1104`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1100-L1104) |
| `bias_field_correction` | Single field: empty; DSI code ignores the parameter. | Console/error output. | Iterative bias-field correction in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"bias_field_correction")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1105-1115`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1105-L1115) |
| `rotate_to_image` | Single field: absolute target NIfTI path. | Console/error output. | Rigidly register/rotate to target in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"rotate_to_image","E:\data\target.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1116-1130`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1116-L1130) |
| `warp_to_image` | Single field: absolute target NIfTI path. | Console/error output. | Affinely and nonlinearly warp to target in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"warp_to_image","E:\data\target.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1131-1145`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1131-L1145) |
| `apply_to_image` | Single field: absolute image to sample using prior registration. | Console/error output. | Apply stored mapping in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"apply_to_image","E:\data\other.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1146-1157`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1146-L1157) |
| `refine_label` | Single field: absolute reference NIfTI path. | Console/error output. | Refine labels from reference in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"refine_label","E:\data\reference.nii.gz")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1158-1172`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1158-L1172) |
| `morphology_opening` | Single field: empty. | Console/error output. | Morphological opening in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_opening")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1173-1177`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1173-L1177) |
| `morphology_closing` | Single field: empty. | Console/error output. | Morphological closing in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_closing")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1178-1182`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1178-L1182) |
| `morphology_fill_holes` | Single field: empty. | Console/error output. | Fill holes in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_fill_holes")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1183-1190`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1183-L1190) |
| `morphology_fill_holes_by_slice` | Single field: empty. | Console/error output. | Fill holes slice by slice in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_fill_holes_by_slice")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1191-1198`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1191-L1198) |
| `morphology_negate` | Single field: empty. | Console/error output. | Invert label mask in memory. **Completion:** Synchronous; registration/filtering may exceed timeout. **Caveat:** Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio. | Computation | `Invoke-Dsi -Fields @("CMD",$imageId,"morphology_negate")` | `view_image::command → variant_image::command/TIPL`; [`view_image.ui:1199-1206`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1199-L1206) |

### MAT/FIB/SRC field commands

| Command | Syntax / parameters | Output | Effect and completion | Safety | Example | Handler and source |
|---|---|---|---|---|---|---|
| `mat_remove` | One parameter field: existing field name. | Console/error output. | Remove a MAT field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_remove","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_resize` | One parameter field: `existing_field rows columns`. | Console/error output. | Resize a MAT field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_resize","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_set_name` | One parameter field: `existing_field new_name`. | Console/error output. | Rename a MAT field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_set_name","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_add_string` | One parameter field: `existing_field new_name`. | Console/error output. | Insert a string field before the named field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_add_string","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_add_float` | One parameter field: `existing_field new_name`. | Console/error output. | Insert a float field before the named field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_add_float","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_add_int` | One parameter field: `existing_field new_name`. | Console/error output. | Insert a uint32 field before the named field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_add_int","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_add_short` | One parameter field: `existing_field new_name`. | Console/error output. | Insert a uint16 field before the named field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_add_short","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_add_int64` | One parameter field: `existing_field new_name`. | Console/error output. | Insert a uint64 field before the named field in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_add_int64","fa0 new_field")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `mat_set_value` | One parameter field: `existing_field whitespace-separated-values`. | Console/error output. | Replace a field's values/text in memory. **Completion:** Synchronous. | Destructive | `Invoke-Dsi -Fields @("CMD",$imageId,"mat_set_value","fa0 1 2 3")` | `view_image::command → modify_fib`; [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |

The first token of every `mat_*` parameter must name an existing field because
the backend resolves it before performing any operation. `mat_add_*` inserts
relative to that row; its second token is the new field name. There is no remote
field-list command even though the GUI has an information table, so do not
guess field names. Save to a new file first; confirm before replacing the
source container.

## Complete command reference

The preceding operational tables are the full, detailed reference. This
consolidated index contains every documented command spelling/variant and its
argument-field count. Dynamic prefixes are counted only for the source-defined
GUI variants: 14 `add_surface*` commands, four camera-store slots, four
camera-restore slots, three tract flips, six tract cuts, and 32 region actions.
`set_param`/`set_params` values are separately enumerated in the complete
parameter schema.

| Scope | Command | Parameter fields | Safety | Completion | Handler | Source |
|---|---|---:|---|---|---|---|
| `atlas` | `add_region_from_atlas` | `1` | Computation | Synchronous extraction; verify `list_region`. | `RegionTableWidget::command` | [`add_region_from_atlas`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L785-L846) |
| `atlas` | `list_atlas` | `0` | Read-only | Immediate list. | `tracking_window::command` | [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `auto` | `enable_auto_tract` | `0` | Computation | Synchronous atlas load. | `tracking_window::command` | [automatic tracking](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776) |
| `auto` | `list_auto_tract` | `0` | Read-only | Synchronous list. | `tracking_window::command` | [automatic tracking](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776) |
| `auto` | `run_auto_track` | `1-2` | Computation | Asynchronous; `OKAY` means started only. | `tracking_window::command` | [automatic tracking](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776) |
| `auto` | `run_tracking` | `1-3` | Computation | Asynchronous; poll `list_tract` until `running=0` and inspect `LOG`. | `TractTableWidget::command` | [current implementation](https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L451-L460) |
| `device` | `copy_device` | `0-1` | GUI-state change | Immediate. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `delete_all_devices` | `0` | Destructive | Immediate. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `delete_device` | `0-1` | Destructive | Immediate. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `move_device` | `1-2` | GUI-state change | Immediate. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `new_device` | `0-1` | GUI-state change | Immediate; anisotropic data may open a modal warning. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `pull_device` | `0-1` | GUI-state change | Immediate. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `push_device` | `0-1` | GUI-state change | Immediate. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `save_all_devices` | `1` | File creation | Synchronous; verify output. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `device` | `set_acpc` | `0` | Computation | Synchronous mapping; requires MNI mapping. | `DeviceTableWidget::command` | [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675) |
| `image-core` | `brain_extraction` | `1` | Computation | Synchronous download/inference; likely to exceed timeout. | `view_image::command → variant_image::command` | [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `image-core` | `change_type` | `1` | Computation | Synchronous. | `view_image::command → variant_image::command` | [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `image-core` | `deface` | `1` | Computation | Synchronous download/inference; likely to exceed timeout. | `view_image::command → variant_image::command` | [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `image-core` | `save` | `1` | File creation | Synchronous for one image; a multi-file session may open an apply-to-other-images modal. | `view_image::command → modify_fib/save path` | [`view_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L77-L323) |
| `image-core` | `save_mini` | `1` | File creation | Synchronous; only meaningful for a MAT/FIB/SRC-backed image. | `view_image::command → modify_fib/save path` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-core` | `segmentation` | `1` | Computation | Synchronous download/inference; likely to exceed timeout. | `view_image::command → variant_image::command` | [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148) |
| `image-mat` | `mat_add_float` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_add_int` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_add_int64` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_add_short` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_add_string` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_remove` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_resize` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_set_name` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-mat` | `mat_set_value` | `1` | Destructive | Synchronous. | `view_image::command → modify_fib` | [`modify_fib()` MAT commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035) |
| `image-transform` | `add_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:744-758`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L744-L758) |
| `image-transform` | `add_value` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:667-677`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L667-L677) |
| `image-transform` | `apply_to_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1146-1157`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1146-L1157) |
| `image-transform` | `bias_field_correction` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1105-1115`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1105-L1115) |
| `image-transform` | `concatenate_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:987-1001`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L987-L1001) |
| `image-transform` | `crop_to_fit` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:629-639`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L629-L639) |
| `image-transform` | `downsampling` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:767-774`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L767-L774) |
| `image-transform` | `equation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1051-1061`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1051-L1061) |
| `image-transform` | `flip_x` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:783-790`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L783-L790) |
| `image-transform` | `flip_y` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:791-798`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L791-L798) |
| `image-transform` | `flip_z` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:799-806`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L799-L806) |
| `image-transform` | `gaussian_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:886-896`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L886-L896) |
| `image-transform` | `header_flip_x` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:933-937`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L933-L937) |
| `image-transform` | `header_flip_y` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:938-942`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L938-L942) |
| `image-transform` | `header_flip_z` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:943-947`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L943-L947) |
| `image-transform` | `header_swap_xy` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:948-952`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L948-L952) |
| `image-transform` | `header_swap_xz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:953-957`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L953-L957) |
| `image-transform` | `header_swap_yz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:958-962`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L958-L962) |
| `image-transform` | `histogram_sharpening` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1100-1104`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1100-L1104) |
| `image-transform` | `lower_threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:648-658`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L648-L658) |
| `image-transform` | `max_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1010-1024`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1010-L1024) |
| `image-transform` | `mean_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:878-885`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L878-L885) |
| `image-transform` | `min_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1025-1039`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1025-L1039) |
| `image-transform` | `minus_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:831-845`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L831-L845) |
| `image-transform` | `morphology_closing` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1178-1182`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1178-L1182) |
| `image-transform` | `morphology_defragment` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:862-869`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L862-L869) |
| `image-transform` | `morphology_defragment_by_size` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1040-1050`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1040-L1050) |
| `image-transform` | `morphology_dilation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:846-853`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L846-L853) |
| `image-transform` | `morphology_edge` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:708-715`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L708-L715) |
| `image-transform` | `morphology_edge_xy` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:716-723`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L716-L723) |
| `image-transform` | `morphology_edge_xz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:724-731`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L724-L731) |
| `image-transform` | `morphology_erosion` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:870-877`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L870-L877) |
| `image-transform` | `morphology_fill_holes` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1183-1190`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1183-L1190) |
| `image-transform` | `morphology_fill_holes_by_slice` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1191-1198`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1191-L1198) |
| `image-transform` | `morphology_negate` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1199-1206`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1199-L1206) |
| `image-transform` | `morphology_opening` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1173-1177`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1173-L1177) |
| `image-transform` | `morphology_smoothing` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:759-766`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L759-L766) |
| `image-transform` | `multiply_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:595-609`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L595-L609) |
| `image-transform` | `multiply_value` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:678-688`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L678-L688) |
| `image-transform` | `normalize` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:700-707`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L700-L707) |
| `image-transform` | `normalize_otsu_median` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1073-1080`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1073-L1080) |
| `image-transform` | `otsu_threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1081-1091`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1081-L1091) |
| `image-transform` | `refine_label` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1158-1172`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1158-L1172) |
| `image-transform` | `regrid` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:584-594`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L584-L594) |
| `image-transform` | `reshape` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1002-1009`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1002-L1009) |
| `image-transform` | `resize` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:610-617`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L610-L617) |
| `image-transform` | `resize_at_center` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1092-1099`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1092-L1099) |
| `image-transform` | `rotate_to_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1116-1130`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1116-L1130) |
| `image-transform` | `select_value` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:979-986`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L979-L986) |
| `image-transform` | `set_mni` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1062-1072`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1062-L1072) |
| `image-transform` | `set_transformation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:659-666`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L659-L666) |
| `image-transform` | `set_translocation` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:640-647`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L640-L647) |
| `image-transform` | `smoothing_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:905-915`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L905-L915) |
| `image-transform` | `sobel_filter` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:897-904`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L897-L904) |
| `image-transform` | `swap_xy` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:807-814`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L807-L814) |
| `image-transform` | `swap_xz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:815-822`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L815-L822) |
| `image-transform` | `swap_yz` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:823-830`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L823-L830) |
| `image-transform` | `threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:854-861`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L854-L861) |
| `image-transform` | `transform` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:916-923`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L916-L923) |
| `image-transform` | `translocate` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:618-628`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L618-L628) |
| `image-transform` | `upper_threshold` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:689-699`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L689-L699) |
| `image-transform` | `upsampling` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:775-782`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L775-L782) |
| `image-transform` | `warp_to_image` | `0-1` | Computation | Synchronous; registration/filtering may exceed timeout. | `view_image::command → variant_image::command/TIPL` | [`view_image.ui:1131-1145`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1131-L1145) |
| `main` | `list_recent` | `0` | Read-only | Immediate list. | `MainWindow::command` | [current implementation](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1914-L1922) |
| `main` | `run_cli` | `1` | Varies | Synchronous CLI action on GUI thread; verify outputs. | `MainWindow::command` | [current implementation](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1924-L1933) |
| `main` | `hub download` | `4` | File creation | Deferred file write; verify path and stable size. | `MainWindow::command` | [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `main` | `hub files` | `2-3` | GUI-state change | Immediate list; retry if Hub data is loading. | `MainWindow::command` | [current implementation](https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1980-L1991) |
| `main` | `hub help` | `0` | Read-only | Immediate. | `MainWindow::command` | [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `main` | `hub open` | `3` | File creation | Deferred: handler may schedule the open after `OKAY`; poll `LIST`. | `MainWindow::command` | [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `main` | `hub repos` | `0` | GUI-state change | Immediate unless Hub initialization itself is still loading. | `MainWindow::command` | [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `main` | `hub tags` | `1` | GUI-state change | Immediate list; retry if output says loading. | `MainWindow::command` | [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937) |
| `parameters` | `list_param` | `1` | Read-only | Immediate. | `tracking_window::command` | [parameter commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L889-L901) |
| `parameters` | `set_param` | `2` | GUI-state change | Immediate state mutation. | `tracking_window::command` | [current implementation](https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L908-L922) |
| `parameters` | `set_params` | `1` | GUI-state change | Applies multiple values, then requests one redraw. | `tracking_window::command` | [current implementation](https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L908-L922) |
| `region-action` | `region_action_1st_ex_all` | `1` | Destructive | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_all_ex_1st` | `1` | Destructive | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_all_inter_1st` | `1` | Destructive | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_all_to_1st` | `1` | Destructive | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_closing` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_defragment` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_dilation` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_dilation_by_threshold` | `2` | Destructive | Synchronous computation. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_dilation_by_voxel` | `2` | Destructive | Synchronous computation. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_erosion` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_erosion_by_threshold` | `2` | Destructive | Synchronous computation. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_flipx` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_flipy` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_flipz` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_negate` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_opening` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_refine_all` | `1` | Destructive | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_separate` | `1` | Destructive | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_shiftnx` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_shiftny` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_shiftnz` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_shiftx` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_shifty` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_shiftz` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_smoothing` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::do_action` | [basic ROI actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331) |
| `region-action` | `region_action_sort_name` | `1` | GUI-state change | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_sort_size` | `1` | GUI-state change | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_sort_x` | `1` | GUI-state change | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_sort_y` | `1` | GUI-state change | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_sort_z` | `1` | GUI-state change | Synchronous; refresh `list_region`. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_threshold` | `2` | Destructive | Synchronous computation. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-action` | `region_action_threshold_current` | `2` | Destructive | Synchronous computation. | `RegionTableWidget::do_action` | [advanced region actions](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642) |
| `region-create` | `list_region` | `0` | Read-only | Immediate. | `RegionTableWidget::command` | [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `region-create` | `new_region` | `0` | GUI-state change | Immediate. | `RegionTableWidget::command` | [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `region-create` | `new_region_from_mni` | `1` | Computation | Synchronous. | `RegionTableWidget::command` | [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `region-create` | `new_region_from_sphere` | `1` | Computation | Synchronous. | `RegionTableWidget::command` | [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `region-create` | `new_region_from_threshold` | `1` | Computation | Synchronous computation. | `RegionTableWidget::command` | [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `region-create` | `new_region_whole_brain_seed` | `0-1` | Computation | Synchronous computation. | `RegionTableWidget::command` | [region create/list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440) |
| `region-io` | `load_region_color` | `1` | GUI-state change | Synchronous; verify save output. | `RegionTableWidget::command` | [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `region-io` | `open_mni_region` | `1` | GUI-state change | Synchronous file load. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-io` | `open_region` | `1` | GUI-state change | Synchronous file load. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-io` | `save_4d_region` | `1` | File creation | Synchronous; verify outputs. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-io` | `save_all_regions` | `1` | File creation | Synchronous; verify outputs. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-io` | `save_all_regions_to_folder` | `1` | File creation | Synchronous; verify outputs. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-io` | `save_region` | `1-2` | File creation | Synchronous; verify output. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-io` | `save_region_color` | `1` | File creation | Synchronous; verify save output. | `RegionTableWidget::command` | [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `region-io` | `save_region_info` | `1-2` | File creation | Synchronous; verify output. | `RegionTableWidget::command` | [region save/open](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701) |
| `region-manage` | `check_all_regions` | `0` | GUI-state change | Immediate. | `RegionTableWidget::command` | [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `region-manage` | `check_region` | `1-2` | GUI-state change | Immediate; refresh `list_region`. | `RegionTableWidget::command` | [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `region-manage` | `copy_region` | `0-1` | GUI-state change | Immediate; refresh list. | `RegionTableWidget::command` | [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `region-manage` | `delete_all_regions` | `0` | Destructive | Immediate. | `RegionTableWidget::command` | [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `region-manage` | `delete_region` | `0-1` | Destructive | Immediate. | `RegionTableWidget::command` | [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `region-manage` | `merge_regions` | `0-1` | Destructive | Synchronous. | `RegionTableWidget::command` | [atlas/merge/delete](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921) |
| `region-manage` | `move_down_region` | `1` | GUI-state change | Immediate; refresh `list_region`. | `RegionTableWidget::command` | [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `region-manage` | `move_region` | `1-2` | GUI-state change | Immediate. | `RegionTableWidget::command` | [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `region-manage` | `move_slice_to_region` | `0-1` | GUI-state change | Immediate. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-manage` | `move_up_region` | `1` | GUI-state change | Immediate; refresh `list_region`. | `RegionTableWidget::command` | [region selection/move](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529) |
| `region-manage` | `set_region_color` | `1` | GUI-state change | Immediate redraw. | `RegionTableWidget::command` | [`set_region_color`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L902-L910) |
| `region-manage` | `uncheck_all_regions` | `0` | GUI-state change | Immediate. | `RegionTableWidget::command` | [region colors](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768) |
| `region-stats` | `save_device_statistics` | `0-1` | File creation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `save_region_statistics` | `0-1` | File creation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `save_t2r` | `0-1` | File creation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `save_tract_recognition` | `0-2` | File creation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `save_tract_statistics` | `0-2` | File creation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `show_device_statistics` | `0-1` | Computation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `show_region_statistics` | `0-1` | Computation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `show_t2r` | `0-1` | Computation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `show_tract_recognition` | `0-2` | Computation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `region-stats` | `show_tract_statistics` | `0-2` | Computation | Synchronous computation; verify file when path supplied. | `RegionTableWidget::command` | [region statistics](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038) |
| `render` | `open_camera` | `1` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `restore_camera1` | `0` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `restore_camera2` | `0` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `restore_camera3` | `0` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `restore_camera4` | `0` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `rotate` | `1` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_3view_screen` | `1` | File creation | Synchronous; verify image. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_camera` | `1` | File creation | Synchronous; verify output. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_h3view_screen` | `1` | File creation | Synchronous; verify image. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_hd_screen` | `2` | File creation | Synchronous; verify image dimensions. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_rotation_video` | `1` | File creation | Broken; never use as proof of file creation. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_screen` | `1` | File creation | Synchronous; verify image. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `save_v3view_screen` | `1` | File creation | Synchronous; verify image. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `set_camera` | `1` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `set_stereoscopic` | `0` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `set_view` | `1` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `set_zoom` | `1` | GUI-state change | Immediate redraw. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `store_camera1` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `store_camera2` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `store_camera3` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `render` | `store_camera4` | `0` | GUI-state change | State is stored immediately, but the modal must be dismissed. | `GLWidget::command` | [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453) |
| `slice` | `add_mni_slice` | `1` | Computation | Load may start asynchronous registration; poll `list_slice` and `LOG`. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `slice` | `add_slice` | `1` | Computation | Load may start asynchronous registration; poll `list_slice` and `LOG`. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `slice` | `delete_slice` | `0-1` | Destructive | Immediate. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `slice` | `enable_slice` | `0-1` | GUI-state change | Immediate redraw. | `tracking_window::command` | [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `slice` | `list_slice` | `0` | Read-only | Immediate. | `tracking_window::command` | [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `slice` | `move_slice` | `0-1` | GUI-state change | Immediate redraw. | `tracking_window::command` | [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `slice` | `open_slice_mapping` | `1-2` | GUI-state change | Synchronous; verify file for save commands. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `slice` | `save_roi_screen` | `1` | File creation | Synchronous; verify output. | `tracking_window::command` | [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `slice` | `save_slice_image` | `2` | File creation | Synchronous; verify output. | `tracking_window::command` | [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `slice` | `save_slice_mapping` | `1-2` | File creation | Synchronous; verify file for save commands. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `slice` | `save_slice_mni_image` | `2` | File creation | Synchronous; verify output. | `tracking_window::command` | [slice controls](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499) |
| `slice` | `save_slice_volume` | `1-2` | File creation | Synchronous; verify file for save commands. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `slice` | `set_roi_view` | `1` | GUI-state change | Immediate; an invalid integer silently changes nothing. | `tracking_window::command` | [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `slice` | `set_slice` | `0-1` | GUI-state change | Selection is immediate; derived data may remain asynchronous. | `tracking_window::command` | [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `slice` | `set_slice_by_name` | `1` | GUI-state change | Immediate. | `tracking_window::command` | [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `slice` | `set_slice_contrast` | `0-2` | GUI-state change | Immediate redraw. | `tracking_window::command` | [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `slice` | `set_slice_dir_color` | `0-2` | GUI-state change | Immediate redraw. | `tracking_window::command` | [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `slice` | `set_slice_overlay` | `0-2` | GUI-state change | Immediate redraw. | `tracking_window::command` | [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `slice` | `set_slice_stay` | `0-2` | GUI-state change | Immediate redraw. | `tracking_window::command` | [slice display commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910) |
| `slice` | `skull_strip_slice` | `0-1` | Computation | Synchronous computation; may time out. | `tracking_window::command` | [custom-slice commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034) |
| `surface` | `add_surface` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_anterior` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_anterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_left` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_left_anterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_left_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_left_posterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_posterior` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_posterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_right` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_right_anterior_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_right_lower` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `surface` | `add_surface_upper` | `0-2` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [surface commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157) |
| `tracking-files` | `correct_bias_field` | `0` | Computation | Synchronous computation; may exceed client timeout. | `tracking_window::command` | [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `tracking-files` | `open_fib` | `1` | GUI-state change | Synchronous load; then refresh `LIST`. | `tracking_window::command` | [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `tracking-files` | `open_mapping` | `1` | GUI-state change | Synchronous file load. | `tracking_window::command` | [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `tracking-files` | `save_fib_as` | `1` | File creation | Synchronous; verify the output file. | `tracking_window::command` | [tracking file commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188) |
| `tracking-files2` | `load_rendering_setting` | `1` | GUI-state change | Synchronous. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `load_setting` | `1` | GUI-state change | Synchronous. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `load_tracking_setting` | `1` | GUI-state change | Synchronous. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `load_workspace` | `1` | Destructive | Synchronous file load. | `tracking_window::command` | [workspace commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618) |
| `tracking-files2` | `presentation_mode` | `0` | GUI-state change | Immediate. | `tracking_window::command` | [workspace commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618) |
| `tracking-files2` | `restore_rendering` | `0` | GUI-state change | Immediate. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `restore_tracking` | `0` | GUI-state change | Immediate. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `save_rendering_setting` | `1` | File creation | Synchronous; verify file. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `save_setting` | `1` | File creation | Synchronous; verify file. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `save_tracking_setting` | `1` | File creation | Synchronous; verify file. | `tracking_window::command` | [settings commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733) |
| `tracking-files2` | `save_workspace` | `1` | File creation | Synchronous and potentially large; verify directory contents. | `tracking_window::command` | [workspace commands](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618) |
| `tract-color` | `color_all_cluster` | `0` | GUI-state change | Immediate redraw. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-color` | `load_cluster_color` | `1` | GUI-state change | Synchronous. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-color` | `load_cluster_values` | `1` | GUI-state change | Synchronous. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-color` | `load_tract_color` | `1-2` | GUI-state change | Synchronous. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-color` | `load_tract_values` | `1-2` | GUI-state change | Synchronous. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-color` | `save_cluster_color` | `1` | File creation | Synchronous. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-color` | `save_tract_color` | `1-2` | File creation | Synchronous. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-color` | `select_cluster_color` | `1-2` | GUI-state change | Immediate redraw. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-discovery` | `list_tract` | `0` | Read-only | Immediate snapshot. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-discovery` | `load_tract_atlas` | `0-1` | Computation | Synchronous mapping/computation; may time out. | `TractTableWidget::command` | [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `tract-discovery` | `open_mni_tract` | `1-2` | GUI-state change | Synchronous file load. | `TractTableWidget::command` | [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `tract-discovery` | `open_tract` | `1-2` | GUI-state change | Synchronous file load. | `TractTableWidget::command` | [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `tract-discovery` | `open_tract_name` | `1` | GUI-state change | Immediate. | `TractTableWidget::command` | [tracking/open/atlas](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682) |
| `tract-discovery` | `set_dt_index` | `2` | GUI-state change | Immediate. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_by_x` | `0-1` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_by_x2` | `0-1` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_by_y` | `0-1` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_by_y2` | `0-1` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_by_z` | `0-1` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_by_z2` | `0-1` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_end_portion` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_lps_end` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `cut_tract_rai_end` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `delete_branch` | `0` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `flip_tract_x` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `flip_tract_y` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `flip_tract_z` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `redo_tract` | `0` | GUI-state change | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `trim_tract` | `0` | Destructive | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-edit` | `undo_tract` | `0` | GUI-state change | Synchronous parallel edit. | `TractTableWidget::command` | [tract discovery/editing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561) |
| `tract-io` | `endpoint_to_region` | `0-1` | Computation | Synchronous; refresh `list_region`. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_all_tracts` | `1` | File creation | Synchronous; verify output(s). | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_all_tracts_to_folder` | `1` | File creation | Synchronous; verify output(s). | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_mni_tract` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_mni_tract_endpoint` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_slice_tract` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_slice_tract_endpoint` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_tdi` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `tract-io` | `save_tdi2` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `tract-io` | `save_template_tract` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_tract` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_tract_endpoint` | `1-2` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `save_tract_values` | `2-3` | File creation | Synchronous; verify output. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-io` | `tract_to_region` | `0-1` | Computation | Synchronous; refresh `list_region`. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-manage` | `check_tract` | `2` | GUI-state change | Immediate. | `TractTableWidget::command` | [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `tract-manage` | `check_uncheck_all_tract` | `0-1` | GUI-state change | Immediate. | `TractTableWidget::command` | [TDI and checks](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452) |
| `tract-manage` | `copy_tract` | `0-1` | GUI-state change | Synchronous. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-manage` | `delete_all_tracts` | `0` | Destructive | Immediate. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-manage` | `delete_tract` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-manage` | `filter_tract` | `0-1` | Destructive | Synchronous. | `TractTableWidget::command` | [tract filtering/manage](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972) |
| `tract-manage` | `update_tract` | `0-1` | GUI-state change | Immediate. | `TractTableWidget::command` | [tract save/region conversion](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881) |
| `tract-process` | `cluster_tract_by_em` | `1-2` | Destructive | Synchronous computation; refresh list. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-process` | `cluster_tract_by_hy` | `1-2` | Destructive | Synchronous computation; refresh list. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-process` | `cluster_tract_by_km` | `1-2` | Destructive | Synchronous computation; refresh list. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-process` | `cluster_tract_by_label` | `1-2` | Destructive | Synchronous computation; refresh list. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-process` | `delete_repeated_tract` | `0-1` | Destructive | Synchronous computation. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `delete_tract_by_length` | `0-1` | Destructive | Synchronous computation. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `merge_all_tracts` | `0` | Destructive | Synchronous; refresh list. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `merge_tract_by_name` | `0` | Destructive | Synchronous; refresh list. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `recognize_and_cluster_tract` | `1-2` | Destructive | Synchronous computation; refresh list. | `TractTableWidget::command` | [tract clustering](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178) |
| `tract-process` | `recognize_and_rename_tract` | `0` | Destructive | Synchronous; refresh list. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `reconnect_tract` | `1-2` | Destructive | Synchronous computation. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `resample_tract` | `0-1` | Destructive | Synchronous computation. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `separate_deleted_tract` | `1` | Destructive | Synchronous computation. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `tract-process` | `sort_tract_by_name` | `0` | GUI-state change | Synchronous; refresh list. | `TractTableWidget::command` | [tract processing](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389) |
| `unet` | `list_unet` | `0` | Read-only | Immediate after model-menu refresh. | `tracking_window::command` | [atlas and slice lists](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314) |
| `unet` | `segment_brain` | `1` | Computation | Synchronous computation and download; likely to exceed five seconds. Verify with `list_region`. | `tracking_window::command` | [`segment_brain`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L315-L437) |

## End-to-end AI-agent workflows

The examples below use the `Invoke-Dsi` helper and fresh IDs from `LIST`.

### 1. Open a local FIB and wait for its tracking window

1. Confirm the absolute input path exists.
2. Send it as a raw filename to the running instance.
3. If the reply is `BUSY`, wait and retry only after the current user-approved
   operation is known to have finished. If it is `TIMEOUT`, inspect `LIST` and
   `LOG` before any retry.
4. Poll `LIST` with bounded backoff until a new `tracking` row whose title
   matches the input appears.

```powershell
$fib = 'E:\data\subject01.fz'
if (-not (Test-Path -LiteralPath $fib)) { throw "Missing $fib" }
$open = Invoke-DsiRequest $fib
if ($open -notmatch '^(OKAY|BUSY)') { throw $open }

$deadline = (Get-Date).AddMinutes(2)
do {
    Start-Sleep -Milliseconds 500
    $listReply = Read-DsiTextReply (Invoke-Dsi @('LIST'))
    $rows = $listReply.Text -split '\r?\n' | Select-Object -Skip 1 |
        ForEach-Object {
            $c = $_ -split "`t",3
            [pscustomobject]@{Type=$c[0];Id=$c[1];Title=$c[2]}
        }
    $target = $rows | Where-Object { $_.Type -eq 'tracking' -and $_.Title -like '*subject01*' } |
        Select-Object -First 1
} until ($target -or (Get-Date) -gt $deadline)
if (-not $target) { throw 'Tracking window did not become ready' }
$trackingId = $target.Id
```

### 2. Hub discovery, download, and open

1. Resolve `$mainId`.
2. Run `hub repos`; select the exact repository value.
3. Run `hub tags`; select the exact tag.
4. Run `hub files`; select the exact filename and inspect its displayed size.
5. Ask for confirmation if the download is large.
6. For `hub download`, confirm the destination does not already exist; invoke;
   then poll `Test-Path` until size and last-write time are stable.
7. For `hub open`, poll `LIST` for a new tracking/image window because the open
   may be deferred.

```powershell
Invoke-Dsi -Fields @('CMD',$mainId,'hub','repos')
Invoke-Dsi -Fields @('CMD',$mainId,'hub','tags','owner/repository')
Invoke-Dsi -Fields @('CMD',$mainId,'hub','files','owner/repository','v1','CST')
Invoke-Dsi -Fields @('CMD',$mainId,'hub','download','owner/repository','v1','CST.fz','E:\hub')
```

### 3. Select a slice and export it

1. Run `list_slice`; choose the exact zero-based row and name.
2. Run `set_slice`.
3. Run `list_slice` again and require `current=1` for that row.
4. Confirm the new output path.
5. Run `save_slice_image` with separate output and name fields.
6. Require `Test-Path`, nonzero size, and no error in the command reply/log.

```powershell
Invoke-Dsi -Fields @('CMD',$trackingId,'list_slice')
Invoke-Dsi -Fields @('CMD',$trackingId,'set_slice','2')
Invoke-Dsi -Fields @('CMD',$trackingId,'save_slice_image','E:\out\qa.nii.gz','qa')
```

### 4. Run an eligible UNet segmentation

1. Select the intended anatomical slice and wait for it to be ready.
2. Run `list_unet`.
3. Reject TumorSynth unconditionally.
4. Select a row with `available=1`; show the user model/name/description and
   confirm any download/expensive inference.
5. Record `list_region` before the command.
6. Run `segment_brain`. Treat `TIMEOUT` as unknown.
7. Poll `LOG` and `list_region` until the expected label rows appear; report
   actual rows, not merely `OKAY`.

### 5. Add atlas regions safely

1. Run `list_atlas` and select exact template/atlas IDs.
2. Obtain label IDs from a trusted, version-matched source. The present API
   cannot discover them; stop rather than guess.
3. Record `list_region`.
4. Run `add_region_from_atlas` with one packed field such as `0 1 18&19`.
5. Verify the added rows with `list_region`.

### 6. Automatic tractography

1. Run `enable_auto_tract`, then `list_auto_tract`.
2. Select an exact returned tract name.
3. If applying ROI constraints, run `list_region` and build the grammar from
   current indices; for example, `18:0&21:1`.
4. Confirm the potentially expensive computation.
5. Record `list_tract`, then run `run_auto_track`.
6. Poll `list_tract` and `LOG`. Do not declare success until a new row exists
   and its counts stop changing over several polls; disclose that the API lacks
   a definitive job state.
7. Save to a new path and verify the file only after the user accepts the
   resulting counts.

### 7. Manual tracking with current GUI parameters

1. Save tracking settings to a temporary/new `.ini` if reproducibility matters.
2. Obtain the opaque parameter ID from DSI Studio's recorded command/history;
   do not synthesize its format.
3. Run `list_region`; create the ROI grammar.
4. Confirm computation, then run `run_tracking`.
5. Monitor as in automatic tracking.

### 8. Reproducible rendering and screenshot

1. Save the starting camera with `save_camera`.
2. Read each parameter to change with `list_param`.
3. Apply bounded `set_param` values using the complete schema.
4. Run `update_tract` if cached tract geometry/color is involved.
5. Set a known camera using `set_camera`; avoid alternating `set_view` calls.
6. Save the screenshot to a new path and verify dimensions/file size.
7. Restore parameters and camera when the user asked for a temporary view.

### 9. Save and load a workspace

Saving creates many files. Confirm destination and run `save_workspace`; verify
the expected subdirectories and `commands.csv`. Loading may replace current
tracts, regions, and devices. Show `list_tract` and `list_region`, obtain
explicit destructive confirmation, then run `load_workspace` and refresh all
lists.

### 10. Transform and save an image

1. Resolve the exact image window by title.
2. Save to a new backup/output before destructive transforms.
3. Run one transform per request and inspect its reply/log.
4. Registration commands may exceed five seconds; do not retry on timeout
   without inspecting the GUI and logs.
5. Save to a new absolute path and verify it.
6. If the image window was opened from a multi-file group, be prepared for the
   save-time “apply to other images” modal; unattended batch saving is unsafe.

## Missing commands recommended for AI control

These are source-level recommendations, not currently supported commands. Each
proposal names the exact current file and line at commit `9e00c9c23f49df581a78bc1c9928134d262092ad`; do not attempt
to invoke the proposed syntax until the code is implemented and rebuilt.

### P0 — validate `delete_slice` before indexing

**Revise:** [`tracking/tracking_window_action.cpp:1022-1034`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1022-L1034), function `tracking_window::command()`, before dereferencing
`slices[slice_index]`.

**Reason:** a remote invalid index can access outside the vector and crash the
process. This is a safety fix, not a new command.

```cpp
size_t slice_index = run->from_cmd(1,ui->SliceModality->currentIndex());
if(slice_index >= slices.size())
    return run->failed("invalid slice index " + cmd[1]);
auto custom_slice = std::dynamic_pointer_cast<CustomSliceModel>(slices[slice_index]);
```

### P0 — terminal tracking result and indexed `cancel_tracking`

`list_tract` now exposes each row's `running` flag. The remaining gap is a
terminal result/error and a way to stop one job. Preserve the final status when
`ThreadData` ends, add `state` and `error` columns to `list_tract`, and insert
the following branch after the current list handler
([current list handler](https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L487-L500)):

```cpp
if(cmd[0] == "cancel_tracking") {
    int row = currentRow();
    if(!get_cur_row(cmd[1],row) || !thread_data[size_t(row)])
        return run->failed("not running");
    thread_data[size_t(row)]->end_thread();
    return run->succeed();
}
```

Until a terminal field exists, report only `running`/`idle`; never infer
`completed` from `running=0`.

### P0 — `list_atlas_label`

**Insert:** [`tracking/tracking_window_action.cpp:189-205`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L205), function `tracking_window::command()`, between `list_atlas` and
`list_slice` (after current line 197).

**Syntax:** `list_atlas_label <template_id> <atlas_id>` as two parameter fields.

**Output:** `label<TAB>name`, where `label` is the exact zero-based ID accepted
by `add_region_from_atlas`.

```cpp
if(cmd[0] == "list_atlas_label") {
    size_t template_id = QString::fromStdString(cmd[1]).toULongLong();
    size_t atlas_id = QString::fromStdString(cmd[2]).toULongLong();
    if(template_id != handle->template_id || atlas_id >= handle->atlas_list.size())
        return run->failed("invalid template/atlas index");
    tipl::out() << "label\tname";
    const auto& labels = handle->atlas_list[atlas_id]->get_list();
    for(size_t i = 0;i < labels.size();++i)
        tipl::out() << i << "\t" << labels[i];
    return run->succeed();
}
```

**Reason:** this closes the only unsafe discovery gap in the atlas-to-region
workflow. The GUI already enumerates the same list at
[`AtlasDialog` label list](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/atlasdialog.cpp#L31-L31).

### P1 — `get_param`, `list_render_param`, and `list_tracking_param`

**Revise/insert:** [`tracking/tracking_window_action.cpp:889-901`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L889-L901), function `tracking_window::command()`, immediately
before the current `list_param` branch. Keep `list_param` as a compatibility
alias for `get_param`.

**Syntax/output:**

```text
get_param <name>                  -> name<TAB>value
list_render_param                 -> name<TAB>value rows
list_tracking_param               -> name<TAB>value rows
```

```cpp
if(cmd[0] == "get_param" || cmd[0] == "list_param") {
    if(cmd[1].empty())
        return run->failed("missing parameter name");
    tipl::out() << cmd[1] << "\t" << (*this)[cmd[1].c_str()].toString().toStdString();
    return run->succeed();
}
if(cmd[0] == "list_render_param" || cmd[0] == "list_tracking_param") {
    QStringList roots = cmd[0] == "list_tracking_param"
        ? QStringList{"Tracking","Tracking_dT","Tracking_adv"}
        : QStringList{"ROI","Rendering","Slice","Tract","Region","Surface","Device","Label","ODF"};
    for(const auto& root : roots)
        for(const auto& name : renderWidget->treemodel->get_param_list(root))
            tipl::out() << name.toStdString() << "\t"
                        << (*this)[name].toString().toStdString();
    return run->succeed();
}
```

**Reason:** exhaustive discovery and readback are necessary before bounded
parameter mutation. Also validate names and return `ERROR` for unknown IDs.

### P1 — `get_camera` and nonmodal camera slots

**Insert:** [`opengl/glwidget.cpp:2280-2294`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2280-L2294), function `GLWidget::command()`, after the local `get_camera`
lambda and before `open_camera`.

**Syntax/output:** `get_camera` returns one line containing 16 floats.

```cpp
if(cmd[0] == "get_camera")
    return tipl::out() << get_camera(),run->succeed();
```

**Revise:** [`opengl/glwidget.cpp:2313-2318`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2313-L2318), same function, remove the modal `QMessageBox` from
`store_camera*` and return `run->succeed()` rather than `run->canceled()`.

**Reason:** An AI agent can snapshot and restore a camera without temporary files or
blocking the GUI.

### P1 — `list_device`

**Insert:** [`tracking/devicetablewidget.cpp:500-526`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L526), function `DeviceTableWidget::command()`, before `new_device`.

**Syntax/output:** `list_device` returns
`index<TAB>shown<TAB>name<TAB>type<TAB>x<TAB>y<TAB>z<TAB>phi<TAB>theta`.

```cpp
if(cmd[0] == "list_device") {
    tipl::out() << "index\tshown\tname\ttype\tx\ty\tz\tphi\ttheta";
    for(int row = 0;row < rowCount();++row)
        tipl::out() << row << "\t" << (item(row,0)->checkState()==Qt::Checked)
                    << "\t" << item(row,0)->text().toStdString()
                    << "\t" << item(row,1)->text().toStdString()
                    << "\t" << item(row,4)->text().toStdString()
                    << "\t" << item(row,5)->text().toStdString()
                    << "\t" << item(row,6)->text().toStdString()
                    << "\t" << item(row,7)->text().toStdString()
                    << "\t" << item(row,8)->text().toStdString();
    return run->succeed();
}
```

### P1 — image `get_state`, `list_mat`, display setters, undo, and redo

**Insert:** [`view_image.cpp:77-103`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L77-L103), function `view_image::command()`, before the current empty-image
early return, and expose private state through small accessors if needed.

**Proposed syntax/output:**

```text
get_state
dimension<TAB>voxel_size<TAB>pixel_type<TAB>is_mni<TAB>orientation<TAB>slice<TAB>volume<TAB>zoom<TAB>min<TAB>max

set_image_view <orientation> <slice> <volume> <zoom> <min> <max>
list_mat
index<TAB>name<TAB>rows<TAB>columns<TAB>type
undo_image
redo_image
```

**Reason:** image mutation currently has no remote readback, MAT field names
cannot be discovered, and GUI-only undo/redo slots live at
[image undo/redo slots](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L1264-L1308). Keep `set_image_view` field values explicit and nonmodal; return the
resulting state.

### P1 — global `HELP`/`SCHEMA` and readiness

**Revise:** [`main.cpp:577-603`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/main.cpp#L577-L603), main-server request dispatch, between `LOG` and raw filename
handling. **Implement:** a new `ai_request_schema()` beside
`ai_request_list()` in [`mainwindow.cpp:41-146`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L41-L146).

**Syntax/output:**

```text
HELP
OKAY
LIST
LOG
SCHEMA
CMD<TAB>@agent_id<TAB>window_id<TAB>command<TAB>parameter...

SCHEMA
OKAY
<versioned JSON command schema>

STATE
OKAY
busy<TAB>progress<TAB>main_ready<TAB>window_count
```

```cpp
else if(request == "HELP")
    clientSocket->write("OKAY\nLIST\nLOG\nSTATE\nSCHEMA\nCMD\t...");
else if(request == "STATE")
    ai_request_state(clientSocket);
else if(request == "SCHEMA")
    ai_request_schema(clientSocket);
```

**Reason:** hardcoded clients currently depend on source review, and `LIST`
alone does not say whether models, mappings, Hub metadata, or background work
are ready. Version the schema by commit and describe argument fields, safety,
async behavior, and output columns.

### P1 — job IDs, `jobs`, and `cancel`

**Minimal first insertion:** [`mainwindow.cpp:70-136`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L70-L136), function `ai_request_command()`, where command
execution and replies are centralized. Long term, wrap asynchronous tracking,
downloads, registration, and segmentation in a shared job registry.

**Syntax/output:**

```text
JOBS
job_id<TAB>window_id<TAB>command<TAB>state<TAB>progress<TAB>message

CANCEL<TAB>job_id
OKAY or ERROR<TAB>reason
```

**Reason:** a five-second client timeout is not a job model. Do not implement a
fake registry that only tracks the GUI's global progress flag; start with
tracking IDs, then add typed adapters for Hub replies and model inference.

### P2 — command history without unsafe replay

**Insert:** [`tracking/tracking_window_action.cpp:1158-1160`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1158-L1160), function `tracking_window::command()`, before the unknown-command
return at line 1159. The recorded vector is declared at
[`command_history`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window.h#L22-L31).

**Syntax/output:** `get_history` returns
`index<TAB>command`; `clear_history` clears it after confirmation.

```cpp
if(cmd[0] == "get_history") {
    tipl::out() << "index\tcommand";
    for(size_t i = 0;i < history.commands.size();++i)
        tipl::out() << i << "\t" << history.commands[i];
    return run->succeed();
}
if(cmd[0] == "clear_history")
    return history.commands.clear(),run->succeed();
```

Do **not** add blind `replay_history`. Replaying saved destructive/file
commands without re-resolving indices, paths, readiness, and confirmation is
unsafe. If added later, replay must pass each command back through the normal
validated dispatcher and pause for safety-category confirmation.

### P2 — fix `save_rotation_video`

**Revise:** [`opengl/glwidget.cpp:2405-2451`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2405-L2451), function `GLWidget::command()`, remove the unconditional return
at lines 2407-2410, validate AVI creation, and return an error on encoder/open/
write failure. Prefer a job ID and progress/cancel support before exposing this
to an AI agent.


## Machine-readable appendix

This JSON is generated from the same 310-entry command inventory used for the
human-readable tables. `argument_fields` counts fields after the command field;
ranges such as `0-2` are strings. `async=true` means the handler starts or
defers work, or immediate acknowledgement cannot represent completion.

```json
{
  "source_commit": "ecacbd0478e8b7d383a9cd9a5606cc08e6d78a58",
  "base_audit_commit": "9e00c9c23f49df581a78bc1c9928134d262092ad",
  "server_name": "dsi-studio",
  "request_prefixes": {
    "LIST": {
      "fields": 2,
      "wire": "LIST<TAB>@agent_id",
      "legacy_fields": 1,
      "meaning": "discover targetable windows"
    },
    "LOG": {
      "fields": 2,
      "wire": "LOG<TAB>@agent_id",
      "legacy_fields": 1,
      "meaning": "rolling console history"
    },
    "CMD": {
      "minimum_fields": 4,
      "legacy_minimum_fields": 3,
      "wire_single": "CMD<TAB>@agent_id<TAB>window_id<TAB>command<TAB>parameter...",
      "wire_batch": "CMD<TAB>@agent_id<TAB>window_id<TAB>[[\"command\",\"parameter\"],[\"command\",...]]"
    },
    "raw_filename": {
      "fields": 1,
      "meaning": "forward one absolute path to MainWindow::openFile"
    }
  },
  "statuses": {
    "OKAY": {
      "client_exit": 0,
      "meaning": "handler acknowledged"
    },
    "ERROR": {
      "client_exit": 1,
      "meaning": "handler/route failure"
    },
    "BUSY": {
      "client_exit": 1,
      "meaning": "raw open blocked by global progress"
    },
    "TIMEOUT": {
      "client_exit": 1,
      "meaning": "completion unknown after five seconds"
    },
    "NO_INSTANCE": {
      "client_exit": 1,
      "meaning": "LIST found no server"
    },
    "JSON_BATCH": {
      "client_exit": "0 only when every returned okay is true",
      "meaning": "per-command batch results; execution stops at first failure"
    }
  },
  "window_types": [
    "main",
    "tracking",
    "image"
  ],
  "output_schemas": {
    "LIST": [
      "type",
      "window_id",
      "window_title"
    ],
    "list_atlas": [
      "template",
      "atlas",
      "name",
      "regions"
    ],
    "list_slice": [
      "index",
      "current",
      "name",
      "ready",
      "running",
      "downloaded",
      "registered"
    ],
    "list_unet": [
      "index",
      "available",
      "model",
      "name",
      "description"
    ],
    "list_auto_tract": [
      "name"
    ],
    "list_region": [
      "index",
      "shown",
      "name",
      "type",
      "color",
      "dimension x resolution"
    ],
    "list_tract": [
      "index",
      "running",
      "shown",
      "name",
      "tracts",
      "deleted",
      "seeds"
    ],
    "hub repos": [
      "index",
      "repository"
    ],
    "hub tags": [
      "index",
      "tag"
    ],
    "hub files": [
      "row",
      "filename",
      "display-size",
      "cached"
    ],
    "CMD batch": [
      "index",
      "okay",
      "output",
      "error (failure only)"
    ]
  },
  "agent_session": {
    "id": "stable unique case-sensitive string beginning with @",
    "lifetime": "one agent session",
    "legacy_warning": "requests without an ID share one prompt queue and are unsafe for simultaneous agents"
  },
  "prompt_delivery": {
    "text_reply": "PROMPT<TAB><JSON> immediately after the first status line",
    "batch_reply": "optional prompt property on the last result object",
    "clear_rule": "clear only the matching agent queue after the complete reply is written"
  },
  "commands": [
    {
      "scope": "main",
      "name": "list_recent",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "MainWindow::command",
      "output": "Recent `.sz` and `.fz` paths, one per line.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1914-L1922"
    },
    {
      "scope": "main",
      "name": "run_cli",
      "argument_fields": "1",
      "safety": "Varies",
      "async": false,
      "available": true,
      "handler": "MainWindow::command",
      "output": "CLI progress, warnings, and errors.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1924-L1933",
      "caveat": "One complete DSI Studio command line; --action is required; wildcard or --loop processing may affect many files."
    },
    {
      "scope": "main",
      "name": "hub help",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "MainWindow::command",
      "output": "Usage line.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937"
    },
    {
      "scope": "main",
      "name": "hub repos",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "MainWindow::command",
      "output": "`index<TAB>repository` rows.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937"
    },
    {
      "scope": "main",
      "name": "hub tags",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "MainWindow::command",
      "output": "`index<TAB>tag` rows; may print loading warning.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937"
    },
    {
      "scope": "main",
      "name": "hub files",
      "argument_fields": "2-3",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "MainWindow::command",
      "output": "`row<TAB>filename<TAB>display-size<TAB>cached`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/438176e0aa47139cf54bd5f0c22e69e31e2ff11f/mainwindow.cpp#L1980-L1991"
    },
    {
      "scope": "main",
      "name": "hub open",
      "argument_fields": "3",
      "safety": "File creation",
      "async": true,
      "available": true,
      "handler": "MainWindow::command",
      "output": "Console messages only.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937"
    },
    {
      "scope": "main",
      "name": "hub download",
      "argument_fields": "4",
      "safety": "File creation",
      "async": true,
      "available": true,
      "handler": "MainWindow::command",
      "output": "Console messages only.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937"
    },
    {
      "scope": "tracking-files",
      "name": "open_fib",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console messages.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188"
    },
    {
      "scope": "tracking-files",
      "name": "correct_bias_field",
      "argument_fields": "0",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/progress output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188"
    },
    {
      "scope": "tracking-files",
      "name": "save_fib_as",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188"
    },
    {
      "scope": "tracking-files",
      "name": "open_mapping",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L151-L188"
    },
    {
      "scope": "tracking-files2",
      "name": "presentation_mode",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618"
    },
    {
      "scope": "tracking-files2",
      "name": "save_workspace",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618"
    },
    {
      "scope": "tracking-files2",
      "name": "load_workspace",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L500-L618",
      "caveat": "Confirm immediately before running."
    },
    {
      "scope": "tracking-files2",
      "name": "save_setting",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "save_rendering_setting",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "save_tracking_setting",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "load_setting",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "load_rendering_setting",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "load_tracking_setting",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "restore_rendering",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "tracking-files2",
      "name": "restore_tracking",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L619-L733"
    },
    {
      "scope": "slice",
      "name": "list_slice",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Header `index<TAB>current<TAB>name<TAB>ready<TAB>running<TAB>downloaded<TAB>registered`; flags are `0`/`1`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L198-L221"
    },
    {
      "scope": "slice",
      "name": "set_slice",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": true,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314"
    },
    {
      "scope": "slice",
      "name": "set_slice_by_name",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Error if not found.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910"
    },
    {
      "scope": "slice",
      "name": "enable_slice",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499"
    },
    {
      "scope": "slice",
      "name": "move_slice",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499"
    },
    {
      "scope": "slice",
      "name": "set_roi_view",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910"
    },
    {
      "scope": "slice",
      "name": "set_slice_contrast",
      "argument_fields": "0-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910"
    },
    {
      "scope": "slice",
      "name": "set_slice_dir_color",
      "argument_fields": "0-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Error `canceled` when no valid change occurs.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910"
    },
    {
      "scope": "slice",
      "name": "set_slice_overlay",
      "argument_fields": "0-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Error `canceled` when no valid change occurs.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910"
    },
    {
      "scope": "slice",
      "name": "set_slice_stay",
      "argument_fields": "0-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Error `canceled` when no valid change occurs.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L778-L910"
    },
    {
      "scope": "slice",
      "name": "save_slice_image",
      "argument_fields": "2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499"
    },
    {
      "scope": "slice",
      "name": "save_slice_mni_image",
      "argument_fields": "2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499"
    },
    {
      "scope": "slice",
      "name": "save_roi_screen",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L438-L499"
    },
    {
      "scope": "slice",
      "name": "add_slice",
      "argument_fields": "1",
      "safety": "Computation",
      "async": true,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034"
    },
    {
      "scope": "slice",
      "name": "add_mni_slice",
      "argument_fields": "1",
      "safety": "Computation",
      "async": true,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034"
    },
    {
      "scope": "slice",
      "name": "skull_strip_slice",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034"
    },
    {
      "scope": "slice",
      "name": "save_slice_mapping",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034"
    },
    {
      "scope": "slice",
      "name": "open_slice_mapping",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034"
    },
    {
      "scope": "slice",
      "name": "save_slice_volume",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034"
    },
    {
      "scope": "slice",
      "name": "delete_slice",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L911-L1034",
      "caveat": "The handler does not bounds-check before indexing; use a fresh valid index from `list_slice`."
    },
    {
      "scope": "unet",
      "name": "list_unet",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Header `index<TAB>available<TAB>model<TAB>name<TAB>description`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314"
    },
    {
      "scope": "unet",
      "name": "segment_brain",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress, label, and error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L315-L437",
      "caveat": "Do not use TumorSynth."
    },
    {
      "scope": "atlas",
      "name": "list_atlas",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Header `template<TAB>atlas<TAB>name<TAB>regions`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L189-L314"
    },
    {
      "scope": "atlas",
      "name": "add_region_from_atlas",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Console/errors; created rows appear in `list_region`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L785-L846",
      "caveat": "There is no current label-discovery command; never invent label IDs."
    },
    {
      "scope": "auto",
      "name": "enable_auto_tract",
      "argument_fields": "0",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776"
    },
    {
      "scope": "auto",
      "name": "list_auto_tract",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Header `name`, then exact accepted tract names.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776"
    },
    {
      "scope": "auto",
      "name": "run_auto_track",
      "argument_fields": "1-2",
      "safety": "Computation",
      "async": true,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Immediate start/error output; progress later appears in `LOG`/`list_tract`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L734-L776"
    },
    {
      "scope": "auto",
      "name": "run_tracking",
      "argument_fields": "1-3",
      "safety": "Computation",
      "async": true,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Immediate start/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L451-L460",
      "caveat": "With only a tract name, current GUI tracking settings are used; a next field containing ':' is treated as ROI grammar."
    },
    {
      "scope": "surface",
      "name": "add_surface",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_right",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_left",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_upper",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_anterior",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_posterior",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_right_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_left_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_right_anterior_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_left_anterior_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_left_posterior_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_anterior_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "surface",
      "name": "add_surface_posterior_lower",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L1035-L1157"
    },
    {
      "scope": "parameters",
      "name": "list_param",
      "argument_fields": "1",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "One line: `name: value`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L889-L901"
    },
    {
      "scope": "parameters",
      "name": "set_param",
      "argument_fields": "2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L908-L922"
    },
    {
      "scope": "parameters",
      "name": "set_params",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "tracking_window::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/1e79a4e6d3eb8c61eca1e6e13d92f9770255cf4d/tracking/tracking_window_action.cpp#L908-L922",
      "caveat": "One name=value&name=value field; fragments without '=' are ignored."
    },
    {
      "scope": "region-create",
      "name": "list_region",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Header and rows: `index shown name type color dimension resolution` (tab-separated).",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440"
    },
    {
      "scope": "region-create",
      "name": "new_region",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row appears.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440"
    },
    {
      "scope": "region-create",
      "name": "new_region_whole_brain_seed",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row/progress.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440"
    },
    {
      "scope": "region-create",
      "name": "new_region_from_threshold",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row/progress.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440"
    },
    {
      "scope": "region-create",
      "name": "new_region_from_mni",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440"
    },
    {
      "scope": "region-create",
      "name": "new_region_from_sphere",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L440"
    },
    {
      "scope": "region-manage",
      "name": "check_region",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529"
    },
    {
      "scope": "region-manage",
      "name": "move_up_region",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529"
    },
    {
      "scope": "region-manage",
      "name": "move_down_region",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529"
    },
    {
      "scope": "region-manage",
      "name": "move_region",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L473-L529"
    },
    {
      "scope": "region-manage",
      "name": "set_region_color",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Error `canceled` if no regions exist.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L902-L910",
      "caveat": "It cannot target an arbitrary region index; use only immediately after creating the intended last row."
    },
    {
      "scope": "region-manage",
      "name": "check_all_regions",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768"
    },
    {
      "scope": "region-manage",
      "name": "uncheck_all_regions",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768"
    },
    {
      "scope": "region-manage",
      "name": "copy_region",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921"
    },
    {
      "scope": "region-manage",
      "name": "merge_regions",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Rows removed/updated.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921"
    },
    {
      "scope": "region-manage",
      "name": "delete_region",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Row removed.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921"
    },
    {
      "scope": "region-manage",
      "name": "delete_all_regions",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "All rows removed.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L769-L921"
    },
    {
      "scope": "region-manage",
      "name": "move_slice_to_region",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038"
    },
    {
      "scope": "region-io",
      "name": "save_region",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "save_region_info",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "save_4d_region",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "save_all_regions",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "save_all_regions_to_folder",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "open_region",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "open_mni_region",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "New row/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L530-L701"
    },
    {
      "scope": "region-io",
      "name": "load_region_color",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768"
    },
    {
      "scope": "region-io",
      "name": "save_region_color",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L702-L768"
    },
    {
      "scope": "region-stats",
      "name": "show_region_statistics",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "save_region_statistics",
      "argument_fields": "0-1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "show_device_statistics",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "save_device_statistics",
      "argument_fields": "0-1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "show_t2r",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "save_t2r",
      "argument_fields": "0-1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "show_tract_statistics",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "save_tract_statistics",
      "argument_fields": "0-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "show_tract_recognition",
      "argument_fields": "0-2",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-stats",
      "name": "save_tract_recognition",
      "argument_fields": "0-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::command",
      "output": "Tabular statistics or error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L922-L1038",
      "caveat": "A `show_*` command without an output path opens a modal dialog."
    },
    {
      "scope": "region-action",
      "name": "region_action_flipx",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_flipy",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_flipz",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_shiftx",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_shiftnx",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_shifty",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_shiftny",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_shiftz",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_shiftnz",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_smoothing",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_erosion",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_dilation",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_opening",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_closing",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_defragment",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_negate",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/Regions.cpp#L294-L331"
    },
    {
      "scope": "region-action",
      "name": "region_action_1st_ex_all",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_all_ex_1st",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_all_inter_1st",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_all_to_1st",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_refine_all",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_sort_name",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_sort_x",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_sort_y",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_sort_z",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_sort_size",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_separate",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_dilation_by_voxel",
      "argument_fields": "2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_threshold",
      "argument_fields": "2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_threshold_current",
      "argument_fields": "2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_dilation_by_threshold",
      "argument_fields": "2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "region-action",
      "name": "region_action_erosion_by_threshold",
      "argument_fields": "2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "RegionTableWidget::do_action",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L1299-L1642"
    },
    {
      "scope": "tract-discovery",
      "name": "list_tract",
      "argument_fields": "0",
      "safety": "Read-only",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Header `index running shown name tracts deleted seeds` (tab-separated).",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/21146a6f491a61893a8e4866a03b1e09a75d12cd/tracking/tract/tracttablewidget.cpp#L487-L500"
    },
    {
      "scope": "tract-discovery",
      "name": "set_dt_index",
      "argument_fields": "2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-discovery",
      "name": "open_tract",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "New row/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682"
    },
    {
      "scope": "tract-discovery",
      "name": "open_mni_tract",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "New row/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682"
    },
    {
      "scope": "tract-discovery",
      "name": "open_tract_name",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682"
    },
    {
      "scope": "tract-discovery",
      "name": "load_tract_atlas",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "New rows/progress.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L562-L682"
    },
    {
      "scope": "tract-edit",
      "name": "delete_branch",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "undo_tract",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "redo_tract",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "trim_tract",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_end_portion",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_lps_end",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_rai_end",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "flip_tract_x",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None/count update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "flip_tract_y",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None/count update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "flip_tract_z",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None/count update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_by_x",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_by_x2",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_by_y",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_by_y2",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_by_z",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-edit",
      "name": "cut_tract_by_z2",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L561"
    },
    {
      "scope": "tract-manage",
      "name": "filter_tract",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-manage",
      "name": "copy_tract",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-manage",
      "name": "delete_tract",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-manage",
      "name": "delete_all_tracts",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "All rows removed.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-manage",
      "name": "update_tract",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/render update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-manage",
      "name": "check_tract",
      "argument_fields": "2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452"
    },
    {
      "scope": "tract-manage",
      "name": "check_uncheck_all_tract",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452"
    },
    {
      "scope": "tract-io",
      "name": "save_tract",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_mni_tract",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_template_tract",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_slice_tract",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_tract_endpoint",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_mni_tract_endpoint",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_slice_tract_endpoint",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_tdi",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452"
    },
    {
      "scope": "tract-io",
      "name": "save_tdi2",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1390-L1452"
    },
    {
      "scope": "tract-io",
      "name": "save_tract_values",
      "argument_fields": "2-3",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_all_tracts_to_folder",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "save_all_tracts",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "tract_to_region",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "New region row(s).",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-io",
      "name": "endpoint_to_region",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "New region row(s).",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L683-L881"
    },
    {
      "scope": "tract-color",
      "name": "load_tract_color",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-color",
      "name": "load_tract_values",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-color",
      "name": "save_tract_color",
      "argument_fields": "1-2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L882-L972"
    },
    {
      "scope": "tract-color",
      "name": "load_cluster_color",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-color",
      "name": "load_cluster_values",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-color",
      "name": "save_cluster_color",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-color",
      "name": "select_cluster_color",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-color",
      "name": "color_all_cluster",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-process",
      "name": "cluster_tract_by_label",
      "argument_fields": "1-2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows/counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-process",
      "name": "recognize_and_cluster_tract",
      "argument_fields": "1-2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows/counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-process",
      "name": "cluster_tract_by_hy",
      "argument_fields": "1-2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows/counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-process",
      "name": "cluster_tract_by_km",
      "argument_fields": "1-2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows/counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-process",
      "name": "cluster_tract_by_em",
      "argument_fields": "1-2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows/counts update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L973-L1178"
    },
    {
      "scope": "tract-process",
      "name": "delete_repeated_tract",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "resample_tract",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "delete_tract_by_length",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "separate_deleted_tract",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "reconnect_tract",
      "argument_fields": "1-2",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Counts/rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "recognize_and_rename_tract",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "merge_all_tracts",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "merge_tract_by_name",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "tract-process",
      "name": "sort_tract_by_name",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "TractTableWidget::command",
      "output": "Rows update.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L1180-L1389"
    },
    {
      "scope": "device",
      "name": "new_device",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "New table row.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "move_device",
      "argument_fields": "1-2",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "push_device",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "pull_device",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "copy_device",
      "argument_fields": "0-1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "New row.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "set_acpc",
      "argument_fields": "0",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "Rows/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "delete_device",
      "argument_fields": "0-1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "Row removed.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "delete_all_devices",
      "argument_fields": "0",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "All rows removed.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "device",
      "name": "save_all_devices",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "DeviceTableWidget::command",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675"
    },
    {
      "scope": "render",
      "name": "set_zoom",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "set_view",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "rotate",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "set_stereoscopic",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "None.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "open_camera",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error if unreadable/short.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_camera",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error on write failure.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "set_camera",
      "argument_fields": "1",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error `canceled` when empty/short.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "store_camera1",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Shows a modal information box, then returns `ERROR` `canceled`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453",
      "caveat": "Not safe for an unattended AI agent despite changing state."
    },
    {
      "scope": "render",
      "name": "restore_camera1",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error if slot is empty.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "store_camera2",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Shows a modal information box, then returns `ERROR` `canceled`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453",
      "caveat": "Not safe for an unattended AI agent despite changing state."
    },
    {
      "scope": "render",
      "name": "restore_camera2",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error if slot is empty.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "store_camera3",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Shows a modal information box, then returns `ERROR` `canceled`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453",
      "caveat": "Not safe for an unattended AI agent despite changing state."
    },
    {
      "scope": "render",
      "name": "restore_camera3",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error if slot is empty.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "store_camera4",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Shows a modal information box, then returns `ERROR` `canceled`.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453",
      "caveat": "Not safe for an unattended AI agent despite changing state."
    },
    {
      "scope": "render",
      "name": "restore_camera4",
      "argument_fields": "0",
      "safety": "GUI-state change",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error if slot is empty.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_screen",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error on image-save failure.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_3view_screen",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error on image-save failure.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_h3view_screen",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error on image-save failure.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_v3view_screen",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error on image-save failure.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_hd_screen",
      "argument_fields": "2",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "GLWidget::command",
      "output": "Error on image-save failure.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453"
    },
    {
      "scope": "render",
      "name": "save_rotation_video",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": false,
      "handler": "GLWidget::command",
      "output": "Returns `OKAY` when a path is supplied.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453",
      "caveat": "The unconditional return at lines 2407-2410 bypasses all encoding code."
    },
    {
      "scope": "image-core",
      "name": "change_type",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command",
      "output": "None/error.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148"
    },
    {
      "scope": "image-core",
      "name": "save",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib/save path",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L77-L323",
      "caveat": "Confirm overwrite and avoid unattended multi-file saves."
    },
    {
      "scope": "image-core",
      "name": "save_mini",
      "argument_fields": "1",
      "safety": "File creation",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib/save path",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-core",
      "name": "brain_extraction",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command",
      "output": "Download/progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148",
      "caveat": "Image windows have no model-list command. Use only a trusted available stem; never TumorSynth."
    },
    {
      "scope": "image-core",
      "name": "segmentation",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command",
      "output": "Download/progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148",
      "caveat": "Image windows have no model-list command. Use only a trusted available stem; never TumorSynth."
    },
    {
      "scope": "image-core",
      "name": "deface",
      "argument_fields": "1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command",
      "output": "Download/progress/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148",
      "caveat": "Image windows have no model-list command. Use only a trusted available stem; never TumorSynth."
    },
    {
      "scope": "image-transform",
      "name": "regrid",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L584-L594",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "multiply_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L595-L609",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "resize",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L610-L617",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "translocate",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L618-L628",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "crop_to_fit",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L629-L639",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "set_translocation",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L640-L647",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "lower_threshold",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L648-L658",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "set_transformation",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L659-L666",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "add_value",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L667-L677",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "multiply_value",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L678-L688",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "upper_threshold",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L689-L699",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "normalize",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L700-L707",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_edge",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L708-L715",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_edge_xy",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L716-L723",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_edge_xz",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L724-L731",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "add_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L744-L758",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_smoothing",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L759-L766",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "downsampling",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L767-L774",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "upsampling",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L775-L782",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "flip_x",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L783-L790",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "flip_y",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L791-L798",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "flip_z",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L799-L806",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "swap_xy",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L807-L814",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "swap_xz",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L815-L822",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "swap_yz",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L823-L830",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "minus_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L831-L845",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_dilation",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L846-L853",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "threshold",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L854-L861",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_defragment",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L862-L869",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_erosion",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L870-L877",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "mean_filter",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L878-L885",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "gaussian_filter",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L886-L896",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "sobel_filter",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L897-L904",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "smoothing_filter",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L905-L915",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "transform",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L916-L923",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "header_flip_x",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L933-L937",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "header_flip_y",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L938-L942",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "header_flip_z",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L943-L947",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "header_swap_xy",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L948-L952",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "header_swap_xz",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L953-L957",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "header_swap_yz",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L958-L962",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "select_value",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L979-L986",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "concatenate_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L987-L1001",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "reshape",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1002-L1009",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "max_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1010-L1024",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "min_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1025-L1039",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_defragment_by_size",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1040-L1050",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "equation",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1051-L1061",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "set_mni",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1062-L1072",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "normalize_otsu_median",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1073-L1080",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "otsu_threshold",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1081-L1091",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "resize_at_center",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1092-L1099",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "histogram_sharpening",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1100-L1104",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "bias_field_correction",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1105-L1115",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "rotate_to_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1116-L1130",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "warp_to_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1131-L1145",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "apply_to_image",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1146-L1157",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "refine_label",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1158-L1172",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_opening",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1173-L1177",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_closing",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1178-L1182",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_fill_holes",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1183-L1190",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_fill_holes_by_slice",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1191-L1198",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-transform",
      "name": "morphology_negate",
      "argument_fields": "0-1",
      "safety": "Computation",
      "async": false,
      "available": true,
      "handler": "view_image::command → variant_image::command/TIPL",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.ui#L1199-L1206",
      "caveat": "Low-level transform semantics are delegated to TIPL when not implemented directly in DSI Studio."
    },
    {
      "scope": "image-mat",
      "name": "mat_remove",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_resize",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_set_name",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_add_string",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_add_float",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_add_int",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_add_short",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_add_int64",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    },
    {
      "scope": "image-mat",
      "name": "mat_set_value",
      "argument_fields": "1",
      "safety": "Destructive",
      "async": false,
      "available": true,
      "handler": "view_image::command → modify_fib",
      "output": "Console/error output.",
      "source": "https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L961-L1035"
    }
  ]
}
```

## Audit report

- **Source commit:** [`9e00c9c23f49df581a78bc1c9928134d262092ad`](https://github.com/frankyeh/DSI-Studio/commit/9e00c9c23f49df581a78bc1c9928134d262092ad)
- **Dispatch implementations inspected:** 11 total:
  - remotely effective: [`MainWindow::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/mainwindow.cpp#L1849-L1937),
    [`tracking_window::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tracking_window_action.cpp#L118-L1160),
    [`GLWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/opengl/glwidget.cpp#L2233-L2453),
    [`TractTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/tract/tracttablewidget.cpp#L451-L1454),
    [`RegionTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/region/regiontablewidget.cpp#L319-L1042),
    [`DeviceTableWidget::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/tracking/devicetablewidget.cpp#L500-L675),
    [`view_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/view_image.cpp#L77-L323),
    [`variant_image::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/cmd/img.cpp#L13-L148), and
    [`modify_fib()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/tracking/fib_data.cpp#L942-L1109);
  - inspected but not remotely targetable:
    [`reconstruction_window::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/reconstruction/reconstruction_window.cpp#L345-L442)
    and [`src_data::command()`](https://github.com/frankyeh/DSI-Studio/blob/9e00c9c23f49df581a78bc1c9928134d262092ad/libs/dsi/image_model.cpp#L767-L835).
- **Operational command names/variants documented:** 310.
- **Rendering/tracking parameters documented:** 185
  (`178` from `options.txt` plus seven top-level visibility
  parameters).

### Behaviors still ambiguous or not machine-verifiable

1. Generic image-transform parsing and exact default behavior are delegated to
   external TIPL code not present in this DSI Studio source snapshot.
2. Tracking has no terminal job result: `list_tract` reports `running`, but
   `running=0` cannot prove completed versus failed/canceled.
3. `OKAY` from raw filename forwarding does not prove `openFile()` succeeded,
   and Hub open/download has deferred final actions.
4. `TIMEOUT` does not distinguish slow synchronous work, a modal dialog, a
   blocked GUI, or a still-running asynchronous operation.
5. Atlas labels, devices, image/MAT state, camera state, and full parameter
   lists are not all discoverable through current commands.
6. Several empty arguments open modal dialogs; the protocol has no modal-state
   or readiness report.
7. `set_param` does not expose validation success and may not trigger every
   cache-specific update that the interactive editor triggers.
8. `set_view` alternates face orientation, `store_camera*` mutates state then
   returns `ERROR`, and `save_rotation_video` reports success without writing.

### Top five command additions

1. Terminal tracking result/error plus indexed `cancel_tracking`.
2. `list_atlas_label`.
3. versioned `HELP`, `SCHEMA`, and `STATE`.
4. `get_param`, `list_render_param`, and `list_tracking_param`.
5. `get_camera` plus image `get_state`/`list_mat` and `list_device`.

TumorSynth is intentionally documented only as **temporarily unavailable** and
is excluded from every runnable workflow.
