# DSI Studio AI Setup

Read this file completely. Search `DSI_STUDIO_AI_MANUAL.md` only for commands
needed by the request; do not read its entire inventory.

## Identity

Use one UUID for the entire AI conversation. Generate or obtain it once when
the session starts, retain it in the agent context, and reuse it exactly.
Never generate a UUID inside an individual command script.

Provider prefixes are `@C` for Codex, `@A` for Claude Code, and another
two-character ID such as `@G` or `@L` for other agents. The client derives the
full agent ID as the provider prefix plus the first 12 UUID characters.

## PowerShell client

Set the identity, then dot-source the shipped client:

```powershell
$DsiProvider = '@A'       # Use the provider ID for the current agent.
$DsiSession = '<stable UUID retained for this conversation>'
. .\DSI_STUDIO_AI_CLIENT.ps1

$list = Invoke-Dsi @{request='LIST'}
```

Repeat the same provider and session values if a later tool call starts a new
PowerShell process. Always use `Invoke-Dsi`; do not reconstruct the pipe code,
create temporary scripts, or send incomplete raw JSON.

Each `Invoke-Dsi` call opens one `dsi-studio` named-pipe connection, sends
exactly one request, reads one complete reply, and closes. Run on the same
Windows computer as DSI Studio. Do not launch `dsi_studio.exe` per command.

## Requests

```powershell
# Discover windows.
$list = Invoke-Dsi @{request='LIST'}

# Use the numeric ID returned by LIST.
$reply = Invoke-Dsi @{
    request='CMD'; window='2'; command=@('list_region')
}

# Ordered same-window batch.
$reply = Invoke-Dsi @{
    request='CMD'; window='2'
    command=@(@('list_slice'),@('list_region'),@('list_tract'))
}

# Incremental diagnostics and final user-facing reply.
$log = Invoke-Dsi @{request='LOG'}
$log = Invoke-Dsi @{request='LOG'; chat='Task completed.'}
```

Always use the numeric window ID returned by the latest `LIST`; never use a
window type, title, filename, guessed ID, or stale ID as `window`.

JSON fields are `agent`, `session`, `cwd`, `request`, `window`, `command`, and
optional `chat`. The client supplies `agent`, `session`, and `cwd`. Requests
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

Use `Invoke-Dsi -File` when only the main window exists:

```powershell
Invoke-Dsi -File 'C:\data\subject.fz'
$list = Invoke-Dsi @{request='LIST'}
```

Poll `LIST` for the new numeric `tracking` or `image` window ID. `open_fib`
requires an existing tracking window and cannot create the first one.

In DSI Studio, **FIB means `.fz`**. Never substitute `.sz`; `.sz` is an SRC
file. `Invoke-Dsi -File` can open one `.fz`, `.sz`, or image file.

To open multiple images in one O1 window, send one flat command to the numeric
main-window ID:

```powershell
Invoke-Dsi @{
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
