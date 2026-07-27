# DSI Studio AI Command Manual

Read `DSI_STUDIO_AI_SETUP.md` completely. Read the operating rules and common
syntax below, then search this manual only for commands needed by the request.
Do not reread the entire file for each action.

## Operating rules

- Each named-pipe connection sends exactly one request and then closes.
- Send JSON for every AI request. Raw non-JSON text is reserved for one existing
  local file path. If DSI Studio rejects direct text, reread this manual and
  resend a valid JSON request.
- Use a native client for `\\.\pipe\dsi-studio` first. Do not use
  `dsi_agent.ps1`, `dsi_studio.exe`, or another wrapper unless direct access is
  unavailable or fails and the user approves the fallback.
- Send separate non-empty `agent` and `session` fields and reuse the exact pair.
  `agent` must include `Codex` or `Claude` and must not contain `@`.
- For a new Codex chat launched by DSI Studio, process the initiating task in
  the first run. There is no bootstrap run and DSI Studio does not launch the
  same task twice. Use `CODEX_THREAD_ID` as `session` immediately; do not wait
  for a second launch or generate another ID.
- DSI Studio also reads `thread.started.thread_id` to store and later resume the
  same Codex thread with `codex exec resume`.
- For an externally initiated Codex Desktop chat, use a task UUID only when it
  is explicitly present in runtime context. Never guess or scan for one.
- For Claude Code, read `~/.claude/sessions/<pid>.json` and use `sessionId`, not
  `name`. DSI Studio resumes it with `claude -p --resume <sessionId>`.
- Ollama is a model provider, not an agent identity.
- After understanding the initiating prompt, send one concise `TITLE`. Send a
  later `TITLE` only when the user permits renaming.
- Call top-level `LIST` first. It does not use a window ID.
- Every `CMD`, including every `list_*` command, requires a numeric `window`
  returned by the latest `LIST`. Never use a type, title, filename, guessed ID,
  or stale ID.
- Use GUI commands. Do not use `run_cli` unless the user explicitly requests
  CLI execution.
- Discover names, indices, and parameter IDs before mutation. Never guess them.
- `okay:true` means the handler accepted the command; asynchronous work may
  still be running.
- Poll the relevant list command after an asynchronous operation. Use `LOG`
  only for failures or states that targeted discovery cannot explain.
- If a required window disappears or returns `window not found`, assume the
  user closed it. Do not reopen or retry it. Ask whether to continue.
- Confirm destructive actions and overwrites. Verify every output file.
- Do not answer modal dialogs remotely; tell the user what action is required.
- Attach concise progress `chat` to an already-needed request. Avoid separate
  status-only requests except for a required decision, blocked/waiting state,
  or the final reply.
- Inspect every complete reply. A queued `PROMPT` may follow `LIST`, `LOG`,
  `CHAT`, `TITLE`, or appear in the last `CMD` result.

## Discovery

| Need | Command |
|---|---|
| Windows | Top-level JSON `LIST` |
| Recent FIB files | Main: `list_recent_fib` |
| Recent SRC files | Main: `list_recent_src` |
| Slices/readiness/registration | Tracking: `list_slice` |
| Regions | Tracking: `list_region` |
| Full tract details | Tracking: `list_tract` |
| Compact tract polling | Tracking: `list_tract status` |
| Valid parameter IDs | Tracking: `list_param` |
| One parameter value | Tracking: `list_param <id>` |
| Atlases | Tracking: `list_atlas` |
| Segmentation models | Tracking: `list_unet` |
| Automatic tract names | Tracking: `list_auto_tract` |
| Incremental diagnostics | Top-level JSON `LOG` |

`list_slice` fields are:

```text
index current name ready registering downloaded registered
```

Built-in/native volumes report `ready=1`. For custom volumes,
`registering=1` means registration is still running; `registered=0` does not
mean the image is broken.

`list_tract status` returns only:

```text
running bundles
```

Use it while tracking is active. Request full `list_tract` after completion
only when bundle details are needed.

`list_param` without an ID lists all valid IDs. Query only needed values after
that discovery call.

## Request formats

### LIST

```json
{"agent":"Codex","session":"<uuid>","cwd":"<path>","request":"LIST"}
```

### CMD

A single command:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["list_region"]}
```

A safe same-window batch:

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":[["list_param"],["list_slice"]]}
```

Do not batch destructive, asynchronous, output-dependent, or modal-opening
commands.

### LOG

```json
{"agent":"Codex","session":"<uuid>","request":"LOG"}
```

`LOG` returns at most 4096 new console characters and advances the session's
cursor. AI-facing `LOG` and `CMD` text has ANSI escape sequences removed.

### CHAT

```json
{"agent":"Codex","session":"<uuid>","request":"CHAT","chat":"Task completed."}
```

Send the final answer once with `CHAT`.

### TITLE

```json
{"agent":"Codex","session":"<uuid>","request":"TITLE","title":"Concise task name"}
```

Another `TITLE` replaces the current title. Send it later only when the user
permits renaming.

### Open one local file

Send one absolute path as raw pipe text. Raw text is accepted only when it
resolves to an existing file.

## Reply formats

- `LIST`, `LOG`, `CHAT`, `TITLE`, file-open replies, and validation errors are
  text.
- `LIST` begins with `OKAY`; following lines are
  `type<TAB>numeric-id<TAB>title`.
- A `CMD` reply is a JSON array of
  `{index,okay,output,error?}` objects.
- List-command data remains tab-separated text inside `output`; do not invent
  properties such as `.windows`, `.tracks`, or `.regions`.
- `CHAT` and `TITLE` return no console history, but either may append a queued
  `PROMPT` after `OKAY`.
- `CMD` may attach `prompt` to its last result object.
- Paths returned by `LIST`, `list_recent_fib`, and `list_recent_src` use `/` as
  the canonical separator. Windows accepts these paths; compare them without
  converting back to `\`.

## Common command syntax

Every command and parameter is a separate JSON string.

| Task | Command array |
|---|---|
| Hub repositories | `["hub","repos"]` |
| Hub tags | `["hub","tags",repo]` |
| Hub files | `["hub","files",repo,tag,filter,offset,limit]` |
| Open Hub file | `["hub","open",repo,tag,filename-or-index]` |
| Download Hub file | `["hub","download",repo,tag,filename-or-index,directory]` |
| Open images together | Main: `["open_image",path1,path2,...]` |
| Select slice | `["set_slice",index]` |
| Select slice by name | `["set_slice_by_name",name]` |
| Move slices | `["move_slice","x y z"]` |
| Segment a slice | `["segment_brain",model,slice]` |
| Add atlas region | `["add_region_from_atlas",region]` |
| Set region name | `["set_region_name",index,name]` |
| Set region type | `["set_region_type",index,type]` |
| Set region color | `["set_region_color",index,color]` |
| Show only regions | `["show_only_regions","0&2&5"]` |
| Show only tracts | `["show_only_tracts","0&2&5"]` |
| List parameter IDs | `["list_param"]` |
| Read parameter | `["list_param",id]` |
| Set parameter | `["set_param",id,value]` |
| Set parameters | `["set_params","id=value&id=value"]` |
| Start tracking | `["run_tracking",name,optional-settings-or-ROI,optional-tolerance]` |
| Compact tracking poll | `["list_tract","status"]` |
| Automatic tracking | `["run_auto_track",tract,optional-ROI]` |
| Rotate 3D view | `["rotate","degrees x y z"]` |
| Save rendering | `["save_hd_screen",path,"width height"]` |
| Run CLI, explicit request only | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` |

`hub files` returns `index`, `file`, `size`, and `downloaded`. For `hub open`
and `hub download`, pass either the exact filename or the numeric index from
that same `hub files` result. Indices apply only to the currently selected
repository and tag.

## File and window workflows

To open one `.fz`, `.sz`, or image when only the main window exists, send its
absolute path as raw pipe text, then call `LIST` to obtain the new numeric
window ID. `open_fib` requires an existing tracking window and cannot create
the first one.

In DSI Studio, FIB means `.fz`; `.sz` is an SRC file.

To open multiple images in one image window, send one flat `open_image` command
to the numeric main-window ID. Do not send separate commands, target an image
window, split a path into fields, or substitute `add_image`.

After a window opens or closes, refresh `LIST`. Otherwise reuse the latest
window IDs.

Most Hub FIB files contain an HTTP reference to their native T1w. After opening
the FIB, use `list_slice`, then select the returned T1w entry. DSI Studio will
download and load it automatically.

## Asynchronous work

Tracking is asynchronous. Use `list_tract status` until `running=0`, then call
full `list_tract` only when details are needed.

Segmentation is complete when `list_region` shows the expected output.

Slice loading/registration is complete when `list_slice` reports the expected
`ready`, `registering`, and `registered` state.

If a response says loading is in progress, attach one concise waiting update to
the next required request and poll the relevant list command. Never repeat a
failed, timed-out, unavailable, or unexpected operation automatically.

## Token efficiency

- Process a new Codex task in its first run. Never create or wait for a bootstrap
  run.
- Use `CODEX_THREAD_ID` immediately instead of spending requests discovering or
  replacing the session ID.
- Reuse `LIST` until windows change.
- Use parameterless `list_param` once, then query only needed IDs.
- Poll with `list_tract status`, not full `list_tract`.
- Poll targeted state rather than `LOG`.
- Batch safe independent synchronous commands for one window.
- Attach progress chat to an existing request.
- Send `cwd` once and omit it until it changes.
- Reuse discovered names and indices until relevant state changes.
- Stop after verification and one final `CHAT`.

## Safety and verification

Region types are `0=ROI`, `1=ROA`, `2=End`, `3=Seed`, `4=Terminative`,
`5=NotEnd`, and `6=Limiting`. Colors are unsigned packed Qt ARGB integers.

Do not use TumorSynth until its current model bug is fixed.

Obtain permission before overwriting files. Do not answer confirmation dialogs
remotely. Verify expected output paths, files, tract bundles, regions, and
renderings after completion.

If DSI Studio resumes an agent, reconnect using the exact same `agent` and
`session`, inspect every reply for `PROMPT`, and exit naturally when none
remains.
