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
- Call top-level `LIST` first. It returns global activity plus every window's
  numeric ID and busy state; it does not use a window ID.
- Every `CMD`, including every `list_*` command, requires a numeric `window`
  returned by `LIST`. Never use a type, title, filename, guessed ID, or stale ID.
- Use GUI commands. Do not use `run_cli` unless the user explicitly requests
  CLI execution.
- Discover names, indices, and parameter IDs before mutation. Never guess them.
- `okay:true` means the handler accepted the command; asynchronous work may
  still be running.
- Poll top-level `LIST` for global and per-window activity. Use the relevant
  `list_*` command only when detailed state or output verification is needed.
- Use `LOG` only for failures or states that `LIST` and targeted discovery
  cannot explain.
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
| Global and per-window activity | Top-level JSON `LIST` |
| Recent FIB files | Main: `list_recent_fib` |
| Recent SRC files | Main: `list_recent_src` |
| Slices/readiness/registration | Tracking: `list_slice` |
| Regions and ROI types | Tracking: `list_region` |
| Full tract-bundle details | Tracking: `list_tract` |
| Targeted tract polling | Tracking: `list_tract status` |
| Valid tracking/GUI parameter IDs | Tracking: `list_param` |
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

`LIST` already reports the active tracking-job count for every tracking window.
Use `list_tract status` only when a targeted tracking-window reply is useful;
request full `list_tract` after completion when bundle details are needed.

`list_param` without an ID lists all valid IDs. Query only needed values after
that discovery call. Change tracking behavior with `set_param` or `set_params`
before starting tracking.

## Request formats

### LIST

```json
{"agent":"Codex","session":"<uuid>","cwd":"<path>","request":"LIST"}
```

The reply is compact tab-separated text:

```text
OKAY<TAB>busy<TAB>level<TAB>status
type<TAB>id<TAB>busy<TAB>tracking-jobs<TAB>title
...
```

Example:

```text
OKAY	1	2	segment_brain (3/5)
main	1	0	0	DSI Studio
tracking	2	1	0	C:/data/a.fz
tracking	3	1	2	C:/data/b.fz
tracking	4	0	0	C:/data/c.fz
```

The first line reports application-wide state:

- `busy=1` when a TIPL operation or any supported window is busy.
- `level` is the active TIPL nesting depth, excluding the persistent application
  root. It is `1` when only asynchronous work such as fiber tracking is active.
- `status` is the deepest available TIPL status. When asynchronous tracking is
  the only known activity, it is `fiber tracking`.

Each following line reports a window:

- `type` is `main`, `tracking`, or `image`.
- `id` is the numeric value required by `CMD`.
- `busy=1` indicates an active AI command, command sequence, fiber tracking, or
  custom-slice registration associated with that window.
- `tracking-jobs` is the number of active tracking bundles in that window.
- `title` is the normalized window title/path.

Global `busy` can be `1` while all window rows are `0` when TIPL reports work
that cannot be reliably assigned to one window. Reuse the numeric IDs until a
window opens or closes, but call `LIST` while waiting to obtain current activity.

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
commands. Do not send an empty command array.

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
- `LIST` begins with
  `OKAY<TAB>busy<TAB>level<TAB>status`; each following line is
  `type<TAB>numeric-id<TAB>busy<TAB>tracking-jobs<TAB>title`.
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
| Segment current slice | `["segment_brain",model]` |
| Segment selected slice | `["segment_brain",model,slice-name-or-index]` |
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
| Start tracking | `["run_tracking",bundle-name]` |
| Start tracking with regions | `["run_tracking",bundle-name,"region-index:type&region-index:type"]` |
| Targeted tracking poll | `["list_tract","status"]` |
| Automatic tracking | `["run_auto_track",tract-name,optional-ROI]` |
| Rotate 3D view | `["rotate","degrees x y z"]` |
| Save rendering | `["save_hd_screen",path,"width height"]` |
| Run CLI, explicit request only | `["run_cli","--action=rec --loop=C:\\data\\*.sz --method=4"]` |

`hub files` returns `index`, `file`, `size`, and `downloaded`. For `hub open`
and `hub download`, pass either the exact filename or the numeric index from
that same `hub files` result. Indices apply only to the currently selected
repository and tag.

### Brain segmentation

Use `list_unet` to obtain an available model and `list_slice` to obtain the
current slice names and indices. `segment_brain` accepts either form:

```text
["segment_brain","model"]
["segment_brain","model","T1w"]
["segment_brain","model","7"]
```

When the slice argument is omitted, DSI Studio segments the current slice. When
it is supplied, DSI Studio first tries an exact slice-name match and then treats
the value as a numeric index from the latest `list_slice`. Prefer the numeric
index when slice names are duplicated.

The command internally selects the requested slice. If that slice references a
remote image, selection triggers its download and loading. DSI Studio waits for
custom-slice registration before running segmentation, so a separate
`set_slice_by_name` call is not required. To preload and inspect the state
separately, use `set_slice <index>` and poll `LIST` until the target tracking
window is no longer busy, then confirm detailed slice fields with `list_slice`.

`segment_brain` is synchronous. Its `CMD` reply arrives after inference and
region creation finish. While it runs, poll `LIST`; when DSI Studio can process
Qt events, `level` and `status` report its current TIPL progress. A delayed
`LIST` reply alone does not prove that the operation failed. Verify successful
output with `list_region`.

### Fiber tracking

`run_tracking` has no tracking-method argument. DSI Studio uses its tracking
algorithm with the directional information already stored in the loaded FIB
and the current tracking parameters. GQI, DTI, and Q-ball describe how the
FIB's directional information was reconstructed; do not pass them to
`run_tracking`.

The required `bundle-name` is only the label assigned to the new tract bundle.
For example:

```text
["run_tracking","Whole Brain"]
```

This starts tracking using the current settings. To change settings, first use
`list_param` to discover valid IDs, then use `set_param` or `set_params`.

To apply regions, call `list_region` and construct the optional third argument
from returned region indices and types:

```text
["run_tracking","Corticospinal Tract","0:3&1:0&2:1"]
```

Here each item is `region-index:type`. Do not pass a tracking method, FIB
reconstruction method, or tract name from an atlas in this field.

The fourth internal `run_tracking` parameter is reserved for automatic
tracking tolerance. Agents should use `run_auto_track` rather than supplying
that parameter directly.

## File and window workflows

To open one `.fz`, `.sz`, or image when only the main window exists, send its
absolute path as raw pipe text, then call `LIST` to obtain the new numeric
window ID. `open_fib` requires an existing tracking window and cannot create
the first one.

In DSI Studio, FIB means `.fz`; `.sz` is an SRC file.

To open multiple images in one image window, send one flat `open_image` command
to the numeric main-window ID. Do not send separate commands, target an image
window, split a path into fields, or substitute `add_image`.

After a window opens or closes, refresh `LIST`. Otherwise retain the IDs while
using later `LIST` replies for current busy state.

Most Hub FIB files contain an HTTP reference to their native T1w. After opening
the FIB, call `list_slice`. Pass the returned T1w name or index directly to
`segment_brain`, or use `set_slice <index>` first when download and registration
should be observed separately.

## Asynchronous and long-running work

Fiber tracking is asynchronous. A successful `run_tracking` reply means
tracking started. Poll `LIST`; the target row's `tracking-jobs` reaches zero
when no active tracking bundle remains. Call full `list_tract` afterward only
when bundle details are needed.

For a synchronous long-running command, the target window has `busy=1`, and the
global `level`/`status` describe available TIPL progress. Do not repeat the
operation because its original `CMD` connection is still pending. Use `LIST`
for status and process every returned `PROMPT`.

Slice loading and registration make the tracking-window row busy. When it
becomes idle, use `list_slice` if exact `downloaded`, `ready`, `registering`, or
`registered` fields must be verified.

Never automatically repeat a failed, timed-out, unavailable, or unexpected
operation.

## Token efficiency

- Process a new Codex task in its first run. Never create or wait for a bootstrap
  run.
- Use `CODEX_THREAD_ID` immediately instead of spending requests discovering or
  replacing the session ID.
- Retain window IDs until windows change; poll compact top-level `LIST` only
  while waiting for global or per-window activity to change.
- Use the `tracking-jobs` column instead of `list_tract status` when only
  completion is needed.
- Use parameterless `list_param` once, then query only needed IDs.
- Poll targeted detailed state rather than `LOG`.
- Batch safe independent synchronous commands for one window.
- Attach progress chat to an existing request, but do not attach chat to every
  status poll.
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
