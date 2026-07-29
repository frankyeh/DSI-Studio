# DSI Studio

Use one `dsi_agent.ps1` invocation per request. Start by doing this:

```powershell
$session = $env:CODEX_THREAD_ID
./dsi_agent.ps1 -Agent Codex -Session $session -Target TITLE "Open recent FIB"
./dsi_agent.ps1 -Agent Codex -Session $session -Target main list_recent_fib
```

Adapt the title and command to the task. Never search for the session, access or reuse the named pipe directly, inspect the wrapper, or launch another shell. For Codex, ignore direct-pipe examples in the setup file. `main` is fixed; call `LIST` only for a tracking/image window ID. Read the manual and relevant examples only as needed, update `TITLE` when the task changes substantially, verify completion, and ask before destructive actions.