# DSI Studio

Before reading any other file, learn by doing these three requests in order:

```powershell
$session = $env:CODEX_THREAD_ID
./dsi_agent.ps1 -Agent Codex -Session $session -Target TITLE "<concise title derived from the user's task>"
./dsi_agent.ps1 -Agent Codex -Session $session -Target CHAT "<brief improvised message that you are reading the manual before continuing>"
./dsi_agent.ps1 -Agent Codex -Session $session -Target main list_recent_fib
```

Derive `TITLE` from the user's task and improvise the first `CHAT`; never copy either placeholder literally. After all three complete, read `DSI_STUDIO_AI_MANUAL.md` and only the relevant examples, then continue the task. Use one wrapper invocation per request. Never search for the session, access or reuse the pipe directly, inspect the wrapper, or launch another shell. `main` is fixed; call `LIST` only for a tracking/image window ID. Update `TITLE` when the task changes substantially, verify completion, and ask before destructive actions.