# DSI Studio

Use only:

```powershell
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target <TITLE|LIST|LOG|CHAT|window-id> [command/values...]
```

Send `TITLE` first. Call `LIST` only when a task needs current window IDs or status; do not use it merely to announce readiness. For commands, `Target` is the exact current window ID and the first value is the command. Never access the pipe directly, launch another shell, or inspect the wrapper. Read the manual and relevant examples only as needed. Verify completion and ask before destructive actions.