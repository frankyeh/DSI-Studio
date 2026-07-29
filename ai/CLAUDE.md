# DSI Studio

Use only:

```powershell
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target <TITLE|LIST|LOG|CHAT|window-id> [command/values...]
```

Send `TITLE` first and update it when the task changes substantially. Call `LIST` only when current status or a non-`main` window ID is needed. For commands, `Target` is the exact window ID and the first value is the command. Never access the pipe directly, launch another shell, or inspect the wrapper. Read the manual and relevant examples only as needed. Verify completion and ask before destructive actions.