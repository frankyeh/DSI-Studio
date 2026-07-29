# DSI Studio

Use only the allowed PowerShell wrapper; it handles the named pipe:

```powershell
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target <LIST|LOG|TITLE|CHAT|window-id> [command/values...]
```

For a command, `Target` is the exact window ID and the first value is the command. Never access the pipe directly, launch `powershell.exe`/`pwsh.exe`, or inspect the wrapper. Read `DSI_STUDIO_AI_MANUAL.md` and relevant examples only as needed. Call `LIST` before commands, send `TITLE` only after the task is known, verify completion, and ask before destructive actions.