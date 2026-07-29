@AGENTS.md

# Claude Code

Use the PowerShell tool to communicate with DSI Studio through:

```powershell
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target <TARGET> <VALUES...>
```

Invoke `./dsi_agent.ps1` directly. Never wrap it in `powershell.exe` or `pwsh.exe`.

`Agent`, `Session`, and `Target` are mandatory. Use the exact Claude session UUID supplied by the current DSI Studio process. Use the wrapper instead of constructing a separate named-pipe client.
