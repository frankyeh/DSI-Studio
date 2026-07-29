# DSI Studio

Before reading any other file, learn by doing these three requests in order:

```powershell
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target TITLE "<concise title derived from the user's task>"
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target CHAT "I am reading the DSI Studio manual before continuing."
./dsi_agent.ps1 -Agent Claude -Session <SESSION> -Target main list_recent_fib
```

Use the exact session supplied by DSI Studio. Derive the first `TITLE` from the user's actual task; never copy the placeholder literally. After all three complete, read `DSI_STUDIO_AI_MANUAL.md` and only the relevant examples, then continue the task. Use one wrapper invocation per request. Never access or reuse the pipe directly, inspect the wrapper, or launch another shell. `main` is fixed; call `LIST` only for a tracking/image window ID. Update `TITLE` when the task changes substantially, verify completion, and ask before destructive actions.
