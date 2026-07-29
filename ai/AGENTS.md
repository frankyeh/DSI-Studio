# DSI Studio AI control

When asked to operate DSI Studio:

1. Read `DSI_STUDIO_AI_SETUP.md`, then the manual and relevant examples only as needed.
2. Send `TITLE` first after understanding the task.
3. Call `LIST` only when current status or a tracking/image window ID is needed; `main` is fixed.
4. Use exact IDs returned by `LIST`, verify completion, and ask before destructive actions.
5. Do not launch another DSI Studio instance or modify the installation unless requested.

## Codex identity

When running as Codex, use only the exact current task/thread UUID exposed by the runtime or reported as `thread_id` by `thread.started`. Never scan for, guess, or generate it. Send it as `Session` in every `dsi_agent.ps1` invocation.