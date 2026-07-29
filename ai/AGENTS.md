# DSI Studio AI control

This folder contains the files used to control an already-running DSI Studio instance.

When asked to operate DSI Studio:

1. Read `DSI_STUDIO_AI_SETUP.md` completely and follow it.
2. Read `DSI_STUDIO_AI_MANUAL.md` and only the topic-specific example files relevant to the request.
3. Use the documented connection method; do not launch another DSI Studio instance.
4. Call `LIST` before `CMD`, use the exact returned window IDs, and verify completion.
5. Ask before destructive operations.
6. Do not modify the DSI Studio installation unless explicitly requested.

## Codex identity

When running as Codex, use the exact UUID of the current Codex task/thread exposed by its injected runtime context or task-specific runtime path. When DSI Studio launches Codex with JSON output, this is the same `thread_id` reported by `thread.started`. Codex Desktop may expose it as the UUID component of an injected task path such as `...\visualizations\YYYY\MM\DD\<uuid>`. Use only an ID explicitly associated with the current task; do not scan for, guess, or generate one. Send it as `Session` in every `dsi_agent.ps1` invocation.