# DSI Studio AI Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

The complete command inventory is organized by command area so each command has one authoritative documentation location:

- [Main window, Hub, FIB, workspace, settings, and parameters](DSI_STUDIO_AI_COMMAND_EXAMPLES_GENERAL.md)
- [Slices and segmentation](DSI_STUDIO_AI_COMMAND_EXAMPLES_SLICE.md)
- [Regions and tract-to-region analysis](DSI_STUDIO_AI_COMMAND_EXAMPLES_REGION.md)
- [Tracts, tracking, AutoTrack, clustering, recognition, and TDI](DSI_STUDIO_AI_COMMAND_EXAMPLES_TRACT.md)
- [Devices and AC-PC locators](DSI_STUDIO_AI_COMMAND_EXAMPLES_DEVICE.md)
- [Rendering, camera, surfaces, and display](DSI_STUDIO_AI_COMMAND_EXAMPLES_RENDERING.md)
- [Image-window and TIPL generic image operations](DSI_STUDIO_AI_COMMAND_EXAMPLES_IMAGE.md)

Rows with a common example have a recommended or source-verified example. Blank example cells preserve commands from the previous complete manual inventory without inventing argument syntax; inspect current source before using those commands.

## `chat` with `CMD`

For a meaningful command, include a useful top-level `chat` update that tells the user what was verified and what the command is about to do.

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","SynthSeg V2","7"],"chat":"I verified that the T1w slice is loaded and ready. I’m starting SynthSeg now; it may take a while to finish."}
```

The `chat` field is shown to the user and does not alter command execution. Routine polling and trivial discovery commands may omit it to avoid repetitive updates.
