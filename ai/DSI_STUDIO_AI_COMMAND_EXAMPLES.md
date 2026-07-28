# DSI Studio AI Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

Examples are organized by command area so each command has one authoritative documentation location:

- [General, workspace, FIB, mapping, and settings](DSI_STUDIO_AI_COMMAND_EXAMPLES_GENERAL.md)
- [Slice, segmentation, AutoTrack, and parameters](DSI_STUDIO_AI_COMMAND_EXAMPLES_SLICE.md)
- [Regions and tract-to-region analysis](DSI_STUDIO_AI_COMMAND_EXAMPLES_REGION.md)
- [Tracts, clustering, recognition, and TDI](DSI_STUDIO_AI_COMMAND_EXAMPLES_TRACT.md)
- [Devices and AC-PC locators](DSI_STUDIO_AI_COMMAND_EXAMPLES_DEVICE.md)

## `chat` with `CMD`

For a meaningful command, include a useful top-level `chat` update that tells the user what was verified and what the command is about to do.

```json
{"agent":"Codex","session":"<uuid>","request":"CMD","window":"2","command":["segment_brain","SynthSeg V2","7"],"chat":"I verified that the T1w slice is loaded and ready. I’m starting SynthSeg now; it may take a while to finish."}
```

The `chat` field is shown to the user and does not alter command execution. Routine polling and trivial discovery commands may omit it to avoid repetitive updates.
