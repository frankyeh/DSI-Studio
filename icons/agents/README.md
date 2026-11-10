# AI agent icons

These SVG assets are bundled in `icons.qrc` under `:/icons/icons/agents/`.

| Resource | Identification | Source |
| --- | --- | --- |
| `codex.svg` | OpenAI mark for Codex | [gilbarbara/logos](https://github.com/gilbarbara/logos/blob/main/logos/openai-icon.svg) |
| `claude.svg` | Claude mark | [gilbarbara/logos](https://github.com/gilbarbara/logos/blob/main/logos/claude-icon.svg) |
| `muse.svg` | Letter M for Muse | Original SVG created for DSI Studio |
| `antigravity.svg` | Official Antigravity one-color icon | [Google Antigravity press assets](https://antigravity.google/press) |
| `grok.svg` | Grok mark | [gilbarbara/logos](https://github.com/gilbarbara/logos/blob/main/logos/grok-icon.svg) |
| `web.svg` | Generic globe | Created for DSI Studio |

Use the agent's existing provider string for identity, tooltips, and accessibility. Logos are decorative UI assets, not identifiers for the session protocol.

For both enabled and disabled controls, set the same `QIcon(":/icons/icons/agents/<provider>.svg")` on the Qt widget: Qt uses the icon's Normal mode when enabled and synthesizes the Disabled mode when the widget is disabled. No separate disabled SVG files are needed. Keep the existing state indicator (Ready / Sign in / Error) independent of the logo.

These marks identify third-party products and remain subject to their owners' trademarks and brand guidelines.
