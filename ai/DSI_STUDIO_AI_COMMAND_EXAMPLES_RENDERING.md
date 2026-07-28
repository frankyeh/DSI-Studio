# DSI Studio AI Rendering Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

This file contains the complete rendering, camera, surface, and display inventory preserved from the previous manual. Blank example cells mean that the previous manual did not provide source-verified argument syntax; inspect the current rendering command source before use.

| Command | Common example | Important behavior |
|---|---|---|
| `rotate` | `["rotate","15 1 0 0"]` | Rotate the 3D view by degrees around axis `x y z`. |
| `save_hd_screen` | `["save_hd_screen","C:/output/tracts.png","1920 1080"]` | Save a high-resolution rendering at the supplied size. |
| `set_view` |  | Set a predefined or explicit camera view. |
| `set_zoom` |  | Set camera zoom. |
| `set_camera` |  | Set camera parameters. |
| `get_camera` |  | Return current camera parameters. |
| `open_camera` |  | Load camera settings from a file. |
| `save_camera` |  | Save camera settings to a file. |
| `store_camera` |  | Store the current camera in the default slot. |
| `store_camera1` |  | Store camera in slot 1. |
| `store_camera2` |  | Store camera in slot 2. |
| `restore_camera` |  | Restore the default stored camera. |
| `restore_camera1` |  | Restore camera slot 1. |
| `restore_camera2` |  | Restore camera slot 2. |
| `save_screen` |  | Save the current 3D screen. |
| `add_surface` |  | Add/create a surface object; some variants are prefix-dispatched. |
| `delete_surface` |  | Delete a surface object. |
| `load_surface` |  | Load a surface file. |
| `save_surface` |  | Save a surface file. |
| `set_surface_color` |  | Set surface color. |
| `set_surface_alpha` |  | Set surface opacity. |
| `set_surface_visible` |  | Set surface visibility. |
| `set_device_color` |  | Set device color. |
