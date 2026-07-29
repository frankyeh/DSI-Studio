# DSI Studio AI Device Command Examples and Inventory

Use these with the standard top-level `CMD` request. Command names and text,
path, or composite parameters are strings. Send standalone numeric parameters
as JSON numbers.

This file contains the complete device inventory confirmed in the current source.

| Command | Common example | Important behavior |
|---|---|---|
| `new_device` | `["new_device","80 100 80"]` | Create a device at a supplied voxel position; omission places it near image center. The three coordinates remain one composite string. |
| `move_device` | `["move_device","82 101 79",0]` | Move one device to a supplied voxel coordinate. Omit the index to use the current row. |
| `push_device` | `["push_device",0]` | Move one device by -0.5 mm along its direction. Omit the index to use the current row. |
| `pull_device` | `["pull_device",0]` | Move one device by +0.5 mm along its direction. Omit the index to use the current row. |
| `copy_device` | `["copy_device",0]` | Copy one device and offset its x-position by one voxel. Omit the index to use the current row. |
| `set_acpc` | `["set_acpc"]` | Replace AC, PC, and Inter locators from fixed MNI coordinates mapped into subject space. |
| `delete_device` | `["delete_device",0]` | Delete one device. Omit the index to use the current row. |
| `delete_all_devices` | `["delete_all_devices"]` | Delete all devices. |
| `save_all_devices` | `["save_all_devices","C:/output/electrodes.dv.csv"]` | Save checked devices in table order. Without a path, open a save dialog. |
| `show_device_statistics` | `["show_device_statistics"]` | Display device statistics in a modal dialog. |
| `save_device_statistics` | `["save_device_statistics","C:/output/device_stat.txt"]` | Save device statistics to a text file. |

## Source-confirmed notes

- Device positions supplied to `new_device` and `move_device` are voxel coordinates in current FIB space.
- `push_device` and `pull_device` use a physical 0.5 mm movement converted using voxel size.
- `move_device`, `push_device`, `pull_device`, `copy_device`, and `delete_device` use the current row when the index is omitted.
- `set_acpc` removes existing locator entries before creating replacements.
- `save_all_devices` writes only checked devices.
- Modal `show_device_statistics` is unsuitable for unattended operation; prefer `save_device_statistics`.
