# DSI Studio AI Device Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `new_device` | `["new_device","80 100 80"]` | Creates a device at the supplied voxel position. Omitting the position places it randomly near image center. |
| `move_device` | `["move_device","82 101 79","0"]` | Moves device `0` to the supplied voxel coordinate. |
| `push_device` | `["push_device","0"]` | Moves device `0` by `-0.5` mm along its direction vector. |
| `pull_device` | `["pull_device","0"]` | Moves device `0` by `+0.5` mm along its direction vector. |
| `copy_device` | `["copy_device","0"]` | Copies device `0`, offsets x by one voxel, and preserves length, direction, and type. |
| `set_acpc` | `["set_acpc"]` | Replaces existing `AC`, `PC`, and `Inter` locators using fixed MNI coordinates mapped into subject space. |
| `delete_device` | `["delete_device","0"]` | Permanently removes one device. Confirm destructive actions first. |
| `delete_all_devices` | `["delete_all_devices"]` | Permanently removes all devices. Confirm destructive actions first. |
| `save_all_devices` | `["save_all_devices","C:/output/electrodes.dv.csv"]` | Writes only checked devices in table order. |
| `save_device_statistics` | `["save_device_statistics","C:/output/device_stat.txt"]` | Computes statistics for all loaded devices and writes them to the supplied file. |

## Source-confirmed notes

- Device positions supplied to `new_device` and `move_device` are voxel coordinates in current FIB space.
- `push_device` and `pull_device` use a physical 0.5 mm movement, converted to voxel displacement using voxel size.
- `set_acpc` removes existing locator entries before creating replacements.
- `save_all_devices` exports checked devices only.
