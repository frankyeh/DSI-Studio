# DSI Studio AI Device Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `new_device` | `["new_device","80 100 80"]` | Creates a new device at the supplied `x y z` voxel position. Omitting the position places it randomly near the image center. On non-isotropic data, DSI Studio shows a warning dialog before continuing, so provide user guidance if that warning appears. |
| `move_device` | `["move_device","82 101 79","0"]` | Moves device `0` so its position becomes the supplied `x y z` voxel coordinate. The optional third element selects the device index; omit it to use the current device. |
| `push_device` | `["push_device","0"]` | Moves device `0` by `-0.5` mm along its direction vector, converted to voxel displacement using the FIB voxel size. Omit the index to use the current device. |
| `pull_device` | `["pull_device","0"]` | Moves device `0` by `+0.5` mm along its direction vector, converted to voxel displacement using the FIB voxel size. Omit the index to use the current device. |
| `copy_device` | `["copy_device","0"]` | Copies device `0`, offsets its voxel x-position by `+1`, preserves its length, direction, and type, assigns a new generated color, and creates a new type-based name. |
| `set_acpc` | `["set_acpc"]` | Requires a working MNI mapping. It deletes any existing devices named `AC`, `PC`, or `Inter`, then recreates the three `Locator` devices from fixed MNI coordinates mapped into subject space. |
| `delete_device` | `["delete_device","0"]` | Permanently removes one device row. Omit the index to delete the current device; confirm this destructive action first. |
| `delete_all_devices` | `["delete_all_devices"]` | Permanently removes all device rows and device objects. Confirm this destructive action first. |
| `save_all_devices` | `["save_all_devices","C:/output/electrodes.dv.csv"]` | Writes only checked devices to the supplied CSV file in device-table order. It returns canceled when no devices exist. Supplying the filename avoids the save dialog. |

## Source-confirmed notes

- Device positions supplied to `new_device` and `move_device` are voxel coordinates in the current FIB space.
- `push_device` and `pull_device` use a physical movement of 0.5 mm along the normalized device direction, then divide by voxel size to obtain voxel displacement.
- `set_acpc` is not additive for the locator names: existing `AC`, `PC`, and `Inter` entries are removed before new ones are created.
- `save_all_devices` exports checked devices only; unchecked devices are omitted without error.
