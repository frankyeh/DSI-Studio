# Additional DSI Studio AI Tract Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `separate_deleted_tract` | `["separate_deleted_tract","0"]` | Moves the deleted trajectories from tract bundle `0` into a new bundle, then clears the deleted-tract storage of the original bundle. It returns canceled when the selected bundle has no deleted trajectories. |
| `recognize_and_rename_tract` | `["recognize_and_rename_tract"]` | Runs asymmetric tract-atlas recognition on every checked nonempty bundle and renames each to its highest-ranked recognized tract name. It requires the tract atlas to load successfully. |
| `merge_all_tracts` | `["merge_all_tracts"]` | Merges all checked bundles into the first checked bundle and deletes the remaining checked rows. It returns canceled unless at least two bundles are checked. Confirm this destructive operation first. |
| `merge_tract_by_name` | `["merge_tract_by_name"]` | Merges bundles that have identical names, keeping the earlier row and deleting later matching rows. This scans the tract table and changes its row structure. |
| `sort_tract_by_name` | `["sort_tract_by_name"]` | Sorts all tract-table rows alphabetically by name while moving their checked states, thread data, tract models, and rendering objects together. |
| `save_tdi` | `["save_tdi","C:/output/cst_tdi.nii.gz","0"]` | Saves a tract-density image for bundle `0` in the current slice space, using the current slice dimensions, voxel size, transformation, and slice mapping. Supplying the filename avoids the save dialog. |
| `save_tdi2` | `["save_tdi2","C:/output/cst_tdi_2x.nii.gz","0"]` | Saves a tract-density image at twice the native FIB resolution: dimensions are doubled and voxel sizes are halved. The optional third element selects the tract bundle. |
| `check_tract` | `["check_tract","0","1"]` | Sets one tract bundle's checked state. Use `1` to check/show and `0` to uncheck/hide; omit the index to target the current tract. |
| `check_uncheck_all_tract` | `["check_uncheck_all_tract","1"]` | Checks every tract bundle when the argument is `1`, or unchecks every bundle when it is `0`. Omitting the argument toggles all bundles based on the first row's current state. |
| `color_all_cluster` | `["color_all_cluster"]` | Assigns a generated distinct color to every tract bundle and switches `tract_color_style` to manually assigned colors. |

## Source-confirmed notes

- `separate_deleted_tract` transfers, rather than copies, the deleted trajectories into a new bundle.
- `merge_all_tracts` and `merge_tract_by_name` delete rows after merging and therefore require normal destructive-action confirmation.
- `save_tdi` uses the current slice coordinate system, while `save_tdi2` uses a fixed two-times FIB-resolution grid.
- `check_uncheck_all_tract` accepts explicit `"1"` or `"0"`; omitting it invokes toggle behavior.
