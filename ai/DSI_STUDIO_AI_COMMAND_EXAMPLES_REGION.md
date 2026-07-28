# DSI Studio AI Region Command Examples

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

| Command | Common example | Important behavior |
|---|---|---|
| `new_region` | `["new_region"]` | Creates an empty region named `new region` in the current slice space and makes it current. |
| `new_region_whole_brain_seed` | `["new_region_whole_brain_seed"]` | Creates a Seed region from the FA map using the current Otsu-threshold ratio. An optional second element overrides that ratio. |
| `new_region_from_threshold` | `["new_region_from_threshold","0.6"]` | Creates a new region by thresholding the current slice with the supplied value. |
| `new_region_from_sphere` | `["new_region_from_sphere","80 100 80 5"]` | Creates a sphere from current-slice voxel coordinates `x y z` and radius in voxels. |
| `new_region_from_mni` | `["new_region_from_mni","0 -10 21 5"]` | Creates a sphere from MNI coordinates in millimeters and radius in voxels. MNI mapping must be available. |
| `add_region_from_atlas` | `["add_region_from_atlas","0 2 5&6"]` | Uses `template-id atlas-id label-id-list`. Omit the label list to add every region in the atlas. |
| `copy_region` | `["copy_region","0"]` | Duplicates one region immediately after the source row. Omit the index to copy the current region. |
| `move_up_region` | `["move_up_region","3"]` | Swaps region row `3` with the preceding row. |
| `move_down_region` | `["move_down_region","3"]` | Swaps region row `3` with the following row. |
| `move_region` | `["move_region","80 100 80","3"]` | Moves region `3` so its center of mass reaches the supplied voxel-space location. |
| `move_slice_to_region` | `["move_slice_to_region","3"]` | Moves the current slice position to the selected region center. |
| `check_region` | `["check_region","0","1"]` | Checks or unchecks one region. Use `1` to show and `0` to hide. |
| `check_all_regions` | `["check_all_regions"]` | Checks and displays every region. |
| `uncheck_all_regions` | `["uncheck_all_regions"]` | Unchecks and hides every region. |
| `merge_regions` | `["merge_regions","0&1&2"]` | Unions the listed regions into the first and removes the later rows. Use ascending valid indices. |
| `delete_region` | `["delete_region","3"]` | Permanently removes one region. Confirm destructive actions first. |
| `delete_all_regions` | `["delete_all_regions"]` | Permanently removes every region. Confirm destructive actions first. |
| `set_region_name` | `["set_region_name","0","Tumor Core"]` | Renames region row `0`. The name cannot be empty. |
| `set_region_type` | `["set_region_type","0","3"]` | Sets region type: `0` ROI, `1` ROA, `2` End, `3` Seed, `4` Terminative, `5` NotEnd, `6` Limiting. |
| `save_region` | `["save_region","C:/output/seed.nii.gz","0"]` | Saves one region. Omit the index to use the current region. Unsupported or absent extensions receive `.nii.gz`. |
| `save_all_regions` | `["save_all_regions","C:/output/regions.nii.gz"]` | Saves checked regions into one 3D label image. At least one region must be checked. |
| `save_4d_region` | `["save_4d_region","C:/output/regions_4d.nii.gz"]` | Saves each checked region as a separate 4D volume and writes a companion label file. |
| `save_region_statistics` | `["save_region_statistics","C:/output/region_stat.txt"]` | Computes statistics for checked regions and writes them directly to the supplied file. |
| `show_region_statistics` | `["show_region_statistics"]` | Opens a modal result dialog. Prefer the save variant for unattended operation. |
| `tract_to_region` | `["tract_to_region","0"]` | Converts all trajectories of one tract bundle into a voxel region. |
| `endpoint_to_region` | `["endpoint_to_region","0"]` | Converts a tract bundle's two endpoint sets into two new regions. |
| `save_t2r` | `["save_t2r","C:/output/tract_to_region.txt"]` | Calculates connectivity for checked tracts against checked regions. Both are required. |
| `show_t2r` | `["show_t2r"]` | Opens a modal tract-to-region result dialog. Prefer `save_t2r` for unattended operation. |
