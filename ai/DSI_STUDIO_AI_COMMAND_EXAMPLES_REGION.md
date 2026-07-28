# DSI Studio AI Region Command Examples and Inventory

Use these with the standard top-level `CMD` request. Every command name and parameter must remain a quoted JSON string.

This file contains the complete region inventory preserved from the previous manual. Blank example cells mean that the previous manual listed the command but did not provide source-verified argument syntax.

`region_action_<operation>` uses `command[1]` for one region index or an `&`-separated index list. Commands that need an additional threshold or voxel radius use `command[2]`. Do not infer other argument formats from the command name.

| Command | Common example | Important behavior |
|---|---|---|
| `list_region` | `["list_region"]` | List region index, visibility, name, type, color, dimensions, and resolution. |
| `list_atlas` | `["list_atlas"]` | List available templates and atlases before atlas-region creation. |
| `add_region_from_atlas` | `["add_region_from_atlas","0 2 5&6"]` | Add one or more atlas labels; discover valid template, atlas, and label IDs first. |
| `set_region_name` | `["set_region_name","0","Tumor Core"]` | Rename a region by quoted index. |
| `set_region_type` | `["set_region_type","0","3"]` | Set region role: `0=ROI`, `1=ROA`, `2=End`, `3=Seed`, `4=Terminative`, `5=NotEnd`, `6=Limiting`. |
| `set_region_color` | `["set_region_color","0","4294901760"]` | Set packed Qt ARGB color for a region. |
| `show_only_regions` | `["show_only_regions","0&3&5"]` | Show only the listed `&`-separated region indices and hide all others. |
| `new_region` | `["new_region"]` | Create an empty region in current slice space. |
| `new_region_whole_brain_seed` | `["new_region_whole_brain_seed"]` | Create a whole-brain seed from the current FA/Otsu threshold. |
| `new_region_from_threshold` | `["new_region_from_threshold","0.6"]` | Create a region by thresholding the current slice. |
| `new_region_from_mni` | `["new_region_from_mni","0 -10 21 5"]` | Create a spherical region from MNI coordinates and voxel radius. |
| `new_region_from_sphere` | `["new_region_from_sphere","80 100 80 5"]` | Create a spherical region from image-space coordinates and voxel radius. |
| `open_region` |  | Open one or more region files in native space. |
| `open_mni_region` |  | Open region file(s) and map them from MNI space. |
| `save_region` | `["save_region","C:/output/seed.nii.gz","0"]` | Save one region; optional index selects the target. |
| `save_4d_region` | `["save_4d_region","C:/output/regions_4d.nii.gz"]` | Save checked regions as a 4D NIfTI and companion label file. |
| `save_all_regions` | `["save_all_regions","C:/output/regions.nii.gz"]` | Save checked regions as one 3D label NIfTI. |
| `save_all_regions_to_folder` |  | Save each checked region as a separate file in a folder. |
| `save_region_info` |  | Save voxel coordinates, directions, and quantitative values for one region. |
| `load_region_color` |  | Load RGB/RGBA colors for regions from a text file. |
| `save_region_color` |  | Save region colors to a text file. |
| `delete_region` | `["delete_region","3"]` | Delete one region by index or current selection. |
| `delete_all_regions` | `["delete_all_regions"]` | Delete all regions. |
| `copy_region` | `["copy_region","0"]` | Duplicate one region. |
| `merge_regions` | `["merge_regions","0&1&2"]` | Merge supplied or checked region indices into the first region. |
| `check_region` | `["check_region","0","1"]` | Set one region's checked/shown state. |
| `check_all_regions` | `["check_all_regions"]` | Check/show all regions. |
| `uncheck_all_regions` | `["uncheck_all_regions"]` | Uncheck/hide all regions. |
| `move_up_region` | `["move_up_region","3"]` | Move one region up in table order. |
| `move_down_region` | `["move_down_region","3"]` | Move one region down in table order. |
| `move_region` | `["move_region","80 100 80","3"]` | Move a region center to a specified location in region space. |
| `move_slice_to_region` | `["move_slice_to_region","3"]` | Move slice crosshairs to a region center. |
| `show_region_statistics` | `["show_region_statistics"]` | Display statistics for checked regions in a modal dialog. |
| `save_region_statistics` | `["save_region_statistics","C:/output/region_stat.txt"]` | Save statistics for checked regions. |
| `show_t2r` | `["show_t2r"]` | Display tract-to-region connectivity for checked tracts and regions. |
| `save_t2r` | `["save_t2r","C:/output/tract_to_region.txt"]` | Save tract-to-region connectivity to a text file. |
| `region_action_shiftx` |  | Shift selected region(s) +1 voxel in X. |
| `region_action_shiftnx` |  | Shift selected region(s) -1 voxel in X. |
| `region_action_shifty` |  | Shift selected region(s) +1 voxel in Y. |
| `region_action_shiftny` |  | Shift selected region(s) -1 voxel in Y. |
| `region_action_shiftz` |  | Shift selected region(s) +1 voxel in Z. |
| `region_action_shiftnz` |  | Shift selected region(s) -1 voxel in Z. |
| `region_action_flipx` |  | Flip selected region(s) along X. |
| `region_action_flipy` |  | Flip selected region(s) along Y. |
| `region_action_flipz` |  | Flip selected region(s) along Z. |
| `region_action_smoothing` |  | Morphologically smooth selected region(s). |
| `region_action_erosion` |  | Erode selected region(s). |
| `region_action_dilation` |  | Dilate selected region(s). |
| `region_action_opening` |  | Apply morphological opening. |
| `region_action_closing` |  | Apply morphological closing. |
| `region_action_defragment` |  | Keep the principal connected component. |
| `region_action_negate` |  | Invert selected region mask(s). |
| `region_action_dilation_by_voxel` |  | Dilate by voxel radius; `command[2]` is the radius. |
| `region_action_threshold` |  | Replace selected region(s) with a thresholded current-slice mask; `command[2]` is threshold. |
| `region_action_threshold_current` |  | Threshold only voxels already inside selected region(s). |
| `region_action_dilation_by_threshold` |  | Grow selected region(s) using current-slice intensity threshold. |
| `region_action_erosion_by_threshold` |  | Shrink selected region(s) using current-slice intensity threshold. |
| `region_action_separate` |  | Split one region into connected components. |
| `region_action_sort_name` |  | Sort selected/checked regions by name; repeating reverses order. |
| `region_action_sort_x` |  | Sort selected/checked regions by X position. |
| `region_action_sort_y` |  | Sort selected/checked regions by Y position. |
| `region_action_sort_z` |  | Sort selected/checked regions by Z position. |
| `region_action_sort_size` |  | Sort selected/checked regions by volume. |
| `region_action_1st_ex_all` |  | Subtract every later region from the first. |
| `region_action_all_ex_1st` |  | Subtract the first region from every later region. |
| `region_action_all_inter_1st` |  | Intersect every later region with the first. |
| `region_action_all_to_1st` |  | Assign/fill later labels within the first region. |
| `region_action_refine_all` |  | Refine all supplied region labels using the current slice. |

## Safety notes

- Discover indices and roles with `list_region` before mutation.
- Confirm deletion, merging, overwrite, and mask-replacement operations.
- Modal `show_*` commands block for user interaction; prefer the corresponding `save_*` command for unattended work.
