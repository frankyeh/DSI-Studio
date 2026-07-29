# DSI Studio AI Region Command Examples and Inventory

Use these with the standard top-level `CMD` request. Command names and text,
path, or composite parameters are strings. Send standalone numeric parameters
as JSON numbers.

This file contains the complete region inventory confirmed in the current source.

`region_action_<operation>` uses `command[1]` for one region index or an `&`-separated index list. Commands that need an additional threshold or voxel radius use `command[2]`. Send a single index or numeric value as a number; keep an `&`-separated list as a string.

| Command | Common example | Important behavior |
|---|---|---|
| `list_region` | `["list_region"]` | List region index, visibility, name, type, color, dimensions, and resolution. |
| `list_atlas` | `["list_atlas"]` | List available templates and atlases before atlas-region creation. |
| `add_region_from_atlas` | `["add_region_from_atlas","0 2 5&6"]` | Add labels 5 and 6 from atlas 2 of template 0; discover valid template, atlas, and label IDs first. |
| `add_region_from_atlas` | `["add_region_from_atlas","0 2"]` | Add every label from atlas 2 of template 0. |
| `set_region_name` | `["set_region_name",0,"Tumor Core"]` | Rename a region by index. |
| `set_region_type` | `["set_region_type",0,3]` | Set region role: `0=ROI`, `1=ROA`, `2=End`, `3=Seed`, `4=Terminative`, `5=NotEnd`, `6=Limiting`. |
| `set_region_color` | `["set_region_color",0,4294901760]` | Set packed Qt ARGB color for a region. |
| `show_only_regions` | `["show_only_regions","0&3&5"]` | Show only the listed `&`-separated region indices and hide all others. |
| `new_region` | `["new_region"]` | Create an empty region in current slice space. |
| `new_region_whole_brain_seed` | `["new_region_whole_brain_seed"]` | Create a whole-brain seed using the current `otsu_threshold` parameter multiplied by the FIB FA Otsu threshold. |
| `new_region_whole_brain_seed` | `["new_region_whole_brain_seed",0.6]` | Create the whole-brain seed using an explicit Otsu ratio of `0.6`, independent of the current setting. |
| `new_region_from_threshold` | `["new_region_from_threshold",0.6]` | Create a region by thresholding the current slice. See footnote 1. |
| `new_region_from_mni` | `["new_region_from_mni","0 -10 21 5"]` | Create a spherical region from MNI coordinates and voxel radius. |
| `new_region_from_sphere` | `["new_region_from_sphere","80 100 80 5"]` | Create a spherical region from image-space coordinates and voxel radius. |
| `open_region` | `["open_region","C:/data/seed.nii.gz"]` | Open one native-space region file; one command may also load a multi-label NIfTI. |
| `open_mni_region` | `["open_mni_region","C:/data/atlas_roi.nii.gz"]` | Map to MNI space, then load the supplied region file into subject space. |
| `save_region` | `["save_region","C:/output/seed.nii.gz",0]` | Save region 0. An unsupported or missing extension is changed by appending `.nii.gz`. |
| `save_4d_region` | `["save_4d_region","C:/output/regions_4d.nii.gz"]` | Save checked regions as a 4D NIfTI and companion label file. |
| `save_all_regions` | `["save_all_regions","C:/output/regions.nii.gz"]` | Save checked regions as one 3D label NIfTI. |
| `save_all_regions_to_folder` | `["save_all_regions_to_folder","C:/output/regions"]` | Save each checked region as a separate file using the current ROI output format. |
| `save_region_info` | `["save_region_info","C:/output/seed_info.txt",0]` | Save coordinates, fiber directions, and quantitative values for one region index. |
| `load_region_color` | `["load_region_color","C:/data/region_colors.txt"]` | Load RGB or RGBA values in region-table order. |
| `save_region_color` | `["save_region_color","C:/output/region_colors.txt"]` | Save one RGBA line for every region in table order. |
| `delete_region` | `["delete_region",3]` | Delete one region by index or current selection. |
| `delete_all_regions` | `["delete_all_regions"]` | Delete all regions. |
| `copy_region` | `["copy_region",0]` | Duplicate one region and insert the copy immediately after it. |
| `merge_regions` | `["merge_regions","0&1&2"]` | Merge regions 0, 1, and 2 into region 0 and remove the later rows. |
| `merge_regions` | `["merge_regions"]` | Merge all currently checked regions into the first checked region; at least two must be checked. |
| `check_region` | `["check_region",0,1]` | Set one region's checked/shown state. |
| `check_all_regions` | `["check_all_regions"]` | Check/show all regions. |
| `uncheck_all_regions` | `["uncheck_all_regions"]` | Uncheck/hide all regions. |
| `move_up_region` | `["move_up_region",3]` | Move one region up in table order. |
| `move_down_region` | `["move_down_region",3]` | Move one region down in table order. |
| `move_region` | `["move_region","80 100 80",3]` | Move region 3 so its center is at the specified location in that region's space. Empty regions return success without moving. |
| `move_slice_to_region` | `["move_slice_to_region",3]` | Move slice crosshairs to a region center. |
| `show_region_statistics` | `["show_region_statistics"]` | Display statistics for checked regions in a modal dialog. |
| `save_region_statistics` | `["save_region_statistics","C:/output/region_stat.txt"]` | Save statistics for checked regions. |
| `show_t2r` | `["show_t2r"]` | Display tract-to-region connectivity for checked tracts and regions. |
| `save_t2r` | `["save_t2r","C:/output/tract_to_region.txt"]` | Save tract-to-region connectivity to a text file. |
| `region_action_shiftx` | `["region_action_shiftx",0]` | Shift region 0 by +1 voxel in X. |
| `region_action_shiftnx` | `["region_action_shiftnx",0]` | Shift region 0 by -1 voxel in X. |
| `region_action_shifty` | `["region_action_shifty",0]` | Shift region 0 by +1 voxel in Y. |
| `region_action_shiftny` | `["region_action_shiftny",0]` | Shift region 0 by -1 voxel in Y. |
| `region_action_shiftz` | `["region_action_shiftz",0]` | Shift region 0 by +1 voxel in Z. |
| `region_action_shiftnz` | `["region_action_shiftnz",0]` | Shift region 0 by -1 voxel in Z. |
| `region_action_flipx` | `["region_action_flipx",0]` | Flip region 0 along X. |
| `region_action_flipy` | `["region_action_flipy",0]` | Flip region 0 along Y. |
| `region_action_flipz` | `["region_action_flipz",0]` | Flip region 0 along Z. |
| `region_action_smoothing` | `["region_action_smoothing",0]` | Morphologically smooth region 0. |
| `region_action_erosion` | `["region_action_erosion",0]` | Erode region 0. |
| `region_action_dilation` | `["region_action_dilation",0]` | Dilate region 0. |
| `region_action_opening` | `["region_action_opening",0]` | Apply morphological opening to region 0. |
| `region_action_closing` | `["region_action_closing",0]` | Apply morphological closing to region 0. |
| `region_action_defragment` | `["region_action_defragment",0]` | Keep the principal connected component of region 0. |
| `region_action_negate` | `["region_action_negate",0]` | Invert the mask of region 0. |
| `region_action_dilation_by_voxel` | `["region_action_dilation_by_voxel",0,2]` | Dilate region 0 by a radius of 2 voxels. |
| `region_action_threshold` | `["region_action_threshold",0,0.6]` | Replace region 0 with the current-slice mask above `0.6`; a negative threshold selects the low-pass side. |
| `region_action_threshold_current` | `["region_action_threshold_current",0,0.6]` | Retain only existing region-0 voxels above `0.6`. |
| `region_action_dilation_by_threshold` | `["region_action_dilation_by_threshold",0,0.6]` | Grow region 0 using the current slice and threshold `0.6`. |
| `region_action_erosion_by_threshold` | `["region_action_erosion_by_threshold",0,0.6]` | Shrink region 0 using the current slice and threshold `0.6`. |
| `region_action_separate` | `["region_action_separate",0]` | Split region 0 into connected-component regions. |
| `region_action_sort_name` | `["region_action_sort_name","0&1&2"]` | Sort the supplied rows by name; repeating the same sort reverses the order. |
| `region_action_sort_x` | `["region_action_sort_x","0&1&2"]` | Sort the supplied rows by X position. |
| `region_action_sort_y` | `["region_action_sort_y","0&1&2"]` | Sort the supplied rows by Y position. |
| `region_action_sort_z` | `["region_action_sort_z","0&1&2"]` | Sort the supplied rows by Z position. |
| `region_action_sort_size` | `["region_action_sort_size","0&1&2"]` | Sort the supplied rows by region volume. |
| `region_action_1st_ex_all` | `["region_action_1st_ex_all","0&1&2"]` | Subtract regions 1 and 2 from region 0. |
| `region_action_all_ex_1st` | `["region_action_all_ex_1st","0&1&2"]` | Subtract region 0 from regions 1 and 2. |
| `region_action_all_inter_1st` | `["region_action_all_inter_1st","0&1&2"]` | Intersect regions 1 and 2 with region 0. |
| `region_action_all_to_1st` | `["region_action_all_to_1st","0&1&2"]` | Assign and smooth later labels within the first region. |
| `region_action_refine_all` | `["region_action_refine_all","0&1&2"]` | Refine all supplied labels using the current slice intensity image. |

## Safety notes

- Discover indices and roles with `list_region` before mutation.
- Confirm deletion, merging, overwrite, and mask-replacement operations.
- Modal `show_*` commands block for user interaction; prefer the corresponding `save_*` command for unattended work.
- `add_region_from_atlas` changes the active template ID before adding labels.

## Region Window parameter reference

These are the `ROI` parameters from the embedded `:/data/options.txt` resource. The source tree labels the `ROI` root as **Region Window**, so these parameters belong here rather than in the rendering reference.

Use:

```json
["list_param","roi_zoom"]
["set_param","roi_zoom",5.0]
["set_params","roi_zoom=5.0&roi_opacity=0.8&roi_draw_edge=1"]
```

Send numeric values as JSON numbers with `set_param`. `set_params` keeps its combined assignment expression as one string. Enum values are zero-based indices.

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `orientation_convention` | Orientation Convention | `0`=Radiology; `1`=Neurology | `0` (Radiology) |
| `roi_zoom` | Zoom | float `0.2–40`; step `0.5` | `5.0` |
| `roi_draw_edge` | Draw Edge | `0`=Off; `1`=On | `0` (Off) |
| `roi_composition` | Composition | `0`=SourceAtop; `1`=DestinationAtop; `2`=Xor; `3`=Plus; `4`=Multiply; `5`=Screen; `6`=Overlay; `7`=Darken; `8`=Lighten; `9`=ColorDodge; `10`=ColorBun; `11`=HardLight; `12`=SoftLight; `13`=Difference; `14`=Exclusion | `0` (SourceAtop) |
| `roi_opacity` | Opacity | float `0–1`; step `0.1` | `1` |
| `roi_edge_width` | Edge Width | integer `1–5`; step `1` | `1` |
| `roi_track` | Show Tracts | `0`=Off; `1`=On | `1` (On) |
| `roi_track_count` | Visible Tracts Count | integer `1000–500000`; step `1000` | `5000` |
| `roi_fiber` | Fiber Direction | `0`=Off; `1`=On; `2`=1st; `3`=2nd | `1` (On) |
| `roi_fiber_color` | Fiber Color | `0`=RGB; `1`=red; `2`=green; `3`=blue | `0` (RGB) |
| `roi_fiber_width` | Fiber Width | float `0.1–1`; step `0.1` | `0.2` |
| `roi_fiber_length` | Fiber Length | float `0.1–4`; step `0.1` | `2.0` |
| `roi_fiber_antialiasing` | Fiber Antialiasing | `0`=Off; `1`=On | `0` (Off) |
| `roi_label` | "R" label | `0`=Off; `1`=On | `1` (On) |
| `roi_position` | Position Line | `0`=Off; `1`=On | `1` (On) |
| `roi_ruler` | Ruler | `0`=Off; `1`=On | `1` (On) |
| `roi_tic` | Ruler Tic | integer `1–8`; step `1` | `2` |
| `roi_layout` | Slice Layout | `0`=Single Slice; `1`=3 Slices; `2`=Mosaic; `3`=Mosaic 2; `4`=Mosaic 3; `5`=Mosaic 4; `6`=Mosaic 5; `7`=Mosaic 6; `8`=Mosaic 7; `9`=Mosaic 8; `10`=Mosaic 9; `11`=Mosaic 10 | `0` (Single Slice) |
| `roi_mosaic_column` | Mosaic Column Number | integer `0–30`; step `5` | `0` |
| `roi_mosaic_skip_row` | Mosaic Skip Row | integer `0–10`; step `1` | `1` |
| `roi_format` | Default Output Format | `0`=nii.gz; `1`=mat; `2`=txt | `0` (nii.gz) |

## Footnotes

1. `new_region_from_threshold` appends a new region before calling `region_action_threshold`. If threshold selection is canceled or the action fails, the newly appended empty region is not removed. For unattended use, always provide a valid explicit threshold and verify the new region with `list_region`.
