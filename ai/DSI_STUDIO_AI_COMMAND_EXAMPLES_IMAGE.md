# DSI Studio AI Image Command Examples and Inventory

Use these commands with a numeric **image-window** ID returned by top-level `LIST`. Every command name and parameter must remain a quoted JSON string.

This file contains the complete image-window and TIPL generic image command inventory preserved from the previous manual. The previous manual did not provide recommended examples for these commands, so the example column remains blank. Inspect the current image/TIPL command source before supplying parameters; do not infer syntax from the command name.

| Command | Common example | Important behavior |
|---|---|---|
| `change_type` |  | Change image voxel type. |
| `bias_field_correction` |  | Run image-window bias-field correction. |
| `brain_extraction` |  | Run image-window brain extraction. |
| `segmentation` |  | Run image-window segmentation. |
| `deface` |  | Deface the image. |
| `rotate_to_image` |  | Rotate/register the current image to another image. |
| `warp_to_image` |  | Warp the current image to another image. |
| `apply_to_image` |  | Apply an operation or transformation to another image. |
| `morphology_defragment` |  | Keep the principal connected component. |
| `morphology_fill_holes` |  | Fill enclosed holes in 3D. |
| `morphology_fill_holes_by_slice` |  | Fill holes independently by slice. |
| `morphology_defragment_by_size` |  | Remove components below a size ratio; prior manual noted an optional default of `0.05`. |
| `morphology_dilation` |  | Dilate each label. |
| `morphology_erosion` |  | Erode each label. |
| `morphology_opening` |  | Apply opening to each label. |
| `morphology_closing` |  | Apply closing to each label. |
| `morphology_edge` |  | Extract 3D label edges. |
| `morphology_edge_xy` |  | Extract edges in XY planes. |
| `morphology_edge_xz` |  | Extract edges in XZ planes. |
| `morphology_smoothing` |  | Smooth binary or multi-label masks. |
| `sobel_filter` |  | Apply Sobel filtering. |
| `gaussian_filter` |  | Apply Gaussian smoothing. |
| `mean_filter` |  | Apply mean filtering. |
| `smoothing_filter` |  | Apply anisotropic diffusion smoothing. |
| `normalize` |  | Normalize image intensity. |
| `normalize_otsu_median` |  | Normalize using Otsu/median segmentation statistics. |
| `flip_x` |  | Flip voxel data along X. |
| `flip_y` |  | Flip voxel data along Y. |
| `flip_z` |  | Flip voxel data along Z. |
| `select_value` |  | Create a binary mask selecting one exact value. |
| `add_value` |  | Add a scalar constant. |
| `multiply_value` |  | Multiply by a scalar constant. |
| `lower_threshold` |  | Clamp values below the supplied threshold. |
| `upper_threshold` |  | Clamp values above the supplied threshold. |
| `threshold` |  | Binarize values greater than the supplied threshold. |
| `otsu_threshold` |  | Binarize using Otsu threshold multiplied by a supplied ratio. |
| `equation` |  | Apply a TIPL equation expression. |
| `set_transformation` |  | Replace the 4x4 image transformation; parameter contains 16 values. |
| `set_translocation` |  | Set transformation translation components. |
| `set_mni` |  | Set the MNI-space flag (`0` or `1`). |
| `upsampling` |  | Upsample image/labels and update voxel size/transformation. |
| `downsampling` |  | Downsample image/labels and update voxel size/transformation. |
| `header_flip_x` |  | Flip only the header transformation along X. |
| `header_flip_y` |  | Flip only the header transformation along Y. |
| `header_flip_z` |  | Flip only the header transformation along Z. |
| `header_swap_xy` |  | Swap X/Y axes in header metadata. |
| `header_swap_xz` |  | Swap X/Z axes in header metadata. |
| `header_swap_yz` |  | Swap Y/Z axes in header metadata. |
| `swap_xy` |  | Swap voxel X/Y axes and voxel sizes. |
| `swap_xz` |  | Swap voxel X/Z axes and voxel sizes. |
| `swap_yz` |  | Swap voxel Y/Z axes and voxel sizes. |
| `crop_to_fit` |  | Crop to nonzero content with optional margin. |
| `transform` |  | Resample/reorient to a supplied transformation matrix. |
| `translocate` |  | Shift image by voxel offsets and update transformation. |
| `resize` |  | Resize canvas to `width height depth`, anchored at origin. |
| `resize_at_center` |  | Resize canvas around image center. |
| `reshape` |  | Reshape data to new dimensions. |
| `regrid` |  | Resample to one or three supplied voxel sizes. |
| `concatenate_image` |  | Append another image along Z; width and height must match. |
| `refine_label` |  | Refine labels using a reference image file. |
| `load_image` |  | Replace data with another image mapped into current space. |
| `multiply_image` |  | Multiply voxelwise by another mapped image. |
| `add_image` |  | Add another mapped image voxelwise. |
| `minus_image` |  | Subtract another mapped image voxelwise. |
| `max_image` |  | Take voxelwise maximum with another image. |
| `min_image` |  | Take voxelwise minimum with another image. |
| `save` |  | Save image data and metadata. |
| `open` |  | Open/replace image data and metadata. |
