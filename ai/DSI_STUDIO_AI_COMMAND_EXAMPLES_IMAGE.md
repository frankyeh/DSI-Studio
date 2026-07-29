# DSI Studio AI Image Command Examples and Inventory

Use these commands with an `image<hex-address>` window ID returned by top-level
`LIST`. Command names and text, path, or composite parameters are strings. Send
a standalone numeric parameter as a JSON number.

The image-window dispatcher accepts only a command name and at most one parameter. The examples below were checked against `view_image::command()`, `variant_image::command()`, and TIPL `command()` source.

| Command | Common example | Important behavior |
|---|---|---|
| `change_type` | `["change_type",3]` | Change voxel type: `0`=uint8, `1`=uint16, `2`=uint32, `3`=float32. |
| `bias_field_correction` | `["bias_field_correction"]` | Iteratively estimate and remove the bias field using the positive-value mask. |
| `brain_extraction` | `["brain_extraction","<model-ID>"]` | Run the model whose `.nz` filename stem exactly matches `<model-ID>`, then multiply the image by its foreground probability. See footnote 1. |
| `segmentation` | `["segmentation","<model-ID>"]` | Run the exact model ID and replace the current image with its label output; the image becomes uint8 label data. See footnote 1. |
| `deface` | `["deface","<model-ID>"]` | Run the exact model ID and suppress voxels in the generated facial mask while preserving nearby brain tissue. See footnote 1. |
| `rotate_to_image` | `["rotate_to_image","C:/data/template.nii.gz"]` | Rigidly register the current image to the supplied NIfTI using mutual information, then resample it into target space. |
| `warp_to_image` | `["warp_to_image","C:/data/template.nii.gz"]` | Affine-register using correlation, continue with nonlinear registration, then resample the current image into target space. |
| `apply_to_image` | `["apply_to_image","C:/data/other.nii.gz"]` | Load another image and apply the mapping created by the preceding `rotate_to_image` or `warp_to_image`; fails when no mapping exists. |
| `morphology_defragment` | `["morphology_defragment"]` | Keep the principal connected component. Floating images are first converted to a positive-value mask and then preserved by that mask. |
| `morphology_fill_holes` | `["morphology_fill_holes"]` | Fill enclosed holes in 3D. |
| `morphology_fill_holes_by_slice` | `["morphology_fill_holes_by_slice"]` | Fill holes independently by slice. |
| `morphology_defragment_by_size` | `["morphology_defragment_by_size",0.05]` | Remove components below the supplied size ratio; an omitted parameter defaults to `0.05`, but an explicit value is preferred. |
| `morphology_dilation` | `["morphology_dilation"]` | Dilate each nonzero label independently. |
| `morphology_erosion` | `["morphology_erosion"]` | Erode each nonzero label independently. |
| `morphology_opening` | `["morphology_opening"]` | Apply opening to each nonzero label independently. |
| `morphology_closing` | `["morphology_closing"]` | Apply closing to each nonzero label independently. |
| `morphology_edge` | `["morphology_edge"]` | Extract 3D edges for each label. |
| `morphology_edge_xy` | `["morphology_edge_xy"]` | Extract label edges within XY planes. |
| `morphology_edge_xz` | `["morphology_edge_xz"]` | Extract label edges within XZ planes. |
| `morphology_smoothing` | `["morphology_smoothing"]` | Smooth a binary mask or use multi-region smoothing when values exceed `0/1`. |
| `sobel_filter` | `["sobel_filter"]` | Apply TIPL Sobel filtering. |
| `gaussian_filter` | `["gaussian_filter"]` | Apply TIPL Gaussian smoothing. |
| `mean_filter` | `["mean_filter"]` | Apply TIPL mean filtering. |
| `smoothing_filter` | `["smoothing_filter"]` | Apply TIPL anisotropic-diffusion smoothing. |
| `normalize` | `["normalize"]` | Convert to float32 first when needed, then normalize intensity. |
| `normalize_otsu_median` | `["normalize_otsu_median"]` | Convert to float32 first when needed, then normalize using Otsu/median statistics. |
| `flip_x` | `["flip_x"]` | Flip voxel data along X. This does not itself modify the header transformation. |
| `flip_y` | `["flip_y"]` | Flip voxel data along Y. This does not itself modify the header transformation. |
| `flip_z` | `["flip_z"]` | Flip voxel data along Z. This does not itself modify the header transformation. |
| `select_value` | `["select_value",1]` | Replace the image with a binary mask where voxels equal the supplied value. |
| `add_value` | `["add_value",10]` | Add a scalar constant to every voxel. |
| `multiply_value` | `["multiply_value",0.5]` | Multiply every voxel by a scalar constant. |
| `lower_threshold` | `["lower_threshold",0]` | Clamp values below the supplied threshold. |
| `upper_threshold` | `["upper_threshold",1000]` | Clamp values above the supplied threshold. |
| `threshold` | `["threshold",0.5]` | Replace values greater than the threshold with `1` and all others with `0`. |
| `otsu_threshold` | `["otsu_threshold",1.0]` | Binarize at `Otsu threshold × supplied ratio`. |
| `equation` | `["equation","x*2"]` | Multiply the current image by two using TIPL's expression parser. Operators include `+`, `-`, `*`, `/`, `>`, `<`, and `=`; multiplication and division are evaluated first. |
| `set_transformation` | `["set_transformation","1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"]` | Replace the 4×4 image transformation using 16 floats and recalculate voxel size. |
| `set_translocation` | `["set_translocation","0 0 0"]` | Set transformation translation entries `T[3]`, `T[7]`, and `T[11]`. |
| `set_mni` | `["set_mni",1]` | Set the MNI-space flag from the first character: `1`=true; other nonempty values=false. |
| `upsampling` | `["upsampling"]` | Upsample image data or labels by 2× and halve voxel size. |
| `downsampling` | `["downsampling"]` | Downsample image data or labels by 2× and double voxel size. |
| `header_flip_x` | `["header_flip_x"]` | Flip only the header transformation along X. |
| `header_flip_y` | `["header_flip_y"]` | Flip only the header transformation along Y. |
| `header_flip_z` | `["header_flip_z"]` | Flip only the header transformation along Z. |
| `header_swap_xy` | `["header_swap_xy"]` | Swap X/Y axes in header metadata and voxel sizes without swapping voxel data. |
| `header_swap_xz` | `["header_swap_xz"]` | Swap X/Z axes in header metadata and voxel sizes without swapping voxel data. |
| `header_swap_yz` | `["header_swap_yz"]` | Swap Y/Z axes in header metadata and voxel sizes without swapping voxel data. |
| `swap_xy` | `["swap_xy"]` | Swap voxel X/Y axes and voxel sizes. |
| `swap_xz` | `["swap_xz"]` | Swap voxel X/Z axes and voxel sizes. |
| `swap_yz` | `["swap_yz"]` | Swap voxel Y/Z axes and voxel sizes. |
| `crop_to_fit` | `["crop_to_fit",2]` | Crop around the principal positive-value component with a 2-voxel margin in all axes; three margins remain one composite string. |
| `transform` | `["transform","1 0 0 0 0 1 0 0 0 0 1 0"]` | Resample/reorient to a target transformation supplied as its first 12 matrix values. The handler resolves axis flips, voxel-size changes, and translation as needed. |
| `translocate` | `["translocate","1 0 0"]` | Shift image data by voxel offsets and update transformation translation; fractional shifts use interpolation. |
| `resize` | `["resize","256 256 160"]` | Resize the canvas from the origin and copy overlapping data without changing the transformation. |
| `resize_at_center` | `["resize_at_center","256 256 160"]` | Resize the canvas around the image center. |
| `reshape` | `["reshape","256 256 160"]` | Reshape the voxel buffer to the supplied 3D dimensions. The image-window wrapper also has special 4D handling. |
| `regrid` | `["regrid","1 1 1"]` | Resample to one isotropic voxel size or three supplied voxel sizes and update dimensions and transformation. |
| `concatenate_image` | `["concatenate_image","C:/data/second.nii.gz"]` | Append another image along Z; width and height must match. In a 4D image window, it appends another volume instead. |
| `refine_label` | `["refine_label","C:/data/reference.nii.gz"]` | Map the reference image into current space and refine the current labels. |
| `load_image` | `["load_image","C:/data/replacement.nii.gz"]` | Map another image into current space and replace current voxel data. |
| `multiply_image` | `["multiply_image","C:/data/mask.nii.gz"]` | Map another image into current space and multiply voxelwise. |
| `add_image` | `["add_image","C:/data/other.nii.gz"]` | Map another image into current space and add voxelwise. |
| `minus_image` | `["minus_image","C:/data/other.nii.gz"]` | Map another image into current space and subtract voxelwise. |
| `max_image` | `["max_image","C:/data/other.nii.gz"]` | Map another image into current space and retain the voxelwise maximum. |
| `min_image` | `["min_image","C:/data/other.nii.gz"]` | Map another image into current space and retain the voxelwise minimum. |
| `save` | `["save","C:/output/processed.nii.gz"]` | Save image data with transformation, voxel size, and MNI flag. A multi-file image workflow may trigger a modal batch-processing prompt. |
| `open` | `["open","C:/data/replacement.nii.gz"]` | Load image data and metadata through TIPL into the current image object. |

## Source-confirmed cautions

- Image-window `CMD` accepts no more than two command-array elements: the command name and one parameter. A standalone numeric parameter may be a JSON number; combine multiple numeric components inside one composite string.
- Parameterless morphology, filter, flip, up/downsampling, and header/swap commands are real commands; do not add a dummy value.
- `flip_*` changes voxel data, while `header_flip_*` changes metadata only. Pair them only when that combined spatial change is intended.
- `rotate_to_image` and `warp_to_image` replace the current image with its registered/resampled result and retain the mapping for a later `apply_to_image` command.
- Commands ending in `_image` or `_label` require an existing file and map it into the current image space using linear interpolation for images or majority interpolation for labels.
- `save` can show a modal prompt when the image window was opened with additional batch files. Avoid unattended batch saves unless this workflow is expected.

## Footnotes

1. The model catalog is loaded at startup from the packaged `unet/README.md`, not from this repository. Each image-window model action stores `std::filesystem::path(model_url).stem()` as its command argument, and `download_unet_model()` requires an exact match to the downloaded `.nz` filename stem. A tracking window's `["list_unet"]` reports this value in its `model` column. The image window currently has no equivalent model-list command, so discover the ID from `list_unet` when a tracking window is available or from the packaged model catalog.
