# DSI Studio AI Rendering Command Examples and Inventory

Use these with the standard top-level `CMD` request. Command names and text,
path, or composite parameters are strings. Send standalone numeric parameters
as JSON numbers.

This file contains rendering, camera, surface, and display commands confirmed in the current source. Earlier generic rows with no handler were removed only after checking the full tracking-window dispatch chain.

| Command | Common example | Important behavior |
|---|---|---|
| `rotate` | `["rotate","15 1 0 0"]` | Rotate the 3D view by degrees around axis `x y z`. |
| `set_view` | `["set_view",0]` | Reset to numeric view `0`, `1`, or `2`; repeated calls toggle the corresponding 180-degree flipped view. |
| `set_zoom` | `["set_zoom",1.5]` | Set the absolute camera zoom derived from the transformation-matrix determinant; zero is rejected. |
| `set_camera` | `["set_camera","1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"]` | Replace the camera transformation with the first 16 supplied floats. |
| `open_camera` | `["open_camera","C:/work/camera.txt"]` | Load at least 16 camera-matrix floats from a text file. |
| `save_camera` | `["save_camera","C:/work/camera.txt"]` | Save the current 16-float transformation matrix. |
| `store_camera` | `["store_camera"]` | Store the current camera in the default `cameraa` settings slot; the current implementation shows a modal notice and records the command as canceled. |
| `store_camera1` | `["store_camera1"]` | Store the current camera in slot 1; shows a modal notice. |
| `store_camera2` | `["store_camera2"]` | Store the current camera in slot 2; shows a modal notice. |
| `restore_camera` | `["restore_camera"]` | Restore the default `cameraa` settings slot. |
| `restore_camera1` | `["restore_camera1"]` | Restore camera slot 1. |
| `restore_camera2` | `["restore_camera2"]` | Restore camera slot 2. |
| `set_stereoscopic` | `["set_stereoscopic"]` | Switch the OpenGL widget to stereoscopic view mode. |
| `save_screen` | `["save_screen","C:/output/tracts.png"]` | Save the current 3D rendering. |
| `save_hd_screen` | `["save_hd_screen","C:/output/tracts_hd.png","1920 1080"]` | Temporarily resize the GL widget, save at the supplied width and height, then restore its original size. |
| `save_3view_screen` | `["save_3view_screen","C:/output/tracts_3view.png"]` | Save a 2×2 composite containing three 3D views and the current slice scene. |
| `save_h3view_screen` | `["save_h3view_screen","C:/output/tracts_h3view.png"]` | Save four cropped directional views in a horizontal image. |
| `save_v3view_screen` | `["save_v3view_screen","C:/output/tracts_v3view.png"]` | Save four directional views in a vertical image. |
| `save_rotation_video` | `["save_rotation_video","C:/output/rotation.avi"]` | Currently broken: the handler returns immediately after validating the filename, so the AVI-writing block is unreachable. |
| `add_surface` | `["add_surface",7,0.6]` | Create a surface from slice index 7 using threshold `0.6`; omission of the threshold opens a dialog. |
| `add_surface` | `["add_surface",0,25]` | For a built-in slice, map the built-in ICBM152 white-matter image to subject space and create a whole-brain white-matter isosurface at threshold `25`. |
| `add_surface_left` | `["add_surface_left",7,0.6]` | Create a surface after retaining the source portion on the left side of the current X slice position. |
| `add_surface_right` | `["add_surface_right",7,0.6]` | Create a surface after retaining the source portion on the right side of the current X slice position. |
| `add_surface_upper` | `["add_surface_upper",7,0.6]` | Create a surface after retaining the source portion above the current Z slice position. |
| `add_surface_lower` | `["add_surface_lower",7,0.6]` | Create a surface after retaining the source portion below the current Z slice position. |
| `add_surface_posterior` | `["add_surface_posterior",7,0.6]` | Create a surface after retaining the posterior portion relative to the current Y slice position. |
| `add_surface_anterior` | `["add_surface_anterior",7,0.6]` | Create a surface after retaining the anterior portion relative to the current Y slice position. |

## Source-confirmed cautions

- `set_camera` and camera files require at least 16 floats; additional values are ignored.
- `store_camera`, `store_camera1`, and `store_camera2` display modal messages and return the command-history canceled state even though the setting is stored.
- `save_rotation_video` has an early-return bug and should not be used until the unreachable encoding block is fixed.
- Surface appearance and visibility are controlled through `set_param` using `surface_*` and `show_surface`; there are no `load_surface`, `save_surface`, `delete_surface`, `set_surface_color`, `set_surface_alpha`, or `set_surface_visible` command handlers.
- `get_camera` and `set_device_color` also have no command handlers. Use `save_camera` for retrieval and the device table UI for device color.

## Rendering parameter reference

These parameter IDs come from the embedded `:/data/options.txt` resource used by `RenderingTableWidget::initialize()`. Use:

```json
["list_param"]
["list_param","tract_alpha"]
["set_param","tract_alpha",0.5]
["set_params","tract_alpha=0.5&show_tract=1"]
```

Send numeric values as JSON numbers with `set_param`. `set_params` keeps its combined assignment expression as one string. Enum values are zero-based indices. A bare `int` resource type is a `0–10` slider; ranged integer and float entries show their exact minimum, maximum, and step below. Colors use packed Qt RGB/ARGB integers. The options shown for metric and color-map lists are resource defaults and may be replaced for the loaded data.

### Object visibility

| Parameter ID | UI setting | Accepted value | Default | Source |
|---|---|---|---|---|
| `show_slice` | Slice Rendering | boolean `0`=off, `1`=on | on | Created as a checked root item in `TreeModel`. |
| `show_tract` | Tract Rendering | boolean `0`=off, `1`=on | on | Created as a checked root item in `TreeModel`. |
| `show_region` | Region Rendering | boolean `0`=off, `1`=on | on | Created as a checked root item in `TreeModel`. |
| `show_surface` | Surface Rendering | boolean `0`=off, `1`=on | on | Created as a checked root item in `TreeModel`. |
| `show_device` | Device Rendering | boolean `0`=off, `1`=on | on | Created as a checked root item in `TreeModel`. |
| `show_label` | Label Rendering | boolean `0`=off, `1`=on | off | Created as an unchecked root item in `TreeModel`. |
| `show_odf` | ODF Rendering | boolean `0`=off, `1`=on | off | Created as an unchecked root item in `TreeModel`. |

### Background and global rendering

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `scale_voxel` | Scale with voxel size | `0`=Off; `1`=On | `1` (On) |
| `perspective` | Perspective | integer slider `0–10` | `5` |
| `3d_perspective` | 3D Perspective | float `0.5–3`; step `0.5` | `1.0` |
| `bkg_color` | Background Color | packed Qt ARGB integer | `-1` |
| `anti_aliasing` | Anti-aliasing | `0`=Off; `1`=On | `1` (On) |
| `line_smooth` | Line Smooth | `0`=Off; `1`=On | `0` (Off) |
| `point_smooth` | Point Smooth | `0`=Off; `1`=On | `0` (Off) |
| `poly_smooth` | Polygon Smooth | `0`=Off; `1`=On | `0` (Off) |
| `stereoscopy_angle` | Stereoview Angle | float `0.0–5.0`; step `0.2` | `1` |

### Slice rendering

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `slice_alpha` | Opacity | float `0–1`; step `0.1` | `1` |
| `slice_mag_filter` | Mag Filter | `0`=NEAREST; `1`=LINEAR | `1` (LINEAR) |
| `slice_smoothing` | Smoothing | `0`=Off; `1`=On | `0` (Off) |
| `slice_match_bkcolor` | Match Background Color | `0`=Off; `1`=On | `0` (Off) |
| `slice_bend1` | Blend Func1 | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `2` (DST_COLOR) |
| `slice_bend2` | Blend Func2 | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` (ONE_MINUS_SRC_ALPHA) |

### Tract rendering

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `tract_alpha` | Opacity | float `0–1`; step `0.1` | `1` |
| `tract_color_saturation` | Saturation | float `0–1`; step `0.1` | `0.7` |
| `tract_color_brightness` | Brightness | float `0–1`; step `0.1` | `0.5` |
| `tract_color_style` | Style | `0`=Directional; `1`=Assigned; `2`=Local Metrics; `3`=Averaged Metrics; `4`=Max Metrics; `5`=Loaded Value | `0` (Directional) |
| `tract_color_metrics` | Metrics | `0`=qa; `1`=iso | `0` (qa) |
| `tract_color_max_value` | Max Value | float `0–1`; step `0.1` | `1.0` |
| `tract_color_min_value` | Min Value | float `0–1`; step `0.1` | `0.0` |
| `tract_color_map` | Map | `0`=assigned; `1`=files | `0` (assigned) |
| `tract_color_max` | Max Color | packed Qt ARGB integer | `12079178` |
| `tract_color_min` | Min Color | packed Qt ARGB integer | `14465098` |
| `tract_show_color_bar` | Show Color Bar | `0`=Off; `1`=On | `1` (On) |
| `tract_style` | Style | `0`=Line; `1`=Tube; `2`=End; `3`=End1; `4`=End2 | `1` (Tube) |
| `tract_line_width` | Line Width | float `1.0–10`; step `0.5` | `3` |
| `tract_visible_tract` | Visible Tracts | integer `5000–1000000`; step `5000` | `25000` |
| `tract_shader` | Shade | integer `0–20`; step `1` | `4` |
| `tract_tube_detail` | Tube Detail | `0`=Coarse; `1`=Fine; `2`=Finer; `3`=Finest | `1` (Fine) |
| `tube_diameter` | Tube Diameter (voxel) | float `0.01–5`; step `0.1` | `0.2` |
| `end_point_shift` | Endpoint Shift (voxel) | integer `0–10`; step `1` | `0` |
| `tract_light_option` | Light | `0`=One source; `1`=Two sources; `2`=Off | `1` (Two sources) |
| `tract_light_dir` | Light Direction | integer slider `0–10` | `2` |
| `tract_light_shading` | Light Shading | integer slider `0–10` | `10` |
| `tract_light_diffuse` | Light Diffuse | integer slider `0–10` | `10` |
| `tract_light_ambient` | Light Ambient | integer slider `0–10` | `0` |
| `tract_light_specular` | Light Specular | integer slider `0–10` | `0` |
| `tract_specular` | Material Specular | integer slider `0–10` | `0` |
| `tract_emission` | Material Emission | integer slider `0–10` | `0` |
| `tract_shininess` | Material Shininess | integer slider `0–10` | `0` |
| `tract_sel_angle` | Tract Selection Angle | integer `0–90`; step `5` | `45` |
| `tract_bend1` | Blend Func1 | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `6` (DST_ALPHA) |
| `tract_bend2` | Blend Func2 | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `2` (SRC_COLOR) |

### Region rendering and graph

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `region_alpha` | Opacity | float `0–1`; step `0.1` | `0.8` |
| `region_color_style` | Style | `0`=Assigned; `1`=Metrics | `0` (Assigned) |
| `region_color_metrics` | Metrics | `0`=qa; `1`=iso | `0` (qa) |
| `region_color_max_value` | Max Value | float `0–1`; step `0.1` | `1.0` |
| `region_color_min_value` | Min Value | float `0–1`; step `0.1` | `0.0` |
| `region_color_map` | Map | `0`=assigned; `1`=files | `0` (assigned) |
| `region_color_max` | Max Color | packed Qt ARGB integer | `12079178` |
| `region_color_min` | Min Color | packed Qt ARGB integer | `14465098` |
| `region_show_color_bar` | Show Color Bar | `0`=Off; `1`=On | `1` (On) |
| `region_graph` | Graph | `0`=Off; `1`=On | `0` (Off) |
| `region_node_size` | Node Size | integer slider `0–10` | `4` |
| `region_constant_node_size` | Constant Node Size | `0`=Off; `1`=On | `0` (Off) |
| `region_hide_unconnected_node` | Hide Unconnected Node | `0`=Off; `1`=On | `1` (On) |
| `region_edge_size` | Edge Size | integer slider `0–10` | `4` |
| `region_constant_edge_size` | Constant Edge Size | `0`=Off; `1`=On | `0` (Off) |
| `region_pos_edge_color1` | Edge Min Color(Positive) | packed Qt ARGB integer | `-1` |
| `region_pos_edge_color2` | Edge Max Color(Positive) | packed Qt ARGB integer | `8224255` |
| `region_neg_edge_color1` | Edge Min Color(Negative) | packed Qt ARGB integer | `-1` |
| `region_neg_edge_color2` | Edge Max Color(Negative) | packed Qt ARGB integer | `16743293` |
| `region_edge_threshold` | Binary Graph Threshold | float `0.00–1`; step `0.1` | `0.1` |
| `region_mesh_smoothed` | Mesh Rendering | `0`=Original; `1`=Smoothed; `2`=Smoothed2 | `1` (Smoothed) |
| `region_light_option` | Light | `0`=One source; `1`=Two sources; `2`=Three sources | `0` (One source) |
| `region_light_dir` | Light Direction | integer slider `0–10` | `2` |
| `region_light_shading` | Light Shading | integer slider `0–10` | `2` |
| `region_light_diffuse` | Light Diffuse | integer slider `0–10` | `10` |
| `region_light_ambient` | Light Ambient | integer slider `0–10` | `0` |
| `region_light_specular` | Light Specular | integer slider `0–10` | `0` |
| `region_specular` | Material Specular | integer slider `0–10` | `0` |
| `region_emission` | Material Emission | integer slider `0–10` | `1` |
| `region_shininess` | Material Shininess | integer slider `0–10` | `0` |
| `region_bend1` | Blend Func1 | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `4` (SRC_ALPHA) |
| `region_bend2` | Blend Func2 | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` (ONE_MINUS_SRC_ALPHA) |

### Surface rendering

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `surface_color` | Color | packed Qt ARGB integer | `11184810` |
| `surface_alpha` | Opacity | float `0–1`; step `0.05` | `0.2` |
| `surface_mesh_smoothed` | Mesh Rendering | `0`=Original; `1`=Smoothed; `2`=Smoothed2 | `2` (Smoothed2) |
| `surface_light_option` | Light | `0`=One source; `1`=Two sources; `2`=Three sources | `2` (Three sources) |
| `surface_light_dir` | Light Direction | integer slider `0–10` | `5` |
| `surface_light_shading` | Light Shading | integer slider `0–10` | `4` |
| `surface_light_diffuse` | Light Diffuse | integer slider `0–10` | `2` |
| `surface_light_ambient` | Light Ambient | integer slider `0–10` | `0` |
| `surface_light_specular` | Light Specular | integer slider `0–10` | `0` |
| `surface_specular` | Material Specular | integer slider `0–10` | `0` |
| `surface_emission` | Material Emission | integer slider `0–10` | `0` |
| `surface_shininess` | Material Shininess | integer slider `0–10` | `0` |
| `surface_bend1` | Blend Func1 | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `2` (DST_COLOR) |
| `surface_bend2` | Blend Func2 | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` (ONE_MINUS_SRC_ALPHA) |

### Device rendering

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `device_light_option` | Light | `0`=One source; `1`=Two sources; `2`=Three sources | `2` (Three sources) |
| `device_light_dir` | Light Direction | integer slider `0–10` | `5` |
| `device_light_shading` | Light Shading | integer slider `0–10` | `4` |
| `device_light_diffuse` | Light Diffuse | integer slider `0–10` | `6` |
| `device_light_ambient` | Light Ambient | integer slider `0–10` | `0` |
| `device_light_specular` | Light Specular | integer slider `0–10` | `0` |
| `device_specular` | Material Specular | integer slider `0–10` | `0` |
| `device_emission` | Material Emission | integer slider `0–10` | `0` |
| `device_shininess` | Material Shininess | integer slider `0–10` | `0` |
| `device_bend1` | Blend Func1 | `0`=ZERO; `1`=ONE; `2`=DST_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `4` (SRC_ALPHA) |
| `device_bend2` | Blend Func2 | `0`=ZERO; `1`=ONE; `2`=SRC_COLOR; `3`=ONE_MINUS_DST_COLOR; `4`=SRC_ALPHA; `5`=ONE_MINUS_SRC_ALPHA; `6`=DST_ALPHA; `7`=ONE_MINUS_DST_ALPHA | `5` (ONE_MINUS_SRC_ALPHA) |

### Labels and directional axis

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `show_track_label` | Show Track Label | `0`=Off; `1`=On | `1` (On) |
| `show_track_label_location` | Track Label Location | `0`=With Track; `1`=Middle; `2`=Middle Bottom | `0` (With Track) |
| `track_label_color` | Track Label Color | packed Qt ARGB integer | `9868955` |
| `track_label_bold` | Track Label Bold | `0`=Off; `1`=On | `1` (On) |
| `track_label_size` | Tract Label Size | integer `2–100`; step `2` | `12` |
| `show_region_label` | Show Region Label | `0`=Off; `1`=On | `1` (On) |
| `region_label_color` | Region Label Color | packed Qt ARGB integer | `9868955` |
| `region_label_bold` | Region Label Bold | `0`=Off; `1`=On | `1` (On) |
| `region_label_size` | Region Label Size | integer `2–100`; step `2` | `12` |
| `show_device_label` | Show Device Label | `0`=Off; `1`=On | `1` (On) |
| `device_label_color` | Device Color | packed Qt ARGB integer | `9868955` |
| `device_label_bold` | Device Bold | `0`=Off; `1`=On | `1` (On) |
| `device_label_size` | Device Size | integer `2–100`; step `2` | `12` |
| `show_directional_axis` | Axis | `0`=Off; `1`=On | `0` (Off) |
| `axis_line_thickness` | Axis Line Thickness | float `1.0–20.0`; step `0.5` | `10` |
| `axis_line_length` | Axis Line Length | float `1.0–10.0`; step `0.5` | `5` |
| `show_axis_label` | Axis Label | `0`=Off; `1`=On | `1` (On) |
| `axis_label_size` | Axis Label Size | integer `2–48`; step `2` | `26` |
| `axis_label_bold` | Axis Label Bold | `0`=Off; `1`=On | `1` (On) |

### ODF rendering

| Parameter ID | UI setting | Accepted value | Default |
|---|---|---|---|
| `odf_position` | Position | `0`=Along Slide; `1`=Slide Intersection; `2`=All | `0` (Along Slide) |
| `odf_scale` | Size | float `0.1–32`; step `1` | `2` |
| `odf_color` | Color | `0`=Dir; `1`=Blue; `2`=Red | `0` (Dir) |
| `odf_skip` | Interleaved | `0`=none; `1`=2; `2`=4 | `0` (none) |
| `odf_smoothing` | Smoothing | `0`=off; `1`=on | `0` (off) |
| `odf_shape` | Shape | `0`=original; `1`=1st; `2`=2nd | `0` (original) |
| `odf_min_max` | Min-Max Normalization | `0`=off; `1`=on | `1` (on) |
| `odf_light_option` | Light | `0`=One source; `1`=Two sources; `2`=Three sources | `0` (One source) |
| `odf_light_dir` | Light Direction | integer slider `0–10` | `2` |
| `odf_light_shading` | Light Shading | integer slider `0–10` | `2` |
| `odf_light_diffuse` | Light Diffuse | integer slider `0–10` | `10` |
| `odf_light_ambient` | Light Ambient | integer slider `0–10` | `0` |
| `odf_light_specular` | Light Specular | integer slider `0–10` | `0` |
| `odf_specular` | Material Specular | integer slider `0–10` | `0` |
| `odf_emission` | Material Emission | integer slider `0–10` | `1` |
| `odf_shininess` | Material Shininess | integer slider `0–10` | `0` |
