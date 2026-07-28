# DSI Studio Fiber-Tracking Guide for AI Agents

Tractography follows reconstructed diffusion orientations. A streamline is a
computational trajectory, not an observed axon or a measure of connectivity
strength. Anatomical validity requires source QC, justified tracking settings,
explicit region logic, and visual inspection.

## Tutorials

- https://www.youtube.com/watch?v=xyFNXB9nJ90
- https://www.youtube.com/watch?v=oJK8jwTHVhc
- https://www.youtube.com/watch?v=V2pxI2tooPs

## Choose the Workflow

| Goal | Preferred approach |
|---|---|
| Data/reconstruction QC | Whole-brain tracking |
| Standard named pathway | AutoTrack when the atlas contains it |
| Custom or distorted pathway | Manual Seed/ROI/ROA constraints |
| Region-to-region connectivity | End-region constraints |
| Cohort standardization | AutoTrack with fixed settings and QC |

Begin by inspecting whole-brain tracking. Major pathways should be coherent,
plausibly symmetric, and free of systematic flips. If many bundles fail, inspect
the acquisition, reconstruction, b-table, orientation, and mask before changing
tract-specific regions.

## Tracking Parameters

Start from DSI Studio defaults. Change one setting at a time for a documented
reason; do not tune parameters until a desired-looking tract appears.

| Parameter | Lower value | Higher value | Rule |
|---|---|---|---|
| Tracking threshold | Extends into uncertain tissue | Stops earlier and may miss low-anisotropy fibers | Change only for widespread premature stopping or excessive low-anisotropy tracking |
| Angular threshold | Straighter and conservative | Follows sharper curves but permits false turns | Match expected tract anatomy |
| Step size | Finer and slower | Coarser and faster | Use default or a validated voxel-aware protocol |
| Smoothing | Follows local directions | Adds directional persistence | Treat as trajectory regularization, not display smoothing |
| Minimum length | Retains short/noisy fragments | Removes short valid pathways too | Match expected anatomy |
| Maximum length | Allows long trajectories | Limits loops and erroneous continuation | Set a plausible anatomical bound |

Tract count is the number of accepted streamlines. Seed count limits attempts.
Neither represents axons or biological connection strength.

## Build Regions from Anatomy

Prefer anatomical segmentation over drawing regions from scratch:

1. Segment an aligned T1w image in the tracking window.
2. If T1w is unavailable, segment the isotropic diffusion image (`iso`).
3. Select and manually merge segmented labels to form the needed region sets.
4. Inspect boundaries and registration in all three planes.
5. Assign every region an explicit tracking role.

Many segmentation models are modality agnostic, but successful inference does
not prove anatomical validity.

| Region role | Effect |
|---|---|
| Seed | Starts trajectories |
| ROI | Retains trajectories passing through it |
| ROA | Rejects trajectories entering it |
| End | Requires termination in the region |
| Terminative | Stops propagation when reached |
| NotEnd | Rejects termination in the region |
| Limiting | Constrains propagation to the region |

Seed and ROI are not interchangeable. Multiple inclusive ROIs usually express
an AND condition. Every region should have a stated anatomical purpose.

Avoid oversized regions that include adjacent pathways, undersized regions that
miss anatomy or registration variation, and ROAs that intersect valid fibers.
Spatial overlap alone does not establish tract identity because unrelated
trajectories can cross the same voxels.

## Manual Tracking

Use manual tracking for distorted anatomy, lesions, pathways absent from the
atlas, or a required explicit anatomical definition.

Before `run_tracking`:

```json
["list_param","tracking"]
["list_region"]
["run_tracking","<bundle-name>"]
```

Call `list_region` only after regions were created, loaded, segmented, or
restored. A newly opened FIB normally has no regions.

Record the Seed, ROI, ROA, endpoint, and Terminative logic. Inspect the initial
bundle before adding filters or exclusion regions.

## AutoTrack

Use AutoTrack for standard named pathways and reproducible cohort workflows.
Always discover the exact internal atlas identifier:

```json
["list_auto_tract"]
["run_auto_track","ProjectionBrainstem_CorticospinalTractL"]
```

Atlas names use underscore-separated hierarchical prefixes such as
`Association_*`, `ProjectionBrainstem_*`, and `Commissure_*`. Never guess or use
human-readable shorthand such as `Corticospinal Tract`.

Recommended dense AutoTrack sampling:

```text
tract limit: 50,000
seed limit: 50,000,000
```

The seed limit prevents difficult, low-yield pathways from running indefinitely.
If it is reached first, fewer than 50,000 tracts may be produced. Adjust
AutoTrack tolerance cautiously: larger values accept more variation and false
positives; smaller values may reject distorted or variable anatomy.

## TIP Cleanup

Topology-informed pruning removes isolated, noisy trajectories and works best
on dense bundles.

- Bundles below 1,000 tracts are generally unsuitable for TIP.
- AutoTrack applies its configured `tip_iteration` automatically.
- `trim_tract` applies one additional iteration to **every checked bundle**.

Recommended cleanup:

1. Set the limits and disable automatic TIP, then run AutoTrack:
   `["set_params","max_tract_count=50000&max_seed_count=50000000&tip_iteration=0"]`.
2. Uncheck every non-target bundle.
3. Apply `["trim_tract"]` four or five times, inspecting the result after each
   round.
4. Run `["delete_repeated_tract","1"]`; `1` voxel is the default distance.
5. Apply secondary `["trim_tract"]` rounds one at a time until approximately
   10,000 clean trajectories remain. Stop earlier if the valid tract core
   deteriorates.

## Result Cleanup and Visualization

After tracking finishes:

1. Poll `["list_tract","status"]` until `status=done`.
2. Use `list_tract` to identify the target and whole-brain bundle indices.
3. Complete the TIP and repeated-tract cleanup above with only the target
   checked.
4. Run `["color_all_cluster"]` to assign distinct bundle colors.
5. Hide the whole-brain bundle with
   `["check_tract","<whole-brain-index>","0"]`.
6. Display one mapped bundle with
   `["show_only_tracts","<target-index>"]`.
7. Add anatomical context using the subject-mapped built-in white-matter
   isosurface: `["add_surface","0","25"]`.

Choose a useful inspection view for each bundle rather than reusing one camera.
For the left arcuate fasciculus, inspect from a left-anterior-superior oblique
position. Start with `["set_view","0"]`, then adjust in small increments:

```json
["rotate","15 1 0 0"]
["rotate","20 0 1 0"]
```

Verify orientation and capture several oblique views. Do not obscure mapped
bundles with whole-brain streamlines.

TIP and repeated-tract deletion modify checked bundles. Preserve the original
or obtain user approval when cleanup must remain recoverable.

## Quality-Control Guide

| Observation | Check or response |
|---|---|
| Most tracts stop early | Reconstruction, mask, and tracking threshold |
| Tracts enter gray matter or CSF | Threshold may be too low |
| Expected curve is missing | ROI placement and local orientations before raising angle |
| Implausible sharp turns | Reduce angular threshold and inspect crossings |
| Bundle is fragmented | Sampling, threshold, restrictive regions, and minimum length |
| Many unrelated branches | Region size, waypoint logic, and conservative ROAs |
| Bundle vanishes after ROA | ROA likely intersects valid anatomy |
| TIP removes nearly everything | Bundle is too sparse or tracking settings are poor |
| AutoTrack takes too long | Low yield; retain a finite seed limit |
| Many AutoTrack pathways fail | Data, b-table, orientation, or reconstruction problem |
| One pathway fails | Tract yield, tolerance, or anatomical distortion |

Endpoint tracking is sensitive to gray-white segmentation, gyral bias,
anisotropy near cortex, and endpoint-mask extent. Failure to reach cortex does
not necessarily prove absence of a connection.

## Reproducibility

Record:

- source FIB and reconstruction space;
- tracking method, index, threshold, angle, step size, and smoothing;
- minimum/maximum length and stopping criteria;
- tract and seed limits;
- every region name, source, space, role, and anatomical purpose;
- AutoTrack identifier, tolerance, and TIP iterations;
- repeated-tract threshold and other cleanup;
- random seed when exact repeatability is required;
- output tract files and QC findings.
