# DSI Studio Tractography Principles for AI Agents

This guide summarizes the tractography principles presented in the DSI Studio tutorial videos, with emphasis on parameter settings, ROI usage, quality control, and reproducible agent workflows.

Tutorials:

- https://www.youtube.com/watch?v=xyFNXB9nJ90
- https://www.youtube.com/watch?v=oJK8jwTHVhc
- https://www.youtube.com/watch?v=V2pxI2tooPs

---

## 1. What Tractography Represents

Tractography reconstructs plausible fiber trajectories by following local diffusion orientations. A streamline is a computational trajectory, not a directly observed axon.

The result depends on:

- diffusion data quality;
- reconstruction quality;
- local fiber orientations;
- tracking parameters;
- seeding strategy;
- ROI constraints;
- post-tracking filtering.

An agent should not assume that a tractography result is anatomically valid merely because streamlines were generated.

Before running tracking, identify the analysis goal:

- whole-brain quality inspection;
- manual ROI-based pathway dissection;
- atlas-based automatic tract recognition;
- connectome construction;
- tract-specific quantitative analysis.

The tracking workflow and parameter settings should follow the analysis goal.

---

## 2. Begin with Whole-Brain Tracking

Before extracting a specific tract, generate or inspect whole-brain tractography.

Whole-brain tracking should show:

- coherent major association, projection, and commissural pathways;
- plausible left-right symmetry;
- no systematic directional flipping;
- no obvious failure in known crossing-fiber regions;
- no dominant abnormal orientation caused by an incorrect b-table;
- adequate white-matter coverage.

If many tracts fail simultaneously, the likely cause is acquisition, reconstruction, image orientation, or b-table quality rather than the ROI definition of one tract.

**Agent rule:** Do not repeatedly modify ROIs to rescue a tract when whole-brain tracking is abnormal. Inspect reconstruction and data quality first.

---

## 3. Tracking Threshold

The tracking threshold determines whether the local diffusion signal is sufficient for tracking to continue.

### Higher threshold

- produces fewer and more conservative streamlines;
- terminates tracking earlier;
- may reduce tracking in gray matter, lesions, edema, crossing regions, or areas with reduced anisotropy.

### Lower threshold

- permits more extensive tracking;
- may recover pathways in low-anisotropy regions;
- increases the chance of entering uncertain tissue and generating false trajectories.

The threshold should be interpreted relative to the selected tracking index, such as QA or FA.

**Agent rule:** Begin with the default or automatically estimated threshold. Change it only for a documented reason, such as widespread premature termination or excessive tracking into low-anisotropy tissue.

Do not tune the threshold until a desired-looking tract appears.

---

## 4. Angular Threshold

The angular threshold limits how sharply a streamline may turn between tracking steps.

### Smaller angular threshold

- favors straighter trajectories;
- is more conservative;
- may fail to follow sharply curved pathways.

### Larger angular threshold

- permits greater curvature;
- may help with curved pathways;
- increases false turns in crossing-fiber regions.

The appropriate value depends on tract anatomy. Projection and callosal pathways generally require less curvature than sharply bending pathways such as the uncinate fasciculus or Meyer’s loop.

**Agent rule:** Do not increase the angular threshold solely because a tract is incomplete. First inspect ROI placement, local fiber orientations, and whether the expected pathway truly bends at that location.

---

## 5. Step Size

Step size is the distance advanced at each tracking iteration.

### Smaller step size

- follows local curvature more finely;
- produces more points;
- increases computation;
- may respond more strongly to local noise.

### Larger step size

- is faster;
- produces smoother and coarser trajectories;
- may skip local curvature or small structures.

**Agent rule:** Use the default or voxel-size-aware setting unless a fixed research protocol requires another value. A smaller step size is not automatically more accurate.

---

## 6. Smoothing

Smoothing mixes the current local direction with the previous tracking direction.

### Lower smoothing

- follows local diffusion orientations more directly;
- may produce less visually smooth trajectories.

### Higher smoothing

- increases directional persistence;
- produces smoother trajectories;
- may ignore genuine local curvature or continue through unsupported directions.

Smoothing is a trajectory regularization parameter, not merely a display option.

---

## 7. Length Limits

Minimum and maximum length constrain accepted streamlines.

### Minimum length

A minimum length removes:

- short noisy fragments;
- tracks confined to a small region;
- disconnected segments;
- short trajectories near uncertain boundaries.

A high minimum length may incorrectly remove valid short pathways, including:

- U-fibers;
- short commissural fibers;
- cranial nerves;
- short brainstem connections;
- local lesion-adjacent pathways.

### Maximum length

A maximum length removes unusually long trajectories caused by:

- looping;
- permissive angular thresholds;
- erroneous continuation;
- propagation through unrelated systems.

**Agent rule:** Choose length limits according to expected anatomy and image scale. Do not apply the same minimum length to every tract.

---

## 8. Tract Count, Seed Count, and Yield Rate

DSI Studio can terminate tracking based on:

- a requested number of accepted tracts;
- a requested number of seeds;
- another stopping criterion.

These are not equivalent.

### Tract count

Tracking continues until the requested number of accepted streamlines is obtained.

This is useful for obtaining a sufficiently dense bundle, but difficult pathways may require a very large number of seed attempts.

### Seed count

A seed limit caps the total number of tracking attempts. It prevents difficult tract searches from running indefinitely.

### Recommended AutoTrack limits

For AutoTrack and TIP-based cleanup:

- use a tract limit of **10,000 tracts**;
- use a seed limit of approximately **1,000 times the tract limit**;
- for 10,000 requested tracts, use a seed limit of approximately **10,000,000 seeds**.

Some difficult tracts have a very low seed-to-tract yield. The accepted tract yield may be around 1% or lower. A large seed limit provides enough attempts to approach the requested tract count while still preventing excessively long execution.

If the requested tract count is reached first, tracking stops normally. If the seed limit is reached first, the output may contain fewer than 10,000 tracts.

**Agent rule:** For AutoTrack, use both a tract limit and a seed limit. The tract limit provides adequate bundle density, while the seed limit prevents difficult tracts from taking excessively long.

Do not interpret streamline count as axon count, fiber count, or biological connectivity strength.

---

## 9. Topology-Informed Pruning

Topology-informed pruning, or TIP, removes noisy and topologically isolated streamlines.

TIP works best when the reconstructed bundle contains sufficient streamline density. A bundle should preferably contain approximately **10,000 tracts** before TIP is applied.

If the tract count is too low:

- the bundle may have insufficient local topological support;
- valid streamlines may appear isolated;
- TIP may remove most or all of the tract.

TIP is therefore used mainly with AutoTrack, where a dense initial tract set can be generated before pruning.

### Recommended TIP workflow

1. Request up to 10,000 accepted tracts.
2. Set the seed limit to approximately 1,000 times the tract limit.
3. Run AutoTrack.
4. Confirm that enough tracts were generated.
5. Apply the requested TIP iterations.
6. Inspect whether valid tract cores remain.

**Agent rule:** Do not apply TIP aggressively to a sparse bundle. If the tract count is low, increase sampling or review tracking and ROI settings before pruning.

TIP improves bundle coherence but does not prove anatomical validity.

---

# ROI Principles

## Prefer Segmentation-Derived Regions

Whenever possible, derive Seed, ROI, ROA, End, and Terminative regions from
anatomical segmentation rather than drawing them from scratch. Prefer
segmentation of an aligned T1w image in the tracking window. Segmented labels
can be selected and manually merged to create most anatomical region sets, then
assigned the required tracking roles.

If no T1w image is available, segment the isotropic diffusion image (`iso`).
Many brain-segmentation models are modality agnostic and can work with either
T1w or `iso`, but always inspect the label boundaries and registration before
using them for tracking. Do not assume successful inference means anatomically
valid regions.

## 10. ROI Types Have Different Functions

An agent must distinguish ROI roles. Regions are not interchangeable.

### Seed region

A seed region specifies where tracking begins.

Use a seed when:

- the pathway is known to pass through a specific tract core;
- tracking should focus on a limited anatomical area;
- whole-brain seeding would be inefficient.

A streamline seeded in a region does not need to terminate there.

### Inclusive ROI

An inclusive ROI requires the streamline to pass through the region.

Multiple inclusive ROIs generally implement an AND condition:

> Retain streamlines passing through ROI A and ROI B.

Use inclusive ROIs as anatomical waypoints.

### ROA or exclusion region

An ROA rejects any streamline entering the region.

Use it to remove:

- contralateral fibers;
- cerebellar fibers;
- unrelated projection systems;
- anatomically impossible branches;
- fibers crossing a known exclusion boundary.

### End region

An end region requires streamline termination in the region rather than simple passage through it.

This distinction is important for region-to-region connectivity. A streamline may pass through a region without ending there.

### Terminative region

A terminative region stops tracking when a streamline reaches it.

This changes tracking during propagation, whereas an inclusive ROI usually filters trajectories according to whether they pass through a region.

---

## 11. Seed and ROI Are Not Interchangeable

A seed controls where trajectories are initiated.

An ROI controls which trajectories are retained.

Using the same mask as a seed and as an ROI can produce different streamline distributions.

Example:

- seeding in the internal capsule densely initiates fibers from that area;
- using the internal capsule as an ROI retains fibers generated elsewhere that pass through it.

**Agent rule:** Always state the role of each region explicitly.

---

## 12. ROI Combinations Encode Anatomical Hypotheses

ROI-based tractography should implement a neuroanatomical definition.

A manual tract dissection may include:

- one seed region;
- one or more waypoint ROIs;
- one or more exclusion ROAs;
- optional endpoint regions;
- minimum and maximum length constraints.

Example:

> Generate fibers passing through the left posterior limb of the internal capsule and the left cerebral peduncle, then exclude fibers entering the cerebellum or crossing the midline.

Each ROI should have a stated anatomical purpose.

---

## 13. ROI Size and Placement

### Oversized ROI

An ROI extending into adjacent structures may retain unrelated pathways.

This is particularly important in compact regions such as:

- centrum semiovale;
- temporal stem;
- external and extreme capsules;
- brainstem;
- thalamus;
- internal capsule.

### Undersized ROI

A very small ROI may miss valid fibers because of:

- anatomical variation;
- registration mismatch;
- partial volume;
- tract dispersion;
- limited spatial resolution.

### Boundary placement

ROIs placed directly at gray-white boundaries, lesions, or low-anisotropy margins may be unstable because tracking often terminates near these areas.

**Agent rule:** Verify ROIs in all three orthogonal planes. ROIs should be specific enough to avoid adjacent pathways but large enough to tolerate modest anatomical variation.

---

## 14. Spatial Intersection Does Not Establish Tract Identity

A tract-derived region represents spatial occupancy, not fiber orientation.

A new streamline may intersect the same voxels while traveling in a perpendicular or unrelated direction.

Therefore:

> Passing through the same voxels does not mean belonging to the same anatomical tract.

Do not use a tract-derived ROI as proof that newly generated fibers belong to the original tract.

For named pathway recognition, trajectory-aware atlas methods such as AutoTrack may be more appropriate.

---

## 15. Exclusion ROIs Should Be Conservative

An ROA can remove many streamlines at once. Poorly placed exclusion regions may eliminate valid anatomy.

Before adding an ROA, the agent should identify:

- the false-positive branch being removed;
- why it is anatomically incompatible;
- whether the exclusion intersects valid fibers;
- whether the same exclusion can be applied consistently across subjects.

Avoid repeatedly adding ROAs merely to sculpt a visually clean bundle.

---

## 16. Endpoint Constraints

Endpoint constraints are useful when the scientific question concerns structural connections between two regions.

They are sensitive to:

- low anisotropy near cortex;
- gray-white interface segmentation;
- gyral bias;
- tracking threshold;
- reconstruction resolution;
- endpoint mask dilation or erosion.

Failure to reach the cortex does not necessarily indicate absence of an anatomical connection.

The agent should distinguish:

- tract core mapping;
- region-to-region connectivity;
- named tract recognition.

---

# Manual ROI Tracking and AutoTrack

## 17. Manual ROI Tracking

Use manual ROI tracking when:

- anatomy is distorted;
- lesions alter tract position;
- the pathway is not represented in an atlas;
- the investigator requires an explicit anatomical definition;
- ROIs can be placed reliably.

Advantages:

- flexible;
- transparent anatomical logic;
- useful for unusual structures.

Limitations:

- operator dependence;
- difficult reproducibility;
- sensitivity to ROI size and placement;
- labor-intensive for large cohorts.

---

## 18. AutoTrack

AutoTrack is appropriate when:

- a standard named pathway is requested;
- a compatible tract atlas exists;
- reproducible cohort processing is needed;
- manual ROI variability should be reduced.

AutoTrack generally:

1. maps subject diffusion information to template space;
2. seeds within or near the atlas tract volume;
3. generates candidate streamlines;
4. compares candidate trajectories with atlas tract trajectories;
5. retains trajectories matching the target bundle;
6. applies optional pruning and filtering.

Important settings include:

- tract limit;
- seed limit;
- minimum length;
- AutoTrack tolerance;
- TIP iterations.

### AutoTrack tolerance

A larger tolerance:

- accepts more anatomical variation;
- may retain more false-positive trajectories.

A smaller tolerance:

- is more restrictive;
- may fail in distorted or variable anatomy.

**Agent rule:** Use AutoTrack for standard named pathways, but always inspect the result. Atlas recognition does not replace quality control.

---

# Recommended Agent Workflow

## 19. Step-by-Step Workflow

### Step 1: Identify the requested output

Classify the task as:

- whole-brain tractography;
- named tract;
- ROI-defined tract;
- endpoint connectivity;
- connectome analysis;
- tract statistics;
- presurgical visualization.

### Step 2: Inspect source data

Confirm:

- a valid `.fib.gz` or `.fz` file is loaded;
- an aligned T1w image is available, or use `iso` for segmentation;
- reconstruction space;
- voxel size;
- tracking index;
- image orientation;
- whole-brain tracking quality.

### Step 3: Select the tracking strategy

- Whole brain: broad seeding without tract-specific ROI restrictions.
- Named canonical tract: use AutoTrack when available.
- Custom pathway: use seed, ROI, ROA, and endpoint logic.
- Distorted anatomy: use subject-specific manual constraints and careful inspection.

### Step 4: Start from defaults

Begin with DSI Studio default settings for:

- tracking threshold;
- angular threshold;
- step size;
- smoothing;
- length limits.

Change one parameter at a time and document the reason.

### Step 5: Define ROI logic

Prefer T1w-based segmentation, or `iso` segmentation when T1w is unavailable.
Select and merge anatomical labels to form the needed region sets before
assigning each tracking role.

For every region, record:

- name;
- source;
- coordinate space;
- region type;
- anatomical purpose.

Example:

```text
Seed: left precentral white matter
ROI: left posterior limb of internal capsule
ROI2: left cerebral peduncle
ROA: corpus callosum
ROA2: cerebellum
```

### Step 6: Set sampling limits

For AutoTrack:

```text
tract limit: 10,000
seed limit: 10,000,000
```

This provides enough streamlines for TIP while limiting execution time for difficult tracts.

### Step 7: Generate and inspect the initial bundle

Check:

- tract count;
- tract course;
- unexpected branches;
- endpoint distribution;
- relationship to lesions;
- left-right symmetry when appropriate.

### Step 8: Apply justified filtering

Use:

- minimum length to remove fragments;
- ROAs to remove anatomically defined false branches;
- endpoint checks for connection analysis;
- AutoTrack tolerance for atlas matching;
- TIP only when streamline density is sufficient.

### Step 9: Verify TIP output

After TIP:

- confirm that the main tract core remains;
- check whether most or all tracts were removed;
- if the bundle disappears, review tract count, sampling, tolerance, threshold, and ROI settings.

### Step 10: Record all parameters

Save or report:

```text
tracking method
tracking index
threshold
turning angle
step size
smoothing
minimum and maximum length
seed region
inclusive ROIs
ROAs
endpoint regions
terminative regions
tract limit
seed limit
TIP iterations
AutoTrack tract name
AutoTrack tolerance
output tract file
```

### Step 11: Preserve reproducibility

For repeated analyses, use:

- command history;
- parameter codes;
- CLI commands;
- identical ROI sources;
- consistent file naming;
- fixed random seed when exact repeatability is required.

---

# Parameter Adjustment Guide

| Observation | Likely cause | Preferred response |
|---|---|---|
| Most tracts stop too early | Threshold too high or poor reconstruction | Inspect whole-brain tracking and reconstruction before lowering threshold |
| Tracts enter gray matter or CSF extensively | Threshold too low | Increase threshold modestly |
| A real sharp bend is not followed | Angular threshold too restrictive | Increase angle cautiously |
| Many implausible sharp turns appear | Angular threshold too permissive | Reduce angle and inspect crossing regions |
| Tract is fragmented | Low sampling, high threshold, restrictive ROIs, or high minimum length | Increase sampling and inspect constraints |
| Too many unrelated pathways appear | ROI too broad or missing waypoint/ROA | Refine ROI logic |
| Tract disappears after adding an ROA | ROA intersects valid anatomy | Inspect the ROA in three planes and reduce it |
| Short noisy fibers remain | Minimum length too low | Increase minimum length conservatively |
| Isolated noisy fibers remain | Topological outliers | Apply TIP after confirming adequate tract count |
| TIP removes nearly all fibers | Initial tract count too low or bundle too sparse | Increase sampling or revise tracking settings before pruning |
| AutoTrack takes too long | Low seed-to-tract yield | Keep the 10,000 tract limit and use a finite seed limit |
| AutoTrack stops below 10,000 tracts | Seed limit reached before tract limit | Review yield, tolerance, threshold, and tract difficulty |
| AutoTrack fails for many pathways | Data, b-table, orientation, or reconstruction problem | Return to whole-brain quality control |
| AutoTrack fails for one difficult tract | Low tract yield or anatomical distortion | Increase tolerance or sampling cautiously and document uncertainty |
| Tract counts differ across subjects | Sampling and anatomy differ | Use standardized tract and seed limits and avoid biological interpretation of count |

---

# Core Principles

1. Whole-brain tracking is the first diagnostic test.
2. Tracking parameters control uncertainty; they do not create missing anatomy.
3. Seed, ROI, ROA, endpoint, and terminative regions have distinct functions.
4. Every ROI should encode a stated anatomical hypothesis.
5. Spatial intersection does not establish tract identity.
6. Streamline count is a sampling result, not a biological fiber count.
7. Start with default parameters and modify one setting for a documented reason.
8. AutoTrack improves standardization but still requires visual quality control.
9. TIP removes noisy and isolated trajectories but requires a sufficiently dense bundle.
10. Prefer approximately 10,000 initial tracts before TIP.
11. For AutoTrack, combine a 10,000-tract limit with a seed limit approximately 1,000 times larger.
12. If the tract count is too low, TIP may remove most or all streamlines.
13. Every tractography result should retain its parameter and ROI provenance.

---

# Concise Agent Instruction

```text
Before running tractography, identify whether the task is whole-brain,
named-tract, ROI-based, endpoint connectivity, or connectome analysis.

Inspect whole-brain tracking first. Start with default tracking parameters.
Prefer T1w-based segmentation for regions, or `iso` segmentation when T1w is
unavailable. Merge anatomical labels as needed, then define every region
explicitly as seed, ROI, ROA, endpoint, or terminative region and state its
anatomical role.

For AutoTrack, use a tract limit of 10,000 and a seed limit of approximately
10,000,000. Difficult tracts may have a very low seed-to-tract yield, so the
seed limit prevents excessively long execution while allowing enough attempts
to build a dense bundle.

Apply TIP mainly to dense AutoTrack results. A bundle should preferably contain
about 10,000 tracts before TIP. If the tract count is too low, TIP may remove
most or all tracts.

Do not modify parameters merely to obtain a desired-looking pathway. Inspect
the initial trajectories before applying length filters, exclusion regions,
endpoint checks, AutoTrack tolerance, or TIP.

Record all parameters, region definitions, stopping criteria, and output files
so the result can be reproduced.
```
