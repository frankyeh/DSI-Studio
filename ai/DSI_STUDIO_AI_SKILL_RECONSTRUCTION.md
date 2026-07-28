# DSI Studio Reconstruction Guide for AI Agents

Reconstruction converts diffusion signals into fiber orientations and metrics:

```text
DICOM or NIfTI + bval/bvec → SZ → reconstruction → FZ
```

Preserve the raw input and each important processing stage:

```text
subject_raw.sz
subject_preprocessed.sz
subject_gqi.fz
subject_qsdr.fz
```

Never overwrite the only raw SZ file. A successfully written FZ can still be
anatomically wrong.

## Tutorials

- [Acquisition and pipeline](https://www.youtube.com/watch?v=Sn2eH07axF4)
- [Reconstruction tutorial](https://www.youtube.com/watch?v=-J8qBMiHQHk)
- [DTI quality control](https://www.youtube.com/watch?v=stL4GMeTC1I)
- [Diffusion models and metrics](https://www.youtube.com/watch?v=wbrMJHD5mKs)
- [NIfTI to tractography](https://www.youtube.com/watch?v=iuBtgGLohsg)

## Required Workflow

### 1. Define the analysis

Choose the reconstruction from the downstream goal:

| Method | Use when |
|---|---|
| **GQI** | Native-space tractography, crossing fibers, tractometry, or subject-specific connectomes |
| **DTI** | Tensor metrics are required or the acquisition cannot support robust crossing-fiber reconstruction |
| **QSDR** | Group analysis, connectometry, or another workflow requiring a common template space |

Do not choose QSDR merely for convenience. Native-space GQI is preferable for
individual anatomy, lesions, distortion, and presurgical work.

### 2. Inspect the source before reconstruction

Confirm:

- dimensions, voxel size, volume count, b-values, and b-vector count;
- brain coverage and image orientation;
- motion, distortion, slice dropout, signal spikes, and background noise;
- neighboring-DWI correlation;
- correspondence between every image volume and b-table row.

Stop and report severe corruption. Reconstruction parameters cannot recover
missing slices, inadequate coverage, very low SNR, insufficient directions, or
susceptibility information that was never acquired.

### 3. Apply only justified corrections

- With suitable reverse-phase data, use TOPUP for susceptibility distortion and
  EDDY for eddy-current distortion and motion.
- Without reverse-phase data, EDDY can address motion and eddy currents but
  cannot fully correct susceptibility distortion.
- Avoid repeated interpolation, registration, smoothing, or resampling.
- Isotropic resampling is optional and does not create true spatial resolution.
- Save corrected data as a new SZ file.

State which artifact each operation addresses.

### 4. Verify image and b-table orientation

Check left-right, anterior-posterior, superior-inferior, laterality, and template
compatibility. Image flips or axis swaps must transform b-vectors consistently.
After any orientation operation, recheck anatomical landmarks and record the
change; an apparently plausible image may still be mirrored.

Automatic b-table checking is evidence, not proof. Confirm its result using
anatomy, local fiber directions, and whole-brain tractography. Be cautious with
low-SNR, low-direction, partial-coverage, animal, or severely pathological data.

### 5. Inspect the mask

The mask should include the entire brain and peripheral white matter without
admitting excessive background or disconnected fragments. Inspect axial,
coronal, and sagittal views.

- A small mask truncates pathways and cortical endpoints.
- A broad mask adds background orientations and wastes computation.

### 6. Select reconstruction settings

#### GQI

Start near:

- **1.25** for typical in-vivo human diffusion MRI;
- **0.6** for typical ex-vivo diffusion MRI.

These are starting points, not constants. When optimization is needed, compare
several sampling-length ratios. Select the highest value that resolves expected
crossings without producing false secondary fibers in coherent regions such as
the corpus callosum. A low value may merge crossings; an excessive value may
create spurious orientations.

#### DTI

Use DTI for FA, MD, AD, RD, tensor elements, and principal eigenvectors. A
single tensor cannot resolve multiple fiber populations. For multishell data,
consider restricting conventional tensor fitting to suitable lower b-values
and record the shells used.

#### QSDR

Select the correct species template and a resolution appropriate to the
acquisition and brain size. Smaller output voxels increase computation and file
size but do not restore missing information. Verify nonlinear registration,
laterality, and attached T1w/T2w alignment.

### 7. Request only needed outputs

Possible outputs include `fa`, `md`, `ad`, `rd`, `tensor`, `gfa`, `rdi`, and
`odf`. Full ODF storage can greatly enlarge the FZ file and is unnecessary for
ordinary tractography unless a downstream method explicitly needs it.

### 8. Validate the FZ

Open the result and inspect:

- anisotropy contrast and laterality;
- dominant directions in coherent white matter;
- plausible multiple directions in known crossing regions;
- the mask boundary;
- QSDR alignment when applicable;
- whole-brain tractography and major commissural/projection pathways.

Do not validate reconstruction from one QA, FA, or color map alone. Widespread
tract failure usually indicates source quality, b-table, orientation, mask, or
reconstruction problems rather than one tract definition.

### 9. Batch only compatible inputs

Use one batch only when every dataset should receive the same corrections,
orientation operations, mask strategy, reconstruction method, template,
sampling length, resolution, and outputs. Separate heterogeneous acquisitions,
species, orientations, or subjects needing unique corrections.

## Common Failures

| Observation | Check first |
|---|---|
| Major tracts have globally wrong directions | Image orientation and b-table flips/swaps |
| Implausible secondary directions in coherent white matter | Sampling length, noise, and b-table |
| Crossings are not resolved | Sampling length and acquisition angular sampling |
| Fibers appear outside the brain | Mask extent and background noise |
| Peripheral pathways are missing | Restrictive mask |
| QSDR anatomy is distorted | Template, orientation, registration, and pathology |
| Many AutoTrack bundles fail | SRC quality, motion, b-table, mask, and whole-brain tracking |
| Left and right are reversed | Flip/swap history and handedness |
| FZ is unexpectedly large | Full ODF or unused metrics |

Do not repeatedly tune parameters to make a poor acquisition look attractive.

## Example Commands

Default native-space GQI:

```bash
dsi_studio --action=rec --source=subject.sz --method=4 --param0=1.25 --output=subject_gqi.fz
```

Batch GQI:

```bash
dsi_studio --action=rec --source=*.sz --method=4 --param0=1.25 --output=fib/
```

EDDY followed by GQI:

```bash
dsi_studio --action=rec --source=subject.sz --cmd="[Step T2][Corrections][EDDY]" --method=4 --param0=1.25 --output=subject_gqi.fz
```

TOPUP/EDDY preparation using reverse-phase data:

```bash
dsi_studio --action=rec --source=subject_raw.sz --rev_pe=subject_rev_b0.nii.gz --save_src=subject_preprocessed.sz
```

QSDR:

```bash
dsi_studio --action=rec --source=subject.sz --method=7 --output=subject_qsdr.fz
```

Available parameters may vary by version. Prefer commands captured from the
current GUI command history when reproducing an interactive workflow.

## Record for Reproducibility

Record:

- DSI Studio version and all input/output filenames;
- acquisition dimensions, voxel size, volumes, shells, and phase encoding;
- reverse-phase input, TOPUP, EDDY, motion, and bad-slice results;
- neighboring-DWI correlation;
- every image and b-table flip or axis swap;
- mask source and edits;
- resampling;
- reconstruction method and GQI sampling length;
- QSDR template and resolution;
- DTI shell selection;
- requested metrics, ODF setting, and attached images;
- post-reconstruction and whole-brain tracking QC.
