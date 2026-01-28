# DSI Studio
[![GitHub release](https://img.shields.io/github/v/release/frankyeh/DSI-Studio)](https://github.com/frankyeh/DSI-Studio/releases)
[![Last commit](https://img.shields.io/github/last-commit/frankyeh/DSI-Studio)](https://github.com/frankyeh/DSI-Studio/commits/master)

Follow: [![Twitter](https://img.shields.io/twitter/follow/FangChengYeh?style=social&logo=twitter)](https://twitter.com/FangChengYeh)  
Subscribe: [![YouTube](https://img.shields.io/youtube/channel/subscribers/UCN6gohY_zeBpK6SwJ7hnz1Q?style=social)](https://www.youtube.com/c/FrankYeh)

**Official Website:** https://dsi-studio.labsolver.org  
**User Forum:** https://groups.google.com/g/dsi-studio

---

## Overview

DSI Studio is a standalone software package for **diffusion MRI (dMRI)** reconstruction, **deterministic tractography**, and **connectome analysis**. It is designed to be lightweight and practical for everyday use, while still supporting advanced analysis features for large studies and reproducible pipelines.

DSI Studio supports:
- End-to-end workflows from raw diffusion data to tractography and connectomes
- Interactive analysis in the GUI and automated batch processing via CLI
- Multiple reconstruction models (DTI, GQI, QSDR, and related derivatives)
- Export of tracts, voxelwise maps, region-to-region connectomes, and tract-to-region connectomes

If you use DSI Studio in your work, please cite the relevant methods (see [Citations](#citations)).

---

## Key Features

### Diffusion Reconstruction
- Import diffusion datasets from **DICOM** or **NIfTI**
- Reconstruct diffusion models and generate fiber data (`.fz`)
- Support for common workflows including:
  - **DTI** (tensor-derived metrics)
  - **GQI** (model-free ODF reconstruction)
  - **QSDR** (template space reconstruction for group analysis)

### Tractography and Bundle Analysis
- Deterministic fiber tracking with configurable thresholds and stopping criteria
- ROI-based tractography with interactive editing
- Automated bundle mapping using atlas-based definitions (template-based workflows)
- Tools for tract visualization, clustering, and shape-related analyses

### Connectome Mapping
- Generate **region-to-region** connectivity matrices
- Generate **tract-to-region** connectomes (atlas-driven representations)
- Export results in common formats for downstream analysis

### Quality Control and Practical Workflow Tools
- Built-in QC utilities (including diffusion signal consistency measures)
- Parameter tracking: output files store the key settings used to generate them
- Structured file formats (`.sz`, `.fz`, `.tt.gz`) designed for reproducibility and portability

### GUI + CLI
- GUI for interactive workflows and visual inspection
- CLI for reproducible pipelines, batch processing, and HPC scripting

---

## Quick Start

### Install (No Compilation Needed)

DSI Studio is distributed as a **standalone executable**.

1. Download from: https://dsi-studio.labsolver.org/download.html  
2. Choose your platform:
   - `dsi_studio_64.exe` (Windows)
   - `dsi_studio_mac.dmg` (macOS)
   - `dsi_studio_ubuntu.zip` (Linux)
3. Launch the executable.

Notes:
- **Windows/Linux:** no installation required  
- **macOS:** you may need to grant execution permission (see download page)  
- **GPU build:** requires an NVIDIA GPU and CUDA toolkit installed

---

## System Requirements

### Supported Platforms
- **Windows:** 64-bit (Windows 10 or newer)
- **macOS:** Intel or Apple Silicon (macOS 13+)
- **Linux:** Ubuntu 18.04 or newer (tested on 20.04, 22.04)

### Dependencies
- None for CPU builds (standalone executable)
- CUDA toolkit required for GPU builds

### Recommended Hardware
- CPU: 4+ cores
- RAM: 8+ GB (more recommended for large datasets)
- GPU: NVIDIA GPU recommended for GPU builds and large-scale tracking

---

## Demo (Try It in Minutes)

### Option A: Use Built-in Sample Fiber Data
1. Launch `dsi_studio`
2. Go to the **Fiber Data** tab
3. Select a `.fz` file and click **Open** to enter the tracking window

<img src="https://github.com/user-attachments/assets/13049248-282d-4295-9cfb-880f1f89315c" width="420"/>

4. Click **Fiber Tracking** to start tractography

<img src="https://github.com/user-attachments/assets/392383ac-3425-4413-88c1-8d356f7c710b" width="420"/>

5. Visualize and export results from the top menu

<img src="https://github.com/user-attachments/assets/7b78441a-6b76-4bc8-bfb0-ff1af82aac52" width="420"/>

**Typical runtime:** ~1–3 minutes on a standard desktop (depends on settings and dataset)

### Option B: Download Data from Fiber Data Hub
You can also browse and download preprocessed fiber datasets from:  
https://brain.labsolver.org

---

## Workflow with Your Own Data (GUI)

A typical GUI workflow is:

1. **Import diffusion data:** `File → Open → DICOM/NIfTI`
2. **Convert to `.sz`:** use **Step T1** conversion
3. **Reconstruct to `.fz`:** choose a reconstruction method (e.g., DTI, GQI)
4. **Run tractography:** use ROIs, atlas-based bundles, or your own definitions
5. **Export:** tracts, maps, connectomes, and summary reports

Documentation entry point: https://dsi-studio.labsolver.org

---

## Command-Line Interface (CLI)

DSI Studio supports CLI scripting for reproducible batch processing.

- CLI documentation (T1): https://dsi-studio.labsolver.org/doc/cli_t1.html  
- CLI documentation (general): https://dsi-studio.labsolver.org/doc/

Example use cases:
- Batch reconstruction across a cohort
- Automated tractography with standardized parameters
- Connectome export for statistical pipelines

---

## Outputs and File Types

Common file types you may encounter:

- `.sz` : converted source data container (input stage)
- `.fz` : reconstructed fiber data used for tracking and analysis
- `.tt.gz` : tractography output
- `.nii.gz` : exported voxelwise maps (e.g., anisotropy measures)
- `.connectivity.mat` / matrices : connectome outputs (depending on export options)

---

## Reproducibility

DSI Studio supports reproducible analysis by:
- Saving parameters alongside output files
- Enabling the same operations through both GUI and CLI
- Standardizing intermediate files (`.sz`, `.fz`) to reduce variability across runs

---

## Support and Community

- Tutorial videos: https://practicum.labsolver.org  
- User forum (bugs, suggestions, troubleshooting): https://groups.google.com/g/dsi-studio  
- Manual: https://dsi-studio.labsolver.org/manual  
- Issue tracker: https://github.com/frankyeh/DSI-Studio/issues  

When reporting an issue, please include:
- OS and version
- DSI Studio version (release tag)
- Steps to reproduce
- Screenshots/logs if available

---

## Contributing and Development Notes

This repository hosts the source code for DSI Studio. Most users will use the standalone binaries from the official site. If you would like to contribute:

- Use GitHub issues for bug reports and feature requests
- Keep changes focused and well-documented
- Include a short rationale and (when possible) minimal test cases

---

## License

Please refer to the repository license and the official website for current licensing and usage terms.

---

## Citations

Please cite the methods you used (select only those applied to your study). Citation names are formatted consistently as **Yeh, FC**.

**DSI Studio platform and Fiber Data Hub (2025)**  
> Yeh, FC. DSI Studio: an integrated tractography platform and fiber data hub for accelerating brain research. *Nature Methods*. 2025 Aug;22(8):1617-1619. doi:10.1038/s41592-025-02762-8.

**Population-based atlas and tract-to-region connectome (2022)**  
> Yeh, FC. Population-based tract-to-region connectome of the human brain and its hierarchical topology. *Nature Communications*. 2022 Aug 22;13(1):1-3.

**Shape analysis (2020)**  
> Yeh, FC. Shape Analysis of the Human Association Pathways. *NeuroImage*. 2020.

**Augmented fiber tracking (2020)**  
> Yeh, FC. Shape Analysis of the Human Association Pathways. *NeuroImage*. 2020.

**Differential tractography (2019)**  
> Yeh, FC, et al. Differential tractography as a track-based biomarker for neuronal injury. *NeuroImage*. 2019;202:116131.

**Topology-informed pruning (TIP, 2019)**  
> Yeh, FC, et al. Automatic Removal of False Connections in Diffusion MRI Tractography Using Topology-Informed Pruning (TIP). *Neurotherapeutics*. 2019.

**Connectometry (2016)**  
> Yeh, FC, Badre D, Verstynen T. Connectometry: A statistical approach harnessing the analytical potential of the local connectome. *NeuroImage*. 2016;125:162-171.

**Restricted diffusion imaging (RDI, 2016)**  
> Yeh, FC, Liu L, Hitchens TK, Wu YL. Mapping Immune Cell Infiltration Using Restricted Diffusion MRI. *Magnetic Resonance in Medicine*. 2016.

**Local connectome fingerprint (LCF, 2016)**  
> Yeh, FC, et al. Quantifying differences and similarities in whole-brain white matter architecture using local connectome fingerprints. *PLoS Computational Biology*. 2016;12(11):e1005203.

**Individual connectometry (2013)**  
> Yeh, FC, et al. Diffusion MRI connectometry automatically reveals affected fiber pathways in individuals with chronic stroke. *NeuroImage: Clinical*. 2013;2:912-921.

**Generalized deterministic tracking (2013)**  
> Yeh, FC, et al. Deterministic diffusion fiber tracking improved by quantitative anisotropy. *PLoS ONE*. 2013;8(11):e80713. doi:10.1371/journal.pone.0080713.

**QSDR / NTU-90 atlas (2011)**  
> Yeh, FC, Tseng WI. NTU-90: a high angular resolution brain atlas constructed by q-space diffeomorphic reconstruction. *NeuroImage*. 2011;58(1):91-99.

**GQI (2010)**  
> Yeh, FC, Wedeen VJ, Tseng WI. Generalized q-sampling imaging. *IEEE Transactions on Medical Imaging*. 2010;29(9):1626-1635.
```
