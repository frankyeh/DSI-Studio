# DSI-Studio 
[![GitHub release](https://img.shields.io/github/v/release/frankyeh/DSI-Studio)](https://github.com/frankyeh/DSI-Studio/releases)  
[![Last commit](https://img.shields.io/github/last-commit/frankyeh/DSI-Studio)](https://github.com/frankyeh/DSI-Studio/commits/master)

Follow: [![Twitter](https://img.shields.io/twitter/follow/FangChengYeh?style=social&logo=twitter)](https://twitter.com/FangChengYeh)  
Subscribe: [![YouTube](https://img.shields.io/youtube/channel/subscribers/UCN6gohY_zeBpK6SwJ7hnz1Q?style=social)](https://www.youtube.com/c/FrankYeh)

📘 **Documentation:** [https://dsi-studio.labsolver.org/manual](https://dsi-studio.labsolver.org/manual)  
💬 **User Forum:** [https://groups.google.com/g/dsi-studio](https://groups.google.com/g/dsi-studio)

---

## 🧠 What is DSI Studio?

DSI Studio is a lightweight and user-friendly software for diffusion MRI analysis, tractography, and connectome mapping. It enables researchers and clinicians to:

- Perform deterministic and advanced fiber tracking
- Reconstruct diffusion models 
- Create tract-based connectivity matrices
- Visualize and interactively edit brain tracts
- Export a wide variety of metrics and outputs

---

## 💻 System Requirements

### Supported Platforms
- **Windows**: 64-bit (Windows 10 or newer)
- **macOS**: Intel or Apple Silicon (macOS 13+)
- **Linux**: Ubuntu 18.04 or newer (tested on 20.04, 22.04)

### Software Dependencies
- None. DSI Studio is distributed as a **standalone executable** (no installation or compilation needed)
- GPU version requires installation of CUDA toolkit.
 
### Hardware Recommendations
- CPU with ≥4 cores
- ≥8 GB RAM
- NVIDIA GPU recommended for GPU version

---

## 🚀 Installation Instructions

1. Visit the [official download page](https://dsi-studio.labsolver.org/download.html)
2. Select the appropriate binary for your platform:
   - `dsi_studio_64.exe` (Windows)
   - `dsi_studio_mac.dmg` (macOS)
   - `dsi_studio_ubuntu.zip` (Linux)
3. Extract the ZIP file (if needed)
4. Run the executable (no installation required)

🕒 **Install Time**: Less than 1 minute

---

## 🧪 Demo: Try it Now

### Demo Dataset
Demo data and sample `.fz` files are available from:  
👉 [Fiber Data Hub](https://brain.labsolver.org)

### How to Run
1. Launch `dsi_studio`
2. Find one `.fz` file at `Fiber Data Tab` tab
<img src="https://github.com/user-attachments/assets/13049248-282d-4295-9cfb-880f1f89315c" width="400"/>

3. Click `Open XXX.fz` button to bring up tracking window
<img src="https://github.com/user-attachments/assets/392383ac-3425-4413-88c1-8d356f7c710b" width="400"/>
  
4. Click `Fiber Tracking` button to visualize tractography and export results
<img src="https://github.com/user-attachments/assets/7b78441a-6b76-4bc8-bfb0-ff1af82aac52" width="400"/>


### Output
- Tract files (`.tt.gz)
- Anisotropy maps (`.nii.gz`)
- Connectivity matrices

⏱️ **Runtime**: ~1–3 minutes on a standard desktop

---

## ▶️ How to Use with Your Data

1. Use File → Open → DICOM/NIfTI to import your raw diffusion MRI data
2. Create `.sz` using the “Step T1” conversion
3. Reconstruct using preferred method (e.g., GQI, DTI) to create `.fz` files
4. Run tractography using custom or template ROIs
5. Export tracts, metrics, and connectome data

---

## ⚙️ Command-Line Interface

DSI Studio supports full CLI scripting for batch processing.  
Docs: [https://dsi-studio.labsolver.org/doc/cli](https://dsi-studio.labsolver.org/doc/cli)

---

## 🔁 Reproducibility

- Parameters are saved with output files
- All reconstructions and tracking are reproducible via GUI or CLI

---

## 📬 Support and Community

- **Tutorial videos**: [https://practicum.labsolver.org](https://practicum.labsolver.org)  
- **Forum (bug report, suggestions, troubleshooting)**: [https://groups.google.com/g/dsi-studio](https://groups.google.com/g/dsi-studio)  
- **Documentation**: [https://dsi-studio.labsolver.org/manual](https://dsi-studio.labsolver.org/manual)  
- **Issue tracker**: [https://github.com/frankyeh/DSI-Studio/issues](https://github.com/frankyeh/DSI-Studio/issues)

---

## 📄 Citation

Please cite the methods you used in DSI Studio (select only those applied to your study):

> Yeh F.C., et al. *Population-averaged atlas of the human structural connectome and its network topology.* NeuroImage, 178, 57–68 (2018).  
> [https://doi.org/10.1016/j.neuroimage.2018.05.027](https://doi.org/10.1016/j.neuroimage.2018.05.027)

Population-based atlas and tracto-to-region connectome (2022): This study constructs a population-based probablistic tractography atlas and its associated tract-to-region connectome.
> Yeh FC. Population-based tract-to-region connectome of the human brain and its hierarchical topology. Nature communications. 2022 Aug 22;13(1):1-3.

Shape Analysis (2020): Shape analysis is a morphology based quantification of tractography.
> Yeh, Fang-cheng. "Shape Analysis of the Human Association Pathways." Neuroimage (2020).

Augmented fiber tracking (2020): The “augmented fiber tracking” are three strategies used to boost reproducibility of deterministic fiber tracking.
> Yeh, Fang-cheng. "Shape Analysis of the Human Association Pathways." Neuroimage (2020).

SRC file quality control (2019): The “neighboring DWI correlation” is introduced in this study as a QC metrics for DWI.
> Yeh, Fang-Cheng, et al. "Differential tractography as a track-based biomarker for neuronal injury." NeuroImage 202 (2019): 116131.

Topology informed pruning (TIP, 2019): A topology-based approach to remove false fiber trajectories.
> Yeh, F. C., Panesar, S., Barrios, J., Fernandes, D., Abhinav, K., Meola, A., & Fernandez-Miranda, J. C. (2019). Automatic Removal of False Connections in Diffusion MRI Tractography Using Topology-Informed Pruning (TIP). Neurotherapeutics, 1-7.

connectometry (2016): connectometry is a statistical framework for testing the significance of correlational tractography.
> Yeh, Fang-Cheng, David Badre, and Timothy Verstynen. "Connectometry: A statistical approach harnessing the analytical potential of the local connectome." NeuroImage 125 (2016): 162-171.

> Restricted diffusion imaging (RDI, 2016): RDI is a model-free method that calculates the density of diffusing spins restricted within a given displacement distance.
Yeh, Fang-Cheng, Li Liu, T. Kevin Hitchens, and Yijen L. Wu, "Mapping Immune Cell Infiltration Using Restricted Diffusion MRI", Magn Reson Med. accepted, (2016)

> Local connectome fingerprint (LCF, 2016): Local conectome fingerprint provides a subject-specific measurement for characterizing the white matter architectures and quantifying differences/similarity.
Yeh, F. C., Vettel, J. M., Singh, A., Poczos, B., Grafton, S. T., Erickson, K. I., ... & Verstynen, T. D. (2016). Quantifying differences and similarities in whole-brain white matter architecture using local connectome fingerprints. PLoS computational biology, 12(11), e1005203.

> Individual connectometry (2013): Individual connectometry is atlas-based analysis method that tracks the deviant pathways of one individual (e.g. a patient) by comparing subject’s data with a normal population.
Yeh, Fang-Cheng, Pei-Fang Tang, and Wen-Yih Isaac Tseng. "Diffusion MRI connectometry automatically reveals affected fiber pathways in individuals with chronic stroke." NeuroImage: Clinical 2 (2013): 912-921.

> Generalized deterministic tracking algorithm (2013): The fiber tracking algorithm implemented in DSI Studio is a generalized version of the deterministic tracking algorithm that uses quantitative anisotropy as the termination index.
Yeh, Fang-Cheng, et al. "Deterministic diffusion fiber tracking improved by quantitative anisotropy." (2013): e80713. PLoS ONE 8(11): e80713. doi:10.1371/journal.pone.0080713

Q-space diffeormophic reconstruction (QSDR, 2011): QSDR is a model-free method that calculates the orientational distribution of the density of diffusing water in a standard space.
> Yeh, Fang-Cheng, and Wen-Yih Isaac Tseng, "NTU-90: a high angular resolution brain atlas constructed by q-space diffeomorphic reconstruction." Neuroimage 58.1 (2011): 91-99.

Generalized q-sampling imaging (GQI, 2010): GQI is a model-free method that calculates the orientational distribution of the density of diffusing water.
> Yeh, Fang-Cheng, Van Jay Wedeen, and Wen-Yih Isaac Tseng, "Generalized q-sampling imaging" Medical Imaging, IEEE Transactions on 29.9 (2010): 1626-1635.



Let me know if you'd like to add example commands, a reproducibility checklist, or a BibTeX citation block!
