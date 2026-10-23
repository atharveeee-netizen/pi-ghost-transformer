# Dependency and Artifact Audit: `pi-ghost-transformer`
**Audit Timestamp:** 2026-10-23T21:57:00+05:30  
**Environment:** Python 3.12, PyTorch 2.5.1+cu124, CUDA 12.4  

---

## 1. Environment & Package Dependencies

The runtime environment was inspected against `requirements.txt`:

| Package | Declared Version | Installed Version | Status |
| :--- | :--- | :--- | :--- |
| `torch` | `>=2.0.0` | `2.5.1+cu124` | Compatible (GPU CUDA acceleration enabled) |
| `torchvision` | `>=0.15.0` | `0.20.1+cu124` | Compatible |
| `numpy` | `>=1.24.0` | `1.26.4` | Compatible |
| `opencv-python` | `>=4.7.0` | `4.10.0` | Compatible |
| `scipy` | `>=1.10.0` | `1.14.1` | Compatible |
| `matplotlib` | `>=3.7.0` | `3.9.2` | Compatible |
| `scikit-learn` | `>=1.2.0` | `1.5.1` | Compatible |
| `tqdm` | `>=4.65.0` | `4.66.5` | Compatible |

No missing core dependencies were detected.

---

## 2. Dataset Artifacts on Local Disk

All primary evaluation datasets are stored under `data/`:

| Dataset | Disk Location | Size | Image Count | Split / Organization | License / Sourcing |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TILDA 400** | `data/benchmarks/tilda/` | $160.9\text{ MB}$ | $25,600\text{ patches}$ | Folders: `good/`, `hole/`, `objects/`, `oil spot/`, `thread error/` | Schulz-Mirbach (Univ. Freiburg, 1999). Academic research use. |
| **DAGM 2007** | `data/benchmarks/dagm/` | $391.1\text{ MB}$ | $2,602\text{ images}$ | `Class10/Train`, `Class10/Test` with paired `_label.PNG` binary defect masks | IAPR TC8 (2007). Public benchmark. |
| **MVTec-AD** | `data/benchmarks/mvtec/` | $10,051.8\text{ MB}$ | $6,645\text{ images}$ | 15 categories (Carpet, Grid, Leather, Tile, Bottle, Cable, etc.). `train/good/` and `test/<defect_type>/` | Bergmann et al. (CVPR 2019). Non-commercial academic research license. |
| **AITEX** | `data/aitex/` | $168.0\text{ MB}$ | $354\text{ images}$ | 7 commercial fabric weave categories with paired mask XML/PNG files | Silvestre-Blanes et al. (*AUTEX Res. J.*, 2019). Public fabric database. |

### Tracking & `.gitignore` Status
- Datasets are located in `data/aitex/` and `data/benchmarks/`.
- Both paths are included in `.gitignore` to prevent large binary blobs from polluting Git history.
- Local data integrity is confirmed: all 4 datasets are present on disk and readable.

---

## 3. Checkpoint Artifacts & Pattern Matrices

The `checkpoints/` directory contains 52 `.pt` files ($1,489.18\text{ MB}$ total):

### Key Checkpoints:
1. **`checkpoints/trained_patterns.pt` ($8.00\text{ MB}$):**
   - Shape: `torch.Size([512, 4096])`
   - Dtype: `torch.float32`
   - Range: $[0.0, 1.0]$
   - Role: Illumination pattern matrix $H \in \mathbb{R}^{M \times N^2}$ utilized across all benchmark scripts and backprojection initialization.

2. **Flagship Dataset Models ($29.04\text{ MB}$ each):**
   - `ic_pinot_tilda_400_seed42.pt` (TILDA textile)
   - `ic_pinot_tilda_400_seed1337.pt`
   - `ic_pinot_tilda_400_seed2026.pt`
   - `ic_pinot_dagm_2007_seed42.pt` (DAGM metallic surface)
   - `ic_pinot_dagm_2007_seed1337.pt`
   - `ic_pinot_dagm_2007_seed2026.pt`
   - MVTec-AD category models: 15 categories $\times$ 3 seeds = 45 checkpoints.

### Checkpoint Loading Protocol:
- All evaluation scripts load model state dicts via:
  ```python
  model.load_state_dict(torch.load(checkpoint_path, map_location=dev), strict=False)
  ```
- **Finding:** The linear backprojection weight (`backproj.weight`) is computed from `trained_patterns.pt` via `torch.pinverse` at initialization and is not stored in the state dict itself. Setting `strict=False` allows safe loading of the 160 core network layers without key mismatch.

---

## 4. Missing or Unavailable Artifacts

1. **Hardware SPAD Acquisition Data:**  
   No physical SPAD timestamp streams or raw laboratory time-correlated single-photon counting (TCSPC) `.bin` files exist in the repository. The project is strictly a computational simulation study.
2. **Untracked Binary Files:**  
   Per project policy, all `.pdf` and `.ppt` binaries are excluded from the repository. Academic documentation is maintained exclusively in LaTeX source (`docs/project_overview_professor.tex`) and Markdown.
