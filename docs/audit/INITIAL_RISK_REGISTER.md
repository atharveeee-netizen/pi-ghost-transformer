# Initial Risk Register & Methodological Defect Log
**Audit Standard:** SYZYGY & Q1 Journal Scientific Review Protocol  
**Repository:** `pi-ghost-transformer`  
**Date:** 2026-10-23T21:58:00+05:30  
**Git Commit:** `5b97d79`  

---

## Summary of Findings at a Glance

| ID | Category | Title | Severity | Status |
|:---:|:---|:---|:---:|:---:|
| **RISK-01** | Mathematical / UQ | Reconstruction Blend vs. Evidential Uncertainty Inconsistency | **Critical** | Confirmed Bug |
| **RISK-02** | Physical Optics | 1,024x Photon Budget Discrepancy in Line-Scan Comparison | **Critical** | Confirmed Bug |
| **RISK-03** | Physical Optics | Asymmetric Intra-Frame Motion Blur Simulation | **Critical** | Confirmed Defect |
| **RISK-04** | Data Integrity | Silent Zero-Image Fallback on Missing/Corrupted Samples | **Major** | Confirmed Bug |
| **RISK-05** | Evaluation Protocol | Untrained Pseudo-Inverse Baseline Evaluated as Deep SPI | **Major** | Confirmed Defect |
| **RISK-06** | Data Leakage | Model Selection Evaluated Directly on Held-Out Test Set | **Major** | Confirmed Flaw |
| **RISK-07** | Calibration / Optics | Blind Defocus Input Defaults to Ground-Truth Zero State | **Major** | Confirmed Flaw |
| **RISK-08** | Anomaly Detection | Sub-Random Detection Performance on Unseen Fabrics (LOFO) | **Major** | Confirmed Empirical Failure |
| **RISK-09** | Architecture | Eponymous Fourier Neural Operator Absent from Forward Graph | **Major** | Confirmed Disconnect |

---

## Detailed Risk & Defect Specifications

### RISK-01: Reconstruction Blend vs. Evidential Uncertainty Inconsistency
* **File & Function:** [`models/ic_pinot.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/models/ic_pinot.py#L210-L213), `forward()`
* **Exact Affected Behavior:**  
  The evidential decoder outputs Normal-Inverse-Gamma parameters $[\gamma, v, \alpha, \beta]$ representing predictive distribution $p(x \mid y) \sim \text{NIG}(\gamma, v, \alpha, \beta)$. The model then computes:
  ```python
  recon_mean = (0.7 * x_back + 0.3 * nig_outputs["gamma"]).clamp(0.0, 1.0)
  nig_outputs["gamma"] = recon_mean
  ```
  The mean parameter $\gamma$ is overwritten with a linear combination of the backprojection adjoint $x_{\text{back}}$ and the decoder mean. However, $v$, $\alpha$, and $\beta$ remain unchanged and describe the variance of the *unblended* decoder.
* **Evidence:** The aleatoric ($\frac{\beta}{\alpha - 1}$) and epistemic ($\frac{\beta}{v(\alpha - 1)}$) uncertainties no longer correspond to the distribution of `recon_mean`.
* **Severity:** **Critical**
* **Reproduction Command:**
  ```bash
  python tests/smoke_test_ic_pinot.py
  ```
* **Status:** Confirmed.
* **Proposed Fix:** Re-evaluate whether uncertainty applies before or after fusion, implement a mathematically grounded residual correction or learned fusion where uncertainty is defined over the final estimate, and ablate against unblended heads.

---

### RISK-02: 1,024x Photon Budget Discrepancy in Line-Scan Comparison
* **File & Function:** [`physics/forward_model.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/physics/forward_model.py#L206) (`forward_expected`), [`benchmarks/benchmark_baselines.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/benchmarks/benchmark_baselines.py#L127) (`simulate_linescan_camera`)
* **Exact Affected Behavior:**  
  The line-scan camera baseline is simulated under `total_photons = 5120.0` across 4,096 pixels ($1.25\text{ photons/pixel}$). In contrast, `forward_model.py` multiplies `mean_photon_count = 10.0` by the unnormalized measurement sum ($\approx 1024$), collecting $\approx 10,240\text{ photons/pattern}$. Over 512 patterns, SPI collects $\approx 5.24\times 10^6$ photons per frame ($1,024\times$ higher budget).
* **Severity:** **Critical**
* **Status:** Confirmed.
* **Proposed Fix:** Clearly define photon budget per frame versus per pattern; evaluate all baselines under matched total photon counts per frame and sweep budgets.

---

### RISK-03: Asymmetric Intra-Frame Motion Blur Simulation
* **File & Function:** [`physics/forward_model.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/physics/forward_model.py) vs [`benchmarks/benchmark_baselines.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/benchmarks/benchmark_baselines.py#L129-L132)
* **Exact Affected Behavior:**  
  1D conveyor motion blur ($L=7$) is applied exclusively to the line-scan camera baseline. In the SPI forward model, all 512 patterns are modulated against a stationary target image, ignoring the $23\text{--}46\text{ mm}$ conveyor translation that occurs during the $23.0\text{ ms}$ pattern projection sequence.
* **Severity:** **Critical**
* **Status:** Confirmed.
* **Proposed Fix:** Implement intra-frame spatial translation in the SPI forward simulator ($\Delta x_m = \frac{v \cdot m}{f_{\text{DMD}}}$), or explicitly restrict the research scope to stroboscopic pulsed illumination for quasi-static inspection.

---

### RISK-04: Silent Zero-Image Fallback on Missing/Corrupted Samples
* **File & Function:** [`data/benchmark_loader.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/data/benchmark_loader.py#L56-L57), `TILDADataset.__getitem__`, `DAGMDataset.__getitem__`, `MVTecTextureDataset.__getitem__`
* **Exact Affected Behavior:**  
  When an image fails to load via `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`, the code silently returns a black zero-tensor (`np.zeros((self.image_size, self.image_size), dtype=np.float32)`) without throwing an exception or warning.
* **Severity:** **Major**
* **Status:** Confirmed.
* **Proposed Fix:** Replace silent fallback with explicit `FileNotFoundError` or `ValueError` stating the corrupted path, and add validation checks on split manifests.

---

### RISK-05: Untrained Baseline in Deep SPI Evaluation
* **File & Function:** [`benchmarks/benchmark_baselines.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/benchmarks/benchmark_baselines.py#L77-L104), `class DeepSPI`
* **Exact Affected Behavior:**  
  `DeepSPI` instantiates an uninitialized CNN refinement block (`nn.Conv2d`, `BatchNorm2d`) following a pseudoinverse layer. It is never trained or loaded from a checkpoint in the baseline script, effectively comparing a trained model against random weights plus backprojection.
* **Severity:** **Major**
* **Status:** Confirmed.
* **Proposed Fix:** Implement a formal baseline training pipeline that trains ReconNet/DeepSPI on the exact same training split before benchmark evaluation.

---

### RISK-06: Model Selection Evaluated Directly on Held-Out Test Set
* **File & Function:** [`training/train_ic_pinot_master.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/training/train_ic_pinot_master.py#L184-L215)
* **Exact Affected Behavior:**  
  Checkpoints are selected and saved based on peak `val_psnr` evaluated directly on `test_loader`. No separate validation split exists, resulting in test-set leakage during model checkpointing.
* **Severity:** **Major**
* **Status:** Confirmed.
* **Proposed Fix:** Create immutable train / validation / test splits (e.g. 70% / 15% / 15%), tune and select checkpoints solely on validation data, and report final metrics on the untouched test split.

---

### RISK-07: Blind Defocus Input Defaults to Ground-Truth Zero State
* **File & Function:** [`models/ic_pinot.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/models/ic_pinot.py#L186-L188), `forward()`
* **Exact Affected Behavior:**  
  When `optical_params is None`, the model executes:
  ```python
  optical_params = torch.zeros(B, self.physics_dim, device=bucket_measurements.device)
  ```
  In test benchmarks where defocus is set to 0.0, the zero vector equals the oracle ground-truth parameter vector, masquerading as blind evaluation.
* **Severity:** **Major**
* **Status:** Confirmed.
* **Proposed Fix:** Separate evaluation into Matched Operator (Mode A), Mismatched/Perturbed Operator (Mode B), and Truly Blind Evaluation with Blind Parameter Estimation (Mode C).

---

### RISK-08: Sub-Random Detection Performance on Unseen Fabrics (AITEX LOFO)
* **File & Function:** [`benchmarks/benchmark_aitex_lofo.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/benchmarks/benchmark_aitex_lofo.py), [`results/aitex_lofo_results.json`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/results/aitex_lofo_results.json)
* **Exact Affected Behavior:**  
  AITEX Leave-One-Fabric-Out cross-validation across 7 commercial fabrics yields:
  $$\text{Macro Defect AUROC} = \mathbf{36.80\%}, \quad \text{Macro F1} = \mathbf{0.382}$$
  An AUROC of 36.8% is worse than random guessing ($50.0\%$), proving that unsupervised reconstruction residual/epistemic uncertainty fails under material-domain shift without calibration.
* **Severity:** **Major**
* **Status:** Confirmed Empirical Failure.
* **Proposed Fix:** Integrate measurement consistency signals, selective prediction policies, and domain-shift calibration to reject or abstain on unreliable out-of-domain inspections.

---

### RISK-09: Eponymous Fourier Neural Operator Absent from Forward Graph
* **File & Function:** [`physics/fno.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/physics/fno.py) vs [`models/ic_pinot.py`](file:///C:/Users/25beevdt047/.gemini/antigravity/scratch/pi-ghost-transformer/atharveeee-netizen-pi-ghost-transformer-fa653160ee132db6e223b7ed2f4c92baea76dbd0/models/ic_pinot.py)
* **Exact Affected Behavior:**  
  While `OpticalFourierNeuralOperator` is defined in `physics/fno.py`, it is never imported, instantiated, or trained inside `ICPINOT`. The network consists of standard Conv1D, Transformers, and Swin-UNet blocks.
* **Severity:** **Major**
* **Status:** Confirmed Disconnect.
* **Proposed Fix:** Fully aligned in Phase 0 documentation by retitling the architecture to **Neural Optical Transformer**, eliminating misleading operator claims.
