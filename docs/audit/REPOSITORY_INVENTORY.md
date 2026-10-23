# Repository Inventory: `pi-ghost-transformer`
**Audit Timestamp:** 2026-10-23T21:55:00+05:30  
**Git Commit:** `5b97d79`  
**Branch:** `master` / `main`  
**Auditor:** Senior Research Engineering & Scientific Methodology Audit

---

## 1. Directory Structure & File Inventory

The repository contains 10 primary functional Python modules, 52 model checkpoints, 4 benchmark datasets on disk (10.7 GB total), and test/logging utilities:

```
pi-ghost-transformer/
├── config.py                           # Master experiment & physics configuration dataclasses
├── requirements.txt                    # Minimal environment dependencies
├── LICENSE                             # MIT License
├── README.md                           # Project technical overview and benchmark tables
├── .gitignore                          # Exclusions for weights, logs, datasets, binary documents
├── benchmarks/                         # Benchmark & sanity test suites
│   ├── ablations.py                    # Architecture ablation suite (backprojection, attention, NIG)
│   ├── benchmark_aitex_lofo.py         # 7-fold Leave-One-Fabric-Out (LOFO) cross-validation
│   ├── benchmark_baselines.py          # Classical, optimization, and baseline comparisons
│   ├── benchmark_in_context_adaptation.py # K-shot zero-gradient test-time adaptation
│   ├── benchmark_mvtec_auroc_pro.py    # MVTec-AD anomaly detection (AUROC and PRO metrics)
│   ├── benchmark_stress_matrix.py      # Noise, defocus, and scattering stress grid
│   ├── benchmark_uncertainty_calibration.py # ECE and Brier score uncertainty calibration
│   ├── sanity_checks.py                # 3-fold sanity checks (constant-mean, shuffle, monotonic)
│   └── sim_to_real_transfer.py         # Domain gap transfer benchmark
├── checkpoints/                        # Model weights and illumination patterns (1,489 MB)
│   ├── trained_patterns.pt             # Binary/grayscale pattern matrix (512, 4096)
│   ├── ic_pinot_tilda_400_seed*.pt     # TILDA textile checkpoints (seeds 42, 1337, 2026)
│   ├── ic_pinot_dagm_2007_seed*.pt     # DAGM metal checkpoints (seeds 42, 1337, 2026)
│   └── ic_pinot_mvtec-*_seed*.pt       # 15 MVTec category checkpoints (3 seeds each)
├── data/                               # Dataset loaders and raw data directories (10,772 MB)
│   ├── aitex/                          # 354 raw defect & defect-free fabric images
│   ├── aitex_loader.py                 # PyTorch dataset for AITEX fabric crops
│   ├── benchmark_loader.py             # Dataloaders for TILDA, DAGM, MVTec-AD
│   ├── benchmarks/                     # Benchmark image trees
│   │   ├── dagm/                       # 2,602 images (391.1 MB)
│   │   ├── mvtec/                      # 6,645 images (10,051.8 MB)
│   │   └── tilda/                      # 25,600 patch images (160.9 MB)
│   └── synthetic.py                    # Procedural yarn / weave defect generator
├── docs/                               # Academic reporting and technical briefing
│   ├── generate_professor_pdf.py       # ReportLab PDF compilation script
│   └── project_overview_professor.tex  # 4-page academic technical briefing source
├── figures/                            # Visual assets and qualitative comparison figures
├── logs/                               # Execution logs from training and evaluation
├── models/                             # PyTorch neural network modules
│   ├── attention.py                    # PhysicsConditionedCrossAttention (PCCA)
│   ├── evidential.py                   # EvidentialHead2D (Normal-Inverse-Gamma NIG output)
│   ├── ic_pinot.py                     # ICPINOT unified flagship model architecture
│   ├── in_context.py                   # InContextPromptEncoder (K-shot exemplar conditioning)
│   ├── loss.py                         # NIG evidential loss & flux-normalized Poisson NLL
│   ├── pighostformer.py                # Baseline PI-GhostFormer model
│   └── swin.py                         # SwinTransformerBlock spatial decoder layers
├── physics/                            # Forward optics, noise, and PSF simulation
│   ├── fno.py                          # OpticalFourierNeuralOperator & SpectralConv2d
│   ├── forward_model.py                # GhostImagingForwardModel & MeasurementMatrix
│   ├── noise.py                        # PhotonNoiseModel (Poisson, dark count, SPAD effects)
│   └── psf.py                          # ZernikePSF modal wavefront aberration generator
├── results/                            # JSON output manifests from benchmark runs
├── tables/                             # LaTeX tables auto-generated from result manifests
├── tests/                              # Unit & integration smoke tests
│   ├── smoke_test.py                   # End-to-end synthetic forward & training test
│   └── smoke_test_ic_pinot.py          # 4-stage component verification test
├── training/                           # Training orchestrators
│   ├── run_master_campaign.py          # Multi-seed multi-dataset orchestration runner
│   ├── train.py                        # Single-dataset training loop
│   ├── train_benchmarks.py             # Benchmark fine-tuning loop
│   ├── train_full_benchmarks_deep.py   # Deep 300-epoch SGDR training pipeline
│   └── train_ic_pinot_master.py        # Master training pipeline with evidential warmup
└── utils/                              # Metrics, plotting, and analysis scripts
```

---

## 2. Component Entry Points

| Task | Primary Entry Point | Invocation Command | Dependencies |
|:---|:---|:---|:---|
| **Training (Flagship)** | `training/train_ic_pinot_master.py` | `python training/train_ic_pinot_master.py --dataset tilda --epochs 300` | `data/`, `config.py`, CUDA GPU |
| **Model Verification** | `tests/smoke_test_ic_pinot.py` | `python tests/smoke_test_ic_pinot.py` | PyTorch, `models/`, `physics/` |
| **Sanity Suite** | `benchmarks/sanity_checks.py` | `python benchmarks/sanity_checks.py --checkpoint checkpoints/ic_pinot_tilda_400_seed42.pt` | `data/benchmarks/tilda`, checkpoints |
| **Baselines Benchmark** | `benchmarks/benchmark_baselines.py` | `python benchmarks/benchmark_baselines.py` | `data/benchmarks/`, checkpoints |
| **AITEX LOFO CV** | `benchmarks/benchmark_aitex_lofo.py` | `python benchmarks/benchmark_aitex_lofo.py` | `data/aitex/`, checkpoints |
| **MVTec AUROC/PRO** | `benchmarks/benchmark_mvtec_auroc_pro.py` | `python benchmarks/benchmark_mvtec_auroc_pro.py` | `data/benchmarks/mvtec/`, checkpoints |
| **Uncertainty Calibration** | `benchmarks/benchmark_uncertainty_calibration.py` | `python benchmarks/benchmark_uncertainty_calibration.py` | checkpoints, test splits |
| **Stress Matrix** | `benchmarks/benchmark_stress_matrix.py` | `python benchmarks/benchmark_stress_matrix.py` | checkpoints, `physics/` |

---

## 3. End-to-End Pipeline Dataflow Trace

Tracing a single input sample through the complete pipeline:

1. **Source Image Loading (`data/benchmark_loader.py`):**
   - Disk File: `data/benchmarks/tilda/good/tilda_001.png`
   - Reading: `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` $\to$ `cv2.resize(..., (64, 64))` $\to$ normalize `img.astype(np.float32) / 255.0`
   - Tensor Output: Shape `(1, 64, 64)`, `torch.float32`, dynamic range $[0.498, 0.698]$, mean $0.639$.
   - Label: `torch.int64`, value $0$ (defect-free) or $1$ (defective).

2. **Optical Forward Modeling (`physics/forward_model.py`):**
   - Zernike Defocus Vector: $\theta \in \mathbb{R}^{11}$, index 3 set to defocus amplitude ($a_4 = 1.0\text{ waves}$).
   - PSF Convolution: `apply_psf_blur(img, psf)` blurs $64 \times 64$ patch with $21 \times 21$ kernel.
   - Measurement Matrix Modulation: $H \in \mathbb{R}^{512 \times 4096}$ modulates flattened image $x \in \mathbb{R}^{4096}$.
   - Poisson Shot Noise: Expected counts $\bar{y} = \lambda \cdot (H x) + d$. With $\lambda = 10.0$, expected counts per measurement are $\approx 12,500$ photons.
   - Realized Sample: $y_{\text{noisy}} = \text{Poisson}(\bar{y}) + \mathcal{N}(0, \sigma_{\text{read}}^2)$, normalized by $\lambda \to$ vector $y \in \mathbb{R}^{1 \times 512}$, range $[1223.1, 1289.9]$.

3. **Neural Reconstruction & Latency (`models/ic_pinot.py`):**
   - Physical Adjoint Backprojection: $x_{\text{back}} = H^\dagger y \in \mathbb{R}^{1 \times 1 \times 64 \times 64}$.
   - 1D Signal Encoder: Conv1D stem $\to$ 2-layer TransformerEncoder $\to$ feature tokens $Z \in \mathbb{R}^{1 \times 512 \times 256}$.
   - Spatial Query Cross-Attention (PCCA): 64 spatial queries attend to $Z$, conditioned on optical parameters $\theta$.
   - Hierarchical Swin-UNet Spatial Decoder: Latent expansion $8 \times 8 \to 16 \times 16 \to 32 \times 32 \to 64 \times 64$.
   - Evidential Normal-Inverse-Gamma (NIG) Head: Predicts $[\gamma, v, \alpha, \beta]$ per pixel.
   - Fused Reconstruction: $\hat{x} = 0.7 x_{\text{back}} + 0.3 \gamma \in \mathbb{R}^{1 \times 1 \times 64 \times 64}$, clamped to $[0, 1]$.
   - Compute Time: $7.2\text{ ms}$ on NVIDIA RTX GPU.

4. **Uncertainty Decomposition:**
   - Aleatoric Uncertainty (sensor shot noise): $\sigma^2_a = \beta / (\alpha - 1) \in [0.0004, 0.0041]$.
   - Epistemic Uncertainty (model/novelty variance): $\sigma^2_e = \beta / [v(\alpha - 1)] \in [0.0092, 0.0126]$.

5. **Anomaly Scoring & Evaluation:**
   - Pixel-level residual error map: $e = |x - \hat{x}| \in \mathbb{R}^{1 \times 1 \times 64 \times 64}$.
   - Image-level anomaly score: Mean top-$K$ residual energy or evidential epistemic sum.
   - Classification Head Logits: Linear projection from bottleneck features $\to$ `[logit_good, logit_defect]`.
