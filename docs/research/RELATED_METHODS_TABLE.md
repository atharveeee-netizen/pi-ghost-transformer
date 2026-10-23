# Comprehensive Related Methods & Comparators Matrix
**Audit Standard:** Q1 Systematic Methodological Benchmark  
**Date:** 2026-10-23  

---

## Systematic Multi-Dimensional Comparator Table

| Dimension | **Shang et al. (2023)** [Nat. Commun. Eng.] | **Li et al. (2025)** [Opt. Lett.] | **Li (2026)** [Opt. Lasers Eng.] | **Higham et al. (2018)** [Sci. Rep.] | **Kulkarni et al. (2016)** [CVPR ReconNet] | **Our Proposed System (Risk-Calibrated SPI)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Year** | 2023 | 2025 | 2026 | 2018 | 2016 | **2026** |
| **Method Name** | Bayesian CNN (BCNN) for SPI | Uncertainty-Aware Physics-Informed GI | Domain-Agnostic Physics-Informed GI | Deep Single-Pixel Video (DGI-CNN) | ReconNet (Block CS CNN) | **Risk-Calibrated SPI (RC-SPI)** |
| **Imaging Regime** | Single-Pixel Imaging ($M \ll N^2$) | Computational Ghost Imaging | Computational Ghost Imaging | Single-Pixel Video | Block Compressive Sensing | **Single-Pixel Computational Imaging ($M=512$, $N^2=4096$)** |
| **Noise & Photon Model** | Gaussian readout noise + Poisson | Additive ambient light + Gaussian | Gaussian noise | Clean / idealized simulated | Gaussian measurement noise | **Poisson shot noise ($\lambda \approx 10$) + Dark counts + SPAD dead time** |
| **Measurement Operator** | Known fixed Hadamard patterns | Known patterns + estimated ambient offset | Known patterns + Gaussian regularizer | Known fixed Hadamard | Known random Gaussian projection | **Known/mismatched patterns + Zernike defocus ($a_4 \in [0, 3]$)** |
| **Optical-Shift Evaluation** | **No** (Assumes perfectly matched optics) | **Partial** (Ambient illumination variation) | **No** (Matched optical path) | **No** | **No** | **Yes** (Defocus sweep $a_4 \in [0, 3]$, blind evaluation, operator mismatch) |
| **Domain-Shift Evaluation** | **No** (In-distribution natural/MNIST images) | **No** (Single scene domain) | **Partial** (Cross-domain scene images) | **No** (Standard bench objects) | **No** (Standard natural image patches) | **Yes** (AITEX 7-fold LOFO cross-validation, TILDA $\to$ MVTec) |
| **Uncertainty Method** | Monte Carlo Dropout (30–50 passes) | Physics-informed residual optimization | Implicit Gaussian variance | **None** (Point estimator) | **None** (Point estimator) | **Single-Pass Evidential NIG + Measurement Residual Consistency** |
| **UQ Calibration Evaluation** | Error correlation plot only | Qualitative reconstruction contrast | **None** | **None** | **None** | **Expected Calibration Error (ECE), Brier Score, Risk-Coverage Curves** |
| **Selective Prediction / Abstention** | **No** | **No** | **No** | **No** | **No** | **Yes** (3-tier policy: Accept, Re-measure, Refer with bounded risk) |
| **Datasets & Hardware** | Simulated MNIST/ImageNet + bench SPI setup | Laboratory optical bench setup | Simulated benchmarks | Physical DMD + high-speed detector | Standard natural image benchmarks | **Simulation study on TILDA (25.6k), DAGM (2.6k), AITEX (354), MVTec (6.6k)** |
| **Baselines Included** | Classical DGI, TVAL3 | Standard DGI, iterative GI | Standard GI, TV-CS | Linear DGI, TVAL3 | D-AMP, TVAL3, MH-BCS | **Flat-Image, Brightness Scalar, Raw Line-Scan, Denoised Line-Scan, TV-CS, Deep SPI** |
| **Code Availability** | Partially available | Code not publicly released | Code on request | Open GitHub repository | Open GitHub repository | **Full open-source repository, configs, checkpoints, and benchmark scripts** |
| **Primary Limitation** | 50 forward passes ($>1.5\text{ s}$); cannot run at factory line speeds | Iterative per-instance optimization ($<0.5\text{ FPS}$) | Heuristic prior; no anomaly detection | Fails under low photons and defocus | Purely feedforward; degrades under operator shift | Simulation study; physical SPAD hardware integration left to future work |
| **Key Differentiation** | We replace slow MC dropout with 7.2 ms single-pass NIG + measurement residual | We evaluate discrete industrial defect anomalies rather than ambient light | We provide explicit risk-coverage calibration and selective inspection | We incorporate physics adjoint fusion, defocus modeling, and uncertainty | We embed measurement operator consistency and evidential uncertainty bounds | **First systematic formulation of risk-controlled selective inspection under joint photon and optical shift** |
