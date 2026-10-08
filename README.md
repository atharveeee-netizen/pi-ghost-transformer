# Physics-Informed Ghost Transformer (PI-GhostFormer)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Physics-Informed Ghost Transformer: Defocus-Invariant Single-Photon Reconstruction for Fabric Micro-Defect Detection**

Codebase for the Q1 Journal Submission targeting *Optics Express* / *IEEE Trans. Computational Imaging*.

## 🌟 Overview

Single-photon computational imaging (ghost imaging) offers immense potential for industrial textile inspection under extreme low-light conditions. However, existing Transformer-based methods treat the optical system as a "black box," using physically incorrect loss functions (MSE) and suffering catastrophic degradation when the camera loses focus.

**PI-GhostFormer** introduces a fully differentiable wave-optics pipeline directly into the neural network architecture. By incorporating a novel **Physics-Conditioned Cross-Attention (PCCA)** mechanism and a **Poisson Negative Log-Likelihood (NLL)** loss, the model achieves state-of-the-art defocus-invariant reconstruction of micro-defects at ultra-low photon counts.

### Key Contributions
1. **Differentiable Zernike Wavefront Layer**: Full end-to-end physics forward model inside PyTorch.
2. **Physics-Conditioned Cross-Attention (PCCA)**: Attention keys/values are modulated by the physical state of the optical system.
3. **64-Token Spatial Decoder**: Eliminates the 1D-to-2D bottleneck by using a learned spatial query grid, preserving high-frequency edge details for micro-defects.
4. **Poisson NLL Loss**: Replaces physically incorrect MSE for single-photon SPAD data.
5. **Learned Measurement Matrix**: Co-optimizes the physical illumination patterns alongside the reconstruction network.
6. **Self-Supervised Physics-Only Mode**: Reconstructs defects without needing ground-truth paired datasets.

## 🚀 Architecture

```text
x_clean → [Zernike Defocus PSF] → [Learned Illumination H] → [SPAD Poisson Noise] → y_measured
                                                                                         ↓
[Defect Class] ← [Hybrid Swin-UNet Decoder] ← [Physics-Conditioned Cross-Attention] ← [1D CNN-Transformer Encoder]
```

## 📁 Repository Structure

```
├── config.py               # Central configuration (Physics, Model, Training)
├── train.py                # Main training loop
├── ablations.py            # Automated ablation study matrix runner
├── physics/
│   ├── forward_model.py    # Differentiable imaging pipeline
│   ├── psf.py              # Zernike polynomial wavefront expansion
│   └── noise.py            # SPAD Poisson/Dark current/Readout noise
├── models/
│   ├── pighostformer.py    # Main hybrid architecture
│   ├── attention.py        # Novel Physics-Conditioned Cross-Attention
│   ├── swin.py             # Swin Transformer blocks
│   └── loss.py             # Poisson NLL and TV regularization
├── data/
│   └── synthetic.py        # Synthetic fabric & anomaly generator
└── utils/
    └── metrics.py          # PSNR, SSIM, F1, AUROC
```

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

## 🏃 Running Experiments

**1. Train the PI-GhostFormer Baseline:**
```bash
python train.py
```

**2. Run the Ablation Matrix (IEEE TCI Requirement):**
```bash
python ablations.py
```
*This systematically disables PCCA, Poisson NLL, spatial decoding, and learned patterns to mathematically prove individual component contributions.*

## 🔬 Experimental Evaluation

The model is evaluated on its ability to reconstruct four primary fabric defect classes (Hole, Stain, Thread Break, Misweave) under:
- **Defocus Range**: 0.0 to 3.0 waves (Zernike $a_4$).
- **Photon Scarcity**: 1 to 1000 mean photons per pixel.
- **Sampling Ratio**: 1% to 25%.

## 📝 Citation

*(To be updated upon publication)*
