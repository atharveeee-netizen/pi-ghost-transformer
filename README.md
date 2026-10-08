# Physics-Informed Ghost Transformer (PI-GhostFormer)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Physics-Informed Ghost Transformer: Defocus-Invariant Single-Photon Reconstruction for Fabric Micro-Defect Detection**

Codebase for the Q1 Journal Submission targeting *Optics Express* / *IEEE Trans. Computational Imaging*.

## Overview

Single-photon computational imaging (ghost imaging) offers immense potential for industrial textile inspection under extreme low-light conditions. However, existing Transformer-based methods treat the optical system as a "black box," using physically incorrect loss functions (MSE) and suffering catastrophic degradation when the camera loses focus.

**PI-GhostFormer** introduces a fully differentiable wave-optics pipeline directly into the neural network architecture. By incorporating a novel **Physics-Conditioned Cross-Attention (PCCA)** mechanism and a **Poisson Negative Log-Likelihood (NLL)** loss, the model achieves state-of-the-art defocus-invariant reconstruction of micro-defects at ultra-low photon counts.

### Key Contributions
1. **Differentiable Zernike Wavefront Layer**: Full end-to-end physics forward model inside PyTorch.
2. **Physics-Conditioned Cross-Attention (PCCA)**: Attention keys/values are modulated by the physical state of the optical system.
3. **64-Token Spatial Decoder**: Eliminates the 1D-to-2D bottleneck by using a learned spatial query grid, preserving high-frequency edge details for micro-defects.
4. **Curriculum Learning (MSE $\rightarrow$ Poisson NLL)**: Solves the gradient instability of training Vision Transformers with physical photon probability distributions.
5. **Learned Measurement Matrix**: Co-optimizes the physical illumination patterns alongside the reconstruction network.
6. **Automated Convergence Testing**: Built-in infinite-patience early stopping logic.

## Architecture

```text
x_clean $\rightarrow$ [Zernike Defocus PSF] $\rightarrow$ [Learned Illumination H] $\rightarrow$ [SPAD Poisson Noise] $\rightarrow$ y_measured
                                                                                         $\downarrow$
[Defect Class] $\leftarrow$ [Hybrid Swin-UNet Decoder] $\leftarrow$ [Physics-Conditioned Cross-Attention] $\leftarrow$ [1D CNN-Transformer Encoder]
```

## Repository Structure

```
.
├── config.py               # Central configuration (Physics, Model, Training)
├── train.py                # Main curriculum training loop (AITEX dataset)
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
│   ├── aitex_loader.py     # Real-world AITEX fabric PyTorch dataset
│   └── synthetic.py        # Fallback synthetic fabric generator
└── utils/
    └── metrics.py          # PSNR, SSIM, F1, AUROC
```

## Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

Both training scripts are configured to dynamically pull from the `data/aitex` directory if the AITEX dataset is present.

**1. Train the PI-GhostFormer Baseline:**
```bash
python train.py
```
*Executes the two-phase Curriculum Learning strategy: 50 epochs of pure MSE structural warmup followed by Poisson NLL physical fine-tuning, stopping automatically upon full convergence.*

**2. Run the Ablation Matrix (IEEE TCI Requirement):**
```bash
python ablations.py
```
*Systematically isolates and tests PCCA, Poisson NLL, spatial decoding, and learned patterns to mathematically validate individual component contributions.*

## Experimental Evaluation

The model evaluates reconstruction of fabric defect classes under:
- **Defocus Range**: 0.0 to 3.0 waves (Zernike $a_4$).
- **Photon Scarcity**: 1 to 1000 mean photons per pixel.
- **Sampling Ratio**: 1% to 25%.

## Resources & Supplementary Materials
The following folders contain supplementary videos, extended documentation, and data required for the IEEE review process:
- [Supplementary Drive Folder 1](https://drive.google.com/drive/folders/1x6wSwfCeKxWRZ0njOqYQxk-OOHgya0T7)
- [Supplementary Drive Folder 2](https://drive.google.com/drive/folders/1a1VA_He4nM-u6kl_Rp-rpCW-rxuMeKho)
- [Supplementary Drive Folder 3](https://drive.google.com/drive/folders/1R-fE4jggLrYRBGISnl4HAFOgUsHNsNXM)

## Citation

*(To be updated upon publication)*
