# Comprehensive Dataset Audit and Provenance Manifest
**Repository:** `atharveeee-netizen/pi-ghost-transformer`  
**Date:** October 2026  
**Auditor:** Senior Research Engineer & Experimental Scientist  

---

## 1. Executive Summary

This document provides a complete audit of all datasets present within the repository and evaluates candidate external datasets for distribution-shift evaluation under strict computational and storage constraints.

### Storage and Computational Capacity
- **Available Free Disk Space:** 22.97 GB on root drive.
- **Physical Memory (RAM):** 63.7 GB total (46.19 GB available).
- **GPU Accelerator:** NVIDIA GeForce RTX 4070 (11.99 GB VRAM).
- **Kaggle API:** Unconfigured locally (`~/.kaggle/kaggle.json` absent).
- **LUSITANOv2 Status:** Evaluated. The full LUSITANOv2 release requires ~37.6 GB uncompressed, which exceeds the currently available 22.97 GB free disk space. To prevent disk overflow and process crash, the audit preserves existing high-resolution textile data (TILDA 400, DAGM 2007, MVTec AD, AITEX AFID) totaling **33,801 images** on disk.

---

## 2. Dataset Inventory and Characteristics

| Dataset Name | Image Count | Defect-Free Count | Defective Count | Resolution & Channels | Storage Size | Splits Used | Annotation Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TILDA 400** | 25,600 | 20,480 (80%) | 5,120 (20%) | $64 \times 64$ Grayscale | 160.94 MB | 80% Train (20,480), 20% Test (5,120) | Class label (0=Good, 1=Defect: hole, objects, oil spot, thread error) |
| **DAGM 2007** | 2,600 | 2,300 (88.5%) | 300 (11.5%) | $64 \times 64$ Grayscale | 391.14 MB | Standard Split (Class 1-6 synthetic surface flaws) | Image label + ellipse bounding parameters |
| **MVTec AD (Textures)** | 5,354 | 4,284 | 1,070 | $64 \times 64$ Grayscale (Resized) | 10,051.86 MB | Train (Good only), Test (Good + Defect) | Binary class label + pixel segmentation masks |
| **AITEX AFID** | 247 | 141 (57.1%) | 106 (42.9%) | $64 \times 64$ Grayscale (Resized from $4096 \times 256$) | 168.05 MB | Stratified 80% Train (196), 20% Test (51) / All (247) | Image label (0=Normal, 1=Defect) + segmentation masks |

---

## 3. Data Integrity and Verification Checks

### 3.1 AITEX Label Parsing Bug Remediation
During the audit, a critical path-matching bug was identified in `data/aitex_loader.py`:
- `NODefect_images` contains the substring `Defect_images`.
- The legacy code used `if "Defect_images" in img_path: label = 1`, causing all 141 defect-free images in `data/aitex/NODefect_images/` to be incorrectly labeled as defective (`label = 1`).
- **Remediation:** Explicit directory check implemented:
  ```python
  if "NODefect_images" in file:
      normal_paths.append((file, 0))
  elif "Defect_images" in file:
      defect_paths.append((file, 1))
  ```
- **Verification:** Verified label balance: 141 normal (57.1%), 106 defective (42.9%), enabling valid AUROC and selective coverage calculations.

### 3.2 Leakage Prevention and Split Isolation
- **TILDA 400:** Split using deterministic pseudo-random permutation (`np.random.seed(42)`). No training samples bleed into test splits.
- **AITEX AFID:** Stratified train/test splits ensure defect ratio is identical across train (84 defect, 112 normal) and test (22 defect, 29 normal).
- **Zero Synthetic Fallbacks:** Removed all silent generation loops (`torch.zeros()`); any missing image path raises an immediate `FileNotFoundError`.

---

## 4. Fundamental Modality Distinction: Real Images vs. Single-Pixel Measurements

All benchmark datasets (TILDA, DAGM, MVTec, AITEX) consist of **2D spatial camera images**, not native hardware single-pixel bucket photodiode recordings.

### Forward Simulation Protocol:
To evaluate single-pixel imaging:
1. Ground-truth images $x \in [0, 1]^{64 \times 64}$ are passed through a physics-based forward operator:
   $$y = \text{Poisson}\left(\mu_{\text{phot}} \cdot H \cdot \text{PSF}(a_4) \cdot x + \text{dark}\right) + \mathcal{N}(0, \sigma_{\text{read}}^2)$$
   where $H \in [0, 1]^{512 \times 4096}$ is the Hadamard/learned illumination pattern matrix, $a_4$ is the optical defocus coefficient, and $\mu_{\text{phot}} = 10.0$ photons/pattern ($N_{\text{total}} = 5,120$ photons).
2. The measurements $y \in \mathbb{R}^{512}$ are compressed scalar bucket signals.
3. The reconstruction network operates solely on $y$ without access to the ground-truth image $x$.

> **Scientific Declaration:** All reported results describe **synthetic single-pixel compressive measurements generated from real industrial textile imagery**. We do NOT claim physical SPAD hardware capture.

---

## 5. Candidate External Dataset Evaluation

### 5.1 LUSITANOv2 (Carrilho et al., 2025)
- **Source:** Zenodo (`10.5281/zenodo.22150068`), MDPI *Electronics*.
- **Properties:** 25,120 production-line line-scan images at $4096 \times 1024$.
- **Storage Requirement:** ~37.6 GB uncompressed.
- **Audit Decision:** **DEFERRED due to disk capacity constraints**. Current available disk space is 22.97 GB. Downloading and extracting 37.6 GB risks complete disk saturation and IO failure. AITEX AFID (real industrial fabric line-scan database) and MVTec AD Carpet/Grid provide sufficient domain-shift complexity within available storage.

### 5.2 Kaggle Multi-Class Fabric Defect Dataset (`ziya07`)
- **Status:** Kaggle credentials not configured in local environment.
- **Audit Decision:** In accordance with non-negotiable Rule 16 ("If a task cannot be completed because of unavailable credentials, document the blocker and continue with independent tasks"), this download is skipped in favor of the 33,801 fully verified local images.
