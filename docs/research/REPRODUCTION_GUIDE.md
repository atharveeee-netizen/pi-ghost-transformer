# Reproduction Guide: Risk-Calibrated Single-Pixel Imaging
**Repository:** `atharveeee-netizen/pi-ghost-transformer`  
**Date:** October 2026  

This guide provides step-by-step instructions to reproduce all benchmarks, unit tests, environment validations, and selective inspection evaluations reported in the manuscript.

---

## 1. Environment Requirements

- **OS:** Windows 10/11 or Linux
- **Python:** 3.10+ (Tested on Python 3.12.4 with Anaconda)
- **PyTorch:** >= 2.0 with CUDA support (Tested on PyTorch 2.7.1+cu118 / RTX 4070)
- **Dependencies:** `torch`, `torchvision`, `numpy`, `opencv-python`, `scipy`, `scikit-learn`

---

## 2. Integrity and Environment Validation

Run the automated 5-stage environment and codebase integrity check:

```bash
python scripts/validate_environment.py
```

**Expected Output:**
- Validates Python and GPU accelerators.
- Confirms `checkpoints/trained_patterns.pt` (shape `[512, 4096]`) and model weights exist.
- Validates dataset paths and decodes sample images from TILDA, DAGM, MVTec, and AITEX.
- Tests physical forward model execution and IC-PINOT evidential outputs.
- Status: `ALL ENVIRONMENT AND REPOSITORY INTEGRITY CHECKS PASSED SUCCESSFULLY!`.

---

## 3. Unit and Physical Invariant Tests

Run the full unit test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Components Verified:**
- `test_forward_model.py`:
  - `test_non_negativity`: Photon bucket counts $\ge 0$.
  - `test_zero_signal`: Zero optical input produces dark count baseline ($0.5$).
  - `test_linear_adjoint_consistency`: $\langle Hx, y \rangle = \langle x, H^T y \rangle$.
  - `test_seed_repeatability`: Deterministic Poisson/Gaussian noise sampling.
  - `test_identity_zero_defocus`: Zernike PSF conservation of energy ($\sum \text{PSF} = 1.0$).
  - `test_pattern_matrix_conditioning`: Measurement matrix full row rank ($512$) and condition number $\kappa \approx 423.8 < 10^4$.
- `test_selective_inspection.py`:
  - `test_measurement_consistency_under_clean_and_defocused`: Verifies residual increases under optical shift.
  - `test_policy_calibration_and_decision`: Empirically tests (Accept, Re-measure, Refer) tier allocation.

---

## 4. Equal-Photon Baseline Benchmark

Executes the formal benchmark across 8 comparator methods under strict matching photon budget ($N_{\text{total}} = 5,120$ photons, $M=512$ patterns):

```bash
python benchmarks/benchmark_baselines.py --output results/baseline_suite_results.json --manifest results/baseline_registry_manifest.json
```

**Output Files:**
- `results/baseline_suite_results.json`: PSNR and SSIM summary across TILDA, DAGM, and MVTec.
- `results/baseline_registry_manifest.json`: Provenance and assumptions for every baseline.

---

## 5. Standardized Multi-Dataset Reconstruction Benchmark

Evaluates blind reconstruction fidelity and inference latency across all four benchmarks:

```bash
python scripts/run_reconstruction_benchmark.py --max_samples 100 --output results/reconstruction_benchmark_results.json
```

---

## 6. Joint Shift & Selective Inspection Benchmark

Evaluates optical defocus shift ($a_4 \in [0.0, 3.0]$), photon starvation ($N \in [1024, 25600]$), and zero-shot cross-fabric domain shift on AITEX:

```bash
python scripts/run_shift_benchmark.py --output results/shift_benchmark_results.json
```

---

## 7. Controlled Architectural Ablations

Evaluates Linear Backprojection, Pure Neural Decoder, Fixed Adjoint Fusion, and In-Context Adaptation:

```bash
python scripts/run_ablations.py --max_samples 100 --output results/controlled_ablations_results.json
```

---

## 8. Calibration and Matched-Coverage Evaluation

Evaluates residual metrics ($S_1, S_2, S_3, S_4$) and compares selective inspection policies at matched coverages ($50\%, 70\%, 85\%, 95\%$):

```bash
python research/overnight/20261023_225327/evaluate_calibration_and_policies.py
```
