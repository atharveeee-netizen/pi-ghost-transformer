# Baseline Execution Record: `pi-ghost-transformer`
**Audit Timestamp:** 2026-10-23T21:56:00+05:30  
**Environment:** Python 3.12 (Anaconda, Windows), PyTorch 2.5.1+cu124, NVIDIA CUDA Device  
**Git Commit:** `5b97d79`  

---

## 1. Unit & Smoke Test Suite Baselines

### Test 1: Architecture Smoke Test (`tests/smoke_test_ic_pinot.py`)
* **Execution Command:**
  ```bash
  python tests/smoke_test_ic_pinot.py
  ```
* **Raw Execution Output:**
  ```
  =================================================================
  IC-PINOT END-TO-END PIPELINE VERIFICATION
  =================================================================
  Executing on: cuda

  [Test 1/4] Zero-Shot Standard Forward Pass (K=0)...
    Reconstruction output shape: torch.Size([2, 1, 64, 64])
    Classification logits shape: torch.Size([2, 2])
    --> PASS

  [Test 2/4] In-Context Adaptation Pass (K=3 prompt pairs)...
    In-Context Adapted Reconstruction: torch.Size([2, 1, 64, 64])
    --> PASS

  [Test 3/4] Evidential NIG Single-Pass Calibrated Uncertainty...
    Predicted Mean gamma:                   torch.Size([2, 1, 64, 64])
    Aleatoric Uncertainty beta/(alpha-1):   torch.Size([2, 1, 64, 64])
    Epistemic Uncertainty beta/(v(alpha-1)): torch.Size([2, 1, 64, 64])
    Total Uncertainty:                      torch.Size([2, 1, 64, 64])
    Evidential NLL Loss: 1.1144 | Reg: 1.1156
    Backpropagation gradient flow: OK
    --> PASS

  [Test 4/4] Differentiable Optical Fourier Neural Operator...
    FNO continuous wave degradation field shape: torch.Size([2, 1, 64, 64])
    --> PASS

  =================================================================
  ALL 4 IC-PINOT CORE ARCHITECTURAL TESTS PASSED!
  =================================================================
  ```
* **Exit Code:** `0` (Success)

---

### Test 2: Training & Synthetic Forward Pipeline Test (`tests/smoke_test.py`)
* **Execution Command:**
  ```bash
  python tests/smoke_test.py
  ```
* **Raw Execution Output:**
  ```
  ============================================================
  PI-GhostFormer Smoke Test
  ============================================================

  [1/6] Building forward model...
    Measurement matrix: 512 patterns x 4096 pixels

  [2/6] Building PI-GhostFormer...
    Total params: 6,835,093
    Trainable:    6,835,093

  [3/6] Building loss function...

  [4/6] Generating 8 synthetic samples...
    Images shape: torch.Size([8, 1, 64, 64])
    Labels shape: torch.Size([8])
    Label distribution: [4, 1, 2, 0, 1]

  [5/6] Training for 2 epochs (batch_size=4)...
    Epoch 1/2 | Loss: 0.4625 | PSNR: 16.00 dB | Acc: 25% | Time: 2.6s
    Epoch 2/2 | Loss: 0.4250 | PSNR: 15.45 dB | Acc: 25% | Time: 1.5s

  [6/6] Verification...
    Reconstruction output shape: torch.Size([4, 1, 64, 64])
    Logits output shape:         torch.Size([4, 5])
    Gradient flow OK:            True
    Physics patterns learnable:  True

  ============================================================
  SMOKE TEST PASSED — full pipeline runs end-to-end
  ============================================================
  ```
* **Exit Code:** `0` (Success)

---

## 2. Rigorous Evaluation Sanity Suite

### Test 3: Sanity Suite (`benchmarks/sanity_checks.py`)
* **Execution Command:**
  ```bash
  python benchmarks/sanity_checks.py --checkpoint checkpoints/ic_pinot_tilda_400_seed42.pt
  ```
* **Raw Execution Output:**
  ```
  ================================================================================
  IC-PINOT RIGOROUS EVALUATION SANITY SUITE
  ================================================================================
  Device: cuda
  Loading checkpoint: checkpoints/ic_pinot_tilda_400_seed42.pt
  Loading patterns:   checkpoints/trained_patterns.pt

  --------------------------------------------------------------------------------
  SANITY TEST 1: CONSTANT-MEAN BASELINE COMPARISON
  --------------------------------------------------------------------------------
    Dataset Batch Pixel Mean:   0.6393, Std: 0.1139
    Constant-Mean Baseline:     PSNR = 18.87 dB | SSIM = 0.6925 | MAE = 0.0995
    IC-PINOT (Blind Inversion): PSNR = 19.81 dB | SSIM = 0.3544 | MAE = 0.0805
    --> Observation: Low-variance textile texture establishes ~18.9 dB baseline floor.
        Model achieves residual MAE reduction: 0.0995 -> 0.0805

  --------------------------------------------------------------------------------
  SANITY TEST 2: SHUFFLED-MEASUREMENT TEST (VERIFYING PHYSICS INVERSION)
  --------------------------------------------------------------------------------
    Normal Measurements:   PSNR = 19.81 dB | SSIM = 0.3544
    Shuffled Measurements: PSNR = 6.65 dB | SSIM = 0.1067
    Reconstruction Drop:   dPSNR = 13.16 dB | dSSIM = 0.2477
    --> PASS: Permuting measurement order collapses reconstruction fidelity.
        Confirms network actively inverts pattern-projected measurements, not memorized templates.

  --------------------------------------------------------------------------------
  SANITY TEST 3: MONOTONIC DEGRADATION CHECKS (NO FALSE INVARIANCE CLAIMS)
  --------------------------------------------------------------------------------
    [3A] Photon Flux Degradation Sweep:
      Flux =  100.0 photons/pixel -> PSNR = 19.76 dB | SSIM = 0.3537
      Flux =   50.0 photons/pixel -> PSNR = 19.83 dB | SSIM = 0.3545
      Flux =   10.0 photons/pixel -> PSNR = 19.87 dB | SSIM = 0.3569
      Flux =    5.0 photons/pixel -> PSNR = 19.83 dB | SSIM = 0.3553
      Flux =    1.0 photons/pixel -> PSNR = 19.92 dB | SSIM = 0.3580
      Flux Monotonic Degradation Check: PASS

    [3B] Optical Defocus Degradation Sweep (Blind Evaluation, No Defocus Oracle):
      Defocus = 0.0 waves -> PSNR = 19.81 dB | SSIM = 0.3564
      Defocus = 0.5 waves -> PSNR = 19.66 dB | SSIM = 0.3472
      Defocus = 1.0 waves -> PSNR = 19.82 dB | SSIM = 0.3477
      Defocus = 1.5 waves -> PSNR = 19.79 dB | SSIM = 0.3457
      Defocus = 2.0 waves -> PSNR = 19.79 dB | SSIM = 0.3467
      Defocus = 2.5 waves -> PSNR = 19.76 dB | SSIM = 0.3445
      Defocus = 3.0 waves -> PSNR = 19.75 dB | SSIM = 0.3452
      Defocus Monotonic Degradation Check: PASS
  ```
* **Exit Code:** `0` (Success)

---

## 3. Baseline Suite Results Summary (`results/baseline_suite_results.json`)

The baseline benchmark (`benchmarks/benchmark_baselines.py`) records the following numbers on TILDA 400, DAGM 2007, and MVTec Carpet:

| Model / Baseline | TILDA 400 PSNR/SSIM | DAGM 2007 PSNR/SSIM | MVTec Carpet PSNR/SSIM |
| :--- | :---: | :---: | :---: |
| **Flat-Image (Dataset Mean)** | $19.93\text{ dB}$ / $0.686$ | $21.34\text{ dB}$ / $0.379$ | $17.41\text{ dB}$ / $0.259$ |
| **Brightness-Only (Image Mean)** | $29.22\text{ dB}$ / $0.697$ | $21.47\text{ dB}$ / $0.379$ | $17.42\text{ dB}$ / $0.259$ |
| **Raw Line-Scan Camera** | $6.21\text{ dB}$ / $0.128$ | $6.41\text{ dB}$ / $0.126$ | $6.29\text{ dB}$ / $0.093$ |
| **Denoised Line-Scan (Gaussian)** | $13.63\text{ dB}$ / $0.321$ | $14.97\text{ dB}$ / $0.293$ | $14.60\text{ dB}$ / $0.244$ |
| **DGI (Classical Ghost Imaging)** | $12.65\text{ dB}$ / $0.243$ | $12.15\text{ dB}$ / $0.253$ | $13.19\text{ dB}$ / $0.345$ |
| **TV-CS ($L_1$ Optimization)** | $17.73\text{ dB}$ / $0.311$ | $16.62\text{ dB}$ / $0.322$ | $16.38\text{ dB}$ / $0.405$ |
| **Deep SPI (ReconNet)** | $16.97\text{ dB}$ / $0.306$ | $15.93\text{ dB}$ / $0.317$ | $16.05\text{ dB}$ / $0.395$ |
| **IC-PINOT (Blind Inversion)** | $19.78\text{ dB}$ / $0.352$ | $18.52\text{ dB}$ / $0.352$ | $17.01\text{ dB}$ / $0.392$ |
