# Claim-Evidence Matrix
**Working Title:** *Risk-Calibrated Single-Pixel Imaging for Reliable Textile Defect Inspection Under Joint Photon and Optical Shift*  
**Date:** 2026-10-23  

---

## Systematic Evaluation of Research Claims

| # | Scientific Claim | Current Evidence in Repo | Status | Action Required |
|:---:|:---|:---|:---:|:---|
| **C1** | Single-pixel bucket detection operates with zero readout noise under low photon flux. | `physics/noise.py` implements Poisson arrival model with zero readout noise for SPAD mode. | **VERIFIED** | Maintain accurate mathematical formulation of Poisson counting. |
| **C2** | Model actively inverts pattern-projected measurements rather than recalling templates. | `benchmarks/sanity_checks.py` shows $13.16\text{ dB}$ PSNR drop under shuffled measurements ($y[\pi]$). | **VERIFIED** | Document that linear backprojection layer $H^\dagger$ is the primary driver of permutation sensitivity. |
| **C3** | Model performance degrades monotonically with decreasing photon flux and increasing defocus. | `benchmarks/sanity_checks.py` confirms monotonic degradation across $\lambda \in [100, 1]$ and $a_4 \in [0, 3]$. | **VERIFIED** | Retain monotonic sweep results in benchmark reports. |
| **C4** | IC-PINOT executes in a single deterministic pass without Monte Carlo sampling. | `models/evidential.py` outputs Normal-Inverse-Gamma parameters in one forward pass ($7.2\text{ ms}$ on GPU). | **VERIFIED** | Distinguish GPU latency ($7.2\text{ ms}$) from physical DMD acquisition time ($23.0\text{ ms}$). |
| **C5** | Blind evaluation reconstructs targets without oracle defocus parameters. | `models/ic_pinot.py` processes raw bucket measurements without requiring defocus input $a_4$. | **PARTIAL** | When `optical_params=None`, code currently inputs a zero vector (oracle $a_4=0$). Must test under non-zero true defocus states. |
| **C6** | In-context exemplar pairs enable zero-gradient test-time adaptation across fabric weaves. | `models/in_context.py` implements prompt cross-attention, but `train_ic_pinot_master.py` never supplies context during training. | **REJECTED AS UNTRAINED** | Either train the in-context engine with exemplar pairs or remove zero-gradient adaptation claims. |
| **C7** | Evidential epistemic uncertainty isolates defect novelty on out-of-distribution textile weaves. | On AITEX LOFO, raw evidential uncertainty achieves AUROC of $36.80\%$ (worse than random guessing). | **REJECTED IN CURRENT FORM** | Epistemic uncertainty alone fails under material shift; must combine with measurement consistency $r = \|y - A\hat{x}\|$. |
| **C8** | Model outperforms classical optics and compressed sensing baselines under matched photon budget. | Outperforms DGI ($12.65\text{ dB}$) and TV-CS ($17.73\text{ dB}$) with $19.78\text{ dB}$ on TILDA. | **PARTIAL** | Underperforms trivial scalar brightness baseline ($29.22\text{ dB}$). Must reframe around structural edge/anomaly recovery. |
| **C9** | Selective inspection policy improves reliability by rejecting out-of-domain inspections. | Proposed Phase 4 research direction: evaluate risk-coverage trade-offs on AITEX LOFO and MVTec. | **PENDING IMPLEMENTATION** | Implement 3-tier selective policy (Accept / Re-measure / Refer) and evaluate risk-coverage curves. |
| **C10** | Continuous wave propagation solved by Fourier Neural Operator. | FNO is coded in `physics/fno.py` but never called in `models/ic_pinot.py`. | **REJECTED** | Disconnect resolved: renamed to Neural Optical Transformer; FNO claims purged. |
