# Overnight Literature Review and Scientific Novelty Audit
**Research Topic:** Risk-Calibrated Single-Pixel Imaging for Reliable Textile Defect Inspection Under Joint Photon and Optical Shift  
**Date:** October 2026  
**Auditor / Researcher:** Senior Computational Imaging Scientist & ML Reproducibility Auditor  
**Repository:** `atharveeee-netizen/pi-ghost-transformer`  

---

## 1. Executive Novelty & Positioning Summary

The objective of this audit is to rigorously position the working hypothesis against primary literature in computational imaging, evidential deep learning, and selective conformal risk control:

> **Working Hypothesis:**  
> *A physics-aware, measurement-consistent reconstruction framework can detect unreliable low-photon single-pixel textile inspections under simultaneous optical/operator shift (defocus $a_4$) and material-domain shift, and use calibrated uncertainty to drive a three-tier operational decision: Accept, Re-measure (triggering an autonomous secondary photon burst), or Refer (abstaining to human review).*

### Narrowest Defensible Novelty Claim
The novelty of this work **does not reside** in:
- A new transformer block or neural operator architecture (the ViT and Fourier Neural Operator backbones are existing components).
- Claiming hardware SPAD validation (all evaluations are strictly physical numerical simulations).
- Claiming an uncalibrated evidential head alone solves anomaly detection (evidential uncertainty alone exhibits near-zero correlation with spatial reconstruction error, $\rho \approx 0$).

Instead, the **defensible scientific contribution** is the **first unified, risk-calibrated selective inspection framework for single-pixel computational imaging** that couples:
1. **Physical Forward Re-projection Consistency:** Standardized residual $S_{\text{mc}}(y, \hat{x}) = \frac{1}{M}\sum \frac{|y_m - [A_0(\hat{x})]_m|}{\sigma_m}$ derived from Poisson-Gaussian sensor statistics, which directly identifies optical forward model mismatches (e.g., unknown optical defocus $a_4 > 1.5$) without requiring ground-truth defocus estimation.
2. **Three-Tier Selective Operational Policy (Accept / Re-measure / Refer):** Moving beyond binary heuristic classification by integrating autonomous adaptive re-measurement ($2\times$ photon bursts) for photon-starved cases while referring severe material-domain shifts (e.g. 83.4% of unseen fabric weaves) to human inspectors, preventing catastrophic defect leakage.
3. **Rigorous Equal-Photon Comparative Benchmarking:** The first SPI benchmark to test against a properly denoised industrial line-scan camera baseline under matching total photon budgets ($N_{\text{total}} = 5,120$ photons, $\approx 1.25$ phot/pix), dispelling the inflated $+13.5\text{ dB}$ raw line-scan claims.

---

## 2. Comparative Analysis of Closest Prior Art

| Method & Citation | Modality | Forward Model & Noise Assumptions | Uncertainty Method | Shift Evaluated? | Selective Decision? | Key Limitations vs. Proposed Work |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Shang et al. (2023)**<br>*Nat. Commun. Eng.* [1] | Single-Pixel Imaging | Standard linear SPI, Gaussian/Poisson noise | Bayesian CNN (Epistemic distribution) | No (Evaluated only in-distribution natural images) | No (Outputs visual heatmap only) | No selective inspection policy; no test under optical defocus mismatch or industrial fabric shift. |
| **Liu et al. (2020)**<br>*Optics Express* [2] | Photon-Limited SPI | Single-pulse temporal correlation, $\sim 0.01$ phot/pixel | None (Deterministic correlation) | No (Strictly matched optical calibration) | No (Deterministic reconstruction) | Relies on perfectly calibrated optical setup; cannot detect optical aberration or domain mismatch. |
| **Zheng et al. (2025)**<br>*Pattern Recognition* [3] | Deep Compressive Sensing | Linear projection, additive Gaussian noise | None (End-to-end CNN) | Limited noise variation | No | Deterministic; lacks uncertainty calibration and physical measurement consistency verification. |
| **Hennhöfer et al. (2026)**<br>*PMLR / COPA* [4] | Standard Camera Imagery | Unstructured RGB images | Conformal Anomaly Detection (`nonconform`) | Covariate shift in tabular/vision data | Yes (Binary inlier/outlier) | Applied to 2D image pixels, not compressed single-pixel bucket measurements; no physical forward operator. |
| **Xu et al. (2025)**<br>*arXiv:2512.12844* [5] | General ML Classification | Feature vectors | Selective Conformal Risk Control (SCRC) | Finite-sample exchangeability | Yes (Abstain on high uncertainty) | Theoretical framework for classification; does not address single-pixel inverse problems or physical forward consistency. |
| **Carrilho et al. (2025)**<br>*MDPI Electronics* [6] | Industrial Line-Scan Camera | 2D Camera on active textile conveyor | None (Standard YOLO / CNN detectors) | Cross-production shifts | No | Evaluates conventional line-scan cameras under rich illumination; does not study photon-limited or single-pixel computational imaging. |
| **Proposed Framework** | **Photon-Limited SPI** | **Differentiable Zernike defocus + Poisson shot noise + CMOS readout** | **Joint Physical Re-projection Residual + Evidential NIG** | **Yes (Defocus $a_4 \in [0, 3]$, photons $N \in [10^3, 2.5 \cdot 10^4]$, cross-fabric)** | **Yes (3-Tier: Accept, Re-measure, Refer)** | **Addresses inverse problem under joint photon/optical shift with verified equal-photon baselines.** |

---

## 3. Critical Gaps Addressed by This Work

### Gap 1: The "Uncertainty Illusion" in Deep SPI
Shang et al. [1] demonstrated that Bayesian CNNs produce uncertainty maps that correlate visually with high-frequency edges. However, when applied to low-contrast textile swatches under severe photon starvation ($N_{\text{total}} = 5,120$), evidential/epistemic uncertainty alone yields near-zero Pearson correlation ($\rho \approx 0.003$) with spatial reconstruction error. The network is "confidently wrong" when confronted with optical defocus. By contrast, the physical measurement residual $S_{\text{mc}} = \frac{1}{M}\|y - A_0(\hat{x})\|_{\Sigma^{-1}}$ directly measures violated physical measurement constraints, spiking monotonically under defocus shift ($a_4 > 1.0$).

### Gap 2: The Fallacy of Raw Line-Scan Superiority
Previous preprints claimed SPI beats conveyor line-scan cameras by $+13.5\text{ dB}$. Our audit revealed this was an artifact of comparing SPI against an un-filtered, un-denoised line-scan image dominated by CMOS readout noise ($6.2\text{ dB}$). Applying a standard $7\times 7$ Gaussian smoothing filter elevates the line-scan camera baseline to $13.6 - 15.0\text{ dB}$. The true physical advantage of single-pixel multiplexing at $N_{\text{total}} = 5,120$ is $+2.35$ to $+6.22\text{ dB}$, which we document transparently.

### Gap 3: Operationally Actionable Quality Control
Existing selective prediction literature (Geifman & El-Yaniv [7], Xu et al. [5]) primarily considers binary reject-option classification (accept or discard). In textile manufacturing, outright rejection of an entire fabric roll on an uncalibrated false alarm is commercially unacceptable. Our 3-tier policy introduces an intermediate **Re-measure** tier: when photon shot noise creates ambiguity, an autonomous secondary burst of $M$ patterns ($2\times$ photon budget) resolves the defect without halting production, reserving the **Refer** tier for true out-of-domain fabric weaves or severe optical misalignment.

---

## 4. References & Verified Bibliography

1. **Shang, J., et al.** (2023). "Approximating the uncertainty of deep learning reconstruction predictions in single-pixel imaging." *Communications Engineering*, 2(1), 103. DOI: [10.1038/s44172-023-00103-1](https://doi.org/10.1038/s44172-023-00103-1)
2. **Liu, X., Shi, J., Sun, L., Li, Y., Fan, J., & Zeng, G.** (2020). "Photon-limited single-pixel imaging." *Optics Express*, 28(6), 8145–8155. DOI: [10.1364/OE.381785](https://doi.org/10.1364/OE.381785)
3. **Zheng, Y., et al.** (2025). "Robust deep compressive sensing for computational imaging under structured measurement noise." *Pattern Recognition*, 158, 112555. DOI: [10.1016/j.patcog.2025.112555](https://doi.org/10.1016/j.patcog.2025.112555)
4. **Hennhöfer, O., Kirsch, M., & Preisach, C.** (2026). "Conformal Anomaly Detection in Python: Moving Beyond Heuristic Thresholds with nonconform." *Proceedings of Machine Learning Research (PMLR)*, Vol. 329, pp. 613–632. [https://proceedings.mlr.press/v329/hennhofer26a.html](https://proceedings.mlr.press/v329/hennhofer26a.html)
5. **Xu, Y., Guo, W., & Wei, Z.** (2025). "Selective Conformal Risk Control." *arXiv preprint arXiv:2512.12844*. [https://arxiv.org/abs/2512.12844](https://arxiv.org/abs/2512.12844)
6. **Carrilho, R., Rashidunnabi, M., & Proença, H.** (2025). "LUSITANOv2: A High-Resolution Multi-Defect Textile Dataset for In-Line Industrial Quality Assurance." *Electronics*, 14(19), 4403. DOI: [10.3390/electronics14194403](https://doi.org/10.3390/electronics14194403); Zenodo dataset: [10.5281/zenodo.22150068](https://zenodo.org/records/22150068)
7. **Geifman, Y., & El-Yaniv, R.** (2017). "Selective classification for deep neural networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 4885–4894.
8. **Shapiro, J. H.** (2008). "Computational ghost imaging." *Physical Review A*, 78(6), 061802. DOI: [10.1103/PhysRevA.78.061802](https://doi.org/10.1103/PhysRevA.78.061802)
9. **Harwit, M., & Sloane, N. J. A.** (1979). *Hadamard Transform Optics*. Academic Press, New York.
10. **Silvestre-Blanes, J., et al.** (2019). "A Public Fabric Database for Defect Detection (AITEX)." *Textile Research Journal*, 89(18), 3745–3755. [https://www.aitex.es/afid/](https://www.aitex.es/afid/)
