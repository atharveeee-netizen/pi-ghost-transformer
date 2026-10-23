# Systematic Literature Review: Risk-Calibrated Single-Pixel Imaging
**Working Research Title:** *Risk-Calibrated Single-Pixel Imaging for Reliable Textile Defect Inspection Under Joint Photon and Optical Shift*  
**Date:** 2026-10-23  
**Auditor:** Senior Research Engineer & Computational Imaging Reviewer  

---

## 1. Scope & Research Objective

This review investigates the intersection of:
1. **Photon-Limited Computational Ghost Imaging / Single-Pixel Imaging (SPI)** under sub-Nyquist sampling ($M \ll N^2$) and Poisson counting statistics;
2. **Measurement-Operator & Optical Defocus Mismatch** (e.g. unknown or drifting Zernike wavefront aberrations, defocus $a_4$, and optical transfer function attenuation);
3. **Material-Domain Shift** (unseen fabric weaves, surface texture variability, out-of-distribution industrial anomalies);
4. **Calibrated Uncertainty Quantification (UQ)** (epistemic/aleatoric decomposition, deep evidential regression, Bayesian approximations);
5. **Selective Inspection Policies** (risk-coverage trade-offs, abstention, remeasurement, and referral).

The research question driving this study is:
> *Can a measurement-consistent reconstruction system identify unreliable low-photon textile inspections under simultaneous optical/operator shift and material-domain shift, and use calibrated uncertainty to make reliable accept, re-measure, or refer decisions?*

---

## 2. Core Literature Categories & Primary Sources

### Category A: Single-Pixel Imaging & Physical Shot-Noise Limits
1. **Shapiro, J. H.** (2008). *Computational ghost imaging*. **Physical Review A**, 78(6), 061802. [DOI: 10.1103/PhysRevA.78.061802](https://doi.org/10.1103/PhysRevA.78.061802).
   - *Key Contribution:* Proved mathematically that classical computational ghost imaging can be achieved using a single bucket detector with predetermined spatial patterns.
   - *Physical Implication:* Clarified that when Poisson shot noise dominates (single-photon counting), multiplexing spatial patterns yields no fundamental signal-to-noise advantage over point-by-point scanning. Multiplex advantage (Fellgett's) holds only in detector-noise-limited regimes.
2. **Harwit, M., & Sloane, N. J.** (1979). *Hadamard Transform Optics*. Academic Press.
   - *Key Contribution:* Formalized the SNR behavior of multiplexed optical masks. Proved that under photon shot noise, multiplexing distributes the Poisson variance of all illuminated regions into each scalar measurement, penalizing high-contrast localized anomalies.
3. **Higham, C. F., Murray-Smith, R., Padgett, M. J., & Edgar, M. P.** (2018). *Deep learning for real-time single-pixel video*. **Scientific Reports**, 8(1), 2369. [DOI: 10.1038/s41598-018-20521-y](https://doi.org/10.1038/s41598-018-20521-y).
   - *Key Contribution:* Demonstrated feedforward CNN autoencoders for real-time SPI reconstruction at 128x128 resolution.
   - *Limitation:* Purely deterministic point estimators; assumed fixed, noise-free Hadamard measurement operators and zero operator mismatch.
4. **Gnanasambandam, A., & Chan, S. H.** (2020). *Image formation and exposure control for single-photon cameras*. **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, 44(12), 9540–9554. [DOI: 10.1109/TPAMI.2020.3040336](https://doi.org/10.1109/TPAMI.2020.3040336).
   - *Key Contribution:* Formulated exact mathematical exposure models and flux limits for Quanta Image Sensors (QIS) and SPADs under non-linear photon arrival statistics and dead-time pile-up.

### Category B: Uncertainty Quantification in Single-Pixel & Inverse Imaging
5. **Shang, R., O’Brien, M. A., Wang, F., Situ, G., & Luke, G. P.** (2023). *Approximating the uncertainty of deep learning reconstruction predictions in single-pixel imaging*. **Communications Engineering**, 2(1), 53. [DOI: 10.1038/s44172-023-00103-1](https://doi.org/10.1038/s44172-023-00103-1).
   - *Key Contribution:* Applied Bayesian CNNs with Monte Carlo Dropout and heteroscedastic NLL loss to quantify pixel-level uncertainty in SPI. Showed that predictive variance correlates strongly with edge reconstruction error.
   - *Limitation:* Requires 30–50 stochastic forward passes per frame (computationally prohibitive at real-time industrial line speed); evaluated only under matched optical operators without domain shift or selective decision-making.
6. **Li, Q., Huang, G., Li, Y., Liu, Z., & Liu, S.** (2025). *Uncertainty-aware physics-informed high-fidelity ghost imaging in ambient illumination*. **Optics Letters**, 50(2), 245–248. [DOI: 10.1364/OL.580593](https://doi.org/10.1364/OL.580593).
   - *Key Contribution:* Integrated environmental ambient light estimation with physics-informed regularization for ghost imaging.
   - *Limitation:* Per-instance optimization (iterative); does not provide single-pass closed-form epistemic/aleatoric uncertainty.
7. **Li, Q.** (2026). *Gaussian priors and physics-informed for domain-agnostic ghost imaging*. **Optics and Lasers in Engineering**, 196, 109405. [DOI: 10.1016/j.optlaseng.2025.109405](https://doi.org/10.1016/j.optlaseng.2025.109405).
   - *Key Contribution:* Combined Gaussian prior modeling with optical forward constraints to improve out-of-domain ghost imaging generalization.
   - *Limitation:* Focuses on general scenes without defect anomaly detection, risk calibration, or selective prediction.
8. **Amini, A., Schwarting, W., Soleimany, A., & Rus, D.** (2020). *Deep evidential regression*. **Advances in Neural Information Processing Systems (NeurIPS)**, 33, 14927–14937.
   - *Key Contribution:* Formulated deterministic Normal-Inverse-Gamma (NIG) prior over Gaussian likelihoods, enabling single-pass extraction of aleatoric and epistemic uncertainty without Monte Carlo sampling.
   - *Critical Caveat:* Subsequent literature (e.g. Meinert et al., 2023) showed that empirical NIG epistemic uncertainty can fail to separate out-of-distribution shift without calibration, requiring rigorous empirical validation.

### Category C: Learned Reconstruction Under Operator Mismatch & Defocus
9. **Kulkarni, K., Lohit, S., Turaga, P., Kerviche, R., & Ashok, A.** (2016). *ReconNet: Non-iterative reconstruction of images from compressively sensed measurements*. **IEEE CVPR**, 449–458.
   - *Key Contribution:* Classic deep compressive sensing inverter mapping block CS measurements to initial images via fully connected layers followed by convolutional refinement.
10. **Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.** (2021). *Fourier neural operator for parametric partial differential equations*. **ICLR**.
    - *Key Contribution:* Parameterizes solution operators in continuous Fourier space. Mesh-independent operator mapping.
    - *Caveat for Project:* On a fixed discrete grid ($64 \times 64$), FNO reduces to frequency-domain filtering and provides no automatic invariance to optical defocus unless trained across defocus distributions.
11. **Ongie, G., Jalal, A., Metzler, C. A., Baraniuk, R. G., Dimakis, A. G., & Willett, R.** (2020). *Deep learning techniques for inverse problems in imaging*. **IEEE Journal on Selected Areas in Information Theory**, 1(1), 39–56.
    - *Key Contribution:* Comprehensive review establishing that learned inverters degrade sharply when the test forward operator $A_{\text{test}}$ deviates from training $A_{\text{train}}$. Emphasized measurement consistency constraints $A^\dagger y$.

### Category D: Selective Classification, Abstention, & Conformal Risk
12. **Geifman, Y., & El-Yaniv, R.** (2017). *Selective classification for deep neural networks*. **NeurIPS**, 30, 4878–4887.
    - *Key Contribution:* Formalized the risk–coverage curve for neural networks, optimizing selective prediction to guarantee bounded test risk at minimal abstention.
13. **Geifman, Y., & El-Yaniv, R.** (2019). *SelectiveNet: A deep neural network with an integrated reject option*. **ICML**, 2151–2159.
    - *Key Contribution:* Architectural framework incorporating a dedicated selection/rejection head for risk-controlled decision-making.
14. **Angelopoulos, A. N., & Bates, S.** (2021). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification*. **arXiv:2107.07511**.
    - *Key Contribution:* Mathematical foundation for finite-sample distribution-free risk guarantees using held-out calibration sets.
15. **Bates, S., Candès, E., Lei, L., & Romano, Y.** (2021). *Distribution-free risk-controlling prediction sets*. **Journal of the ACM (JACM)**, 68(6), 1–34.
    - *Key Contribution:* Generalized conformal prediction to control arbitrary monotonic risk functions (e.g., false-negative rate, reconstruction MAE).

### Category E: Industrial Surface Inspection & Anomaly Benchmarks
16. **Bergmann, P., Fauser, M., Sattlegger, D., & Steger, C.** (2019). *MVTec AD — A comprehensive real-world dataset for unsupervised anomaly detection*. **IEEE CVPR**, 9592–9600.
    - *Key Contribution:* Standardized unsupervised anomaly detection benchmark with pixel-level ground truth masks and Per-Region Overlap (PRO) metric.
17. **Silvestre-Blanes, J., Albero-Albero, T., Miralles, I., Pérez-Llorens, R., & Moreno, J.** (2019). *A public fabric database for defect detection methods and results*. **AUTEX Research Journal**, 19(4), 363–374. [DOI: 10.1515/aut-2018-0035](https://doi.org/10.1515/aut-2018-0035).
    - *Key Contribution:* Real-world industrial fabric database (AITEX) featuring 245 defect images and 141 defect-free images across 7 commercial textile weaves with high-resolution ground truth masks.
18. **Schulz-Mirbach, H.** (1999). *TILDA: A database for the inspection of textile surfaces*. University of Freiburg Technical Report.
    - *Key Contribution:* Pioneering benchmark for textile defect classification across 4 structured woven and knit fabric categories.

---

## 3. Critical Methodological Synthesis

1. **The Disconnect in Prior SPI Uncertainty Work:**
   Existing SPI uncertainty methods (e.g. Shang et al., 2023) focus exclusively on *in-distribution reconstruction variance* using Monte Carlo dropout. They assume:
   - Matched optical operator ($A_{\text{test}} = A_{\text{train}}$);
   - Constant photon flux;
   - Same object distribution.
   No existing SPI work evaluates **joint optical mismatch** (lens defocus, vibration) and **material-domain shift** (unseen fabrics).
2. **The Anomaly Detection Gap:**
   Standard unsupervised anomaly detection methods (e.g. PatchCore, SimpleNet) assume high-resolution, uncompressed, noise-free images from multi-pixel cameras. When deployed on low-photon single-pixel reconstructions, anomaly scoring collapses because compressive reconstruction artifacts trigger false alarms.
3. **The Selective Inspection Opportunity:**
   Rather than forcing a single model to output an unconditional defect prediction under severe degradation, a **measurement-consistent risk-calibrated pipeline** can quantify both measurement discrepancy $\|y - A \hat{x}\|$ and model epistemic uncertainty, selectively abstaining on unreliable inspections (re-measure or refer to human).
