# Novelty & Defensible Research Gap Analysis
**Date:** 2026-10-23  
**Working Research Title:** *Risk-Calibrated Single-Pixel Imaging for Reliable Textile Defect Inspection Under Joint Photon and Optical Shift*  

---

## 1. Narrowest Defensible Research Gap

Existing computational imaging literature exhibits three critical fractures:
1. **The Uncertainty vs. Speed Trade-off in SPI:**  
   Prior uncertainty estimation methods for single-pixel imaging (e.g. Shang et al., *Nat. Commun. Eng.* 2023) rely on Monte Carlo Dropout (30 to 50 stochastic passes), which requires seconds per frame and is completely incompatible with real-time inspection. While evidential deep learning (Amini et al., 2020) provides $O(1)$ single-pass uncertainty, it has never been integrated with physical optical backprojection or calibrated under optical operator mismatch.
2. **The Isolation of Optical Shift from Material-Domain Shift:**  
   Inverse problems research typically tests optical aberrations (e.g. defocus, motion blur) on synthetic natural images with fixed operators. Conversely, computer vision defect detection benchmarks (e.g. MVTec, AITEX) test material shifts on pristine, high-photon camera images. In continuous web manufacturing, **both shifts occur simultaneously**: conveyor vibration perturbs the optical focal distance ($a_4$), illumination power fluctuates, and changing fabric rolls induces unseen textile weave geometries.
3. **The Absence of Selective Inspection Policies in Compressive Imaging:**  
   Prior deep SPI systems force an unconditional point prediction for every single sample. When optical blur or photon starvation destroys measurement fidelity, the network outputs corrupted or hallucinated reconstructions that trigger severe false defect calls or miss critical yarn breaks. There is no existing framework that computes a **physically grounded measurement-consistency signal** ($r = \|y - A_\phi(\hat{x})\|$) alongside predictive uncertainty to drive an operational **Accept / Re-measure / Refer** decision policy.

### Core Research Contribution Statement (Defensible Scope):
> We formulate and empirically evaluate a **Risk-Calibrated Single-Pixel Imaging (RC-SPI)** framework for industrial surface inspection. By uniting physical pseudoinverse adjoint backprojection, single-pass evidential uncertainty decomposition, and measurement-consistency residuals, the framework quantifies reconstruction unreliability under simultaneous photon starvation and optical defocus. Rather than forcing unreliable predictions on out-of-domain fabric weaves, a calibrated selective inspection policy abstains on ambiguous cases and achieves provable empirical error reduction on accepted samples.

---

## 2. Explicit List of Prohibited Claims (Claims That MUST NOT Be Made)

To ensure scientific defensibility and protect against journal rejection, the following claims are **strictly prohibited** across all manuscripts, documentation, and presentations:

1. ❌ **DO NOT CLAIM real hardware validation.**  
   *Reason:* All experiments are conducted via numerical simulation on digitized benchmarks (TILDA, DAGM, MVTec, AITEX). No physical SPAD detector or DMD hardware bench was constructed. The paper must be clearly titled and framed as a *simulation study*.
2. ❌ **DO NOT CLAIM 138 FPS industrial inspection throughput.**  
   *Reason:* 138 FPS ($7.2\text{ ms}$) represents isolated GPU tensor inference on pre-recorded vectors. Sequential binary pattern projection on a $22.2\text{ kHz}$ DMD requires $\ge 23.0\text{ ms}$, capping physical frame rate at $\le 43.4\text{ FPS}$ and end-to-end throughput at $\approx 31.3\text{ FPS}$.
3. ❌ **DO NOT CLAIM that SPI "freezes conveyor motion" without qualification.**  
   *Reason:* At 1 to 2 m/s, sequential projection of 512 patterns spans 23 ms, translating the web by 23 to 46 mm (15 to 30 pixels). The forward model does not simulate intra-frame space-time shearing; it models stationary/stroboscopic windows.
4. ❌ **DO NOT CLAIM a +7 to +13 dB gain over physics/optics.**  
   *Reason:* Trivial scalar brightness predictors achieve $29.22\text{ dB}$ on TILDA (outperforming the neural model's $19.78\text{ dB}$ by $9.44\text{ dB}$) due to low background variance. The model's value lies in high-frequency edge restoration and defect localization, not bulk DC PSNR.
5. ❌ **DO NOT CLAIM Fourier Neural Operator (FNO) continuous wavefield guarantees.**  
   *Reason:* The FNO module is not integrated into the forward inference path of `ICPINOT`. On a discrete $64 \times 64$ grid, neural operators provide no magical resolution-invariance to optical blur.
6. ❌ **DO NOT CLAIM state-of-the-art unsupervised anomaly detection on unseen fabrics.**  
   *Reason:* AITEX Leave-One-Fabric-Out cross-validation achieves a raw AUROC of $36.80\%$, demonstrating that uncalibrated zero-shot reconstruction residuals fail on novel weaves.
7. ❌ **DO NOT CLAIM mathematical guarantees unless the exchangeability and boundedness assumptions are proven.**  
   *Reason:* Conformal and evidential bounds hold under exchangeability; when testing across severe domain shifts, guarantees degrade to empirical risk-coverage calibration.
