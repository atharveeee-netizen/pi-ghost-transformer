"""
Risk-Calibrated Selective Inspection Policy for Single-Pixel Imaging.

Combines:
1. Measurement-Consistency Residual:
   r = ||y - A_0(x_hat)||_2 / sigma_expected
   Catches optical forward mismatch (defocus, blur) and severe out-of-distribution patterns.
2. Evidential Epistemic Uncertainty:
   beta / (v * (alpha - 1))
   Measures model parameter ignorance on novel fabric textures.
3. Three-Tier Decision Policy:
   - ACCEPT: Confident in-distribution sample. Low risk. Accept automated defect prediction.
   - RE-MEASURE: Moderate uncertainty, photon-limited. Trigger additional exposure/patterns.
   - REFER: High model mismatch or out-of-distribution. Abstain and escalate to human inspector.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class MeasurementConsistencyScorer:
    """
    Computes normalized measurement-consistency residual against the physical forward operator.
    """
    def __init__(self, forward_model):
        self.forward_model = forward_model

    def compute_residual(
        self,
        reconstruction: torch.Tensor,
        observed_measurements: torch.Tensor,
        nominal_optical_params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            reconstruction: (B, 1, H, W) in [0, 1]
            observed_measurements: (B, M) normalized observed bucket measurements
            nominal_optical_params: (B, 11) or None (zeros)
        Returns:
            residuals: (B,) normalized scalar discrepancy per sample
        """
        B = reconstruction.shape[0]
        device = reconstruction.device
        if nominal_optical_params is None:
            nominal_optical_params = torch.zeros(B, self.forward_model.num_patterns if hasattr(self.forward_model, 'physics_dim') else 11, device=device)
            if nominal_optical_params.shape[-1] != 11:
                nominal_optical_params = torch.zeros(B, 11, device=device)

        # Expected counts from reconstructed image under nominal forward operator
        with torch.no_grad():
            expected_counts = self.forward_model.forward_expected(reconstruction, nominal_optical_params) # (B, M)
            expected_meas_norm = expected_counts / (self.forward_model.mean_photon_count + 1e-8)

            # Expected variance under Poisson-readout statistics:
            # Var(y_norm) = (Var(counts)) / (scale^2) = (expected_counts + sigma_read^2) / scale^2
            scale = self.forward_model.mean_photon_count + 1e-8
            sigma_read = self.forward_model.noise_model.readout_noise_std
            var_norm = (expected_counts + sigma_read**2) / (scale**2)
            std_norm = torch.sqrt(var_norm.clamp(min=1e-8))

            # Normalized residual: |y - y_hat| / sigma
            diff = (observed_measurements - expected_meas_norm).abs() / std_norm # (B, M)
            residual_score = diff.mean(dim=1) # (B,)

        return residual_score


class SelectiveInspectionPolicy:
    """
    Risk-Calibrated Selective Inspection Policy.
    Maps evidential uncertainty and measurement consistency to (Accept, Re-measure, Refer) actions.
    """
    def __init__(
        self,
        tau_accept: float = 0.5,
        tau_refer: float = 0.85,
        alpha_mc: float = 0.6
    ):
        """
        Args:
            tau_accept: Quantile threshold for Accept vs Re-measure.
            tau_refer: Quantile threshold for Re-measure vs Refer.
            alpha_mc: Relative weight of measurement consistency vs evidential uncertainty.
        """
        self.tau_accept = tau_accept
        self.tau_refer = tau_refer
        self.alpha_mc = alpha_mc
        self.mc_mean = 0.0
        self.mc_std = 1.0
        self.unc_mean = 0.0
        self.unc_std = 1.0
        self.is_calibrated = False

    def calibrate(
        self,
        val_mc_scores: np.ndarray,
        val_uncertainties: np.ndarray
    ):
        """
        Calibrates standardization statistics and decision thresholds on validation split.
        """
        self.mc_mean = float(np.mean(val_mc_scores))
        self.mc_std = float(np.std(val_mc_scores)) + 1e-8
        self.unc_mean = float(np.mean(val_uncertainties))
        self.unc_std = float(np.std(val_uncertainties)) + 1e-8

        # Standardized composite risk scores on validation set
        z_mc = (val_mc_scores - self.mc_mean) / self.mc_std
        z_unc = (val_uncertainties - self.unc_mean) / self.unc_std
        val_risk = self.alpha_mc * z_mc + (1.0 - self.alpha_mc) * z_unc

        # Compute empirical quantile thresholds
        self.thresh_accept = float(np.percentile(val_risk, self.tau_accept * 100))
        self.thresh_refer = float(np.percentile(val_risk, self.tau_refer * 100))
        self.is_calibrated = True

    def compute_risk(
        self,
        mc_scores: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Computes standardized scalar composite inspection risk."""
        device = mc_scores.device
        z_mc = (mc_scores - self.mc_mean) / self.mc_std
        z_unc = (uncertainties - self.unc_mean) / self.unc_std
        risk = self.alpha_mc * z_mc + (1.0 - self.alpha_mc) * z_unc
        return risk

    def decide(
        self,
        risk_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Classifies each inspection into:
            0: ACCEPT (Automated decision accepted)
            1: RE_MEASURE (Request additional photon counts / pattern burst)
            2: REFER (Abstain, escalate to human inspection)
        """
        if not self.is_calibrated:
            # Fallback default thresholds if not calibrated
            t_acc, t_ref = 0.0, 1.5
        else:
            t_acc, t_ref = self.thresh_accept, self.thresh_refer

        decisions = torch.zeros_like(risk_scores, dtype=torch.long)
        decisions[risk_scores > t_acc] = 1 # RE_MEASURE
        decisions[risk_scores > t_ref] = 2 # REFER

        counts = {
            "accept": int((decisions == 0).sum().item()),
            "re_measure": int((decisions == 1).sum().item()),
            "refer": int((decisions == 2).sum().item()),
            "total": int(decisions.numel())
        }
        return decisions, counts
