"""
Evidential Deep Learning for Calibrated Uncertainty Quantification in Ghost Imaging.
Implements Normal-Inverse-Gamma (NIG) distribution modeling based on Amini et al. (NeurIPS 2020).
Predicts [gamma, v, alpha, beta] per-pixel in a single deterministic pass:
  - Mean Reconstruction: gamma
  - Aleatoric Uncertainty: beta / (alpha - 1)
  - Epistemic Uncertainty: beta / (v * (alpha - 1))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead2D(nn.Module):
    """
    Predicts Normal-Inverse-Gamma (NIG) parameters for 2D spatial reconstruction.
    Outputs:
        gamma: mean intensity in [0, 1]
        v: virtual evidence (v > 0)
        alpha: shape parameter (alpha > 1)
        beta: scale parameter (beta > 0)
    """
    def __init__(self, in_channels: int, out_channels: int = 1):
        super().__init__()
        # 4 outputs per spatial pixel: [gamma, v, alpha, beta]
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> dict:
        out = self.conv(x)
        # Split into 4 components
        gamma_raw, v_raw, alpha_raw, beta_raw = torch.chunk(out, 4, dim=1)
        
        # Enforce mathematical constraints of NIG distribution
        gamma = torch.sigmoid(gamma_raw)              # Bounded normalized image [0, 1]
        v = F.softplus(v_raw) + 1e-6                  # Evidence v > 0
        alpha = F.softplus(alpha_raw) + 1.0 + 1e-6    # Alpha > 1 (required for finite variance)
        beta = F.softplus(beta_raw) + 1e-6            # Scale beta > 0
        
        # Calculate closed-form uncertainties
        aleatoric = beta / (alpha - 1.0)
        epistemic = beta / (v * (alpha - 1.0))
        total_uncertainty = aleatoric + epistemic
        
        return {
            "gamma": gamma,
            "v": v,
            "alpha": alpha,
            "beta": beta,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total_uncertainty": total_uncertainty
        }


class EvidentialNIGLoss(nn.Module):
    """
    Negative Log-Likelihood loss for Normal-Inverse-Gamma (NIG) distribution
    with error-regularization penalty.
    """
    def __init__(self, coeff_reg: float = 0.05):
        super().__init__()
        self.coeff_reg = coeff_reg

    def forward(self, nig_dict: dict, target: torch.Tensor, coeff_reg: float = None) -> dict:
        gamma = nig_dict["gamma"]
        v = nig_dict["v"]
        alpha = nig_dict["alpha"]
        beta = nig_dict["beta"]
        
        # Ensure target is broadcastable
        if target.dim() == 3:
            target = target.unsqueeze(1)
            
        error = (target - gamma).abs()
        omega = 2.0 * beta * (1.0 + v)
        
        # 1. Negative Log-Likelihood under Student-t marginal
        # L_NLL = 0.5 * log(pi / v) - alpha * log(omega) + (alpha + 0.5) * log((target - gamma)^2 * v + omega) + lgamma(alpha) - lgamma(alpha + 0.5)
        nll = (
            0.5 * torch.log(torch.tensor(torch.pi, device=gamma.device) / v)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log((target - gamma)**2 * v + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        
        # 2. Evidence regularizer: penalizes low uncertainty when error is high
        reg = error * (2.0 * v + alpha)
        
        loss_nll = nll.mean()
        loss_reg = reg.mean()
        coeff = self.coeff_reg if coeff_reg is None else coeff_reg
        total_loss = loss_nll + coeff * loss_reg
        
        return {
            "loss_evidential": total_loss,
            "loss_nll": loss_nll,
            "loss_reg": loss_reg
        }
