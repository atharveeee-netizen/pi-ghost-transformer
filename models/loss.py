"""
Training loss functions for PI-GhostFormer.

Implements the novel Poisson Negative Log-Likelihood (NLL) loss for the 
physics-informed reconstruction, combined with standard perceptual and classification losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoissonNLLLoss(nn.Module):
    """
    Poisson Negative Log-Likelihood Loss.
    
    L_Poisson = Σ [ expected - measured * log(expected) ]
    
    This is the physically correct loss for photon-limited data (SPADs),
    replacing the standard MSE used in vanilla Ghost Translation.
    
    Args:
        eps: Small constant to avoid log(0).
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, expected: torch.Tensor, measured: torch.Tensor) -> torch.Tensor:
        """
        Args:
            expected: Expected photon counts from forward model, shape (B, M)
            measured: Actual noisy bucket measurements (scaled to counts), shape (B, M)
            
        Returns:
            Scalar loss value.
        """
        # Ensure expected is strictly positive
        expected = torch.clamp(expected, min=self.eps)
        
        # Poisson NLL: \lambda - y \log(\lambda)
        nll = expected - measured * torch.log(expected)
        return torch.mean(nll)


class TotalVariationLoss(nn.Module):
    """
    Total Variation (TV) Regularization Loss.
    Encourages piece-wise smoothness (sparsity in gradient domain).
    Particularly useful for the self-supervised training mode.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Reconstructed image, shape (B, C, H, W)
            
        Returns:
            Scalar TV loss.
        """
        B, C, H, W = x.size()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return (h_tv + w_tv) / (B * C * H * W)


class PIGhostFormerLoss(nn.Module):
    """
    Combined loss function for the PI-GhostFormer.
    
    L_total = λ_recon * MSE(x_gt, x_recon)
            + λ_physics * PoissonNLL(y_expected, y_measured)
            + λ_classify * CrossEntropy(logits, class_gt)
            + λ_tv * TV(x_recon)
    """
    def __init__(
        self, 
        lambda_recon: float = 1.0,
        lambda_physics: float = 0.5,
        lambda_classify: float = 0.3,
        lambda_tv: float = 0.01,
        self_supervised: bool = False
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_physics = lambda_physics
        self.lambda_classify = lambda_classify
        self.lambda_tv = lambda_tv
        self.self_supervised = self_supervised
        
        self.mse_loss = nn.MSELoss()
        self.poisson_nll = PoissonNLLLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.tv_loss = TotalVariationLoss()
        
    def forward(
        self, 
        recon_img: torch.Tensor, 
        logits: torch.Tensor, 
        expected_buckets: torch.Tensor, 
        measured_buckets: torch.Tensor, 
        gt_img: torch.Tensor = None, 
        gt_class: torch.Tensor = None
    ) -> dict:
        """
        Compute total loss and individual components.
        
        Args:
            recon_img: (B, 1, H, W)
            logits: (B, num_classes)
            expected_buckets: (B, M) from differentiable forward model
            measured_buckets: (B, M) actual noisy measurements (scaled to photon counts)
            gt_img: (B, 1, H, W) ground truth image (None if self-supervised)
            gt_class: (B,) ground truth class labels (None if self-supervised)
            
        Returns:
            Dictionary containing 'total_loss' and individual loss terms.
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Physics Loss (Poisson NLL)
        if self.lambda_physics > 0:
            loss_phys = self.poisson_nll(expected_buckets, measured_buckets)
            losses['loss_physics'] = loss_phys.item()
            total_loss += self.lambda_physics * loss_phys
            
        # 2. Reconstruction Loss (MSE)
        if not self.self_supervised and gt_img is not None and self.lambda_recon > 0:
            loss_recon = self.mse_loss(recon_img, gt_img)
            losses['loss_recon'] = loss_recon.item()
            total_loss += self.lambda_recon * loss_recon
            
        # 3. Classification Loss (Cross Entropy)
        if not self.self_supervised and gt_class is not None and self.lambda_classify > 0:
            loss_cls = self.ce_loss(logits, gt_class)
            losses['loss_classify'] = loss_cls.item()
            total_loss += self.lambda_classify * loss_cls
            
        # 4. Total Variation Regularization (crucial for self-supervised)
        if self.lambda_tv > 0:
            loss_tv = self.tv_loss(recon_img)
            losses['loss_tv'] = loss_tv.item()
            total_loss += self.lambda_tv * loss_tv
            
        losses['total_loss'] = total_loss
        return losses
