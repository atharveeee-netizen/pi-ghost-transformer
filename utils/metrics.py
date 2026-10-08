"""
Evaluation Metrics for PI-GhostFormer.

Implements standard reconstruction and classification metrics
required for high-impact journal submissions.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse).item()


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    Calculate Structural Similarity Index (SSIM).
    A simplified 2D approximation for PyTorch. 
    For final paper results, skimage.metrics.structural_similarity is recommended.
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def calculate_classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict:
    """
    Calculates Accuracy, F1-Score, and AUROC for defect classification.
    """
    probs = F.softmax(logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    targets = targets.cpu().numpy()
    
    acc = np.mean(preds == targets)
    
    # Macro F1 is standard for imbalanced defect datasets
    f1 = f1_score(targets, preds, average='macro', zero_division=0)
    
    # AUROC requires one-hot encoding of targets if multiclass
    try:
        if probs.shape[1] > 2:
            auroc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
        else:
            auroc = roc_auc_score(targets, probs[:, 1])
    except ValueError:
        # Happens if a class is completely missing from the batch
        auroc = float('nan')
        
    return {
        'accuracy': acc,
        'f1_score': f1,
        'auroc': auroc
    }


def evaluate_batch(recon: torch.Tensor, gt: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> dict:
    """Evaluate a single batch returning all metrics."""
    return {
        'psnr': calculate_psnr(recon, gt),
        'ssim': calculate_ssim(recon, gt),
        **calculate_classification_metrics(logits, labels)
    }
