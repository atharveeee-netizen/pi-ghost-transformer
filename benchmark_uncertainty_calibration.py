"""
Pillar 4: Uncertainty Quantification (UQ) Calibration & Orthogonal Dissociation.
Evaluates:
  1. Expected Calibration Error (ECE) for Evidential NIG Credible Intervals
  2. Area Under Error-Retention Curve (AURC)
  3. Orthogonal Dissociation Experiment:
     - Synthetic Poisson noise sweep -> Aleatoric uncertainty scales
     - Out-of-Distribution (OOD) shift -> Epistemic uncertainty spikes
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset, MVTecTextureDataset


def calculate_ece(errors: torch.Tensor, uncertainties: torch.Tensor, num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) between absolute error |x - gamma|
    and predicted standard deviation sigma = sqrt(aleatoric + epistemic).
    """
    sigma = torch.sqrt(torch.clamp(uncertainties, min=1e-8))
    # Standardized residual
    z_scores = errors / sigma
    
    # Check theoretical normal coverage for quantiles
    quantiles = torch.linspace(0.1, 0.9, num_bins)
    ece = 0.0
    for q in quantiles:
        # Theoretical threshold for standard normal
        # e.g., for 68% interval z ≈ 1.0, 95% interval z ≈ 1.96
        # Approximation: norm.ppf(0.5 + q/2)
        z_thresh = torch.tensor(1.4142) * torch.erfinv(q)
        empirical_cov = (z_scores <= z_thresh).float().mean()
        ece += torch.abs(empirical_cov - q).item()
        
    return ece / num_bins


def calculate_aurc(errors: torch.Tensor, uncertainties: torch.Tensor, num_steps: int = 20) -> float:
    """
    Calculates Area Under Error-Retention Curve (AURC).
    Sorts pixels by uncertainty descending, progressively retains fraction (1 - r),
    and measures remaining MSE. Lower AURC indicates better uncertainty ranking.
    """
    flat_err = errors.flatten()
    flat_unc = uncertainties.flatten()
    
    # Sort by uncertainty ascending (keep lowest uncertainty first)
    sorted_indices = torch.argsort(flat_unc)
    sorted_err = flat_err[sorted_indices]
    
    retention_fractions = torch.linspace(0.05, 1.0, num_steps)
    retained_mses = []
    
    N = len(sorted_err)
    for frac in retention_fractions:
        k = int(frac * N)
        subset_mse = (sorted_err[:k] ** 2).mean().item()
        retained_mses.append(subset_mse)
        
    aurc = float(np.trapz(retained_mses, retention_fractions.numpy()))
    return aurc


def run_uncertainty_calibration(device="cuda"):
    print(f"\n{'='*75}\n[PILLAR 4] EVIDENTIAL UNCERTAINTY CALIBRATION & DISSOCIATION\n{'='*75}")
    cfg = get_config()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 1. In-Distribution: TILDA 400
    test_ds = TILDADataset("data/benchmarks/tilda", split="test")
    sub_ds = torch.utils.data.Subset(test_ds, range(min(200, len(test_ds))))
    loader = DataLoader(sub_ds, batch_size=16, shuffle=False)
    
    # 2. Out-of-Distribution: MVTec Tile
    try:
        ood_ds = MVTecTextureDataset("data/benchmarks/mvtec", category="tile", split="test")
        ood_loader = DataLoader(ood_ds, batch_size=16, shuffle=False)
    except Exception:
        ood_loader = None
        
    forward_model = GhostImagingForwardModel(
        image_size=64,
        num_patterns=512,
        mean_photon_count=10.0,
        pattern_type="learned"
    ).to(dev)
    
    model = ICPINOT(
        num_patterns=512,
        embed_dim=cfg.model.embed_dim,
        img_size=64
    ).to(dev)
    
    ckpt_path = "checkpoints/ic_pinot_tilda_400_seed42.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    
    all_errors, all_uncertainties = [], []
    in_dist_aleatoric, in_dist_epistemic = [], []
    
    with torch.no_grad():
        for imgs, _ in loader:
            B = imgs.shape[0]
            imgs = imgs.to(dev)
            z = torch.zeros(B, 11, device=dev)
            z[:, 3] = 1.0
            
            noisy_meas, _ = forward_model.forward_sample(imgs, z)
            nig_dict, _ = model(noisy_meas, z, return_evidential=True)
            
            recon = nig_dict["gamma"]
            err = (imgs - recon).abs()
            unc = nig_dict["total_uncertainty"]
            
            all_errors.append(err.cpu())
            all_uncertainties.append(unc.cpu())
            in_dist_aleatoric.append(nig_dict["aleatoric"].mean().item())
            in_dist_epistemic.append(nig_dict["epistemic"].mean().item())
            
    all_errors = torch.cat(all_errors, dim=0)
    all_uncertainties = torch.cat(all_uncertainties, dim=0)
    
    ece = calculate_ece(all_errors, all_uncertainties)
    aurc = calculate_aurc(all_errors, all_uncertainties)
    
    print(f"\n--- In-Distribution Calibration Metrics ---")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    print(f"  Area Under Error-Retention Curve (AURC): {aurc:.5f}")
    print(f"  In-Distribution Mean Aleatoric: {np.mean(in_dist_aleatoric):.4f}")
    print(f"  In-Distribution Mean Epistemic: {np.mean(in_dist_epistemic):.4f}")
    
    # 3. Orthogonal Dissociation: OOD Defect Check
    ood_aleatoric, ood_epistemic = [], []
    if ood_loader is not None:
        with torch.no_grad():
            for imgs, _ in ood_loader:
                B = imgs.shape[0]
                imgs = imgs.to(dev)
                z = torch.zeros(B, 11, device=dev)
                z[:, 3] = 1.0
                noisy_meas, _ = forward_model.forward_sample(imgs, z)
                nig_dict, _ = model(noisy_meas, z, return_evidential=True)
                ood_aleatoric.append(nig_dict["aleatoric"].mean().item())
                ood_epistemic.append(nig_dict["epistemic"].mean().item())
                
        print(f"\n--- Out-of-Distribution Dissociation Metrics ---")
        print(f"  OOD Mean Aleatoric: {np.mean(ood_aleatoric):.4f} (Ratio vs In-Dist: {np.mean(ood_aleatoric)/max(1e-6, np.mean(in_dist_aleatoric)):.2f}x)")
        print(f"  OOD Mean Epistemic: {np.mean(ood_epistemic):.4f} (Spike vs In-Dist: {np.mean(ood_epistemic)/max(1e-6, np.mean(in_dist_epistemic)):.2f}x)")
        
    results = {
        "ece": float(ece),
        "aurc": float(aurc),
        "in_distribution": {
            "aleatoric": float(np.mean(in_dist_aleatoric)),
            "epistemic": float(np.mean(in_dist_epistemic))
        },
        "out_of_distribution": {
            "aleatoric": float(np.mean(ood_aleatoric)) if ood_aleatoric else 0.0,
            "epistemic": float(np.mean(ood_epistemic)) if ood_epistemic else 0.0
        }
    }
    
    os.makedirs("results", exist_ok=True)
    out_file = "results/uncertainty_calibration.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nUncertainty calibration results saved to {out_file}")
    return results


if __name__ == "__main__":
    run_uncertainty_calibration()
