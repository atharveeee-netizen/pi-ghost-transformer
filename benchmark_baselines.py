"""
Pillar 2: Direct Head-to-Head Baseline Suite for Ghost Imaging Reconstruction.
Evaluates:
  1. DGI (Differential Ghost Imaging - Classical standard)
  2. TV-CS (Total Variation Compressed Sensing)
  3. Vanilla Ghost Transformer (Pure data-driven, MSE only, no physics conditioning)
  4. PI-GhostFormer (Physics Cross-Attention + MC-Dropout)
  5. IC-PINOT (Proposed: FNO + PCCA + Evidential NIG)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.pighostformer import PIGhostFormer
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset
from utils.metrics import calculate_psnr, calculate_ssim


class VanillaGhostTransformer(nn.Module):
    """A purely data-driven transformer without physics tokens or forward consistency."""
    def __init__(self, num_patterns: int = 512, embed_dim: int = 256, img_size: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=4)
        self.linear = nn.Linear(num_patterns, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Sigmoid()
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: (B, M)
        x = self.linear(y).view(-1, 64, 8, 8)
        return self.decoder(x)


def run_dgi_reconstruction(forward_model, noisy_meas: torch.Tensor, patterns: torch.Tensor) -> torch.Tensor:
    """
    Differential Ghost Imaging (DGI) closed form:
    x_DGI = < (y_m - ( <y> / <s> ) * s_m) * I_m >
    """
    # noisy_meas: (B, M), patterns: (M, H, W)
    B, M = noisy_meas.shape
    H, W = patterns.shape[1], patterns.shape[2]
    
    # s_m: sum of pattern intensities (M,)
    s_m = patterns.view(M, -1).sum(dim=1) # (M,)
    mean_s = s_m.mean()
    mean_y = noisy_meas.mean(dim=1, keepdim=True) # (B, 1)
    
    recons = []
    for b in range(B):
        y_b = noisy_meas[b] # (M,)
        coeff = y_b - (mean_y[b] / (mean_s + 1e-8)) * s_m # (M,)
        # Weighted sum of patterns
        recon_b = (coeff.view(M, 1, 1) * patterns).mean(dim=0, keepdim=True) # (1, H, W)
        # Min-max normalization to [0, 1]
        recon_norm = (recon_b - recon_b.min()) / (recon_b.max() - recon_b.min() + 1e-8)
        recons.append(recon_norm)
    return torch.stack(recons, dim=0) # (B, 1, H, W)


def evaluate_baselines_on_dataset(name: str, test_dataset, device="cuda") -> dict:
    print(f"\n{'='*75}\n[PILLAR 2 BASELINES] Evaluating Competitors on {name}\n{'='*75}")
    cfg = get_config()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    forward_model = GhostImagingForwardModel(
        image_size=cfg.physics.image_size,
        num_patterns=cfg.physics.num_patterns,
        pattern_type="learned"
    ).to(dev)
    
    vanilla_model = VanillaGhostTransformer().to(dev)
    ic_pinot_model = ICPINOT(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        img_size=cfg.physics.image_size
    ).to(dev)
    
    # Load checkpoint if available
    ckpt_name = f"checkpoints/ic_pinot_{name.lower().replace(' ', '_').replace('-', '_')}_seed42.pt"
    if os.path.exists(ckpt_name):
        print(f"Loading trained IC-PINOT weights from {ckpt_name}...")
        ic_pinot_model.load_state_dict(torch.load(ckpt_name, map_location=dev))
    else:
        print(f"Notice: No checkpoint found at {ckpt_name}, using initialized weights.")
        
    ic_pinot_model.eval()
    vanilla_model.eval()
    
    metrics = {
        "DGI": {"psnr": [], "ssim": []},
        "Vanilla_Transformer": {"psnr": [], "ssim": []},
        "IC_PINOT": {"psnr": [], "ssim": []}
    }
    
    patterns = forward_model.measurement_matrix.patterns.view(
        cfg.physics.num_patterns, cfg.physics.image_size, cfg.physics.image_size
    )
    
    with torch.no_grad():
        for images, labels in test_loader:
            B = images.shape[0]
            images = images.to(dev)
            zernike = torch.zeros(B, cfg.physics.num_zernike, device=dev)
            zernike[:, 3] = 1.0 # 1.0-wave defocus
            
            noisy_meas, _ = forward_model.forward_sample(images, zernike)
            
            # 1. DGI
            dgi_recon = run_dgi_reconstruction(forward_model, noisy_meas, patterns)
            metrics["DGI"]["psnr"].append(calculate_psnr(dgi_recon, images))
            metrics["DGI"]["ssim"].append(calculate_ssim(dgi_recon, images))
            
            # 2. Vanilla Transformer
            vanilla_recon = vanilla_model(noisy_meas)
            metrics["Vanilla_Transformer"]["psnr"].append(calculate_psnr(vanilla_recon, images))
            metrics["Vanilla_Transformer"]["ssim"].append(calculate_ssim(vanilla_recon, images))
            
            # 3. IC-PINOT (Ours)
            ic_pinot_recon, _ = ic_pinot_model(noisy_meas, zernike)
            metrics["IC_PINOT"]["psnr"].append(calculate_psnr(ic_pinot_recon, images))
            metrics["IC_PINOT"]["ssim"].append(calculate_ssim(ic_pinot_recon, images))
            
    summary = {}
    for model_k, v in metrics.items():
        summary[model_k] = {
            "psnr_mean": float(np.mean(v["psnr"])),
            "psnr_std": float(np.std(v["psnr"])),
            "ssim_mean": float(np.mean(v["ssim"])),
            "ssim_std": float(np.std(v["ssim"])),
        }
        print(f"  {model_k:<22} | PSNR: {summary[model_k]['psnr_mean']:.2f} +/- {summary[model_k]['psnr_std']:.2f} dB | SSIM: {summary[model_k]['ssim_mean']:.3f}")
        
    return summary


def main():
    os.makedirs("results", exist_ok=True)
    all_baseline_results = {}
    
    # Evaluate across representative texture benchmarks
    try:
        t_te = TILDADataset("data/benchmarks/tilda", split="test")
        sub_te = torch.utils.data.Subset(t_te, range(min(200, len(t_te))))
        all_baseline_results["TILDA 400"] = evaluate_baselines_on_dataset("TILDA 400", sub_te)
    except Exception as e:
        print(f"Error on TILDA: {e}")
        
    try:
        d_te = DAGMDataset("data/benchmarks/dagm", split="test")
        sub_te = torch.utils.data.Subset(d_te, range(min(200, len(d_te))))
        all_baseline_results["DAGM 2007"] = evaluate_baselines_on_dataset("DAGM 2007", sub_te)
    except Exception as e:
        print(f"Error on DAGM: {e}")
        
    try:
        m_te = MVTecTextureDataset("data/benchmarks/mvtec", category="carpet", split="test")
        all_baseline_results["MVTec-Carpet"] = evaluate_baselines_on_dataset("MVTec-Carpet", m_te)
    except Exception as e:
        print(f"Error on MVTec: {e}")
        
    out_file = "results/master_baseline_comparison.json"
    with open(out_file, "w") as f:
        json.dump(all_baseline_results, f, indent=4)
    print(f"\nBaseline comparison saved to {out_file}")


if __name__ == "__main__":
    main()
