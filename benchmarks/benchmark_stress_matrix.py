import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

"""
Pillar 3: 3D Physical Stress Matrix for Single-Photon Computational Ghost Imaging.
Systematic validation across three critical physical axes:
  1. Photon Count Flux: λ ∈ {1.0, 5.0, 10.0, 50.0, 100.0, 1000.0} photons/pixel
  2. Defocus Wavefront Aberration: a_4 ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0} waves
  3. Sub-Nyquist Compression Ratio: M/N² ∈ {1.0%, 3.0%, 6.25%, 12.5%, 25.0%}
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
from data.benchmark_loader import TILDADataset
from utils.metrics import calculate_psnr, calculate_ssim


def evaluate_stress_matrix(device="cuda"):
    print(f"\n{'='*75}\n[PILLAR 3] 3D PHYSICAL STRESS MATRIX EXECUTION\n{'='*75}")
    cfg = get_config()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Load test set (TILDA 400 patches)
    test_ds = TILDADataset("data/benchmarks/tilda", split="test")
    sub_ds = torch.utils.data.Subset(test_ds, range(min(200, len(test_ds))))
    loader = DataLoader(sub_ds, batch_size=16, shuffle=False)
    
    # Load IC-PINOT model
    model = ICPINOT(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        img_size=cfg.physics.image_size
    ).to(dev)
    
    ckpt_path = "checkpoints/ic_pinot_tilda_400_seed42.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading IC-PINOT checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    
    results = {
        "photon_flux_sweep": {},
        "defocus_sweep": {},
        "sampling_ratio_sweep": {}
    }
    
    # -------------------------------------------------------------
    # Axis 1: Photon Flux Sweep (λ ∈ {1, 5, 10, 50, 100, 1000})
    # -------------------------------------------------------------
    photon_levels = [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
    print("\n--- Running Axis 1: Photon Flux Sweep ---")
    for flux in photon_levels:
        fwd = GhostImagingForwardModel(
            image_size=64,
            num_patterns=512,
            mean_photon_count=flux,
            pattern_type="learned"
        ).to(dev)
        
        psnrs, ssims = [], []
        with torch.no_grad():
            for imgs, _ in loader:
                B = imgs.shape[0]
                imgs = imgs.to(dev)
                z = torch.zeros(B, 11, device=dev)
                z[:, 3] = 1.0 # 1 wave defocus
                noisy_meas, _ = fwd.forward_sample(imgs, z)
                recon, _ = model(noisy_meas, z)
                psnrs.append(calculate_psnr(recon, imgs))
                ssims.append(calculate_ssim(recon, imgs))
                
        results["photon_flux_sweep"][str(flux)] = {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims))
        }
        print(f"  Photon Flux = {flux:>6.1f} | PSNR: {np.mean(psnrs):.2f} dB | SSIM: {np.mean(ssims):.3f}")
        
    # -------------------------------------------------------------
    # Axis 2: Defocus Wavefront Aberration Sweep (a_4 ∈ [0.0, 3.0])
    # -------------------------------------------------------------
    defocus_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    fwd_standard = GhostImagingForwardModel(
        image_size=64,
        num_patterns=512,
        mean_photon_count=10.0,
        pattern_type="learned"
    ).to(dev)
    
    print("\n--- Running Axis 2: Defocus Wavefront Aberration Sweep ---")
    for a4 in defocus_levels:
        psnrs, ssims = [], []
        with torch.no_grad():
            for imgs, _ in loader:
                B = imgs.shape[0]
                imgs = imgs.to(dev)
                z = torch.zeros(B, 11, device=dev)
                z[:, 3] = a4
                noisy_meas, _ = fwd_standard.forward_sample(imgs, z)
                recon, _ = model(noisy_meas, z)
                psnrs.append(calculate_psnr(recon, imgs))
                ssims.append(calculate_ssim(recon, imgs))
                
        results["defocus_sweep"][str(a4)] = {
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims))
        }
        print(f"  Defocus a_4 = {a4:.1f} waves | PSNR: {np.mean(psnrs):.2f} dB | SSIM: {np.mean(ssims):.3f}")
        
    # -------------------------------------------------------------
    # Axis 3: Sub-Nyquist Sampling Ratio Sweep (M/N² ∈ [1%, 25%])
    # -------------------------------------------------------------
    compression_levels = [
        (41, "1.0%"),
        (123, "3.0%"),
        (256, "6.25%"),
        (512, "12.5%"),
        (1024, "25.0%")
    ]
    print("\n--- Running Axis 3: Sub-Nyquist Compression Ratio Sweep ---")
    for m_count, label in compression_levels:
        fwd_comp = GhostImagingForwardModel(
            image_size=64,
            num_patterns=m_count,
            mean_photon_count=10.0,
            pattern_type="learned"
        ).to(dev)
        
        comp_model = ICPINOT(
            num_patterns=m_count,
            embed_dim=cfg.model.embed_dim,
            img_size=64
        ).to(dev)
        comp_model.eval()
        
        psnrs, ssims = [], []
        with torch.no_grad():
            for imgs, _ in loader:
                B = imgs.shape[0]
                imgs = imgs.to(dev)
                z = torch.zeros(B, 11, device=dev)
                z[:, 3] = 1.0
                noisy_meas, _ = fwd_comp.forward_sample(imgs, z)
                recon, _ = comp_model(noisy_meas, z)
                psnrs.append(calculate_psnr(recon, imgs))
                ssims.append(calculate_ssim(recon, imgs))
                
        results["sampling_ratio_sweep"][label] = {
            "num_patterns": m_count,
            "psnr": float(np.mean(psnrs)),
            "ssim": float(np.mean(ssims))
        }
        print(f"  Sampling Ratio = {label:<6} (M={m_count:<4}) | PSNR: {np.mean(psnrs):.2f} dB | SSIM: {np.mean(ssims):.3f}")
        
    os.makedirs("results", exist_ok=True)
    out_file = "results/physical_stress_matrix.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nPhysical stress matrix results saved to {out_file}")
    return results


if __name__ == "__main__":
    evaluate_stress_matrix()
