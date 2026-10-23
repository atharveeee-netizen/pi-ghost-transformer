"""
Standardized Reconstruction Benchmark Runner across TILDA, DAGM, MVTec, and AITEX.
Computes PSNR, SSIM, L1 error, and throughput (ms/frame) under strict equal photon budget.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset
from data.aitex_loader import AITEXDataset
from utils.metrics import calculate_psnr, calculate_ssim


def evaluate_dataset(
    name: str,
    dataset,
    model: ICPINOT,
    forward_model: GhostImagingForwardModel,
    device: torch.device,
    max_samples: int = 100
) -> dict:
    sub_ds = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(sub_ds, batch_size=16, shuffle=False)
    
    psnrs, ssims, l1s = [], [], []
    latencies = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            B = images.shape[0]
            
            # Forward SPI measurement under nominal state
            t0 = time.perf_counter()
            y, _ = forward_model.forward_sample(images, torch.zeros(B, 11, device=device))
            
            # Blind Reconstruction
            recon, _ = model(y, optical_params=None)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            
            latencies.append((t1 - t0) * 1000.0 / B) # ms per image
            
            psnrs.append(calculate_psnr(images, recon))
            ssims.append(calculate_ssim(images, recon))
            l1s.append(float(torch.abs(images - recon).mean().item()))
            
    res = {
        "dataset": name,
        "samples_evaluated": len(sub_ds),
        "psnr_mean": float(np.mean(psnrs)),
        "psnr_std": float(np.std(psnrs)),
        "ssim_mean": float(np.mean(ssims)),
        "ssim_std": float(np.std(ssims)),
        "l1_mean": float(np.mean(l1s)),
        "l1_std": float(np.std(l1s)),
        "latency_ms_per_frame": float(np.mean(latencies))
    }
    print(f"{name:<20} | PSNR: {res['psnr_mean']:5.2f} +/- {res['psnr_std']:4.2f} dB | SSIM: {res['ssim_mean']:.4f} | L1: {res['l1_mean']:.4f} | {res['latency_ms_per_frame']:5.2f} ms/frame")
    return res


def main():
    parser = argparse.ArgumentParser(description="Run Standardized Reconstruction Benchmark")
    parser.add_argument("--patterns", default="checkpoints/trained_patterns.pt")
    parser.add_argument("--output", default="results/reconstruction_benchmark_results.json")
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patterns = torch.load(args.patterns, map_location=device)
    forward_model = GhostImagingForwardModel(image_size=64, num_patterns=512, pattern_type="learned").to(device)
    forward_model.measurement_matrix.H_raw.data = torch.logit(patterns.clamp(1e-5, 1 - 1e-5))

    datasets = [
        ("TILDA 400", TILDADataset("data/benchmarks/tilda", split="test"), "checkpoints/ic_pinot_tilda_400_seed42.pt"),
        ("DAGM 2007", DAGMDataset("data/benchmarks/dagm", split="test"), "checkpoints/ic_pinot_dagm_2007_seed42.pt"),
        ("MVTec Carpet", MVTecTextureDataset("data/benchmarks/mvtec", category="carpet", split="test"), "checkpoints/ic_pinot_mvtec-carpet_seed42.pt"),
        ("AITEX", AITEXDataset("data/aitex"), "checkpoints/ic_pinot_tilda_400_seed42.pt")
    ]

    print("\n" + "=" * 85)
    print("STANDARDIZED RECONSTRUCTION BENCHMARK (N_total = 5,120 photons, 512 patterns)")
    print("=" * 85)

    all_results = {}
    for name, ds, ckpt_path in datasets:
        model = ICPINOT(num_patterns=512, img_size=64).to(device)
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()
        all_results[name] = evaluate_dataset(name, ds, model, forward_model, device, max_samples=args.max_samples)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)
    print("=" * 85)
    print(f"Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
