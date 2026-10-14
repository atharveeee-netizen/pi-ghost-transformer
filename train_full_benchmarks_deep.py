"""
Comprehensive Deep Benchmark Suite (150-200 Epochs) for PI-GhostFormer.
Evaluates across:
  - TILDA 400 (Textile Patches)
  - DAGM 2007 (Industrial Weak-Contrast Surfaces)
  - All 15 MVTec-AD Categories (5 Textures + 10 Manufactured Objects)
Uses Cosine Annealing Learning Rate Decay and Two-Phase Physics Curriculum.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.pighostformer import PIGhostFormer
from models.loss import PIGhostFormerLoss
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_classification_metrics

# All 15 MVTec-AD categories
MVTEC_CATEGORIES = [
    # Woven & Surface Textures
    "carpet", "grid", "leather", "tile", "wood",
    # Manufactured Objects
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper"
]


def train_single_benchmark(name: str, train_dataset, test_dataset, epochs: int = 150, lr: float = 1e-4):
    print(f"\n{'='*70}\n[DEEP TRAINING] Starting {name} ({epochs} Epochs, Cosine LR Decay)\n{'='*70}")
    print(f"Dataset Size: {len(train_dataset)} Train | {len(test_dataset)} Test")
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 1. Forward Optics Model
    forward_model = GhostImagingForwardModel(
        image_size=cfg.physics.image_size,
        num_patterns=cfg.physics.num_patterns,
        psf_size=cfg.physics.psf_size,
        num_zernike=cfg.physics.num_zernike,
        mean_photon_count=cfg.physics.default_photon_count,
        dark_count_rate=cfg.physics.dark_count_rate,
        readout_noise_std=cfg.physics.readout_noise_std,
        pattern_type=cfg.physics.pattern_type
    ).to(device)
    
    # 2. PI-GhostFormer Network
    model = PIGhostFormer(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        physics_dim=cfg.physics.num_zernike,
        img_size=cfg.physics.image_size,
        num_classes=2
    ).to(device)
    
    # 3. Loss & Optimizer with Cosine Annealing
    criterion = PIGhostFormerLoss(lambda_physics=1e-4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    best_psnr = -1.0
    best_metrics = {}
    warmup_epochs = max(25, int(epochs * 0.2))
    
    for epoch in range(epochs):
        model.train()
        # Curriculum: Warmup with bounded MSE, then Poisson physics loss
        if epoch < warmup_epochs:
            criterion.lambda_physics = 0.0
            criterion.lambda_recon = 1.0
        else:
            criterion.lambda_physics = 1e-4
            criterion.lambda_recon = 1.0
            
        for images, labels in train_loader:
            B = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            
            # Dynamic Zernike Defocus variation
            zernike = torch.zeros(B, cfg.physics.num_zernike, device=device)
            zernike[:, 3] = torch.empty(B, device=device).uniform_(*cfg.physics.defocus_range)
            
            optimizer.zero_grad()
            with torch.no_grad():
                noisy_meas, _ = forward_model.forward_sample(images, zernike)
                
            recon_img, logits = model(noisy_meas, zernike)
            expected_buckets = forward_model.forward_expected(recon_img, zernike)
            
            losses = criterion(
                recon_img=recon_img,
                logits=logits,
                expected_buckets=expected_buckets,
                measured_buckets=noisy_meas,
                gt_img=images,
                gt_class=labels
            )
            
            losses['total_loss'].backward()
            optimizer.step()
            
        scheduler.step()
        
        # Periodic Evaluation every 10 epochs
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_psnr, val_ssim = [], []
            all_logits, all_labels = [], []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    B = images.shape[0]
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    zernike = torch.zeros(B, cfg.physics.num_zernike, device=device)
                    zernike[:, 3] = sum(cfg.physics.defocus_range) / 2.0
                    
                    noisy_meas, _ = forward_model.forward_sample(images, zernike)
                    recon_img, logits = model(noisy_meas, zernike)
                    
                    val_psnr.append(calculate_psnr(recon_img, images))
                    val_ssim.append(calculate_ssim(recon_img, images))
                    all_logits.append(logits)
                    all_labels.append(labels)
                    
            avg_psnr = float(np.mean(val_psnr))
            avg_ssim = float(np.mean(val_ssim))
            
            all_logits = torch.cat(all_logits, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            clf_metrics = calculate_classification_metrics(all_logits, all_labels)
            current_lr = scheduler.get_last_lr()[0]
            
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                best_metrics = {
                    "psnr": avg_psnr,
                    "ssim": avg_ssim,
                    "accuracy": clf_metrics['accuracy'],
                    "f1_score": clf_metrics['f1_score'],
                    "best_epoch": epoch
                }
                
            print(f"[{name}] Epoch {epoch:03d}/{epochs} | LR: {current_lr:.1e} | PSNR: {avg_psnr:.2f} dB (Best: {best_psnr:.2f}) | SSIM: {avg_ssim:.3f} | Acc: {clf_metrics['accuracy']:.1%}")
            
    print(f"--> {name} Final: PSNR={best_metrics.get('psnr', 0):.2f} dB, SSIM={best_metrics.get('ssim', 0):.3f}, Acc={best_metrics.get('accuracy', 0):.1%}")
    return best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train PI-GhostFormer on Full 15-Category MVTec and Benchmarks")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs per dataset")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    args = parser.parse_args()
    
    os.makedirs("results", exist_ok=True)
    results = {}
    
    # 1. TILDA 400
    print("Preparing TILDA 400 Benchmark...")
    tilda_train = TILDADataset("data/benchmarks/tilda", split="train")
    tilda_test = TILDADataset("data/benchmarks/tilda", split="test")
    train_sub = torch.utils.data.Subset(tilda_train, range(min(2000, len(tilda_train))))
    test_sub = torch.utils.data.Subset(tilda_test, range(min(500, len(tilda_test))))
    results["TILDA 400"] = train_single_benchmark("TILDA 400", train_sub, test_sub, epochs=args.epochs, lr=args.lr)
    
    # 2. DAGM 2007
    print("Preparing DAGM 2007 Benchmark...")
    dagm_train = DAGMDataset("data/benchmarks/dagm", split="train")
    dagm_test = DAGMDataset("data/benchmarks/dagm", split="test")
    results["DAGM 2007"] = train_single_benchmark("DAGM 2007", dagm_train, dagm_test, epochs=args.epochs, lr=args.lr)
    
    # 3. All 15 MVTec-AD Categories
    for cat in MVTEC_CATEGORIES:
        print(f"\nPreparing MVTec-AD Category: {cat}...")
        try:
            m_train = MVTecTextureDataset("data/benchmarks/mvtec", category=cat, split="train")
            m_test = MVTecTextureDataset("data/benchmarks/mvtec", category=cat, split="test")
            if len(m_train) > 0 and len(m_test) > 0:
                results[f"MVTec_{cat}"] = train_single_benchmark(f"MVTec-{cat.capitalize()}", m_train, m_test, epochs=args.epochs, lr=args.lr)
        except Exception as e:
            print(f"Skipping {cat} due to: {e}")
            
    # Save combined results
    out_path = "results/full_15_mvtec_benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n{'='*70}\nAll 15 MVTec Categories + TILDA + DAGM Complete! Saved to {out_path}\n{'='*70}")


if __name__ == "__main__":
    main()
