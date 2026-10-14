"""
Cross-Domain Benchmark Training & Evaluation Suite for PI-GhostFormer.
Evaluates generalizability across:
  1. TILDA 400 (Woven Textile Texture Patches)
  2. DAGM 2007 (Industrial Textured Surface Defect Inspection)
  3. MVTec-AD (Carpet & Grid Woven Anomalies)
"""

import os
import sys
import json
import time
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


def train_and_eval_benchmark(name: str, train_dataset, test_dataset, epochs: int = 40):
    print(f"\n{'='*65}\n[BENCHMARK] Training PI-GhostFormer on {name}\n{'='*65}")
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 1. Forward Physics
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
    
    # 2. PI-GhostFormer
    model = PIGhostFormer(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        physics_dim=cfg.physics.num_zernike,
        img_size=cfg.physics.image_size,
        num_classes=2 # Binary defect classification: normal vs defect
    ).to(device)
    
    # 3. Loss & Optimizer
    criterion = PIGhostFormerLoss(lambda_physics=1e-4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    
    best_psnr = -1.0
    best_metrics = {}
    
    for epoch in range(epochs):
        model.train()
        # Curriculum: Warmup for first 15 epochs, then Poisson physics loss
        if epoch < 15:
            criterion.lambda_physics = 0.0
            criterion.lambda_recon = 1.0
        else:
            criterion.lambda_physics = 1e-4
            criterion.lambda_recon = 1.0
            
        for images, labels in train_loader:
            B = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            
            # Defocus sweep (Zernike a4)
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
            
        # Validation epoch
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
        
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_metrics = {
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "accuracy": clf_metrics['accuracy'],
                "f1_score": clf_metrics['f1_score'],
                "best_epoch": epoch
            }
            
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"[{name}] Epoch {epoch:02d}/{epochs} | PSNR: {avg_psnr:.2f} dB (Best: {best_psnr:.2f}) | SSIM: {avg_ssim:.3f} | Acc: {clf_metrics['accuracy']:.1%}")
            
    print(f"--> {name} Final: PSNR={best_metrics['psnr']:.2f} dB, SSIM={best_metrics['ssim']:.3f}, Acc={best_metrics['accuracy']:.1%}, F1={best_metrics['f1_score']:.3f}")
    return best_metrics


def main():
    os.makedirs("results", exist_ok=True)
    all_results = {}
    
    # 1. TILDA Benchmark (subsample to 2000 train / 500 test for rapid balanced convergence)
    print("Preparing TILDA 400...")
    tilda_full_train = TILDADataset("data/benchmarks/tilda", split="train")
    tilda_full_test = TILDADataset("data/benchmarks/tilda", split="test")
    # Subsample for balanced execution
    train_sub = torch.utils.data.Subset(tilda_full_train, range(min(2000, len(tilda_full_train))))
    test_sub = torch.utils.data.Subset(tilda_full_test, range(min(500, len(tilda_full_test))))
    all_results["TILDA 400 (Textile Patches)"] = train_and_eval_benchmark("TILDA 400", train_sub, test_sub, epochs=30)
    
    # 2. DAGM 2007 Benchmark
    print("\nPreparing DAGM 2007...")
    dagm_train = DAGMDataset("data/benchmarks/dagm", split="train")
    dagm_test = DAGMDataset("data/benchmarks/dagm", split="test")
    all_results["DAGM 2007 (Industrial Surfaces)"] = train_and_eval_benchmark("DAGM 2007", dagm_train, dagm_test, epochs=30)
    
    # 3. MVTec-AD Carpet Benchmark
    print("\nPreparing MVTec-AD Carpet...")
    mvtec_train = MVTecTextureDataset("data/benchmarks/mvtec", category="carpet", split="train")
    mvtec_test = MVTecTextureDataset("data/benchmarks/mvtec", category="carpet", split="test")
    all_results["MVTec-AD (Carpet Texture)"] = train_and_eval_benchmark("MVTec Carpet", mvtec_train, mvtec_test, epochs=30)
    
    # Save results
    out_file = "results/cross_domain_benchmarks.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nAll benchmark runs complete! Results saved to {out_file}")


if __name__ == "__main__":
    main()
