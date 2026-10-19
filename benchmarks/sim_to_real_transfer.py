import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

"""
Sim-to-Real Transfer & Few-Shot Generalization Experiment.
Validates whether pre-training on physics simulation allows rapid convergence
with only a handful of real factory defect samples (5, 10, 25, 50 samples).
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.pighostformer import PIGhostFormer
from models.loss import PIGhostFormerLoss
from data.synthetic import SyntheticFabricGenerator
from data.aitex_loader import AITEXDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_classification_metrics


def train_few_shot_model(model, forward_model, criterion, train_samples, val_samples, epochs=25, lr=1e-4):
    device = next(model.parameters()).device
    cfg = get_config()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_loader = DataLoader(train_samples, batch_size=min(16, len(train_samples)), shuffle=True)
    val_loader = DataLoader(val_samples, batch_size=min(32, len(val_samples)), shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            B = images.shape[0]
            images, labels = images.to(device), labels.to(device)
            zernike = torch.zeros(B, cfg.physics.num_zernike, device=device)
            zernike[:, 3] = torch.empty(B, device=device).uniform_(*cfg.physics.defocus_range)
            
            optimizer.zero_grad()
            with torch.no_grad():
                noisy_meas, _ = forward_model.forward_sample(images, zernike)
            recon_img, logits = model(noisy_meas, zernike)
            expected_buckets = forward_model.forward_expected(recon_img, zernike)
            
            losses = criterion(recon_img, logits, expected_buckets, noisy_meas, images, labels)
            losses['total_loss'].backward()
            optimizer.step()
            
    # Evaluation
    model.eval()
    val_psnr, val_ssim, all_logits, all_labels = [], [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            B = images.shape[0]
            images, labels = images.to(device), labels.to(device)
            zernike = torch.zeros(B, cfg.physics.num_zernike, device=device)
            zernike[:, 3] = 1.5
            
            noisy_meas, _ = forward_model.forward_sample(images, zernike)
            recon_img, logits = model(noisy_meas, zernike)
            val_psnr.append(calculate_psnr(recon_img, images))
            val_ssim.append(calculate_ssim(recon_img, images))
            all_logits.append(logits)
            all_labels.append(labels)
            
    metrics = {
        "psnr": float(sum(val_psnr) / len(val_psnr)),
        "ssim": float(sum(val_ssim) / len(val_ssim)),
        **calculate_classification_metrics(torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0))
    }
    return metrics


def main():
    os.makedirs("results", exist_ok=True)
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nSIM-TO-REAL TRANSFER LEARNING BENCHMARK ({device})\n{'='*65}")
    
    forward_model = GhostImagingForwardModel(
        image_size=cfg.physics.image_size,
        num_patterns=cfg.physics.num_patterns,
        psf_size=cfg.physics.psf_size,
        num_zernike=cfg.physics.num_zernike
    ).to(device)
    criterion = PIGhostFormerLoss(lambda_physics=1e-4).to(device)
    
    # 1. Pre-train on Synthetic Simulation
    print("Pre-training on 500 Synthetic Physics Fabric samples...")
    gen = SyntheticFabricGenerator(image_size=64)
    syn_imgs, syn_lbls = gen.generate_batch(500, defect_prob=0.3)
    syn_lbls_binary = (syn_lbls > 0).long()
    syn_dataset = TensorDataset(syn_imgs, syn_lbls_binary)
    
    pretrained_model = PIGhostFormer(num_patterns=512, embed_dim=256, physics_dim=11, img_size=64, num_classes=2).to(device)
    _ = train_few_shot_model(pretrained_model, forward_model, criterion, syn_dataset, syn_dataset, epochs=15)
    print("Pre-training complete.")
    
    # 2. Real AITEX Dataset (Split)
    real_train_full = AITEXDataset(cfg.data.aitex_path, image_size=64, split="train")
    real_val = AITEXDataset(cfg.data.aitex_path, image_size=64, split="test")
    
    shot_budgets = [5, 15, 30, 60]
    comparison_results = {}
    
    for shots in shot_budgets:
        print(f"\n--- Evaluating Few-Shot Transfer with N={shots} Real Samples ---")
        few_shot_subset = torch.utils.data.Subset(real_train_full, range(min(shots, len(real_train_full))))
        
        # A. Scratch Baseline
        scratch_model = PIGhostFormer(num_patterns=512, embed_dim=256, physics_dim=11, img_size=64, num_classes=2).to(device)
        scratch_res = train_few_shot_model(scratch_model, forward_model, criterion, few_shot_subset, real_val, epochs=25)
        
        # B. Sim-to-Real Transfer
        import copy
        transfer_model = copy.deepcopy(pretrained_model)
        transfer_res = train_few_shot_model(transfer_model, forward_model, criterion, few_shot_subset, real_val, epochs=25, lr=5e-5)
        
        comparison_results[f"shots_{shots}"] = {
            "scratch": scratch_res,
            "sim_to_real": transfer_res,
            "psnr_gain": float(transfer_res["psnr"] - scratch_res["psnr"]),
            "f1_gain": float(transfer_res["f1_score"] - scratch_res["f1_score"])
        }
        
        print(f"[Shots={shots:02d}] Scratch: PSNR={scratch_res['psnr']:.2f} dB, F1={scratch_res['f1_score']:.2f} | Sim-to-Real: PSNR={transfer_res['psnr']:.2f} dB, F1={transfer_res['f1_score']:.2f} (Gain: +{transfer_res['psnr'] - scratch_res['psnr']:.2f} dB)")
        
    out_file = "results/sim_to_real_transfer_results.json"
    with open(out_file, "w") as f:
        json.dump(comparison_results, f, indent=4)
        
    print(f"\nSim-to-Real Transfer results successfully saved to {out_file}!")


if __name__ == "__main__":
    main()
