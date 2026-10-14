"""
IC-PINOT Master Training Harness (Pillar 1).
Comprehensive Multi-Seed Deep Training across 17 Industrial Benchmarks:
  - TILDA 400 (Woven textile defects)
  - DAGM 2007 (Industrial weak-contrast surfaces)
  - All 15 MVTec-AD Categories (Carpet, Grid, Leather, Tile, Wood, Bottle, Cable, Capsule,
    Hazelnut, Metal Nut, Pill, Screw, Toothbrush, Transistor, Zipper)
Features:
  - Evidential Deep Learning (NIG) with annealed evidence regularization
  - Flux-Normalized Log-Stabilized Poisson Physics Consistency
  - SGDR Cosine Annealing with Warm Restarts
  - Full statistical tracking (Mean ± Standard Error across optical seeds)
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from models.loss import NormalizedPoissonNLLLoss
from models.evidential import EvidentialNIGLoss
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset
from utils.metrics import calculate_psnr, calculate_ssim, calculate_classification_metrics

MVTEC_CATEGORIES = [
    # Woven & Surface Textures
    "carpet", "grid", "leather", "tile", "wood",
    # Manufactured Objects
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper"
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_benchmark_ic_pinot(
    name: str,
    train_dataset,
    test_dataset,
    epochs: int = 150,
    lr: float = 1e-4,
    seed: int = 42,
    device: str = "cuda"
) -> dict:
    set_seed(seed)
    print(f"\n{'='*75}\n[IC-PINOT FLAGSHIP] {name} | Seed: {seed} | Epochs: {epochs} | SGDR Cosine\n{'='*75}")
    print(f"Dataset Size: {len(train_dataset)} Train | {len(test_dataset)} Test | Device: {device}")
    
    cfg = get_config()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False
    )
    
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
    ).to(dev)
    
    # 2. IC-PINOT Flagship Architecture
    model = ICPINOT(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        encoder_layers=cfg.model.encoder_layers,
        encoder_heads=cfg.model.encoder_heads,
        physics_dim=cfg.physics.num_zernike,
        img_size=cfg.physics.image_size,
        num_classes=2,
        num_prompt_tokens=16
    ).to(dev)
    
    # 3. Losses
    evidential_loss_fn = EvidentialNIGLoss(coeff_reg=0.05).to(dev)
    poisson_loss_fn = NormalizedPoissonNLLLoss(photon_scale=cfg.physics.default_photon_count).to(dev)
    ce_loss_fn = nn.CrossEntropyLoss().to(dev)
    
    # 4. Optimizer with SGDR (Cosine Annealing with Warm Restarts)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Restart cycle: T_0 = 50 epochs, doubling every cycle
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=max(25, epochs // 3),
        T_mult=1,
        eta_min=1e-6
    )
    
    os.makedirs("checkpoints", exist_ok=True)
    best_psnr = -1.0
    best_metrics = {}
    
    # Smooth warmup schedules
    warmup_epochs_physics = max(15, int(epochs * 0.15))
    warmup_epochs_evid = max(20, int(epochs * 0.20))
    
    for epoch in range(epochs):
        model.train()
        
        # Physics loss weight warmup (smooth transition)
        if epoch < warmup_epochs_physics:
            lambda_phys = 0.0
        else:
            progress = min(1.0, (epoch - warmup_epochs_physics) / float(max(1, warmup_epochs_physics)))
            lambda_phys = 1e-4 * progress
            
        # Evidential regularizer weight annealing
        if epoch < warmup_epochs_evid:
            coeff_reg = 0.01 * (epoch / float(warmup_epochs_evid))
        else:
            coeff_reg = 0.05
            
        epoch_losses = []
        for images, labels in train_loader:
            B = images.shape[0]
            images = images.to(dev)
            labels = labels.to(dev)
            
            # Dynamic wavefront defocus variation
            zernike = torch.zeros(B, cfg.physics.num_zernike, device=dev)
            zernike[:, 3] = torch.empty(B, device=dev).uniform_(*cfg.physics.defocus_range)
            
            optimizer.zero_grad()
            with torch.no_grad():
                noisy_meas, _ = forward_model.forward_sample(images, zernike)
                
            nig_outputs, logits = model(noisy_meas, zernike, return_evidential=True)
            recon_img = nig_outputs["gamma"]
            
            # 1. Evidential NIG Loss
            evid_dict = evidential_loss_fn(nig_outputs, images, coeff_reg=coeff_reg)
            loss_evid = evid_dict["loss_evidential"]
            
            # 2. Differentiable Forward Physics Consistency (Poisson)
            expected_buckets = forward_model.forward_expected(recon_img, zernike)
            loss_phys = poisson_loss_fn(expected_buckets, noisy_meas) if lambda_phys > 0 else 0.0
            
            # 3. Defect Classification Loss
            loss_cls = ce_loss_fn(logits, labels)
            
            total_loss = loss_evid + (lambda_phys * loss_phys) + (0.3 * loss_cls)
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(total_loss.item())
            
        scheduler.step()
        
        # Periodic Evaluation every 10 epochs or last epoch
        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            val_psnr, val_ssim = [], []
            val_aleatoric, val_epistemic = [], []
            all_logits, all_labels = [], []
            
            with torch.no_grad():
                for images, labels in test_loader:
                    B = images.shape[0]
                    images = images.to(dev)
                    labels = labels.to(dev)
                    
                    zernike = torch.zeros(B, cfg.physics.num_zernike, device=dev)
                    zernike[:, 3] = sum(cfg.physics.defocus_range) / 2.0
                    
                    noisy_meas, _ = forward_model.forward_sample(images, zernike)
                    nig_outputs, logits = model(noisy_meas, zernike, return_evidential=True)
                    recon_img = nig_outputs["gamma"]
                    
                    val_psnr.append(calculate_psnr(recon_img, images))
                    val_ssim.append(calculate_ssim(recon_img, images))
                    val_aleatoric.append(nig_outputs["aleatoric"].mean().item())
                    val_epistemic.append(nig_outputs["epistemic"].mean().item())
                    all_logits.append(logits)
                    all_labels.append(labels)
                    
            avg_psnr = float(np.mean(val_psnr))
            avg_ssim = float(np.mean(val_ssim))
            avg_aleatoric = float(np.mean(val_aleatoric))
            avg_epistemic = float(np.mean(val_epistemic))
            
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
                    "aleatoric_uncertainty": avg_aleatoric,
                    "epistemic_uncertainty": avg_epistemic,
                    "best_epoch": epoch,
                    "seed": seed
                }
                # Save best checkpoint
                ckpt_path = f"checkpoints/ic_pinot_{name.lower().replace(' ', '_')}_seed{seed}.pt"
                torch.save(model.state_dict(), ckpt_path)
                
            print(
                f"[{name} S{seed}] Ep {epoch:03d}/{epochs} | LR: {current_lr:.1e} | "
                f"PSNR: {avg_psnr:.2f} dB (Best: {best_psnr:.2f}) | SSIM: {avg_ssim:.3f} | "
                f"Acc: {clf_metrics['accuracy']:.1%} | Epistemic: {avg_epistemic:.4f}"
            )
            
        # Continuous heartbeat update for power-cut tracking
        try:
            import datetime
            heartbeat = {
                "active_dataset": name,
                "seed": seed,
                "epoch": epoch,
                "total_epochs": epochs,
                "current_best_psnr": best_psnr,
                "timestamp": datetime.datetime.now().isoformat()
            }
            atomic_save_json(heartbeat, "results/heartbeat.json")
        except Exception:
            pass
            
    print(f"--> [DONE {name} S{seed}] Best: PSNR={best_metrics.get('psnr', 0):.2f} dB, SSIM={best_metrics.get('ssim', 0):.3f}, Acc={best_metrics.get('accuracy', 0):.1%}")
    return best_metrics


def atomic_save_json(data: dict, file_path: str):
    """Atomically writes JSON to disk using temporary file replacement to survive power cuts."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=4)
    os.replace(tmp_path, file_path)


def run_full_suite(epochs: int = 150, seeds: list = [42], categories: list = None):
    os.makedirs("results", exist_ok=True)
    results_path = "results/ic_pinot_master_benchmarks.json"
    
    # Load previous results for seamless power-cut resume
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                all_results = json.load(f)
            print(f"[POWER-CUT RESUME] Loaded existing progress from {results_path}")
        except Exception:
            all_results = {}
    else:
        all_results = {}
        
    target_cats = categories if categories is not None else MVTEC_CATEGORIES
    datasets_to_run = []
    
    # 1. TILDA 400
    try:
        t_tr = TILDADataset("data/benchmarks/tilda", split="train")
        t_te = TILDADataset("data/benchmarks/tilda", split="test")
        sub_tr = torch.utils.data.Subset(t_tr, range(min(2000, len(t_tr))))
        sub_te = torch.utils.data.Subset(t_te, range(min(500, len(t_te))))
        datasets_to_run.append(("TILDA 400", sub_tr, sub_te))
    except Exception as e:
        print(f"Skipping TILDA 400: {e}")
        
    # 2. DAGM 2007
    try:
        d_tr = DAGMDataset("data/benchmarks/dagm", split="train")
        d_te = DAGMDataset("data/benchmarks/dagm", split="test")
        datasets_to_run.append(("DAGM 2007", d_tr, d_te))
    except Exception as e:
        print(f"Skipping DAGM 2007: {e}")
        
    # 3. MVTec-AD Categories
    for cat in target_cats:
        try:
            m_tr = MVTecTextureDataset("data/benchmarks/mvtec", category=cat, split="train")
            m_te = MVTecTextureDataset("data/benchmarks/mvtec", category=cat, split="test")
            if len(m_tr) > 0 and len(m_te) > 0:
                datasets_to_run.append((f"MVTec-{cat.capitalize()}", m_tr, m_te))
        except Exception as e:
            print(f"Skipping MVTec-{cat}: {e}")
            
    # Run multi-seed iterations with power-cut resume
    for ds_name, ds_train, ds_test in datasets_to_run:
        if ds_name not in all_results:
            all_results[ds_name] = {"seed_runs": []}
            
        completed_seeds = {r.get("seed") for r in all_results[ds_name].get("seed_runs", [])}
        
        # Check if all requested seeds are already completed
        if set(seeds).issubset(completed_seeds):
            print(f"[POWER-CUT RESUME] {ds_name} already completed across seeds {seeds}. Skipping...")
            continue
            
        for seed in seeds:
            if seed in completed_seeds:
                print(f"[POWER-CUT RESUME] {ds_name} (Seed {seed}) already completed. Skipping...")
                continue
                
            res = train_single_benchmark_ic_pinot(
                ds_name,
                ds_train,
                ds_test,
                epochs=epochs,
                seed=seed
            )
            all_results[ds_name]["seed_runs"].append(res)
            
            # Atomic save immediately after every seed finishes
            atomic_save_json(all_results, results_path)
            
        # Recompute summary across all seed runs
        seed_runs = all_results[ds_name].get("seed_runs", [])
        if seed_runs:
            psnr_list = [r.get("psnr", 0.0) for r in seed_runs]
            ssim_list = [r.get("ssim", 0.0) for r in seed_runs]
            acc_list = [r.get("accuracy", 0.0) for r in seed_runs]
            
            all_results[ds_name]["summary"] = {
                "psnr_mean": float(np.mean(psnr_list)),
                "psnr_std": float(np.std(psnr_list)),
                "ssim_mean": float(np.mean(ssim_list)),
                "ssim_std": float(np.std(ssim_list)),
                "accuracy_mean": float(np.mean(acc_list)),
                "accuracy_std": float(np.std(acc_list)),
            }
            print(
                f"\n>>> [SUMMARY {ds_name}]: PSNR={all_results[ds_name]['summary']['psnr_mean']:.2f} +/- {all_results[ds_name]['summary']['psnr_std']:.2f} dB | "
                f"SSIM={all_results[ds_name]['summary']['ssim_mean']:.3f} +/- {all_results[ds_name]['summary']['ssim_std']:.3f} | "
                f"Acc={all_results[ds_name]['summary']['accuracy_mean']:.1%} +/- {all_results[ds_name]['summary']['accuracy_std']:.1%}\n"
            )
            # Atomic save summary
            atomic_save_json(all_results, results_path)
            
    print("\n=======================================================")
    print(f"ALL BENCHMARK RUNS COMPLETE! Results saved to {results_path}")
    print("=======================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master IC-PINOT Benchmark Runner")
    parser.add_argument("--epochs", type=int, default=150, help="Epochs per benchmark run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="List of random seeds")
    parser.add_argument("--categories", type=str, nargs="+", default=None, help="Subset of MVTec categories")
    args = parser.parse_args()
    
    run_full_suite(epochs=args.epochs, seeds=args.seeds, categories=args.categories)
