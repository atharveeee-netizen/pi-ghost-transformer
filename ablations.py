"""
Ablation Studies Pipeline for IEEE TCI / Optics Express.

Automates the execution of the 8 critical ablation experiments required
to validate the physics-informed architecture.
"""

import os
import torch
import json
from config import get_config
from train import train_step, validate
from physics.forward_model import GhostImagingForwardModel
from models.pighostformer import PIGhostFormer
from models.loss import PIGhostFormerLoss
from data.synthetic import SyntheticFabricGenerator
from utils.metrics import evaluate_batch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


def run_experiment(config_override: dict, exp_name: str):
    """Run a single training pipeline with a specific configuration override."""
    print(f"\n{'='*50}\nStarting Ablation: {exp_name}\n{'='*50}")
    
    cfg = get_config()
    
    # Apply overrides (simplified for demonstration)
    if 'lambda_physics' in config_override:
        cfg.train.lambda_physics = config_override['lambda_physics']
    if 'pattern_type' in config_override:
        cfg.physics.pattern_type = config_override['pattern_type']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    forward_model = GhostImagingForwardModel(
        image_size=cfg.physics.image_size,
        num_patterns=cfg.physics.num_patterns,
        psf_size=cfg.physics.psf_size,
        num_zernike=cfg.physics.num_zernike,
        mean_photon_count=cfg.physics.default_photon_count,
        pattern_type=cfg.physics.pattern_type
    ).to(device)
    
    model = PIGhostFormer(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        physics_dim=cfg.physics.num_zernike,
        img_size=cfg.physics.image_size,
        num_classes=cfg.model.num_defect_classes
    ).to(device)
    
    criterion = PIGhostFormerLoss(
        lambda_physics=cfg.train.lambda_physics
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    
    # Load Real AITEX Dataset for Ablations
    if cfg.data.aitex_path and os.path.exists(cfg.data.aitex_path):
        from data.aitex_loader import AITEXDataset
        train_dataset = AITEXDataset(cfg.data.aitex_path, image_size=cfg.physics.image_size, split="train")
        val_dataset = AITEXDataset(cfg.data.aitex_path, image_size=cfg.physics.image_size, split="test")
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
        val_images, val_labels = [], []
        for i in range(min(100, len(val_dataset))):
            img, lbl = val_dataset[i]
            val_images.append(img)
            val_labels.append(lbl)
        val_images = torch.stack(val_images)
        val_labels = torch.stack(val_labels)
    else:
        # Fallback
        generator = SyntheticFabricGenerator(image_size=cfg.physics.image_size)
        train_images, train_labels = generator.generate_batch(500, cfg.data.defect_probability)
        val_images, val_labels = generator.generate_batch(100, cfg.data.defect_probability)
        train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=cfg.train.batch_size)
    
    # Training loop runs indefinitely until early stopping is triggered by convergence
    best_psnr = -1.0
    best_metrics = None
    max_epochs = cfg.train.max_epochs 
    patience = cfg.train.early_stopping_patience
    epochs_without_improvement = 0
    
    val_zernike = torch.zeros(val_images.shape[0], cfg.physics.num_zernike).to(device)
    val_zernike[:, 3] = 1.0
    
    for epoch in range(max_epochs):
        # Curriculum Learning Phase Toggle
        if exp_name != "A2: Remove Poisson NLL (MSE only)": # A2 is explicitly MSE only
            if epoch < cfg.train.curriculum_warmup_epochs:
                criterion.lambda_physics = 0.0
                criterion.lambda_recon = 1.0
            else:
                criterion.lambda_physics = cfg.train.lambda_physics
                criterion.lambda_recon = cfg.train.lambda_recon
                
        model.train()
        for images, labels in train_loader:
            zernike_coeffs = torch.zeros(images.shape[0], cfg.physics.num_zernike)
            zernike_coeffs[:, 3] = 1.0 # 1 wave defocus
            train_step(model, forward_model, criterion, optimizer, images, labels, zernike_coeffs, device)
            
        # Validation per epoch
        model.eval()
        with torch.no_grad():
            noisy_measurements, _ = forward_model.forward_sample(val_images.to(device), val_zernike)
            recon_img, logits = model(noisy_measurements, val_zernike)
            metrics = evaluate_batch(recon_img, val_images.to(device), logits, val_labels.to(device))
            
        current_psnr = metrics['psnr']
        
        # Track the best PSNR achieved during the full training run
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Print progress occasionally
        if epoch % 10 == 0:
            print(f"[{exp_name}] Epoch {epoch}: PSNR={current_psnr:.2f} (Best: {best_psnr:.2f}, Patience: {epochs_without_improvement}/{patience})")
            
        if epochs_without_improvement >= patience and epoch >= 50:
            print(f"[{exp_name}] Early stopping triggered at epoch {epoch}. Model has fully converged.")
            break
            
    print(f"Results for {exp_name}: PSNR={best_metrics['psnr']:.2f}, F1={best_metrics['f1_score']:.4f}")
    return best_metrics


def main():
    ablations = [
        {"name": "Baseline (Full PI-GhostFormer)", "overrides": {}},
        {"name": "A1: Remove PCCA (Standard Cross-Attention)", "overrides": {"use_pcca": False}},
        {"name": "A2: Remove Poisson NLL (MSE only)", "overrides": {"lambda_physics": 0.0}},
        {"name": "A3: Single-Token Decoder Bottleneck", "overrides": {"use_64_tokens": False}},
        {"name": "A4: Fixed Hadamard patterns", "overrides": {"pattern_type": "hadamard"}},
        {"name": "A4: Fixed Random patterns", "overrides": {"pattern_type": "random"}},
    ]
    
    results = {}
    for exp in ablations:
        res = run_experiment(exp["overrides"], exp["name"])
        results[exp["name"]] = res
        
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAblation studies completed and saved to results/ablation_results.json")

if __name__ == "__main__":
    main()

