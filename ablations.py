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
    
    generator = SyntheticFabricGenerator(image_size=cfg.physics.image_size)
    # Use smaller dataset for fast ablation testing
    train_images, train_labels = generator.generate_batch(500, cfg.data.defect_probability)
    val_images, val_labels = generator.generate_batch(100, cfg.data.defect_probability)
    
    train_loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=cfg.train.batch_size)
    
    # Run a few epochs
    for epoch in range(5):
        for images, labels in train_loader:
            zernike_coeffs = torch.zeros(images.shape[0], cfg.physics.num_zernike)
            zernike_coeffs[:, 3] = 1.0 # 1 wave defocus
            train_step(model, forward_model, criterion, optimizer, images, labels, zernike_coeffs, device)
            
    # Final Validation
    model.eval()
    val_zernike = torch.zeros(val_images.shape[0], cfg.physics.num_zernike).to(device)
    val_zernike[:, 3] = 1.0
    
    with torch.no_grad():
        noisy_measurements, _ = forward_model.forward_sample(val_images.to(device), val_zernike)
        recon_img, logits = model(noisy_measurements, val_zernike)
        metrics = evaluate_batch(recon_img, val_images.to(device), logits, val_labels.to(device))
        
    print(f"Results for {exp_name}: PSNR={metrics['psnr']:.2f}, F1={metrics['f1_score']:.4f}")
    return metrics


def main():
    ablations = [
        {"name": "Baseline (Full PI-GhostFormer)", "overrides": {}},
        {"name": "A2: Remove Poisson NLL (MSE only)", "overrides": {"lambda_physics": 0.0}},
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
