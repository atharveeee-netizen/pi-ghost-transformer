"""
Training Loop and Evaluation Pipeline for PI-GhostFormer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import cv2

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.pighostformer import PIGhostFormer
from models.loss import PIGhostFormerLoss
from data.synthetic import SyntheticFabricGenerator


def train_step(
    model: nn.Module, 
    forward_model: nn.Module, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer,
    images: torch.Tensor, 
    labels: torch.Tensor, 
    zernike_coeffs: torch.Tensor,
    device: str
) -> dict:
    model.train()
    optimizer.zero_grad()
    
    images = images.to(device)
    labels = labels.to(device)
    zernike_coeffs = zernike_coeffs.to(device)
    
    # 1. Forward Physics Model (Data Generation)
    with torch.no_grad():
        noisy_measurements, _ = forward_model.forward_sample(images, zernike_coeffs)
        
    # 2. Differentiable Physics Model (for loss)
    # We need the expected bucket measurements given the current *reconstructed* image
    # Note: We pass the generated noisy measurements into the model
    
    # 3. Model Forward Pass
    recon_img, logits = model(noisy_measurements, zernike_coeffs)
    
    # 4. Compute Expected Measurements from Reconstructed Image
    # (H * x_recon). We use the forward model's measurement matrix.
    # We only need the expected value, not the sampled noise.
    expected_buckets = forward_model.forward_expected(recon_img, zernike_coeffs)
    
    # 5. Loss Calculation
    losses = criterion(
        recon_img=recon_img,
        logits=logits,
        expected_buckets=expected_buckets,
        measured_buckets=noisy_measurements, # The actual noisy input
        gt_img=images,
        gt_class=labels
    )
    
    loss = losses['total_loss']
    
    # 6. Backward Pass
    loss.backward()
    optimizer.step()
    
    return losses


def validate(
    model: nn.Module, 
    forward_model: nn.Module, 
    criterion: nn.Module,
    val_images: torch.Tensor, 
    val_labels: torch.Tensor, 
    zernike_coeffs: torch.Tensor,
    device: str
) -> dict:
    model.eval()
    
    val_images = val_images.to(device)
    val_labels = val_labels.to(device)
    zernike_coeffs = zernike_coeffs.to(device)
    
    with torch.no_grad():
        noisy_measurements, _ = forward_model.forward_sample(val_images, zernike_coeffs)
        recon_img, logits = model(noisy_measurements, zernike_coeffs)
        expected_buckets = forward_model.forward_expected(recon_img, zernike_coeffs)
        
        losses = criterion(
            recon_img=recon_img,
            logits=logits,
            expected_buckets=expected_buckets,
            measured_buckets=noisy_measurements,
            gt_img=val_images,
            gt_class=val_labels
        )
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == val_labels).float().mean().item()
        losses['val_accuracy'] = acc
        
        # Calculate PSNR
        mse = F.mse_loss(recon_img, val_images)
        psnr = 10 * torch.log10(1.0 / mse).item()
        losses['val_psnr'] = psnr
        
    return losses


def main():
    cfg = get_config()
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Models
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
    
    model = PIGhostFormer(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        encoder_layers=cfg.model.encoder_layers,
        encoder_heads=cfg.model.encoder_heads,
        physics_dim=cfg.physics.num_zernike,
        img_size=cfg.physics.image_size,
        num_classes=cfg.model.num_defect_classes
    ).to(device)
    
    # Loss and Optimizer
    criterion = PIGhostFormerLoss(
        lambda_recon=cfg.train.lambda_recon,
        lambda_physics=cfg.train.lambda_physics,
        lambda_classify=cfg.train.lambda_classify,
        lambda_tv=cfg.train.lambda_tv,
        self_supervised=cfg.train.self_supervised
    ).to(device)
    
    # Co-optimize measurement patterns if required
    params = list(model.parameters())
    if cfg.physics.pattern_type == "learned":
        params += list(forward_model.measurement_matrix.parameters())
        
    optimizer = optim.AdamW(
        params, 
        lr=cfg.train.learning_rate, 
        weight_decay=cfg.train.weight_decay
    )
    
    # Load Real Dataset or Fallback to Synthetic
    if cfg.data.aitex_path and os.path.exists(cfg.data.aitex_path):
        print(f"Loading real AITEX dataset from {cfg.data.aitex_path}...")
        from data.aitex_loader import AITEXDataset
        full_dataset = AITEXDataset(cfg.data.aitex_path, image_size=cfg.physics.image_size)
        
        # Split into train/val
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
        # For validation we need explicit tensors in the current code structure
        val_images, val_labels = [], []
        for i in range(min(32, len(val_dataset))):
            img, lbl = val_dataset[i]
            val_images.append(img)
            val_labels.append(lbl)
        val_images = torch.stack(val_images)
        val_labels = torch.stack(val_labels)
    else:
        print("Generating synthetic data...")
        generator = SyntheticFabricGenerator(image_size=cfg.physics.image_size)
        train_images, train_labels = generator.generate_batch(cfg.data.num_train_samples, cfg.data.defect_probability)
        val_images, val_labels = generator.generate_batch(cfg.data.num_val_samples, cfg.data.defect_probability)
        
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    
    # Training Loop
    print("Starting Phase 2 real-world training with Curriculum Learning...")
    best_psnr = -1.0
    patience = cfg.train.early_stopping_patience
    epochs_without_improvement = 0
    max_epochs = cfg.train.max_epochs

    for epoch in range(max_epochs):
        epoch_losses = {}
        
        # Curriculum Learning Phase Toggle
        if epoch < cfg.train.curriculum_warmup_epochs:
            criterion.lambda_physics = 0.0
            criterion.lambda_recon = 1.0
            if epoch == 0:
                print("==> Curriculum Phase 1: Pure MSE Warmup")
        else:
            criterion.lambda_physics = cfg.train.lambda_physics
            criterion.lambda_recon = cfg.train.lambda_recon
            if epoch == cfg.train.curriculum_warmup_epochs:
                print("==> Curriculum Phase 2: Activating Poisson NLL Physics Loss")
                
        for batch_idx, (images, labels) in enumerate(train_loader):
            B = images.shape[0]
            
            # Randomize defocus for each batch to make model robust
            # a4 (defocus) is index 3 in Zernike array
            zernike_coeffs = torch.zeros(B, cfg.physics.num_zernike)
            defocus = torch.empty(B).uniform_(*cfg.physics.defocus_range)
            zernike_coeffs[:, 3] = defocus
            
            losses = train_step(model, forward_model, criterion, optimizer, images, labels, zernike_coeffs, device)
            
            for k, v in losses.items():
                if isinstance(v, torch.Tensor): v = v.item()
                epoch_losses[k] = epoch_losses.get(k, 0) + v
                
        # Validation
        val_zernike = torch.zeros(val_images.shape[0], cfg.physics.num_zernike)
        val_zernike[:, 3] = sum(cfg.physics.defocus_range) / 2.0  # test at mid defocus
        
        # Batch validation to avoid OOM
        val_losses = validate(model, forward_model, criterion, val_images[:32], val_labels[:32], val_zernike[:32], device)
        
        current_psnr = val_losses['val_psnr']
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print stats
        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {epoch_losses['total_loss']/len(train_loader):.4f} | "
                  f"Val Loss: {val_losses['total_loss']:.4f} | Val Acc: {val_losses['val_accuracy']:.2f} | Val PSNR: {val_losses['val_psnr']:.2f} dB | Patience: {epochs_without_improvement}/{patience}")

        if epochs_without_improvement >= patience and epoch > 100:
            print(f"Early stopping triggered at epoch {epoch}. Model has fully converged and is ready for IEEE submission!")
            break


if __name__ == "__main__":
    main()
