import os, sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

"""
Quick smoke test: verify the full PI-GhostFormer pipeline runs end-to-end.
Uses a tiny dataset and 2 epochs to validate all modules connect correctly.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.pighostformer import PIGhostFormer
from models.loss import PIGhostFormerLoss
from data.synthetic import SyntheticFabricGenerator

def main():
    cfg = get_config()
    device = torch.device("cpu")  # Force CPU for smoke test
    
    print("=" * 60)
    print("PI-GhostFormer Smoke Test")
    print("=" * 60)
    
    # 1. Build forward model
    print("\n[1/6] Building forward model...")
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
    print(f"  Measurement matrix: {cfg.physics.num_patterns} patterns x {cfg.physics.image_size**2} pixels")
    
    # 2. Build model
    print("\n[2/6] Building PI-GhostFormer...")
    model = PIGhostFormer(
        num_patterns=cfg.physics.num_patterns,
        embed_dim=cfg.model.embed_dim,
        encoder_layers=cfg.model.encoder_layers,
        encoder_heads=cfg.model.encoder_heads,
        physics_dim=cfg.physics.num_zernike,
        img_size=cfg.physics.image_size,
        num_classes=cfg.model.num_defect_classes
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable:    {trainable:,}")
    
    # 3. Build loss
    print("\n[3/6] Building loss function...")
    criterion = PIGhostFormerLoss(
        lambda_recon=cfg.train.lambda_recon,
        lambda_physics=cfg.train.lambda_physics,
        lambda_classify=cfg.train.lambda_classify,
        lambda_tv=cfg.train.lambda_tv,
        self_supervised=cfg.train.self_supervised
    ).to(device)
    
    # 4. Generate tiny dataset
    N_SAMPLES = 8
    BATCH_SIZE = 4
    print(f"\n[4/6] Generating {N_SAMPLES} synthetic samples...")
    generator = SyntheticFabricGenerator(image_size=cfg.physics.image_size)
    train_images, train_labels = generator.generate_batch(N_SAMPLES, cfg.data.defect_probability)
    print(f"  Images shape: {train_images.shape}")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Label distribution: {torch.bincount(train_labels).tolist()}")
    
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    # 5. Optimizer
    params = list(model.parameters())
    if cfg.physics.pattern_type == "learned":
        params += list(forward_model.measurement_matrix.parameters())
    optimizer = optim.AdamW(params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
    
    # 6. Train 2 epochs
    print(f"\n[5/6] Training for 2 epochs (batch_size={BATCH_SIZE})...")
    model.train()
    
    for epoch in range(2):
        t0 = time.time()
        for batch_idx, (images, labels) in enumerate(train_loader):
            B = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            
            # Random defocus
            zernike_coeffs = torch.zeros(B, cfg.physics.num_zernike)
            zernike_coeffs[:, 3] = torch.empty(B).uniform_(*cfg.physics.defocus_range)
            zernike_coeffs = zernike_coeffs.to(device)
            
            optimizer.zero_grad()
            
            # Forward physics (data generation, no grad)
            with torch.no_grad():
                noisy_measurements, _ = forward_model.forward_sample(images, zernike_coeffs)
            
            # Model forward
            recon_img, logits = model(noisy_measurements, zernike_coeffs)
            
            # Expected buckets for Poisson NLL
            expected_buckets = forward_model.forward_expected(recon_img, zernike_coeffs)
            
            # Loss
            losses = criterion(
                recon_img=recon_img,
                logits=logits,
                expected_buckets=expected_buckets,
                measured_buckets=noisy_measurements,
                gt_img=images,
                gt_class=labels
            )
            
            losses['total_loss'].backward()
            optimizer.step()
        
        dt = time.time() - t0
        
        # Quick validation
        model.eval()
        with torch.no_grad():
            val_z = torch.zeros(BATCH_SIZE, cfg.physics.num_zernike).to(device)
            val_z[:, 3] = 1.5  # mid defocus
            val_imgs = train_images[:BATCH_SIZE].to(device)
            val_meas, _ = forward_model.forward_sample(val_imgs, val_z)
            val_recon, val_logits = model(val_meas, val_z)
            
            mse = F.mse_loss(val_recon, val_imgs)
            psnr = 10 * torch.log10(1.0 / mse).item()
            preds = torch.argmax(val_logits, dim=1)
            acc = (preds == train_labels[:BATCH_SIZE].to(device)).float().mean().item()
        model.train()
        
        print(f"  Epoch {epoch+1}/2 | Loss: {losses['total_loss'].item():.4f} | "
              f"PSNR: {psnr:.2f} dB | Acc: {acc:.0%} | Time: {dt:.1f}s")
    
    print(f"\n[6/6] Verification...")
    print(f"  Reconstruction output shape: {val_recon.shape}")
    print(f"  Logits output shape:         {val_logits.shape}")
    print(f"  Gradient flow OK:            {model.encoder.stem[0].weight.grad is not None}")
    print(f"  Physics patterns learnable:  {forward_model.measurement_matrix.H_raw.grad is not None}")
    
    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — full pipeline runs end-to-end")
    print("=" * 60)

if __name__ == "__main__":
    main()
