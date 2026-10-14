"""
Physics-Informed Ghost Transformer (PI-GhostFormer).

The complete end-to-end architecture combining:
  1. 1D Signal Encoder (CNN Stem + Transformer)
  2. Physics-Conditioned Cross-Attention (PCCA)
  3. Multi-Scale Hybrid U-Net/Swin Decoder
  4. Classification Head (for defects)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PhysicsConditionedCrossAttention
from .swin import SwinTransformerBlock


class Encoder1D(nn.Module):
    """
    Encodes the 1D bucket measurements.
    Uses a CNN stem followed by standard Transformer layers.
    """
    def __init__(self, in_features: int, embed_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        
        # CNN stem to extract local temporal patterns from the bucket signal
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, M)
        x = x.unsqueeze(1)  # (B, 1, M)
        x = self.stem(x)    # (B, embed_dim, M)
        x = x.transpose(1, 2)  # (B, M, embed_dim)
        x = self.transformer(x) # (B, M, embed_dim)
        return x


class Decoder2D(nn.Module):
    """
    Multi-Scale Hybrid Decoder (CNN + Swin Transformer).
    Upsamples the encoded features to a 2D image.
    """
    def __init__(self, embed_dim: int, img_size: int, out_channels: int = 1):
        super().__init__()
        self.img_size = img_size
        
        self.init_h = 8
        self.init_w = 8
        
        # Stage 1: 8x8 -> 16x16
        
        # Stage 1: 8x8 -> 16x16
        self.stage1_swin = SwinTransformerBlock(embed_dim, num_heads=8, window_size=4)
        self.stage1_up = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1)
        
        # Stage 2: 16x16 -> 32x32
        self.stage2_swin = SwinTransformerBlock(embed_dim // 2, num_heads=4, window_size=4)
        self.stage2_up = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1)
        
        # Stage 3: 32x32 -> 64x64 (img_size)
        self.stage3_swin = SwinTransformerBlock(embed_dim // 4, num_heads=2, window_size=4)
        self.stage3_up = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1)
        
        # Final output projection (Mean intensity reconstruction)
        self.out_conv = nn.Conv2d(embed_dim // 8, out_channels, kernel_size=3, padding=1)
        # Uncertainty head: Predicts spatial aleatoric variance log(sigma^2)
        self.uncertainty_conv = nn.Conv2d(embed_dim // 8, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple:
        # x is the attended sequence: (B, 64, embed_dim)
        B = x.shape[0]
        
        # Stage 1
        x = self.stage1_swin(x, (self.init_h, self.init_w))
        x = x.transpose(1, 2).view(B, -1, self.init_h, self.init_w)
        x = self.stage1_up(x)  # (B, embed_dim/2, 16, 16)
        
        # Stage 2
        x = x.flatten(2).transpose(1, 2)  # (B, 256, embed_dim/2)
        x = self.stage2_swin(x, (16, 16))
        x = x.transpose(1, 2).view(B, -1, 16, 16)
        x = self.stage2_up(x)  # (B, embed_dim/4, 32, 32)
        
        # Stage 3
        x = x.flatten(2).transpose(1, 2)  # (B, 1024, embed_dim/4)
        x = self.stage3_swin(x, (32, 32))
        x = x.transpose(1, 2).view(B, -1, 32, 32)
        x = self.stage3_up(x)  # (B, embed_dim/8, 64, 64)
        
        # Outputs: Mean reconstruction + Uncertainty (variance)
        recon_mean = torch.sigmoid(self.out_conv(x))
        log_var = self.uncertainty_conv(x)
        uncertainty = F.softplus(log_var)  # Ensure positive variance sigma^2
        return recon_mean, uncertainty, x


class PIGhostFormer(nn.Module):
    """
    Physics-Informed Ghost Transformer (PI-GhostFormer).
    """
    def __init__(
        self,
        num_patterns: int = 512,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        physics_dim: int = 11,
        img_size: int = 64,
        num_classes: int = 5
    ):
        super().__init__()
        
        self.encoder = Encoder1D(
            in_features=num_patterns,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads
        )
        
        # PCCA (Physics-Conditioned Cross-Attention)
        # Bridges the encoder and decoder
        self.pcca = PhysicsConditionedCrossAttention(
            embed_dim=embed_dim,
            num_heads=encoder_heads,
            physics_dim=physics_dim
        )
        
        # A learned query grid (8x8 = 64 tokens) for the decoder to attend to the encoded sequence
        self.decoder_query = nn.Parameter(torch.randn(1, 64, embed_dim))
        
        self.decoder = Decoder2D(
            embed_dim=embed_dim,
            img_size=img_size,
            out_channels=1
        )
        
        # Defect Classification Head
        # Global Average Pooling on the reconstructed features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim // 8, 128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, 
        bucket_measurements: torch.Tensor, 
        zernike_coeffs: torch.Tensor,
        return_uncertainty: bool = False
    ):
        """
        Args:
            bucket_measurements: (B, num_patterns)
            zernike_coeffs: (B, num_zernike)
            return_uncertainty: bool, whether to return the predicted uncertainty map
            
        Returns:
            If return_uncertainty is False:
                reconstructed_image: (B, 1, H, W)
                logits: (B, num_classes)
            If return_uncertainty is True:
                reconstructed_image: (B, 1, H, W)
                logits: (B, num_classes)
                uncertainty_map: (B, 1, H, W)  [Predictive spatial variance sigma^2]
        """
        B = bucket_measurements.shape[0]
        
        # 1. Encode 1D measurements
        enc_feats = self.encoder(bucket_measurements)  # (B, M, embed_dim)
        
        # 2. Physics-Conditioned Attention
        query = self.decoder_query.expand(B, -1, -1)  # (B, 64, embed_dim)
        attended_feats = self.pcca(query, enc_feats, zernike_coeffs)  # (B, 64, embed_dim)
        
        # 3. Decode to 2D image and uncertainty
        recon_img, uncertainty_map, features = self.decoder(attended_feats)
        
        # 4. Classify defect from features
        logits = self.classifier(features)  # (B, num_classes)
        
        if return_uncertainty:
            return recon_img, logits, uncertainty_map
        return recon_img, logits

    def estimate_uncertainty_mc_dropout(
        self, 
        bucket_measurements: torch.Tensor, 
        zernike_coeffs: torch.Tensor, 
        num_passes: int = 10
    ) -> dict:
        """
        Monte Carlo (MC) Dropout Uncertainty Quantification.
        Computes both Aleatoric (data/sensor) and Epistemic (model knowledge) uncertainty.
        
        Returns:
            dict containing:
                - 'mean_recon': (B, 1, H, W) ensemble average
                - 'epistemic_uncertainty': (B, 1, H, W) variance across MC passes
                - 'aleatoric_uncertainty': (B, 1, H, W) average predicted variance
                - 'total_uncertainty': (B, 1, H, W) epistemic + aleatoric
                - 'defect_probs': (B, num_classes) softmax defect probabilities
                - 'defect_entropy': (B,) prediction entropy / confidence
        """
        self.train()  # Keep dropout enabled for MC passes
        recons, aleatorics, all_logits = [], [], []
        
        with torch.no_grad():
            for _ in range(num_passes):
                r, l, u = self.forward(bucket_measurements, zernike_coeffs, return_uncertainty=True)
                recons.append(r)
                aleatorics.append(u)
                all_logits.append(l)
                
        # Stack passes: (T, B, 1, H, W)
        recons_stack = torch.stack(recons, dim=0)
        aleatorics_stack = torch.stack(aleatorics, dim=0)
        logits_stack = torch.stack(all_logits, dim=0)
        
        mean_recon = recons_stack.mean(dim=0)
        epistemic = recons_stack.var(dim=0)
        aleatoric = aleatorics_stack.mean(dim=0)
        total_uq = epistemic + aleatoric
        
        # Softmax probabilities and Shannon entropy
        mean_probs = F.softmax(logits_stack, dim=-1).mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        self.eval()
        return {
            'mean_recon': mean_recon,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total_uq,
            'defect_probs': mean_probs,
            'defect_entropy': entropy
        }
