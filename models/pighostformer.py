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
        
        # Final output projection
        self.out_conv = nn.Conv2d(embed_dim // 8, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is the attended sequence: (B, 64, embed_dim)
        B = x.shape[0]
        
        # x is already properly shaped as (B, 64, embed_dim) from the 64 query tokens

        
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
        
        # Output
        out = self.out_conv(x)
        return torch.sigmoid(out), x


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

    def forward(self, bucket_measurements: torch.Tensor, zernike_coeffs: torch.Tensor):
        """
        Args:
            bucket_measurements: (B, num_patterns)
            zernike_coeffs: (B, num_zernike)
            
        Returns:
            reconstructed_image: (B, 1, H, W)
            logits: (B, num_classes)
        """
        B = bucket_measurements.shape[0]
        
        # 1. Encode 1D measurements
        enc_feats = self.encoder(bucket_measurements)  # (B, M, embed_dim)
        
        # 2. Physics-Conditioned Attention
        # The query is a learned 8x8 spatial grid, expanded to batch size
        query = self.decoder_query.expand(B, -1, -1)  # (B, 64, embed_dim)
        
        # The cross attention allows the query grid to pull info from the sequence AND the physics token
        attended_feats = self.pcca(query, enc_feats, zernike_coeffs)  # (B, 64, embed_dim)
        
        # 3. Decode to 2D image and get pre-output features
        recon_img, features = self.decoder(attended_feats)  # recon_img: (B, 1, H, W), features: (B, C, H, W)
        
        # 4. Classify defect from the pre-output features
        logits = self.classifier(features)  # (B, num_classes)
        
        return recon_img, logits
