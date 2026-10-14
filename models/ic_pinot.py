"""
In-Context Physics-Informed Neural Operator Transformer (IC-PINOT).
Unified Flagship Architecture for Single-Photon Computational Ghost Imaging:
  1. Learnable Optical Forward Modeling (FNO / Differentiable Wavefront)
  2. Physics-Conditioned Cross-Attention (PCCA)
  3. In-Context Few-Shot Adaptation Engine (Zero-Gradient Test-Time Transfer)
  4. Native Evidential Deep Learning (NIG) for Calibrated O(1) Uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import PhysicsConditionedCrossAttention
from .swin import SwinTransformerBlock
from .in_context import InContextPromptEncoder
from .evidential import EvidentialHead2D


class Encoder1D(nn.Module):
    """Encodes the 1D bucket measurements using Conv1D Stem + Transformer."""
    def __init__(self, in_features: int, embed_dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
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
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        return x


class Decoder2D(nn.Module):
    """Multi-Scale Hybrid Swin-UNet Decoder with Evidential Head."""
    def __init__(self, embed_dim: int, img_size: int = 64, out_channels: int = 1):
        super().__init__()
        self.init_h = 8
        self.init_w = 8
        
        # Stage 1: 8x8 -> 16x16
        self.stage1_swin = SwinTransformerBlock(embed_dim, num_heads=8, window_size=4)
        self.stage1_up = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=4, stride=2, padding=1)
        
        # Stage 2: 16x16 -> 32x32
        self.stage2_swin = SwinTransformerBlock(embed_dim // 2, num_heads=4, window_size=4)
        self.stage2_up = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=4, stride=2, padding=1)
        
        # Stage 3: 32x32 -> 64x64
        self.stage3_swin = SwinTransformerBlock(embed_dim // 4, num_heads=2, window_size=4)
        self.stage3_up = nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=4, stride=2, padding=1)
        
        # Evidential Head (Predicts NIG parameters: gamma, v, alpha, beta)
        self.evidential_head = EvidentialHead2D(in_channels=embed_dim // 8, out_channels=out_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        B = x.shape[0]
        x = self.stage1_swin(x, (self.init_h, self.init_w))
        x = x.transpose(1, 2).view(B, -1, self.init_h, self.init_w)
        x = self.stage1_up(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = self.stage2_swin(x, (16, 16))
        x = x.transpose(1, 2).view(B, -1, 16, 16)
        x = self.stage2_up(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = self.stage3_swin(x, (32, 32))
        x = x.transpose(1, 2).view(B, -1, 32, 32)
        feat = self.stage3_up(x)
        
        nig_outputs = self.evidential_head(feat)
        return nig_outputs, feat


class ICPINOT(nn.Module):
    """
    In-Context Physics-Informed Neural Operator Transformer.
    """
    def __init__(
        self,
        num_patterns: int = 512,
        embed_dim: int = 256,
        encoder_layers: int = 6,
        encoder_heads: int = 8,
        physics_dim: int = 11,
        img_size: int = 64,
        num_classes: int = 2,
        num_prompt_tokens: int = 16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Measurement Encoder
        self.encoder = Encoder1D(
            in_features=num_patterns,
            embed_dim=embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads
        )
        
        # 2. In-Context Prompt Engine
        self.in_context_engine = InContextPromptEncoder(
            num_patterns=num_patterns,
            embed_dim=embed_dim,
            num_tokens=num_prompt_tokens
        )
        
        # 3. Physics-Conditioned Cross-Attention (PCCA)
        self.pcca = PhysicsConditionedCrossAttention(
            embed_dim=embed_dim,
            num_heads=encoder_heads,
            physics_dim=physics_dim
        )
        
        # 4. Learned Query Grid (8x8 = 64 spatial tokens)
        self.spatial_query_grid = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        
        # 5. Evidential Decoder
        self.decoder = Decoder2D(embed_dim=embed_dim, img_size=img_size, out_channels=1)
        
        # 6. Defect Classifier Head
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
        optical_params: torch.Tensor,
        context_meas: torch.Tensor = None,
        context_imgs: torch.Tensor = None,
        return_evidential: bool = False
    ):
        """
        Args:
            bucket_measurements: (B, num_patterns) query bucket vector y_q
            optical_params: (B, physics_dim) optical state theta (e.g. Zernike defocus a_4)
            context_meas: (B, K, num_patterns) optional K reference measurements
            context_imgs: (B, K, 1, H, W) optional K reference images
            return_evidential: bool, whether to return the full NIG uncertainty dictionary
        """
        B = bucket_measurements.shape[0]
        
        # 1. 1D Signal Encoding
        enc_tokens = self.encoder(bucket_measurements) # (B, M, embed_dim)
        
        # 2. In-Context Prompt Injection
        query = self.spatial_query_grid.expand(B, -1, -1) # (B, 64, embed_dim)
        if context_meas is not None and context_imgs is not None:
            prompt_tokens = self.in_context_engine(context_meas, context_imgs) # (B, num_prompt_tokens, embed_dim)
            enc_tokens = torch.cat([prompt_tokens, enc_tokens], dim=1) # Concatenate context and query sequence
            
        # 3. Physics-Conditioned Cross-Attention
        attended = self.pcca(query, enc_tokens, optical_params) # (B, 64, embed_dim)
        
        # 4. Evidential Reconstruction & Spatial Decoding
        nig_outputs, pre_feats = self.decoder(attended)
        
        # 5. Defect Classification
        logits = self.classifier(pre_feats)
        
        recon_mean = nig_outputs["gamma"]
        if return_evidential:
            return nig_outputs, logits
        return recon_mean, logits
