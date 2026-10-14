"""
In-Context Adaptation Engine for Computational Ghost Imaging (IC-PINOT).
Enables test-time adaptation to unseen fabrics and physical optics drifts
using a few context reference pairs (y_k, x_k) without gradient backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InContextPromptEncoder(nn.Module):
    """
    Encodes K prompt pairs (y_k, x_k) into in-context condition tokens:
      - y_k: 1D bucket measurement vector in R^M
      - x_k: 2D reference ground-truth fabric image in R^{1 x H x W}
    """
    def __init__(self, num_patterns: int = 512, embed_dim: int = 256, num_tokens: int = 16):
        super().__init__()
        self.num_patterns = num_patterns
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        
        # 1. Measurement projection
        self.meas_proj = nn.Sequential(
            nn.Linear(num_patterns, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. Image patch projection
        self.img_proj = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=4),  # 64x64 -> 16x16
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=4), # 16x16 -> 4x4
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embed_dim)
        )
        
        # 3. Context cross-attention to distill into L fixed prompt tokens
        self.learned_prompt_queries = nn.Parameter(torch.randn(1, num_tokens, embed_dim) * 0.02)
        self.prompt_mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_meas: torch.Tensor = None, context_imgs: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            context_meas: (B, K, M) tensor of K context measurements
            context_imgs: (B, K, 1, H, W) tensor of K context images
        Returns:
            context_tokens: (B, num_tokens, embed_dim) prompt conditioning tokens
        """
        if context_meas is None or context_imgs is None:
            return None  # No in-context conditioning supplied (fallback to static)
            
        B, K, M = context_meas.shape
        # Flatten batch and K
        meas_flat = context_meas.view(B * K, M)
        imgs_flat = context_imgs.view(B * K, 1, context_imgs.shape[-2], context_imgs.shape[-1])
        
        meas_tokens = self.meas_proj(meas_flat) # (B*K, embed_dim)
        img_tokens = self.img_proj(imgs_flat)   # (B*K, embed_dim)
        
        # Fused pair representation
        pair_tokens = (meas_tokens + img_tokens).view(B, K, self.embed_dim)
        
        # Distill K pairs into fixed length prompt tokens using learned queries
        prompt_q = self.learned_prompt_queries.expand(B, -1, -1)
        context_tokens, _ = self.prompt_mha(prompt_q, pair_tokens, pair_tokens)
        context_tokens = self.norm(context_tokens + prompt_q)
        
        return context_tokens
