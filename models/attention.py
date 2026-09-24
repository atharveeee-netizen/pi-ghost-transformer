"""
Physics-Conditioned Cross-Attention (PCCA).

This module implements the novel attention mechanism that conditions the reconstruction
on the physical parameters (Zernike coefficients) of the optical system.

Instead of standard self-attention (Q = K = V from the image features),
it uses Cross-Attention where:
  - Q comes from the image features (decoder input)
  - K, V come from the encoder features
  - A Physics Token (PSF embedding) is appended to K and V, allowing the attention
    mechanism to query the physical state (e.g., defocus level).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConditionedCrossAttention(nn.Module):
    """
    Cross-Attention conditioned on physical optical parameters.

    Args:
        embed_dim: Dimension of the feature embeddings.
        num_heads: Number of attention heads.
        physics_dim: Dimension of the input physics parameters (e.g., num_zernike).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        physics_dim: int = 11,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Physics projection network (MLP)
        # Maps raw Zernike coefficients to the same dimension as the feature embeddings
        self.physics_mlp = nn.Sequential(
            nn.Linear(physics_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Projections for Queries (from decoder features)
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # Projections for Keys and Values (from encoder features)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_feats: torch.Tensor,
        key_value_feats: torch.Tensor,
        zernike_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_feats: Target features (e.g., from decoder), shape (B, N_q, D)
            key_value_feats: Source features (e.g., from encoder), shape (B, N_kv, D)
            zernike_coeffs: Physical parameters (Zernike coeffs), shape (B, physics_dim)

        Returns:
            Conditioned features, shape (B, N_q, D)
        """
        B, N_q, D = query_feats.shape
        _, N_kv, _ = key_value_feats.shape

        # 1. Project the physics parameters into a "Physics Token"
        # Shape: (B, D) -> (B, 1, D)
        physics_token = self.physics_mlp(zernike_coeffs).unsqueeze(1)

        # 2. Append the Physics Token to the key_value_feats
        # This makes the physics state accessible as a distinct feature to attend to.
        # Shape: (B, N_kv + 1, D)
        kv_feats_augmented = torch.cat([key_value_feats, physics_token], dim=1)

        # 3. Standard multi-head attention projections
        # Reshape to (B, N, num_heads, head_dim) and transpose to (B, num_heads, N, head_dim)
        q = self.q_proj(query_feats).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_feats_augmented).view(B, N_kv + 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_feats_augmented).view(B, N_kv + 1, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Compute Attention Scores: Q(K^T) / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # 5. Apply attention to Values
        out = torch.matmul(attn, v)  # (B, num_heads, N_q, head_dim)

        # 6. Recombine heads and project
        out = out.transpose(1, 2).contiguous().view(B, N_q, D)
        out = self.out_proj(out)

        return out
