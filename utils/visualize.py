"""
Visualization tools for Physics-Informed Ghost Transformer.

Functions to generate the attention rollout maps and physical state
mappings required for the qualitative results section of the paper.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os


def plot_reconstruction_comparison(clean_img: torch.Tensor,
                                   defocused_psf: torch.Tensor, 
                                   recon_img: torch.Tensor,
                                   save_path: str = "results/reconstruction_comparison.png"):
    """
    Plots the ground truth, the physical point spread function (defocus),
    and the reconstructed image side-by-side.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    clean_np = clean_img.squeeze().cpu().detach().numpy()
    psf_np = defocused_psf.squeeze().cpu().detach().numpy()
    recon_np = recon_img.squeeze().cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ax = axes[0]
    ax.imshow(clean_np, cmap='gray')
    ax.set_title("Ground Truth Fabric")
    ax.axis('off')
    
    ax = axes[1]
    ax.imshow(psf_np, cmap='hot')
    ax.set_title("Physical Defocus (PSF)")
    ax.axis('off')
    
    ax = axes[2]
    ax.imshow(recon_np, cmap='gray')
    ax.set_title("PI-GhostFormer Reconstruction")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_attention_map(attention_weights: torch.Tensor, 
                       save_path: str = "results/attention_map.png"):
    """
    Visualizes the Physics-Conditioned Cross-Attention weights.
    Highlights how the network adapts its spatial focus based on the physics vector.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Assuming attention_weights shape: (batch, num_heads, seq_len, seq_len)
    # We take the mean across heads for the first item in the batch
    att_np = attention_weights[0].mean(dim=0).cpu().detach().numpy()
    
    # If it's a 1D sequence mapping (e.g., 256x256), we reshape it to spatial dimensions if needed
    # For visualization, we just plot the 2D matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(att_np, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.title("Physics-Conditioned Cross-Attention Map")
    plt.xlabel("Key/Value Sequence")
    plt.ylabel("Query Sequence")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
