"""
High-Resolution Evidential Reconstruction & Uncertainty Visualizer for IC-PINOT.
Generates publication-quality 5-column visual comparisons:
  1. Ground Truth Texture (x)
  2. Reconstructed Mean Intensity (gamma)
  3. Predicted Aleatoric Uncertainty (beta / (alpha - 1)) [Photon Shot Noise]
  4. Predicted Epistemic Uncertainty (beta / (v*(alpha - 1))) [Structural Novelty]
  5. Absolute Pixel Reconstruction Error (|x - gamma|)
Saves figures to figures/fig_evidential_reconstruction.png and .pdf.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset


def visualize_dataset_reconstruction(
    model_ckpt: str,
    dataset,
    dataset_name: str,
    num_samples: int = 3,
    device: str = "cuda",
    save_prefix: str = "figures/fig_evidential"
):
    os.makedirs("figures", exist_ok=True)
    cfg = get_config()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    forward_model = GhostImagingForwardModel(
        image_size=64,
        num_patterns=512,
        mean_photon_count=10.0,
        pattern_type="learned"
    ).to(dev)

    model = ICPINOT(
        num_patterns=512,
        embed_dim=cfg.model.embed_dim,
        img_size=64
    ).to(dev)

    if os.path.exists(model_ckpt):
        model.load_state_dict(torch.load(model_ckpt, map_location=dev))
        print(f"Loaded checkpoint for visualization: {model_ckpt}")
    else:
        print(f"Checkpoint not found at {model_ckpt}, using model state.")

    model.eval()

    samples_x, samples_gamma, samples_alea, samples_epis, samples_err = [], [], [], [], []

    with torch.no_grad():
        for i in range(num_samples):
            img, _ = dataset[i]
            x = img.unsqueeze(0).to(dev) # (1, 1, 64, 64)
            z = torch.zeros(1, 11, device=dev)
            z[:, 3] = 1.0 # 1-wave defocus

            noisy_meas, _ = forward_model.forward_sample(x, z)
            nig_outputs, _ = model(noisy_meas, z, return_evidential=True)

            gamma = nig_outputs["gamma"].squeeze().cpu().numpy()
            alea = nig_outputs["aleatoric"].squeeze().cpu().numpy()
            epis = nig_outputs["epistemic"].squeeze().cpu().numpy()
            gt = x.squeeze().cpu().numpy()
            err = np.abs(gt - gamma)

            samples_x.append(gt)
            samples_gamma.append(gamma)
            samples_alea.append(alea)
            samples_epis.append(epis)
            samples_err.append(err)

    fig, axes = plt.subplots(num_samples, 5, figsize=(15, 3.2 * num_samples), dpi=300)
    titles = [
        "Ground Truth ($x$)",
        "IC-PINOT Recon ($\\gamma$)",
        "Aleatoric UQ ($\\sigma^2_a$)",
        "Epistemic UQ ($\\sigma^2_e$)",
        "Absolute Error ($|x-\\gamma|$)"
    ]

    for row in range(num_samples):
        # 1. Ground Truth
        im0 = axes[row, 0].imshow(samples_x[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 0].axis("off")
        if row == 0:
            axes[row, 0].set_title(titles[0], fontsize=12, fontweight="bold")

        # 2. Recon
        im1 = axes[row, 1].imshow(samples_gamma[row], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        if row == 0:
            axes[row, 1].set_title(titles[1], fontsize=12, fontweight="bold")

        # 3. Aleatoric UQ
        im2 = axes[row, 2].imshow(samples_alea[row], cmap="viridis")
        axes[row, 2].axis("off")
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)
        if row == 0:
            axes[row, 2].set_title(titles[2], fontsize=12, fontweight="bold")

        # 4. Epistemic UQ
        im3 = axes[row, 3].imshow(samples_epis[row], cmap="inferno")
        axes[row, 3].axis("off")
        plt.colorbar(im3, ax=axes[row, 3], fraction=0.046, pad=0.04)
        if row == 0:
            axes[row, 3].set_title(titles[3], fontsize=12, fontweight="bold")

        # 5. Error
        im4 = axes[row, 4].imshow(samples_err[row], cmap="magma")
        axes[row, 4].axis("off")
        plt.colorbar(im4, ax=axes[row, 4], fraction=0.046, pad=0.04)
        if row == 0:
            axes[row, 4].set_title(titles[4], fontsize=12, fontweight="bold")

    plt.tight_layout()
    png_path = f"{save_prefix}_{dataset_name.lower().replace(' ', '_')}.png"
    pdf_path = f"{save_prefix}_{dataset_name.lower().replace(' ', '_')}.pdf"
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Publication figure saved to:\n  - {png_path}\n  - {pdf_path}")


def main():
    # 1. Visualize TILDA 400
    try:
        t_te = TILDADataset("data/benchmarks/tilda", split="test")
        visualize_dataset_reconstruction("checkpoints/ic_pinot_tilda_400_seed1337.pt", t_te, "TILDA_400")
    except Exception as e:
        print(f"TILDA visualization error: {e}")

    # 2. Visualize DAGM 2007
    try:
        d_te = DAGMDataset("data/benchmarks/dagm", split="test")
        visualize_dataset_reconstruction("checkpoints/ic_pinot_dagm_2007_seed1337.pt", d_te, "DAGM_2007")
    except Exception as e:
        print(f"DAGM visualization error: {e}")

    # 3. Visualize MVTec Leather
    try:
        l_te = MVTecTextureDataset("data/benchmarks/mvtec", category="leather", split="test")
        visualize_dataset_reconstruction("checkpoints/ic_pinot_mvtec-leather_seed1337.pt", l_te, "MVTec_Leather")
    except Exception as e:
        print(f"Leather visualization error: {e}")


if __name__ == "__main__":
    main()
