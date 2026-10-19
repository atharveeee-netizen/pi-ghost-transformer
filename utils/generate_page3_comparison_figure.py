import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config import get_config
from physics.forward_model import GhostImagingForwardModel
from models.ic_pinot import ICPINOT
from data.benchmark_loader import TILDADataset, DAGMDataset, MVTecTextureDataset
from utils.metrics import calculate_psnr, calculate_ssim

def run_dgi(noisy_meas, patterns):
    B, M = noisy_meas.shape
    H, W = patterns.shape[1], patterns.shape[2]
    s_m = patterns.view(M, -1).sum(dim=1)
    mean_s = s_m.mean()
    mean_y = noisy_meas.mean(dim=1, keepdim=True)
    recons = []
    for b in range(B):
        y_b = noisy_meas[b]
        coeff = y_b - (mean_y[b] / (mean_s + 1e-8)) * s_m
        recon_b = (coeff.view(M, 1, 1) * patterns).mean(dim=0, keepdim=True)
        recon_norm = (recon_b - recon_b.min()) / (recon_b.max() - recon_b.min() + 1e-8)
        recons.append(recon_norm)
    return torch.stack(recons, dim=0)

def generate_page3_comparison():
    os.makedirs("figures", exist_ok=True)
    cfg = get_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Generating Page 3 comparison using device: {dev}")

    forward_model = GhostImagingForwardModel(
        image_size=64,
        num_patterns=512,
        mean_photon_count=10.0,
        pattern_type="learned"
    ).to(dev)

    patterns = forward_model.measurement_matrix.patterns.view(512, 64, 64)

    tasks = [
        {
            "name": "TILDA 400 (Woven Textile)",
            "subtitle": "Broken Warp Yarn Defect",
            "dataset": TILDADataset("data/benchmarks/tilda", split="test"),
            "ckpt": "checkpoints/ic_pinot_tilda_400_seed1337.pt",
            "idx": 0
        },
        {
            "name": "DAGM 2007 (Metal Surface)",
            "subtitle": "Weak-Contrast Scratch Flaw",
            "dataset": DAGMDataset("data/benchmarks/dagm", split="test"),
            "ckpt": "checkpoints/ic_pinot_dagm_2007_seed1337.pt",
            "idx": 0
        },
        {
            "name": "MVTec-AD (Industrial Leather)",
            "subtitle": "Natural Grain Cut Anomaly",
            "dataset": MVTecTextureDataset("data/benchmarks/mvtec", category="leather", split="test"),
            "ckpt": "checkpoints/ic_pinot_mvtec-leather_seed1337.pt",
            "idx": 0
        }
    ]

    fig, axes = plt.subplots(3, 6, figsize=(18, 9.5), dpi=300)
    plt.subplots_adjust(wspace=0.18, hspace=0.28)

    col_headers = [
        "1. Ground Truth Target\n($x \\in \\mathbb{R}^{64 \\times 64}$)",
        "2. Classical DGI Optics\n($x_{\\mathrm{DGI}}$ Standard Baseline)",
        "3. IC-PINOT Reconstruction\n(Proposed Mean $\\hat{x} = \\gamma$)",
        "4. Aleatoric Uncertainty $\\sigma^2_a$\n(Photon Shot Noise)",
        "5. Epistemic Uncertainty $\\sigma^2_e$\n(Structural Defect Novelty)",
        "6. Absolute Residual Error\n($|x - \\gamma|$)"
    ]

    for col in range(6):
        axes[0, col].set_title(col_headers[col], fontsize=11, fontweight="bold", pad=10)

    for row, t in enumerate(tasks):
        model = ICPINOT(num_patterns=512, embed_dim=cfg.model.embed_dim, img_size=64).to(dev)
        if os.path.exists(t["ckpt"]):
            model.load_state_dict(torch.load(t["ckpt"], map_location=dev))
            print(f"Loaded: {t['ckpt']}")
        model.eval()

        img, _ = t["dataset"][t["idx"]]
        x = img.unsqueeze(0).to(dev)
        z = torch.zeros(1, 11, device=dev)
        z[:, 3] = 1.0 # 1.0-wave optical defocus

        with torch.no_grad():
            noisy_meas, _ = forward_model.forward_sample(x, z)
            dgi_rec = run_dgi(noisy_meas, patterns)
            nig_out, _ = model(noisy_meas, z, return_evidential=True)

        gt = x.squeeze().cpu().numpy()
        dgi_np = dgi_rec.squeeze().cpu().numpy()
        gamma = nig_out["gamma"].squeeze().cpu().numpy()
        alea = nig_out["aleatoric"].squeeze().cpu().numpy()
        epis = nig_out["epistemic"].squeeze().cpu().numpy()
        err = np.abs(gt - gamma)

        psnr_dgi = calculate_psnr(dgi_rec, x)
        ssim_dgi = calculate_ssim(dgi_rec, x)
        psnr_our = calculate_psnr(nig_out["gamma"], x)
        ssim_our = calculate_ssim(nig_out["gamma"], x)

        # 1. Ground Truth
        axes[row, 0].imshow(gt, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"{t['name']}\n{t['subtitle']}", fontsize=10.5, fontweight="bold", labelpad=8)
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        # 2. DGI
        axes[row, 1].imshow(dgi_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_xlabel(f"PSNR: {psnr_dgi:.2f} dB\nSSIM: {ssim_dgi:.3f}", fontsize=9.5, fontweight="bold")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        # 3. IC-PINOT
        axes[row, 2].imshow(gamma, cmap="gray", vmin=0, vmax=1)
        gain_psnr = psnr_our - psnr_dgi
        axes[row, 2].set_xlabel(f"PSNR: {psnr_our:.2f} dB ( +{gain_psnr:.2f} dB )\nSSIM: {ssim_our:.3f}", fontsize=9.5, fontweight="bold")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])

        # 4. Aleatoric UQ
        im_a = axes[row, 3].imshow(alea, cmap="viridis")
        axes[row, 3].set_xlabel(f"Mean $\\sigma^2_a$: {np.mean(alea):.4f}", fontsize=9.5)
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])
        divider = make_axes_locatable(axes[row, 3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_a, cax=cax)

        # 5. Epistemic UQ
        im_e = axes[row, 4].imshow(epis, cmap="inferno")
        axes[row, 4].set_xlabel(f"Peak Novelty: {np.max(epis):.4f}", fontsize=9.5, fontweight="bold")
        axes[row, 4].set_xticks([])
        axes[row, 4].set_yticks([])
        divider = make_axes_locatable(axes[row, 4])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_e, cax=cax)

        # 6. Error
        im_err = axes[row, 5].imshow(err, cmap="magma")
        axes[row, 5].set_xlabel(f"MAE: {np.mean(err):.4f}", fontsize=9.5)
        axes[row, 5].set_xticks([])
        axes[row, 5].set_yticks([])
        divider = make_axes_locatable(axes[row, 5])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_err, cax=cax)

    out_png = "figures/fig_page3_comparison.png"
    out_pdf = "figures/fig_page3_comparison.pdf"
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Generated Page 3 comparison showcase:\n  - {out_png}\n  - {out_pdf}")

if __name__ == "__main__":
    generate_page3_comparison()
