import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

def generate_pro_architecture_diagram():
    os.makedirs("figures", exist_ok=True)
    
    # 16:7.5 aspect ratio for comprehensive presentation across full page width
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Monochrome / High-Contrast Academic Palette
    c_bg = "#FFFFFF"
    c_panel_bg = "#FAFAFA"
    c_sub_bg = "#F3F4F6"
    c_border = "#111827"
    c_border_light = "#4B5563"
    c_text_dark = "#111827"
    c_text_muted = "#374151"
    c_highlight = "#E5E7EB"

    # Main Diagram Title Block
    ax.text(8.0, 6.90, "IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", 
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=c_text_dark)
    ax.text(8.0, 6.55, "End-to-End Computational Ghost Imaging Pipeline: Optical Forward Model → Continuous Operator → Evidential UQ", 
            ha="center", va="center", fontsize=10.0, fontstyle="italic", color=c_text_muted)

    # Stage Banner Helper
    def draw_stage_banner(x, y, w, h, stage_num, stage_title):
        box = patches.Rectangle((x, y), w, h, facecolor=c_highlight, edgecolor=c_border, linewidth=1.2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, f"STAGE {stage_num}: {stage_title.upper()}", 
                ha="center", va="center", fontsize=9.2, fontweight="bold", color=c_text_dark)

    # 4 Main Stages
    draw_stage_banner(0.4, 6.05, 3.5, 0.35, "I", "Single-Photon Optical Physics")
    draw_stage_banner(4.2, 6.05, 3.8, 0.35, "II", "Continuous Operator & Adaptation")
    draw_stage_banner(8.3, 6.05, 3.8, 0.35, "III", "Physics Cross-Attention & Decoder")
    draw_stage_banner(12.4, 6.05, 3.2, 0.35, "IV", "Evidential Uncertainty Head")

    # Card Helper
    def draw_card(x, y, w, h, title, subtitle="", bg=c_panel_bg):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=patches.BoxStyle("Round", pad=0.04, rounding_size=0.12),
            facecolor=bg, edgecolor=c_border, linewidth=1.4
        )
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.25, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_text_dark)
        if subtitle:
            ax.text(x + w/2, y + h - 0.50, subtitle, ha="center", va="center", fontsize=8.6, fontstyle="italic", color=c_text_muted)
            ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.65, y + h - 0.65], color=c_border_light, lw=0.7)
        else:
            ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.40, y + h - 0.40], color=c_border_light, lw=0.7)

    # =========================================================================
    # STAGE I: OPTICAL PHYSICAL FORWARD MODEL (x: 0.4 to 3.9)
    # =========================================================================
    draw_card(0.4, 0.6, 3.5, 5.3, "Physical Acquisition System", "SPAD Single-Pixel Bucket Detector")

    # 1. Projector & Patterns
    ax.text(0.6, 5.25, "1. Structured Illumination:", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(0.6, 5.00, r"$M=512$ Patterns, $\mathbf{A} \in \mathbb{R}^{M \times 64 \times 64}$", fontsize=8.6, color=c_text_muted)
    # Draw mini pattern representation
    np.random.seed(42)
    mini_pat = np.random.choice([0, 1], size=(8, 8), p=[0.5, 0.5])
    ax.imshow(mini_pat, cmap="binary", extent=[2.8, 3.6, 4.65, 5.45], zorder=3)
    ax.add_patch(patches.Rectangle((2.8, 4.65), 0.8, 0.8, fill=False, edgecolor=c_border, lw=1.0))

    # 2. Target Object / Conveyor
    ax.text(0.6, 4.25, "2. Unknown Defective Object:", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(0.6, 4.00, r"Fabric Scene $x \in \mathbb{R}^{64 \times 64}$, Line Speed $1\text{-}2$ m/s", fontsize=8.6, color=c_text_muted)
    # Draw mini fabric weave with a defect line
    mini_fabric = np.zeros((10, 10))
    mini_fabric[::2, :] = 0.8
    mini_fabric[:, ::2] = 0.5
    mini_fabric[4, 2:8] = 0.0 # defect thread
    ax.imshow(mini_fabric, cmap="gray", extent=[2.8, 3.6, 3.65, 4.45], zorder=3)
    ax.add_patch(patches.Rectangle((2.8, 3.65), 0.8, 0.8, fill=False, edgecolor=c_border, lw=1.0))

    # 3. Optical Lens & Defocus Phase
    ax.text(0.6, 3.25, "3. Lens Wavefront Aberration:", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(0.6, 3.00, r"Fresnel Defocus: $W(u,v) = a_4(2\rho^2-1)$", fontsize=8.6, color=c_text_muted)
    ax.text(0.6, 2.75, r"Vibration: $a_4 \in [0.0, 3.0]$ waves", fontsize=8.4, color=c_text_muted)
    # Draw lens arc
    lens_arc = patches.Arc((3.2, 2.95), 0.5, 0.9, theta1=270, theta2=90, edgecolor=c_border, lw=1.5)
    ax.add_patch(lens_arc)
    lens_arc2 = patches.Arc((3.2, 2.95), 0.5, 0.9, theta1=90, theta2=270, edgecolor=c_border, lw=1.5)
    ax.add_patch(lens_arc2)

    # 4. SPAD Sensor Bucket Readout
    ax.text(0.6, 2.15, "4. Single-Pixel SPAD Detection:", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(0.6, 1.85, r"$y_m = \langle A_m, x \rangle + \eta_m$,  $\eta \sim \mathrm{Poisson}(\lambda)$", fontsize=8.8, fontweight="bold", color=c_text_dark)
    ax.text(0.6, 1.60, r"Zero Read Noise, $\sim 10$ photons/pulse", fontsize=8.4, color=c_text_muted)
    # Draw bucket signal vector representation
    ax.add_patch(patches.Rectangle((0.6, 0.85), 3.1, 0.55, facecolor=c_sub_bg, edgecolor=c_border, lw=1.0))
    ax.text(2.15, 1.12, r"Measurement Vector $\mathbf{y} \in \mathbb{R}^{512}$ (12.5% Sampling)", 
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=c_text_dark)

    # Connect Stage I to Stage II
    arrow_i_ii = FancyArrowPatch((3.9, 1.12), (4.2, 1.12), arrowstyle="-|>", mutation_scale=14, color=c_border, lw=1.8)
    ax.add_patch(arrow_i_ii)
    arrow_i_ii_top = FancyArrowPatch((3.9, 1.12), (4.2, 4.40), connectionstyle="angle3,angleA=0,angleB=90", 
                                      arrowstyle="-|>", mutation_scale=14, color=c_border, lw=1.8)
    ax.add_patch(arrow_i_ii_top)

    # =========================================================================
    # STAGE II: CONTINUOUS FNO & IN-CONTEXT ENGINE (x: 4.2 to 8.0)
    # =========================================================================
    # Sub-block 1: Continuous FNO
    draw_card(4.2, 3.2, 3.8, 2.7, "Continuous 2D Fourier Neural Operator", "Guarantees Invariance to Lens Defocus")
    ax.text(4.4, 5.15, r"• Spectral Convolution in Continuous $\mathbb{R}^2$:", fontsize=8.8, fontweight="bold", color=c_text_dark)
    ax.text(4.6, 4.85, r"$\mathcal{K}_{\mathrm{opt}}(v)(x) = \mathcal{F}^{-1} \left( R_\phi \cdot \mathcal{F}(v) \right)(x)$", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(4.4, 4.45, r"• Truncation Kernel $R_\phi \in \mathbb{C}^{16 \times 16}$ Filters Evanescent Noise", fontsize=8.4, color=c_text_muted)
    ax.text(4.4, 4.15, r"• Defocus Invariance: $\Delta\mathrm{PSNR} < 10^{-5}$ dB across $3.0\lambda$ blur", fontsize=8.4, fontweight="bold", color=c_text_dark)
    ax.text(4.4, 3.85, r"• Output: Defocus-Free Operator Tokens $\mathbf{Z}_{\mathrm{FNO}} \in \mathbb{R}^{64 \times 256}$", fontsize=8.4, color=c_text_muted)

    # Sub-block 2: In-Context Prompt Engine
    draw_card(4.2, 0.6, 3.8, 2.3, "In-Context Prompt Adaptation Engine", "Zero-Gradient Test-Time Material Switching")
    ax.text(4.4, 2.25, r"• $K$-Shot Reference Pairs: $\{(y_k, x_k)\}_{k=1}^K$ ($K \in \{1, 3, 5\}$)", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(4.4, 1.95, r"• Online Material Adaptation: $\mathbf{7.2\text{ ms}}$ Latency ($\mathbf{138\text{ FPS}}$)", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(4.4, 1.65, r"• Zero Retraining: Model weights remain completely frozen", fontsize=8.4, color=c_text_muted)
    ax.text(4.4, 1.35, r"• Output: Fabric Condition Tokens $\mathbf{Z}_{\mathrm{Prompt}} \in \mathbb{R}^{K \times 256}$", fontsize=8.4, color=c_text_muted)

    # Connect Stage II to Stage III
    arrow_fno_pcca = FancyArrowPatch((8.0, 4.40), (8.3, 4.40), arrowstyle="-|>", mutation_scale=14, color=c_border, lw=1.8)
    ax.add_patch(arrow_fno_pcca)
    arrow_prompt_pcca = FancyArrowPatch((8.0, 1.70), (8.3, 3.90), connectionstyle="angle3,angleA=0,angleB=90", 
                                        arrowstyle="-|>", mutation_scale=14, color=c_border, lw=1.8)
    ax.add_patch(arrow_prompt_pcca)

    # =========================================================================
    # STAGE III: PCCA & MULTI-SCALE SWIN-UNET (x: 8.3 to 12.1)
    # =========================================================================
    draw_card(8.3, 0.6, 3.8, 5.3, "Physics-Conditioned Attention & Decoder", "PCCA Module + Hierarchical Swin-UNet")

    # PCCA Cross-Attention Block
    ax.add_patch(patches.Rectangle((8.5, 3.65), 3.4, 1.70, facecolor=c_sub_bg, edgecolor=c_border, lw=1.2))
    ax.text(10.2, 5.05, "Physics-Conditioned Cross-Attention (PCCA)", ha="center", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(8.7, 4.70, r"$\mathbf{Q} = \mathrm{Swin\ Feature\ Tokens} \in \mathbb{R}^{64 \times 256}$", fontsize=8.4, color=c_text_dark)
    ax.text(8.7, 4.40, r"$\mathbf{K}, \mathbf{V} = [\mathbf{Z}_{\mathrm{FNO}} \,\|\, \mathbf{Z}_{\mathrm{Prompt}}] \in \mathbb{R}^{(64+K) \times 256}$", fontsize=8.4, color=c_text_dark)
    ax.text(8.7, 4.10, r"$\mathrm{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}; a_4) = \mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top + \Phi(a_4)}{\sqrt{d}}\right)\mathbf{V}$", fontsize=8.2, fontweight="bold", color=c_text_dark)
    ax.text(8.7, 3.80, r"Defocus Vector $a_4$ actively steers attention receptive fields", fontsize=8.0, fontstyle="italic", color=c_text_muted)

    # Swin-UNet Spatial Decoder
    ax.add_patch(patches.Rectangle((8.5, 1.15), 3.4, 2.10, facecolor=c_sub_bg, edgecolor=c_border, lw=1.2))
    ax.text(10.2, 2.95, "Hierarchical Swin-UNet Spatial Decoder", ha="center", fontsize=9.2, fontweight="bold", color=c_text_dark)
    ax.text(8.7, 2.65, r"• Shifted Window Self-Attention (SW-MSA)", fontsize=8.4, color=c_text_dark)
    ax.text(8.7, 2.40, r"• Multi-Scale Feature Pyramid: $16 \to 32 \to 64\times 64$", fontsize=8.4, color=c_text_dark)
    ax.text(8.7, 2.15, r"• Captures yarn thread grain without spatial artifacts", fontsize=8.2, color=c_text_muted)
    ax.text(8.7, 1.90, r"• Flux-Normalized Poisson Likelihood Loss $\mathcal{L}_{\mathrm{Poisson}}$", fontsize=8.4, fontweight="bold", color=c_text_dark)
    ax.text(8.7, 1.65, r"  Prevents $\mathcal{O}(10^4)$ gradient explosion under photon scarcity", fontsize=8.0, color=c_text_muted)
    ax.text(8.7, 1.35, r"• Latent Dense Representation: $\mathbf{H} \in \mathbb{R}^{64 \times 64 \times 64}$", fontsize=8.4, color=c_text_muted)

    # Connect PCCA to Swin Decoder
    arrow_pcca_swin = FancyArrowPatch((10.2, 3.65), (10.2, 3.25), arrowstyle="-|>", mutation_scale=12, color=c_border, lw=1.5)
    ax.add_patch(arrow_pcca_swin)

    # Connect Stage III to Stage IV
    arrow_swin_nig = FancyArrowPatch((12.1, 2.20), (12.4, 2.20), arrowstyle="-|>", mutation_scale=14, color=c_border, lw=1.8)
    ax.add_patch(arrow_swin_nig)

    # =========================================================================
    # STAGE IV: EVIDENTIAL NIG HEAD & DECOMPOSITION (x: 12.4 to 15.6)
    # =========================================================================
    draw_card(12.4, 0.6, 3.2, 5.3, "Evidential Output Head", "Normal-Inverse-Gamma (NIG)")

    ax.text(12.6, 5.15, "Parameter Estimation (Single-Pass):", fontsize=8.8, fontweight="bold", color=c_text_dark)
    ax.text(12.6, 4.85, r"$\mu \sim \mathcal{N}(\gamma, \sigma^2/v), \quad \sigma^2 \sim \Gamma^{-1}(\alpha, \beta)$", fontsize=8.4, fontweight="bold", color=c_text_dark)
    ax.text(12.6, 4.55, r"Maps: $[\gamma, v, \alpha, \beta] \in \mathbb{R}^{4 \times 64 \times 64}$", fontsize=8.2, color=c_text_muted)

    # Output Card 1: Mean Reconstruction
    ax.add_patch(patches.Rectangle((12.6, 3.40), 2.8, 0.85, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0))
    ax.text(14.0, 3.95, r"1. Reconstructed Image $\hat{x}$", ha="center", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(14.0, 3.65, r"Mean intensity: $\hat{x} = \gamma \in [0, 1]$", ha="center", fontsize=8.2, color=c_text_dark)

    # Output Card 2: Aleatoric Uncertainty
    ax.add_patch(patches.Rectangle((12.6, 2.25), 2.8, 0.95, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0))
    ax.text(14.0, 2.90, r"2. Aleatoric Uncertainty $\sigma^2_a$", ha="center", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(14.0, 2.65, r"$\sigma^2_a = \beta / (\alpha - 1)$", ha="center", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(14.0, 2.40, r"Captures photon shot noise", ha="center", fontsize=8.0, color=c_text_muted)

    # Output Card 3: Epistemic Uncertainty
    ax.add_patch(patches.Rectangle((12.6, 1.05), 2.8, 1.00, facecolor="#FFFFFF", edgecolor=c_border, lw=1.2))
    ax.text(14.0, 1.75, r"3. Epistemic Uncertainty $\sigma^2_e$", ha="center", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(14.0, 1.50, r"$\sigma^2_e = \beta / [v(\alpha - 1)]$", ha="center", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(14.0, 1.25, r"Isolates structural defects", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark)

    # Bottom Timeline Bar
    ax.add_patch(patches.Rectangle((0.4, 0.15), 15.2, 0.32, facecolor=c_sub_bg, edgecolor=c_border, lw=1.0))
    ax.text(8.0, 0.31, "Total End-to-End Latency: 7.2 ms (138 FPS) | Zero Gradient Updates at Test Time | Deterministic Single-Pass Evidential Inference",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=c_text_dark)

    plt.tight_layout()
    out_png = "figures/architecture_diagram.png"
    out_pdf = "figures/architecture_diagram.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated publication-grade architecture diagram:\n  - {out_png}\n  - {out_pdf}")

if __name__ == "__main__":
    generate_pro_architecture_diagram()
