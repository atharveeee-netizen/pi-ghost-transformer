import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

def draw_isometric_cube(ax, x, y, dx, dy, dz, face_color="#FFFFFF", edge_color="#111111", lw=1.2, alpha=1.0):
    """Draws a clean 2.5D isometric tensor block with front, top, and side faces."""
    # Front face
    front = Polygon([
        [x, y], [x + dx, y], [x + dx, y + dz], [x, y + dz]
    ], closed=True, facecolor=face_color, edgecolor=edge_color, linewidth=lw, alpha=alpha, zorder=4)
    ax.add_patch(front)

    # Top face
    top = Polygon([
        [x, y + dz], [x + dx, y + dz], [x + dx + dy*0.5, y + dz + dy*0.3], [x + dy*0.5, y + dz + dy*0.3]
    ], closed=True, facecolor="#F3F4F6", edgecolor=edge_color, linewidth=lw, alpha=alpha, zorder=4)
    ax.add_patch(top)

    # Side face
    side = Polygon([
        [x + dx, y], [x + dx + dy*0.5, y + dy*0.3], [x + dx + dy*0.5, y + dz + dy*0.3], [x + dx, y + dz]
    ], closed=True, facecolor="#E5E7EB", edgecolor=edge_color, linewidth=lw, alpha=alpha, zorder=4)
    ax.add_patch(side)

def generate_ieee_architecture_diagram():
    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(16.5, 7.5), dpi=300)
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    c_border = "#111827"
    c_text_dark = "#111827"
    c_text_muted = "#374151"
    c_box_bg = "#FAFAFA"
    c_inner_bg = "#FFFFFF"
    c_highlight = "#F3F4F6"

    # Main Diagram Header (Classic IEEE Journal Caption Style)
    ax.text(8.25, 7.18, "IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", 
            ha="center", va="center", fontsize=15.0, fontweight="bold", color=c_text_dark)
    ax.text(8.25, 6.85, "Complete Deep Architecture: Continuous Wavefield Operator → In-Context Prompt Engine → PCCA → Swin-UNet → Evidential NIG Head", 
            ha="center", va="center", fontsize=9.8, fontstyle="italic", color=c_text_muted)

    def draw_container(x, y, w, h, title, subtitle=""):
        box = FancyBboxPatch((x, y), w, h, boxstyle="Round,pad=0.04,rounding_size=0.15",
                             facecolor=c_box_bg, edgecolor=c_border, linewidth=1.4, zorder=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.25, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color=c_text_dark, zorder=3)
        if subtitle:
            ax.text(x + w/2, y + h - 0.48, subtitle, ha="center", va="center", fontsize=8.4, fontstyle="italic", color=c_text_muted, zorder=3)
            ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.62, y + h - 0.62], color=c_border, lw=0.6, zorder=3)
        else:
            ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.38, y + h - 0.38], color=c_border, lw=0.6, zorder=3)

    # =========================================================================
    # 1. OPTICAL MEASUREMENT & PROJECTION (x: 0.4 to 2.8)
    # =========================================================================
    draw_container(0.4, 0.6, 2.4, 6.0, "1. Compressive Input", "Single-Pixel SPAD")

    # SPAD 1D Vector
    ax.text(1.6, 5.75, "SPAD Measurement", ha="center", fontsize=9.0, fontweight="bold", color=c_text_dark)
    ax.text(1.6, 5.50, r"$\mathbf{y} \in \mathbb{R}^{512}$ (12.5% M/N²)", ha="center", fontsize=8.2, color=c_text_muted)
    
    # Draw vertical vector bar
    ax.add_patch(patches.Rectangle((1.35, 3.8), 0.5, 1.4, facecolor="#F3F4F6", edgecolor=c_border, lw=1.2, zorder=4))
    for iy in np.linspace(3.9, 5.0, 6):
        ax.plot([1.35, 1.85], [iy, iy], color="#9CA3AF", lw=0.8, zorder=5)
    ax.text(1.6, 4.5, "...", ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)

    # Linear Embedding Layer
    ax.annotate("", xy=(1.6, 3.2), xytext=(1.6, 3.8),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.5, mutation_scale=12), zorder=4)
    ax.add_patch(patches.Rectangle((0.6, 2.7), 2.0, 0.5, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(1.6, 2.95, "Linear Projection $W_p$\n+ Patch Reshape", ha="center", va="center", fontsize=7.8, fontweight="bold", zorder=5)

    # Output Tensor X_0 (Isometric Cube)
    draw_isometric_cube(ax, 1.0, 1.2, 0.9, 0.5, 0.9, face_color="#FFFFFF")
    ax.text(1.5, 0.85, r"$\mathbf{X}_0 \in \mathbb{R}^{64 \times 16 \times 16}$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark)

    # Connect to FNO and Prompt Engine
    arrow_out1 = FancyArrowPatch((2.2, 1.8), (3.1, 4.6), connectionstyle="angle3,angleA=0,angleB=90",
                                 arrowstyle="-|>", mutation_scale=12, color=c_border, lw=1.5, zorder=5)
    ax.add_patch(arrow_out1)
    arrow_out2 = FancyArrowPatch((2.2, 1.8), (3.1, 1.8), arrowstyle="-|>", mutation_scale=12, color=c_border, lw=1.5, zorder=5)
    ax.add_patch(arrow_out2)

    # =========================================================================
    # 2. CONTINUOUS 2D FNO OPERATOR (x: 3.1 to 6.3, Top)
    # =========================================================================
    draw_container(3.1, 3.4, 3.2, 3.2, "2. Continuous 2D FNO", "Optical Wavefield Modeling")

    # Spectral Path Box
    ax.add_patch(patches.Rectangle((3.3, 4.6), 2.8, 1.3, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(4.7, 5.65, "Spectral Fourier Kernel Path", ha="center", fontsize=8.4, fontweight="bold", color=c_text_dark, zorder=5)
    
    # 2D FFT Block -> Weight R -> 2D IFFT Block
    ax.add_patch(patches.Rectangle((3.45, 4.8), 0.7, 0.6, facecolor="#F3F4F6", edgecolor=c_border, lw=1.0, zorder=5))
    ax.text(3.8, 5.1, "2D FFT\n" + r"$\mathcal{F}$", ha="center", va="center", fontsize=7.8, fontweight="bold", zorder=6)
    
    ax.annotate("", xy=(4.45, 5.1), xytext=(4.15, 5.1),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.2, mutation_scale=10), zorder=5)
    
    ax.add_patch(patches.Rectangle((4.45, 4.8), 0.7, 0.6, facecolor="#E5E7EB", edgecolor=c_border, lw=1.2, zorder=5))
    ax.text(4.8, 5.1, "Kernel\n" + r"$R_\phi \in \mathbb{C}$", ha="center", va="center", fontsize=7.8, fontweight="bold", zorder=6)

    ax.annotate("", xy=(5.45, 5.1), xytext=(5.15, 5.1),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.2, mutation_scale=10), zorder=5)

    ax.add_patch(patches.Rectangle((5.45, 4.8), 0.55, 0.6, facecolor="#F3F4F6", edgecolor=c_border, lw=1.0, zorder=5))
    ax.text(5.72, 5.1, "2D IFFT\n" + r"$\mathcal{F}^{-1}$", ha="center", va="center", fontsize=7.5, fontweight="bold", zorder=6)

    # Local Bypass Path + Addition
    ax.plot([3.2, 3.2, 4.7], [4.6, 3.9, 3.9], color=c_border, lw=1.2, zorder=5)
    ax.add_patch(patches.Rectangle((3.6, 3.75), 1.2, 0.35, facecolor="#FFFFFF", edgecolor=c_border, lw=0.8, zorder=5))
    ax.text(4.2, 3.92, r"Conv 1×1 $W_{\mathrm{local}}$", ha="center", va="center", fontsize=7.4, zorder=6)
    
    # Circle Addition (Cross)
    ax.add_patch(patches.Circle((5.72, 3.9), 0.18, facecolor="#FFFFFF", edgecolor=c_border, lw=1.2, zorder=6))
    ax.text(5.72, 3.9, "+", ha="center", va="center", fontsize=9.0, fontweight="bold", zorder=7)
    
    ax.plot([5.72, 5.72], [4.8, 4.08], color=c_border, lw=1.2, zorder=5)
    
    # Output Token FNO
    ax.text(4.7, 3.52, r"Defocus-Invariant Tokens $\mathbf{Z}_{\mathrm{FNO}} \in \mathbb{R}^{64 \times 256}$", 
            ha="center", fontsize=8.0, fontweight="bold", color=c_text_dark, zorder=5)

    # =========================================================================
    # 3. IN-CONTEXT PROMPT ENGINE (x: 3.1 to 6.3, Bottom)
    # =========================================================================
    draw_container(3.1, 0.6, 3.2, 2.5, "3. In-Context Engine", "Few-Shot Test-Time Adaptation")
    
    ax.text(4.7, 2.55, r"Exemplars: $\{(y_k, x_k)\}_{k=1}^K$", ha="center", fontsize=8.6, fontweight="bold", color=c_text_dark)
    ax.text(4.7, 2.30, "Zero Retraining / Zero Gradients (7.2 ms)", ha="center", fontsize=7.8, color=c_text_muted)
    
    # Prompt Encoder Box
    ax.add_patch(patches.Rectangle((3.4, 1.4), 2.6, 0.65, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(4.7, 1.72, "Dual-Stream Prompt Encoder\n[Conv1D(y) ∥ Conv2D(x)]", ha="center", va="center", fontsize=7.8, fontweight="bold", zorder=5)
    
    ax.text(4.7, 0.95, r"Condition Tokens $\mathbf{Z}_{\mathrm{Prompt}} \in \mathbb{R}^{K \times 256}$", 
            ha="center", fontsize=8.0, fontweight="bold", color=c_text_dark)

    # =========================================================================
    # 4. PHYSICS-CONDITIONED CROSS-ATTENTION (x: 6.6 to 9.8)
    # =========================================================================
    draw_container(6.6, 0.6, 3.2, 6.0, "4. PCCA Cross-Attention", "Optical State Modulation")

    # Zernike Input Node
    ax.add_patch(patches.FancyBboxPatch((7.1, 5.7), 2.2, 0.55, boxstyle="Round,pad=0.02,rounding_size=0.1",
                                        facecolor="#E5E7EB", edgecolor=c_border, lw=1.2, zorder=4))
    ax.text(8.2, 5.97, r"Optical Defocus State $a_4 \in \mathbb{R}$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(8.2, 5.78, "Fresnel Aberration Wavefront", ha="center", fontsize=7.2, color=c_text_muted, zorder=5)

    # Connect FNO & Prompt into Keys/Values
    arrow_fno_p = FancyArrowPatch((6.3, 3.9), (7.0, 4.5), connectionstyle="angle3,angleA=0,angleB=90",
                                  arrowstyle="-|>", mutation_scale=12, color=c_border, lw=1.5, zorder=5)
    ax.add_patch(arrow_fno_p)
    arrow_pr_p = FancyArrowPatch((6.3, 1.7), (7.0, 3.8), connectionstyle="angle3,angleA=0,angleB=90",
                                 arrowstyle="-|>", mutation_scale=12, color=c_border, lw=1.5, zorder=5)
    ax.add_patch(arrow_pr_p)

    # Multi-Head Attention Detailed Schematic
    ax.add_patch(patches.Rectangle((6.8, 1.8), 2.8, 3.6, facecolor="#FFFFFF", edgecolor=c_border, lw=1.2, zorder=4))
    ax.text(8.2, 5.15, "Cross-Attention Matrix", ha="center", fontsize=8.8, fontweight="bold", color=c_text_dark, zorder=5)

    # Q, K, V Projections
    ax.add_patch(patches.Rectangle((7.0, 4.4), 0.65, 0.45, facecolor="#F3F4F6", edgecolor=c_border, lw=0.9, zorder=5))
    ax.text(7.32, 4.62, "Q (Swin)", ha="center", va="center", fontsize=7.2, fontweight="bold", zorder=6)

    ax.add_patch(patches.Rectangle((7.85, 4.4), 0.75, 0.45, facecolor="#F3F4F6", edgecolor=c_border, lw=0.9, zorder=5))
    ax.text(8.22, 4.62, r"K [FNO∥P]", ha="center", va="center", fontsize=7.0, fontweight="bold", zorder=6)

    ax.add_patch(patches.Rectangle((8.75, 4.4), 0.70, 0.45, facecolor="#F3F4F6", edgecolor=c_border, lw=0.9, zorder=5))
    ax.text(9.10, 4.62, r"V [FNO∥P]", ha="center", va="center", fontsize=7.0, fontweight="bold", zorder=6)

    # Defocus Bias Modulation
    ax.annotate("", xy=(8.2, 4.0), xytext=(8.2, 5.7),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.4, mutation_scale=10), zorder=5)
    ax.text(8.25, 4.15, r"+ $\Psi(a_4)$", ha="left", fontsize=8.4, fontweight="bold", color=c_text_dark, zorder=6)

    # Scaled Dot-Product & Softmax Box
    ax.add_patch(patches.Rectangle((7.0, 3.1), 2.4, 0.75, facecolor="#E5E7EB", edgecolor=c_border, lw=1.0, zorder=5))
    ax.text(8.2, 3.55, r"$\mathrm{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top + \Psi(a_4)}{\sqrt{d}}\right)$", ha="center", fontsize=7.8, fontweight="bold", zorder=6)
    ax.text(8.2, 3.25, "Dynamic Wavefront Steered Attn.", ha="center", fontsize=6.8, fontstyle="italic", color=c_text_muted, zorder=6)

    # MatMul with Value
    ax.annotate("", xy=(8.2, 2.65), xytext=(8.2, 3.1),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.2, mutation_scale=10), zorder=5)
    ax.add_patch(patches.Circle((8.2, 2.5), 0.15, facecolor="#FFFFFF", edgecolor=c_border, lw=1.2, zorder=5))
    ax.text(8.2, 2.5, r"$\times$", ha="center", va="center", fontsize=9.0, fontweight="bold", zorder=6)

    # LayerNorm & Projection
    ax.annotate("", xy=(8.2, 2.05), xytext=(8.2, 2.35),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.2, mutation_scale=10), zorder=5)
    ax.text(8.2, 1.95, "LayerNorm + MLP Residual", ha="center", fontsize=7.4, color=c_text_muted, zorder=5)

    ax.text(8.2, 1.0, r"Fused Tokens $\mathbf{H}_{\mathrm{attn}} \in \mathbb{R}^{64 \times 256}$", 
            ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark)

    # =========================================================================
    # 5. HIERARCHICAL SWIN-UNET SPATIAL DECODER (x: 10.1 to 13.0)
    # =========================================================================
    draw_container(10.1, 0.6, 2.9, 6.0, "5. Swin-UNet Decoder", "Multi-Scale Hierarchy")

    # Stage 1 (16x16)
    ax.add_patch(patches.Rectangle((10.3, 4.6), 2.5, 1.25, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(11.55, 5.60, "Stage 1: Bottleneck", ha="center", fontsize=8.4, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(11.55, 5.35, "Shifted Window MSA (SW-MSA)", ha="center", fontsize=7.4, color=c_text_muted, zorder=5)
    draw_isometric_cube(ax, 11.1, 4.75, 0.7, 0.35, 0.45, face_color="#FFFFFF")
    ax.text(11.55, 4.65, r"$16 \times 16 \times 256$", ha="center", fontsize=7.4, fontweight="bold", zorder=5)

    # Arrow to Stage 2 with Patch Expanding
    ax.annotate("", xy=(11.55, 4.15), xytext=(11.55, 4.6),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.2, mutation_scale=10), zorder=4)
    ax.text(12.35, 4.35, "Patch Expand 2×", ha="center", fontsize=7.0, color=c_text_muted, zorder=5)

    # Stage 2 (32x32)
    ax.add_patch(patches.Rectangle((10.3, 2.8), 2.5, 1.25, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(11.55, 3.80, "Stage 2: Mid-Scale Expand", ha="center", fontsize=8.4, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(11.55, 3.55, "Window Multi-Head Attn.", ha="center", fontsize=7.4, color=c_text_muted, zorder=5)
    draw_isometric_cube(ax, 11.0, 2.95, 0.9, 0.45, 0.45, face_color="#FFFFFF")
    ax.text(11.55, 2.85, r"$32 \times 32 \times 128$", ha="center", fontsize=7.4, fontweight="bold", zorder=5)

    # Arrow to Stage 3
    ax.annotate("", xy=(11.55, 2.35), xytext=(11.55, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=c_border, lw=1.2, mutation_scale=10), zorder=4)
    ax.text(12.35, 2.55, "Patch Expand 2×", ha="center", fontsize=7.0, color=c_text_muted, zorder=5)

    # Stage 3 (64x64)
    ax.add_patch(patches.Rectangle((10.3, 1.0), 2.5, 1.25, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(11.55, 2.00, "Stage 3: Full Resolution", ha="center", fontsize=8.4, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(11.55, 1.75, r"High-Res Feature Map $\mathbf{H}$", ha="center", fontsize=7.4, color=c_text_muted, zorder=5)
    draw_isometric_cube(ax, 10.9, 1.15, 1.1, 0.55, 0.45, face_color="#FFFFFF")
    ax.text(11.55, 1.05, r"$64 \times 64 \times 64$", ha="center", fontsize=7.4, fontweight="bold", zorder=5)

    # =========================================================================
    # 6. EVIDENTIAL NIG PARAMETER HEAD & OUTPUTS (x: 13.3 to 16.1)
    # =========================================================================
    draw_container(13.3, 0.6, 2.8, 6.0, "6. Evidential NIG Head", "Normal-Inverse-Gamma")

    # Connect Swin Decoder to NIG Head
    arrow_swin_head = FancyArrowPatch((13.0, 1.6), (13.3, 5.0), connectionstyle="angle3,angleA=0,angleB=90",
                                       arrowstyle="-|>", mutation_scale=12, color=c_border, lw=1.5, zorder=5)
    ax.add_patch(arrow_swin_head)

    # 4 Parameter Splitting Box
    ax.add_patch(patches.Rectangle((13.5, 4.6), 2.4, 1.25, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(14.7, 5.60, "Conv 1×1 Parameter Heads", ha="center", fontsize=8.0, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 5.35, r"$\gamma = \mathrm{Sigmoid}(W_\gamma * H)$", ha="center", fontsize=7.2, color=c_text_dark, zorder=5)
    ax.text(14.7, 5.15, r"$v = \mathrm{Softplus}(W_v * H) > 0$", ha="center", fontsize=7.2, color=c_text_dark, zorder=5)
    ax.text(14.7, 4.95, r"$\alpha = \mathrm{Softplus}(W_\alpha * H) + 1 > 1$", ha="center", fontsize=7.2, color=c_text_dark, zorder=5)
    ax.text(14.7, 4.75, r"$\beta = \mathrm{Softplus}(W_\beta * H) > 0$", ha="center", fontsize=7.2, color=c_text_dark, zorder=5)

    # 3 Distinct Output Cards
    # Card 1: Recon
    ax.add_patch(patches.Rectangle((13.5, 3.4), 2.4, 0.95, facecolor="#F3F4F6", edgecolor=c_border, lw=1.2, zorder=4))
    ax.text(14.7, 4.15, r"1. Reconstructed Image $\hat{x}$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 3.88, r"$\hat{x} = \gamma \in \mathbb{R}^{64 \times 64}$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 3.55, "High-fidelity texture recovery", ha="center", fontsize=7.0, color=c_text_muted, zorder=5)

    # Card 2: Aleatoric
    ax.add_patch(patches.Rectangle((13.5, 2.2), 2.4, 0.95, facecolor="#FFFFFF", edgecolor=c_border, lw=1.0, zorder=4))
    ax.text(14.7, 2.95, r"2. Aleatoric Noise $\sigma^2_a$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 2.68, r"$\sigma^2_a = \beta / (\alpha - 1)$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 2.35, "Captures photon shot noise", ha="center", fontsize=7.0, color=c_text_muted, zorder=5)

    # Card 3: Epistemic
    ax.add_patch(patches.Rectangle((13.5, 1.0), 2.4, 0.95, facecolor="#E5E7EB", edgecolor=c_border, lw=1.4, zorder=4))
    ax.text(14.7, 1.75, r"3. Epistemic Novelty $\sigma^2_e$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 1.48, r"$\sigma^2_e = \beta / [v(\alpha - 1)]$", ha="center", fontsize=8.2, fontweight="bold", color=c_text_dark, zorder=5)
    ax.text(14.7, 1.15, "Isolates structural defects", ha="center", fontsize=7.2, fontweight="bold", color=c_text_dark, zorder=5)

    # Bottom Summary Bar
    ax.add_patch(patches.Rectangle((0.4, 0.12), 15.7, 0.35, facecolor="#F3F4F6", edgecolor=c_border, lw=1.0, zorder=3))
    ax.text(8.25, 0.29, "Single Deterministic Forward Pass (O(1) Latency, 138 FPS) | Zero Gradient Updates at Test Time | Continuous Wavefield Modeling",
            ha="center", va="center", fontsize=8.6, fontweight="bold", color=c_text_dark, zorder=4)

    plt.tight_layout()
    out_png = "figures/architecture_diagram.png"
    out_pdf = "figures/architecture_diagram.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated IEEE architectural schematic:\n  - {out_png}\n  - {out_pdf}")

if __name__ == "__main__":
    generate_ieee_architecture_diagram()
