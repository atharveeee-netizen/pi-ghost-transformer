import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    os.makedirs("figures", exist_ok=True)
    
    # Dimensions chosen for optimal readability at full page width in report
    fig, ax = plt.subplots(figsize=(14.0, 6.4), dpi=300)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    # Academic Monochrome / Grayscale Palette
    c_white = "#FFFFFF"
    c_card_bg = "#F8F9FA"
    c_card_highlight = "#F1F3F5"
    c_black = "#111111"
    c_dark_gray = "#2B2F36"
    c_gray = "#495057"
    c_border = "#212529"
    c_arrow = "#212529"

    # Header / Title Block
    ax.text(7.0, 6.05, "IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", 
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=c_black)
    ax.text(7.0, 5.68, "End-to-End System Pipeline: Wave Optics Forward Model -> Continuous Operator -> Evidential NIG Head", 
            ha="center", va="center", fontsize=10.5, fontstyle="italic", color=c_gray)

    # Box Drawing Function
    def draw_card(x, y, w, h, title, lines, highlight=False, radius=0.12):
        bg = c_card_highlight if highlight else c_card_bg
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=patches.BoxStyle("Round", pad=0.06, rounding_size=radius),
            facecolor=bg, edgecolor=c_border, linewidth=1.6
        )
        ax.add_patch(box)
        
        # Header banner inside box
        header_y = y + h - 0.35
        ax.text(x + w/2, header_y, title, ha="center", va="center", fontsize=12.0, fontweight="bold", color=c_black)
        
        # Divider line
        ax.plot([x + 0.15, x + w - 0.15], [header_y - 0.20, header_y - 0.20], color=c_border, lw=0.9)
        
        # Line text
        start_y = header_y - 0.42
        line_spacing = (h - 0.65) / max(len(lines), 1)
        for i, l in enumerate(lines):
            ax.text(x + w/2, start_y - i * line_spacing, l, ha="center", va="center", 
                    fontsize=10.2, color=c_dark_gray)

    # Arrow Drawing Function
    def draw_conn(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.8, mutation_scale=15)
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.14, label, ha="center", va="bottom", fontsize=10.2, fontweight="bold", color=c_black)

    # -------------------------------------------------------------
    # STAGE 1: PHYSICAL STAGE (x: 0.4 to 2.8)
    # -------------------------------------------------------------
    draw_card(0.4, 3.8, 2.4, 1.5, "Pattern Illumination", [
        "M = 512 Patterns",
        r"$\mathbf{A} \in \mathbb{R}^{M \times H \times W}$",
        "Learned / Hadamard Basis"
    ])

    draw_card(0.4, 2.1, 2.4, 1.5, "Target Fabric", [
        r"Unknown Object $x$",
        r"$x \in \mathbb{R}^{64 \times 64}$",
        "Woven Micro-Defects"
    ])

    draw_card(0.4, 0.4, 2.4, 1.5, "Lens Defocus", [
        "Machinery Vibration",
        r"Fresnel Phase: $h(a_4)$",
        r"Aberration: $0 \leq a_4 \leq 3.0$"
    ])

    # -------------------------------------------------------------
    # STAGE 2: SINGLE-PIXEL SPAD (x: 3.4 to 5.0)
    # -------------------------------------------------------------
    draw_card(3.4, 1.7, 1.7, 2.3, "SPAD Sensor", [
        "Single-Pixel Bucket",
        r"$\mathbf{y} \in \mathbb{R}^M$ Scalar Flux",
        r"$y_m = \langle A_m, x \rangle + \eta$",
        r"$\eta \sim \mathrm{Poisson}(\lambda)$",
        "Zero Read Noise"
    ], highlight=True)

    draw_conn(2.8, 4.5, 3.4, 3.2)
    draw_conn(2.8, 2.8, 3.4, 2.8)
    draw_conn(2.8, 1.1, 3.4, 2.4)

    # -------------------------------------------------------------
    # STAGE 3: OPERATOR & CONTEXT TOKENS (x: 5.6 to 8.0)
    # -------------------------------------------------------------
    draw_card(5.6, 2.95, 2.4, 1.8, "Continuous 2D FNO", [
        "Spectral Kernel " + r"$R_\phi$",
        r"$\mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(v))$",
        "Truncates Evanescent Modes",
        "Defocus Invariance Guarn."
    ])

    draw_card(5.6, 0.75, 2.4, 1.8, "In-Context Engine", [
        "K Exemplars " + r"$(y_k, x_k)$",
        "Zero-Gradient Adaptation",
        "7.2 ms Online Inference",
        "Instant Fabric Switching"
    ])

    draw_conn(5.1, 2.8, 5.6, 3.85, label="Bucket y")
    draw_conn(5.1, 2.8, 5.6, 1.65, label="Bucket y")

    # -------------------------------------------------------------
    # STAGE 4: PCCA & SPATIAL DECODER (x: 8.6 to 10.9)
    # -------------------------------------------------------------
    draw_card(8.6, 2.7, 2.3, 2.0, "PCCA Module", [
        "Physics-Conditioned",
        "Cross-Attention",
        r"$Q = \mathrm{Swin\ Tokens}$",
        r"$K, V = [\mathrm{FNO}, \mathrm{Prompt}]$",
        "Modulated by Defocus " + r"$a_4$"
    ], highlight=True)

    draw_conn(8.0, 3.85, 8.6, 3.9)
    draw_conn(8.0, 1.65, 8.6, 3.4)

    draw_card(8.6, 0.5, 2.3, 1.8, "Swin-UNet Decoder", [
        "Shifted Window Attn.",
        "Multi-Scale Hierarchy",
        r"Latent Tokens $\to$",
        r"Spatial $64 \times 64$ Fabric"
    ])

    draw_conn(9.75, 2.7, 9.75, 2.3)

    # -------------------------------------------------------------
    # STAGE 5: EVIDENTIAL NIG HEAD & DECOMPOSITION (x: 11.4 to 13.7)
    # -------------------------------------------------------------
    draw_card(11.4, 1.4, 2.3, 3.3, "Evidential NIG Head", [
        "Normal-Inverse-Gamma",
        r"Mean Recon: $\gamma \in \mathbb{R}^{64 \times 64}$",
        r"Evidence: $v > 0$",
        r"Shape: $\alpha > 1$, Scale: $\beta > 0$",
        "--------------------",
        r"Clean Image: $\hat{x} = \gamma$",
        r"Shot Noise: $\sigma^2_a = \frac{\beta}{\alpha - 1}$",
        r"Defect Map: $\sigma^2_e = \frac{\beta}{v(\alpha - 1)}$"
    ], highlight=True)

    draw_conn(10.9, 1.4, 11.4, 2.8)

    # Bottom Legend / Indicator
    ax.text(7.0, 0.12, "[ Factory Floor Pipeline: Physical Optics Model -> Continuous Operator -> Few-Shot Prompt -> Evidential Uncertainty (138 FPS) ]", 
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=c_dark_gray)

    plt.tight_layout()
    png_path = "figures/architecture_diagram.png"
    pdf_path = "figures/architecture_diagram.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"High-legibility architecture diagram generated:\n  - {png_path}\n  - {pdf_path}")

if __name__ == "__main__":
    draw_architecture_diagram()
