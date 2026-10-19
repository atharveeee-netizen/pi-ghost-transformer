import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    os.makedirs("figures", exist_ok=True)
    
    # Dimensions chosen for optimal readability at full page width in report
    fig, ax = plt.subplots(figsize=(13.5, 6.2), dpi=300)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.2)
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
    ax.text(6.75, 5.85, "IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", 
            ha="center", va="center", fontsize=14, fontweight="bold", color=c_black)
    ax.text(6.75, 5.50, "End-to-End Architectural Pipeline: Optics Forward Model -> Continuous Operator -> Evidential NIG Head", 
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color=c_gray)

    # Box Drawing Function
    def draw_card(x, y, w, h, title, lines, highlight=False, radius=0.12):
        bg = c_card_highlight if highlight else c_card_bg
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=patches.BoxStyle("Round", pad=0.06, rounding_size=radius),
            facecolor=bg, edgecolor=c_border, linewidth=1.5
        )
        ax.add_patch(box)
        
        # Header banner inside box
        header_y = y + h - 0.32
        ax.text(x + w/2, header_y, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=c_black)
        
        # Divider line
        ax.plot([x + 0.15, x + w - 0.15], [header_y - 0.18, header_y - 0.18], color=c_border, lw=0.8)
        
        # Line text
        start_y = header_y - 0.38
        line_spacing = (h - 0.60) / max(len(lines), 1)
        for i, l in enumerate(lines):
            ax.text(x + w/2, start_y - i * line_spacing, l, ha="center", va="center", 
                    fontsize=8.8, color=c_dark_gray)

    # Arrow Drawing Function
    def draw_conn(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=c_arrow, lw=1.6, mutation_scale=14)
        )
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mx, my + 0.12, label, ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=c_black)

    # -------------------------------------------------------------
    # STAGE 1: PHYSICAL STAGE (x: 0.4 to 2.8)
    # -------------------------------------------------------------
    draw_card(0.4, 3.7, 2.3, 1.45, "Structured Light", [
        "M = 512 Patterns",
        r"$\mathbf{A} \in \mathbb{R}^{M \times H \times W}$",
        "Learned / Hadamard Basis"
    ])

    draw_card(0.4, 2.05, 2.3, 1.45, "Defective Target", [
        r"Unknown Fabric $x$",
        r"$x \in \mathbb{R}^{64 \times 64}$",
        "Woven Micro-Anomalies"
    ])

    draw_card(0.4, 0.4, 2.3, 1.45, "Differentiable Optics", [
        "Lens Defocus Phase",
        r"Fresnel PSF $h(a_4)$",
        r"Aberration: $0 \leq a_4 \leq 3.0$"
    ])

    # -------------------------------------------------------------
    # STAGE 2: SINGLE-PIXEL SPAD (x: 3.2 to 4.7)
    # -------------------------------------------------------------
    draw_card(3.2, 1.7, 1.6, 2.2, "SPAD Sensor", [
        "Single-Pixel Bucket",
        r"$\mathbf{y} \in \mathbb{R}^M$ Scalar Flux",
        r"$y_m = \langle A_m, x \rangle + \eta$",
        r"$\eta \sim \mathrm{Poisson}(\lambda)$",
        "DCR + Jitter Noise"
    ], highlight=True)

    draw_conn(2.7, 4.4, 3.2, 3.1)
    draw_conn(2.7, 2.8, 3.2, 2.8)
    draw_conn(2.7, 1.1, 3.2, 2.5)

    # -------------------------------------------------------------
    # STAGE 3: OPERATOR & CONTEXT TOKENS (x: 5.3 to 7.6)
    # -------------------------------------------------------------
    draw_card(5.3, 2.9, 2.3, 1.7, "Continuous 2D FNO", [
        "Spectral Kernel " + r"$R_\phi$",
        r"$\mathcal{F}^{-1}(R_\phi \cdot \mathcal{F}(v))$",
        "Truncates Evanescent Modes",
        "Defocus Invariance Guarn."
    ])

    draw_card(5.3, 0.8, 2.3, 1.7, "In-Context Engine", [
        "K Exemplars " + r"$(y_k, x_k)$",
        "Zero-Gradient Adaptation",
        "7.2 ms Online Inference",
        "Material Domain Shifts"
    ])

    draw_conn(4.8, 2.8, 5.3, 3.75, label="Bucket y")
    draw_conn(4.8, 2.8, 5.3, 1.65, label="Bucket y")

    # -------------------------------------------------------------
    # STAGE 4: PCCA & SPATIAL DECODER (x: 8.2 to 10.4)
    # -------------------------------------------------------------
    draw_card(8.2, 2.7, 2.2, 1.9, "PCCA Module", [
        "Physics-Conditioned",
        "Cross-Attention",
        r"$Q = \mathrm{Swin\ Tokens}$",
        r"$K, V = [\mathrm{FNO}, \mathrm{Prompt}]$",
        "Conditioned on " + r"$a_4$"
    ], highlight=True)

    draw_conn(7.6, 3.75, 8.2, 3.8)
    draw_conn(7.6, 1.65, 8.2, 3.4)

    draw_card(8.2, 0.5, 2.2, 1.7, "Swin-UNet Decoder", [
        "Shifted Window Attn.",
        "Multi-Scale Hierarchy",
        r"64 Latent Tokens $\to$",
        r"Spatial $64 \times 64$ Resolution"
    ])

    draw_conn(9.3, 2.7, 9.3, 2.2)

    # -------------------------------------------------------------
    # STAGE 5: EVIDENTIAL NIG HEAD & DECOMPOSITION (x: 10.9 to 13.1)
    # -------------------------------------------------------------
    draw_card(10.9, 1.5, 2.2, 3.1, "Evidential NIG Head", [
        "Normal-Inverse-Gamma",
        r"Mean Recon: $\gamma \in \mathbb{R}^{64 \times 64}$",
        r"Evidence: $v > 0$",
        r"Shape: $\alpha > 1$",
        r"Scale: $\beta > 0$",
        "--------------------",
        r"Mean: $\hat{x} = \gamma$",
        r"Aleatoric: $\sigma^2_a = \frac{\beta}{\alpha - 1}$",
        r"Epistemic: $\sigma^2_e = \frac{\beta}{v(\alpha - 1)}$"
    ], highlight=True)

    draw_conn(10.4, 1.35, 10.9, 2.8)

    # Bottom Legend / Indicator
    ax.text(6.75, 0.12, "[ Industrial Pipeline: Physical Forward Model -> Continuous Operator -> Test-Time Prompt -> Evidential Uncertainty (138 FPS) ]", 
            ha="center", va="center", fontsize=8.2, fontweight="bold", color=c_dark_gray)

    plt.tight_layout()
    png_path = "figures/architecture_diagram.png"
    pdf_path = "figures/architecture_diagram.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Sharp monochrome architecture diagram generated:\n  - {png_path}\n  - {pdf_path}")

if __name__ == "__main__":
    draw_architecture_diagram()
