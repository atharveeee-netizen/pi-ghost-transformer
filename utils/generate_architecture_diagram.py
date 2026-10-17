import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_architecture_diagram():
    os.makedirs("figures", exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    c_blue = "#0969DA"
    c_light_blue = "#DDF4FF"
    c_purple = "#8250DF"
    c_light_purple = "#FBEFFF"
    c_green = "#1A7F37"
    c_light_green = "#DAFBE1"
    c_orange = "#BF8700"
    c_light_orange = "#FFF8C5"
    c_red = "#CF222E"
    c_light_red = "#FFEBE9"
    c_dark = "#1F2328"
    c_gray = "#6E7781"

    # Title
    ax.text(7, 6.6, "IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", 
            ha="center", va="center", fontsize=14, fontweight="bold", color=c_dark)
    ax.text(7, 6.25, "Continuous Differentiable Optics + In-Context Prompt Engine + Evidential NIG Uncertainty Quantification", 
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color=c_gray)

    # Box Helper Function
    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", radius=0.15):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=patches.BoxStyle("Round", pad=0.08, rounding_size=radius),
            facecolor=bg_color, edgecolor=border_color, linewidth=1.5
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha="center", va="center", fontsize=9, fontweight="bold", color=c_dark)
            ax.text(x + w/2, y + h*0.30, subtitle, ha="center", va="center", fontsize=7.5, color=c_gray)
        else:
            ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=9, fontweight="bold", color=c_dark)

    # Arrow Helper Function
    def draw_arrow(x1, y1, x2, y2, color=c_gray, text=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6, shrinkA=3, shrinkB=3)
        )
        if text:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mid_x, mid_y + 0.15, text, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=color)

    # 1. Physical Stage (Left)
    # Illumination
    draw_box(0.6, 4.4, 2.2, 1.2, c_light_orange, c_orange, "Structured Light", "M Learned Patterns\nA ∈ R^(M × H × W)")
    # Target Object
    draw_box(0.6, 2.8, 2.2, 1.2, c_light_blue, c_blue, "Target Scene / Defect", "Unknown Object\nx ∈ R^(H × W)")
    # Wave Optics / Defocus
    draw_box(0.6, 1.0, 2.2, 1.4, c_light_red, c_red, "Differentiable Optics", "Zernike Defocus (a₄)\nPSF Wavefront Distortion")

    # Connect to Single-Pixel SPAD
    draw_box(3.4, 2.8, 1.8, 1.4, c_light_red, c_red, "SPAD Sensor", "Single-Pixel Bucket\ny ∈ R^M\n(Poisson + DCR + Jitter)")
    
    draw_arrow(2.8, 5.0, 3.4, 3.7, c_orange)
    draw_arrow(2.8, 3.4, 3.4, 3.5, c_blue)
    draw_arrow(2.8, 1.7, 3.4, 3.3, c_red)

    # 2. Operator & Context Processing (Middle-Left)
    # Continuous FNO
    draw_box(5.8, 3.6, 2.4, 1.3, c_light_purple, c_purple, "Continuous FNO", "Spectral Conv R_ϕ\nWave Propagation Tokens")
    draw_arrow(5.2, 3.5, 5.8, 4.1, c_purple, "Bucket y")
    
    # In-Context Prompt Engine
    draw_box(5.8, 1.4, 2.4, 1.3, c_light_blue, c_blue, "In-Context Engine", "K Exemplars (yₖ, xₖ)\nZero-Gradient Tokens")
    ax.text(4.4, 1.0, "Reference Pairs (K-shot)", ha="center", fontsize=7.5, color=c_blue)
    draw_arrow(4.4, 1.2, 5.8, 1.8, c_blue)

    # 3. Physics-Conditioned Cross-Attention (PCCA)
    draw_box(8.8, 2.5, 2.0, 2.2, c_light_blue, c_blue, "PCCA Module", "Physics-Conditioned\nCross-Attention\nQ = Swin Tokens\nK, V = FNO + Context")
    draw_arrow(8.2, 4.2, 8.8, 3.9, c_purple)
    draw_arrow(8.2, 2.1, 8.8, 3.1, c_blue)

    # 4. Swin-UNet Spatial Decoder
    draw_box(11.3, 3.8, 2.2, 1.1, c_light_green, c_green, "Swin-UNet Decoder", "Multi-Scale Hierarchy\n64 Spatial Tokens")
    draw_arrow(10.8, 3.6, 11.3, 4.3, c_green)

    # 5. Evidential NIG Output Head
    draw_box(11.3, 1.1, 2.2, 1.8, c_light_purple, c_purple, "Evidential NIG Head", "Normal-Inverse-Gamma\nγ: Mean Recon\nv: Evidence (>0)\nα: Shape (>1)\nβ: Scale (>0)")
    draw_arrow(12.4, 3.8, 12.4, 2.9, c_purple)

    # Output Outputs
    ax.text(12.4, 0.65, "Output Decompositions:", ha="center", fontsize=8, fontweight="bold", color=c_dark)
    ax.text(12.4, 0.40, "• Mean x̂ = γ   • Noise σ²_a = β/(α-1)\n• Novelty σ²_e = β/[v(α-1)]", 
            ha="center", fontsize=7.5, color=c_blue)

    plt.tight_layout()
    png_path = "figures/architecture_diagram.png"
    pdf_path = "figures/architecture_diagram.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram generated: {png_path} and {pdf_path}")

if __name__ == "__main__":
    draw_architecture_diagram()
