import os
from PIL import Image, ImageOps

def generate_combined_flowchart():
    os.makedirs("figures", exist_ok=True)
    comp_top = "figures/components/stage1_stage2_top.png"
    comp_bot = "figures/components/stage3_stage4_bottom.png"

    if not os.path.exists(comp_top) or not os.path.exists(comp_bot):
        raise FileNotFoundError(f"Component files not found: {comp_top}, {comp_bot}")

    im_top = Image.open(comp_top).convert("RGB")
    im_bot = Image.open(comp_bot).convert("RGB")

    # Precise subpixel alignment across component seams
    c_top = im_top.crop((50, 42, 390, 417))  # Stage 1 and Stage 2
    c_bot = im_bot.crop((68, 0, 408, 476))   # Stage 3 and Stage 4

    # Vertically stack seamlessly
    combined = Image.new("RGB", (340, c_top.height + c_bot.height), "white")
    combined.paste(c_top, (0, 0))
    combined.paste(c_bot, (0, c_top.height))

    # Add 20px clean border padding
    padded = ImageOps.expand(combined, border=20, fill="white")

    # 3x high-resolution Lanczos scaling for publication-grade vector clarity
    w, h = padded.size
    high_res = padded.resize((w * 3, h * 3), Image.Resampling.LANCZOS)

    out_png = "figures/architecture_diagram.png"
    out_pdf = "figures/architecture_diagram.pdf"
    out_flowchart = "figures/combined_architecture_flowchart.png"

    high_res.save(out_png)
    high_res.save(out_flowchart)
    high_res.save(out_pdf, "PDF", resolution=300.0)

    print(f"Generated clean combined architecture diagram:")
    print(f"  - {out_png} ({high_res.size[0]}x{high_res.size[1]})")
    print(f"  - {out_pdf}")

if __name__ == "__main__":
    generate_combined_flowchart()
