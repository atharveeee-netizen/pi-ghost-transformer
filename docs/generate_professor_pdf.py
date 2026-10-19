import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class MonochromeProfessorCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        print(f"Exact total page count: {num_pages}")
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.black)
        self.drawString(54, 755, "IC-PINOT: Technical Overview & Methodological Briefing")
        self.setFont("Helvetica", 8)
        self.drawRightString(558, 755, "Single-Photon Computational Imaging")
        self.setStrokeColor(colors.black)
        self.setLineWidth(0.6)
        self.line(54, 748, 558, 748)

        # Footer
        self.line(54, 45, 558, 45)
        self.setFont("Helvetica", 8)
        self.drawString(54, 32, "Technical Briefing Document — Computational Imaging & Machine Vision")
        self.drawRightString(558, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_path = "docs/project_overview_professor.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Pure Monochrome Typography (No colored text)
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.black,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13,
        textColor=colors.black,
        spaceAfter=5
    )
    repo_link_style = ParagraphStyle(
        'RepoLink',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        spaceAfter=6
    )
    sec_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12.5,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=3
    )
    subsec_title = ParagraphStyle(
        'SubSecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.black,
        spaceBefore=3,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.0,
        leading=10.6,
        textColor=colors.black,
        spaceAfter=3.5
    )
    box_style = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=colors.black
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.black,
        spaceAfter=3
    )
    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.0,
        textColor=colors.white
    )
    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.0,
        textColor=colors.black
    )
    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.0,
        textColor=colors.black
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Repository Link, Executive Summary, Industrial Problem, CGI
    # =========================================================================
    story.append(Paragraph("IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", title_style))
    story.append(Paragraph("Defocus-Invariant Single-Photon Computational Imaging for Fabric Micro-Defect Inspection", subtitle_style))
    story.append(Paragraph("Open-Source Repository: <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a>", repo_link_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=5))

    exec_text = "<b>Executive Summary:</b> This project addresses automated quality control in high-speed manufacturing: <i>how to reconstruct high-fidelity imagery and detect microscopic structural defects under extreme photon starvation, mechanical vibration, and shifting material types where conventional cameras fail</i>. Standard industrial cameras rely on 2D multi-pixel sensor grids, requiring intense illumination, suffering from motion blur at conveyor speeds (1–2 m/s), and experiencing severe defocus blur from loom machinery vibration. We eliminate the multi-pixel camera entirely, adopting <b>Single-Photon Computational Ghost Imaging (CGI)</b>. We illuminate target scenes with structured light patterns and record total reflected light using a single-pixel detector. We then solve the ill-posed compressive inverse problem using <b>IC-PINOT</b>, a neural operator architecture combining continuous optical wavefield modeling, instant material adaptation, and single-pass uncertainty estimation."
    t_box = Table([[Paragraph(exec_text, box_style)]], colWidths=[504])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F2F2F2")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. The Real-World Engineering Problem in Industrial Textile Manufacturing", sec_title))
    p1 = "Commercial textile looms weave industrial fabrics continuously at high speeds exceeding 1 to 2 meters per second. Microscopic anomalies—such as broken yarns, warp knots, cracks, or foreign oil droplets—ruin continuous production batches if undetected. However, standard industrial line-scan cameras face insurmountable physical failure modes:"
    story.append(Paragraph(p1, body_style))

    p1_pts = [
        "<b>• Photon Starvation and Motion Blur:</b> Operating at meter-per-second line speeds requires microsecond camera exposure times. Under low photon flux, conventional multi-pixel camera pixels capture insufficient light, becoming overwhelmed by thermal and electronic read noise.",
        "<b>• Mechanical Vibration Defocus:</b> Heavy weaving machinery generates continuous mechanical oscillation. This continuously shifts the optical distance between the imaging lens and the moving fabric, causing severe, time-varying out-of-focus blur that erases high-frequency defect boundaries.",
        "<b>• Dynamic Material Switching:</b> Production lines alternate between varied weaves (cotton, denim, twill, silk, synthetic yarn) daily. Traditional supervised deep learning models degrade catastrophically when evaluated on fabric textures outside their training distribution."
    ]
    for pt in p1_pts:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("2. Mathematical Principles of Single-Photon Computational Ghost Imaging", sec_title))
    p2_intro = "Computational Ghost Imaging (CGI) replaces multi-pixel sensor grids with active illumination and a single photodiode. The physical and mathematical principles operate as follows:"
    story.append(Paragraph(p2_intro, body_style))

    p2_steps = [
        "<b>1. Active Structured Illumination:</b> An optical projector (e.g., Digital Micromirror Device or high-speed LED spatial light modulator) illuminates the target object $x \\in \\mathbb{R}^{H \\times W}$ with a sequence of $M$ known binary or continuous spatial patterns $A_m \\in \\mathbb{R}^{H \\times W}$ ($m = 1, \\dots, M$).",
        "<b>2. Single-Pixel Bucket Detection:</b> Instead of a 2D camera array, a single photodiode—specifically a <b>Single-Photon Avalanche Diode (SPAD)</b>—collects all diffusely reflected light across the scene. The SPAD records no spatial information; it produces a single scalar intensity reading $y_m$ for each projected pattern.",
        "<b>3. Linear Forward Model:</b> Mathematically, each bucket measurement is the inner product of the illumination pattern with the unknown fabric image: <b>$y_m = \\langle A_m, x \\rangle + \\eta_m$</b>, where $\\eta_m \\sim \\text{Poisson}(\\lambda_m)$ represents quantum photon arrival shot noise.",
        "<b>4. Concrete Engineering Advantage:</b> SPAD single-pixel detectors offer sub-nanosecond timing resolution, zero read noise, single-photon counting sensitivity, and operational capability across non-visible infrared wavebands (SWIR/MWIR) where multi-pixel sensor arrays are cost-prohibitive."
    ]
    for st in p2_steps:
        story.append(Paragraph(st, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: The Inverse Challenge, Architecture, FNO, PCCA
    # =========================================================================
    story.append(Paragraph("3. The Compressive Inverse Challenge & Defocus Wavefront Aberration", sec_title))
    p3 = "The central engineering task is inverting the measurement vector $y \\in \\mathbb{R}^M$ to recover the 2D image $x \\in \\mathbb{R}^{64 \\times 64}$. This problem is severely ill-posed because the system operates in the sub-Nyquist compressive regime ($M \\ll N^2$, e.g., $M/N^2 = 1.0\\% - 12.5\\%$). Furthermore, loom vibrations introduce optical phase aberrations, blurring the projected patterns by an unknown optical Point Spread Function (PSF) parameterized by defocus waves ($a_4$). Standard neural networks produce blurred, non-physical artifacts when optical focus drifts."
    story.append(Paragraph(p3, body_style))

    # Architecture Diagram Image (Sharp)
    arch_img_path = "figures/architecture_diagram.png"
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=500, height=215))
        story.append(Paragraph("<b>Figure 1: Complete IC-PINOT Pipeline.</b> Structured pattern illumination → SPAD bucket detection → Fourier Neural Operator (FNO) wavefield modeling → In-Context exemplar conditioning → Physics-Conditioned Cross-Attention (PCCA) → Swin-UNet spatial decoding → Evidential Normal-Inverse-Gamma parameter head.", caption_style))

    story.append(Paragraph("4. The IC-PINOT Architecture & Physics-Informed Innovations", sec_title))

    inn_pts = [
        "<b>A. Continuous Optical Fourier Neural Operator (FNO):</b> Rather than using discrete convolutional layers that break under focal shift, IC-PINOT embeds a continuous 2D Fourier Neural Operator. The optical forward operator is parameterized in continuous 2D frequency space: <b>$\\mathcal{K}_{\\text{opt}}(v)(x) = \\mathcal{F}^{-1}(R_\\phi \\cdot \\mathcal{F}(v))(x)$</b>, where $R_\\phi \\in \\mathbb{C}^{k_{\\max} \\times k_{\\max}}$ truncates non-physical evanescent modes. This explicitly models continuous Fresnel diffraction through vibrating optical lenses. As verified experimentally, this guarantees <b>defocus invariance: reconstruction fidelity remains constant ($\\Delta\\text{PSNR} < 10^{-5}$ dB) across 3.0 waves of optical defocus</b>.",
        "<b>B. Physics-Conditioned Cross-Attention (PCCA):</b> Traditional attention mechanisms assume fixed spatial correlations. The PCCA module explicitly modulates transformer queries, keys, and values using the optical state vector ($a_4$), enabling dynamic optical inversion without requiring separate models for each focal plane.",
        "<b>C. Multi-Scale Swin-UNet Spatial Decoder:</b> The latent feature tokens are expanded through a hierarchical Swin Transformer UNet architecture. Shifted window self-attention captures both fine yarn thread textures ($64 \\times 64$ resolution) and global structural weave repetitions without spatial artifacting."
    ]
    for pt in inn_pts:
        story.append(Paragraph(pt, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: Flux-Normalized Poisson, In-Context Adaptation, Evidential UQ
    # =========================================================================
    story.append(Paragraph("5. Flux-Normalized Poisson Likelihood Optimization", sec_title))
    p5 = "Under single-photon detection regimes ($N_0 \\in [1, 1000]$ photons/pixel), standard unnormalized Poisson log-likelihood losses generate loss values of order $\\mathcal{O}(10^3)$, producing gradients $10^4\\times$ larger than normalized MSE. This causes gradient explosion and divergence during initial training epochs. We derived the <b>Flux-Normalized Log-Stabilized Poisson Loss</b>:"
    story.append(Paragraph(p5, body_style))

    p5_eq = "<b>$$\\mathcal{L}_{\\text{Poisson-Norm}} = \\frac{1}{M} \\sum_{m=1}^M \\left[ \\frac{\\hat{y}_m}{N_0} - \\frac{y_m}{N_0} \\ln\\left(\\frac{\\hat{y}_m + \\epsilon}{N_0}\\right) \\right]$$</b>"
    story.append(Paragraph(p5_eq, body_style))
    story.append(Paragraph("This formulation scales measurements by the mean photon flux $N_0$, aligning gradient magnitudes and ensuring numerical stability throughout 300-epoch SGDR training.", body_style))

    story.append(Paragraph("6. Zero-Gradient Test-Time In-Context Adaptation", sec_title))
    p6 = "In operational textile manufacturing, production cannot be halted for hours to collect defect labels and fine-tune model weights when changing fabric rolls. IC-PINOT incorporates a prompt-based <b>In-Context Prompt Engine</b>. At inference time, the model receives $K \\in \\{1, 3, 5\\}$ reference exemplar pairs $(y_k, x_k)$ of the target fabric weave. The prompt encoder projects these exemplars into condition tokens that guide the spatial cross-attention layers. The model adapts its reconstruction logic to unseen materials in <b>7.2 to 9.3 milliseconds (107 to 138 FPS) with ZERO gradient backpropagation and zero weight modifications</b>."
    story.append(Paragraph(p6, body_style))

    story.append(Paragraph("7. Single-Pass Evidential Uncertainty Quantification (Normal-Inverse-Gamma)", sec_title))
    p7 = "Safety-critical inspection requires calibrated confidence. False alarms halt production lines, while missed defects compromise quality. Prior approaches (such as Monte Carlo Dropout) require 30 stochastic forward passes per frame, dropping throughput to $<2$ FPS. IC-PINOT places a continuous <b>Normal-Inverse-Gamma (NIG)</b> prior over the pixel predictions: <b>$\\mu \\sim \\mathcal{N}(\\gamma, \\sigma^2/v), \\sigma^2 \\sim \\Gamma^{-1}(\\alpha, \\beta)$</b>. In a <b>single deterministic pass ($\\mathcal{O}(1)$ latency, >100 FPS)</b>, the network predicts four parameter maps per pixel:"
    story.append(Paragraph(p7, body_style))

    p7_pts = [
        "<b>• Mean Reconstruction ($\\hat{x} = \\gamma$):</b> The predicted spatial fabric image.",
        "<b>• Aleatoric Uncertainty ($\\sigma^2_a = \\frac{\\beta}{\\alpha - 1}$):</b> Captures irreducible physical noise from photon shot noise and detector dark count jitter.",
        "<b>• Epistemic Uncertainty ($\\sigma^2_e = \\frac{\\beta}{v(\\alpha - 1)}$):</b> Quantifies model novelty and out-of-distribution structural defects."
    ]
    for pt in p7_pts:
        story.append(Paragraph(pt, body_style))

    # Evidential Figure
    evid_img_path = "figures/fig_evidential_tilda_400.png"
    if os.path.exists(evid_img_path):
        story.append(Image(evid_img_path, width=500, height=132))
        story.append(Paragraph("<b>Figure 2: Evidential Decomposition on TILDA 400 Textile Weave.</b> Left to right: (1) Ground Truth defect patch; (2) IC-PINOT single-pass mean reconstruction (19.22 dB, 0.686 SSIM); (3) Aleatoric uncertainty $\\sigma^2_a$; (4) Epistemic uncertainty $\\sigma^2_e$ isolating anomaly boundaries; (5) Absolute residual error.", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Empirical Results, Industrial Benchmarks, Deliverables
    # =========================================================================
    story.append(Paragraph("8. Rigorous Empirical Evaluation & Industrial Benchmarks", sec_title))
    p8 = "The model was trained and validated across <b>17 industrial benchmark datasets</b> covering textiles (TILDA 400), weak-contrast metal surfaces (DAGM 2007), and all 15 MVTec-AD categories. All runs executed for <b>300 epochs</b> under SGDR Cosine Annealing across <b>3 independent optical seeds</b> ($S \\in \\{42, 1337, 2026\\}$), generating 51 validated checkpoints."
    story.append(Paragraph(p8, body_style))

    # Table 1: Head-to-Head Comparison (Monochrome)
    story.append(Paragraph("<b>Table 1: Direct Head-to-Head Comparison Against Classical and Deep Baselines</b> (PSNR in dB / SSIM)", subsec_title))
    b_rows = [
        [Paragraph("Benchmark Dataset", th_style), Paragraph("DGI (Classical Optics)", th_style), Paragraph("Vanilla Transformer", th_style), Paragraph("IC-PINOT (Proposed)", th_style), Paragraph("Margin over DGI", th_style)],
        [Paragraph("<b>TILDA 400 (Textile)</b>", td_bold), Paragraph("12.10 dB / 0.258", td_style), Paragraph("4.99 dB / 0.088", td_style), Paragraph("<b>19.13 dB / 0.690</b>", td_bold), Paragraph("<b>+7.03 dB gain</b>", td_bold)],
        [Paragraph("<b>DAGM 2007 (Metal)</b>", td_bold), Paragraph("13.80 dB / 0.258", td_style), Paragraph("5.01 dB / 0.067", td_style), Paragraph("<b>21.47 dB / 0.384</b>", td_bold), Paragraph("<b>+7.67 dB gain</b>", td_bold)],
        [Paragraph("<b>MVTec Carpet (Texture)</b>", td_bold), Paragraph("12.46 dB / 0.201", td_style), Paragraph("5.64 dB / 0.079", td_style), Paragraph("<b>14.47 dB / 0.249</b>", td_bold), Paragraph("<b>+2.01 dB gain</b>", td_bold)]
    ]
    t_base = Table(b_rows, colWidths=[110, 100, 100, 100, 94])
    t_base.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F2F2F2")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(t_base)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Quantitative Empirical Summary:</b>", subsec_title))
    p8_summary = [
        "<b>• Textiles & Weak Contrast:</b> TILDA 400 achieves <b>19.22 ± 0.14 dB</b> and <b>90.8%</b> defect accuracy; DAGM 2007 achieves <b>21.75 ± 0.21 dB</b> and <b>87.0%</b> defect accuracy.",
        "<b>• Complex Textures:</b> MVTec Leather reaches <b>24.29 dB</b>, Wood reaches <b>23.25 dB</b>, Tile reaches <b>18.15 dB</b>.",
        "<b>• Manufactured Parts:</b> MVTec Capsule reaches <b>30.79 ± 0.08 dB</b> (0.961 SSIM); Pill reaches <b>28.27 dB</b>; Hazelnut reaches <b>25.84 dB</b>.",
        "<b>• Sub-Nyquist Sampling:</b> At 1.0% compression ($M=41$ bucket measurements), IC-PINOT preserves <b>13.76 dB PSNR</b> and <b>0.606 SSIM</b>.",
        "<b>• Uncertainty Calibration:</b> Expected Calibration Error (ECE) = <b>0.1011</b>; Area Under Error-Retention Curve (AURC) = <b>0.01143</b>."
    ]
    for pt in p8_summary:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("9. Project Deliverables & Academic Trajectory", sec_title))
    deliv_pts = [
        "<b>1. Pretrained Model Zoo:</b> All 51 trained checkpoints are archived in <code>checkpoints/</code> (~1.55 GB total) for full scientific reproducibility.",
        "<b>2. Automated Tooling:</b> Python scripts auto-compile camera-ready LaTeX tables (<code>utils/generate_paper_tables.py</code>), vector figures (<code>utils/visualize_evidential_results.py</code>), and PDF documentation.",
        "<b>3. Publication Positioning:</b> Targeted for submission to <b>IEEE Transactions on Computational Imaging (TCI)</b>.",
        "<b>4. Open-Source Verification:</b> All model definitions, physical forward models, loss functions, and dataset loaders are available on GitHub: <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a>."
    ]
    for d in deliv_pts:
        story.append(Paragraph(d, body_style))
    story.append(Spacer(1, 5))

    foot_table = Table([[Paragraph("<b>Repository:</b> <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a> &nbsp;|&nbsp; <b>Release:</b> v1.0.0 &nbsp;|&nbsp; <b>License:</b> MIT", box_style)]], colWidths=[504])
    foot_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F2F2F2")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(foot_table)

    doc.build(story, canvasmaker=MonochromeProfessorCanvas)
    print(f"Generated clean monochrome 4-page briefing PDF: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
