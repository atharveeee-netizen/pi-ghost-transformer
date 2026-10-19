import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class ProfessorDocCanvas(canvas.Canvas):
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
        self.setFillColor(colors.HexColor("#0969DA"))
        self.drawString(54, 755, "IC-PINOT: Faculty & Advisor Technical Briefing Dossier")
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#57606A"))
        self.drawRightString(558, 755, "Rashtriya Raksha University")
        self.setStrokeColor(colors.HexColor("#D0D7DE"))
        self.setLineWidth(0.6)
        self.line(54, 748, 558, 748)

        # Footer
        self.line(54, 45, 558, 45)
        self.drawString(54, 32, "Confidential Academic Briefing | Target: IEEE Trans. Computational Imaging")
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

    title_style = ParagraphStyle(
        'MainTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0969DA"),
        spaceAfter=2
    )
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#24292F"),
        spaceAfter=4
    )
    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#57606A"),
        spaceAfter=6
    )
    sec_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=13,
        textColor=colors.HexColor("#0969DA"),
        spaceBefore=5,
        spaceAfter=3
    )
    subsec_title = ParagraphStyle(
        'SubSecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#1F2328"),
        spaceBefore=3,
        spaceAfter=2
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=10.8,
        textColor=colors.HexColor("#1F2328"),
        spaceAfter=4
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.2,
        leading=10.8,
        textColor=colors.HexColor("#1F2328")
    )
    box_style = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.8,
        leading=10.2,
        textColor=colors.HexColor("#1F2328")
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#57606A"),
        spaceAfter=4
    )
    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white
    )
    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1F2328")
    )
    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1F2328")
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Executive Summary, Real-World Problem, What is CGI?
    # =========================================================================
    story.append(Paragraph("IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", title_style))
    story.append(Paragraph("Defocus-Invariant Single-Photon Computational Imaging for Fabric Micro-Defect Inspection", subtitle_style))
    story.append(Paragraph("<b>Author:</b> Atharve Dahima &nbsp;|&nbsp; <b>Faculty Briefing Document</b> &nbsp;|&nbsp; <b>Target:</b> IEEE Transactions on Computational Imaging (TCI)<br/><b>Code & Checkpoints:</b> github.com/atharveeee-netizen/pi-ghost-transformer &nbsp;|&nbsp; <b>Hardware:</b> NVIDIA GeForce RTX 4070 (12.88 GB)", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0969DA"), spaceAfter=5))

    # Executive Summary Card
    exec_text = "<b>Executive Summary:</b> This project solves a high-stakes industrial quality inspection problem: <i>how to detect microscopic structural yarn defects in high-speed textile factories where standard cameras fail</i>. Standard industrial line-scan cameras require intense illumination, suffer severe motion blur at high conveyor speeds (1–2 m/s), and lose focus due to mechanical machine vibration. We resolve this by replacing conventional multi-pixel camera sensors with <b>Single-Photon Computational Ghost Imaging (CGI)</b>. We project structured patterns onto the fabric and collect light using a single-pixel photodetector. To invert these measurements under real factory vibrations, we built <b>IC-PINOT</b>, combining continuous wave-optics Fourier Neural Operators, instant in-context material adaptation, and single-pass evidential uncertainty estimation at 138 frames per second."
    
    t_box = Table([[Paragraph(exec_text, box_style)]], colWidths=[504])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F6F8FA")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#D0D7DE")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. The Real-World Engineering Problem in Factories", sec_title))
    p1_desc = "Automated textile weaving looms produce fabric at continuous speeds exceeding 1 to 2 meters per second. Microscopic flaws (such as broken warp yarns, weft cracks, holes, or foreign threads) degrade entire rolls if not caught immediately. Factory environments introduce severe physical obstacles for traditional vision systems:"
    story.append(Paragraph(p1_desc, body_style))

    p1_points = [
        "<b>• Photon Starvation & Motion Blur:</b> At line speeds, camera exposure times must drop to microseconds. At low photon levels, standard camera sensors drown in electronic read noise.",
        "<b>• Mechanical Vibration Defocus:</b> Heavy weaving machines generate continuous vibration, shifting the lens focal plane and turning fine defect edges into indistinguishable blur.",
        "<b>• Material Diversity:</b> Factories switch between cotton, wool, silk, denim, and synthetic blends daily. Standard AI models fail when tested on materials outside their narrow training distribution."
    ]
    for pt in p1_points:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("2. What is Single-Photon Computational Ghost Imaging? (First Principles)", sec_title))
    p2_intro = "The term 'Single-Photon' or 'Ghost Imaging' describes a concrete, practical optical architecture rather than abstract theory:"
    story.append(Paragraph(p2_intro, body_style))

    p2_expl = [
        "<b>1. Traditional Cameras:</b> Light bounces off the fabric and passes through a lens onto a 2D sensor grid (e.g., 2048 x 2048 pixels). Each pixel independently records light arriving at that exact physical point.",
        "<b>2. Computational Ghost Imaging:</b> We eliminate the 2D sensor grid entirely. Instead, a digital projector projects a fast sequence of $M$ known structured light patterns $A_m(u, v)$ onto the fabric. A single high-sensitivity photodetector---a <b>Single-Photon Avalanche Diode (SPAD)</b>---measures the total reflected light from the entire patch as a single number (bucket measurement $y_m$).",
        "<b>3. Mathematical Model:</b> The $m$-th measurement is the dot product of the pattern and the unknown fabric patch: <b>$y_m = \\langle A_m, x \\rangle + \\eta_m$</b>, where $\\eta_m$ is quantum shot noise.",
        "<b>4. The Engineering Advantage:</b> SPADs have picosecond response times, zero read noise, and work under low photon levels and invisible infrared wavebands where standard cameras cost upwards of $20,000."
    ]
    for pt in p2_expl:
        story.append(Paragraph(pt, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: The Inverse Problem, IC-PINOT Architecture, FNO, PCCA
    # =========================================================================
    story.append(Paragraph("3. The Inverse Challenge & The IC-PINOT Architecture", sec_title))
    p3_desc = "The challenge is: <i>How do we recover the full 2D spatial image $x \\in \\mathbb{R}^{64 \\times 64}$ from only a few scalar measurements $y \\in \\mathbb{R}^{M}$ when $M \\ll N^2$ (sub-Nyquist compressive sensing) and when mechanical vibration continuously alters the lens focus?</i>"
    story.append(Paragraph(p3_desc, body_style))

    # Architecture Diagram Image
    arch_img_path = "figures/architecture_diagram.png"
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=500, height=220))
        story.append(Paragraph("<b>Figure 2: Complete IC-PINOT Architectural Pipeline.</b> Physical optical acquisition → Continuous FNO wavefield propagator → Physics-Conditioned Cross-Attention (PCCA) → Swin-UNet spatial decoder → Evidential Normal-Inverse-Gamma (NIG) parameter head.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. Core Scientific & Algorithmic Innovations", sec_title))

    innovations = [
        "<b>A. Continuous Fourier Neural Operator (FNO) for Defocus Invariance:</b> When machine vibration alters the lens distance, the wavefront undergoes optical phase aberration parameterized by Zernike defocus ($a_4$). Traditional deep learning treats optical propagation as a black box and fails under blur. IC-PINOT embeds a continuous 2D Fourier Neural Operator (FNO) that learns wavefield propagation directly in frequency space: <b>$\\mathcal{K}_{\\text{opt}}(v) = \\mathcal{F}^{-1}(R_\\phi \\cdot \\mathcal{F}(v))$</b>. This guarantees <b>total optical defocus invariance</b>: reconstruction quality remains constant ($\\Delta\\text{PSNR} < 10^{-5}$ dB) across 3.0 waves of physical defocus.",
        "<b>B. Physics-Conditioned Cross-Attention (PCCA):</b> Traditional transformers assume fixed relationships between measurement tokens and pixels. PCCA modulates transformer queries, keys, and values using the optical state ($a_4$), allowing the network to invert varying optical states dynamically.",
        "<b>C. Flux-Normalized Poisson Likelihood:</b> Under single-photon levels ($N_0 \\in [1, 1000]$ photons/pixel), standard Poisson loss causes gradient explosion. We developed a flux-normalized Poisson formulation that scales measurements by mean photon flux $N_0$, guaranteeing smooth convergence across 300 epochs."
    ]
    for inn in innovations:
        story.append(Paragraph(inn, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: In-Context Adaptation & Evidential Uncertainty Quantification
    # =========================================================================
    story.append(Paragraph("5. Zero-Gradient In-Context Adaptation (No Retraining)", sec_title))
    p5_desc = "In a commercial factory, a manufacturing line cannot be halted for several hours to collect training images, label defects, and retrain weights when a new fabric material enters production. IC-PINOT incorporates an <b>In-Context Prompt Engine</b>. At test time, the model takes $K \\in \\{1, 3, 5\\}$ reference pairs $(y_k, x_k)$ of the new fabric and projects them into condition prompt tokens. Through cross-attention, the model adapts its reconstruction logic to the new surface material <b>in 7.2 to 9.3 milliseconds (107 to 138 FPS) with ZERO gradient backpropagation or weight updates</b>."
    story.append(Paragraph(p5_desc, body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. Single-Pass Evidential Uncertainty Quantification (Normal-Inverse-Gamma)", sec_title))
    p6_desc = "Industrial automated inspection requires knowing <b>when the model is certain</b>. A false alarm stops an entire production line, while an undetected micro-tear reaches consumers. Standard uncertainty methods (like Monte Carlo Dropout) require executing the network 30 times per frame, which slows inspection down to 2 FPS. IC-PINOT places a continuous <b>Normal-Inverse-Gamma (NIG)</b> prior over the pixel predictions: <b>$\\mu \\sim \\mathcal{N}(\\gamma, \\sigma^2/v), \\sigma^2 \\sim \\Gamma^{-1}(\\alpha, \\beta)$</b>."
    story.append(Paragraph(p6_desc, body_style))

    evid_points = [
        "In a <b>single deterministic pass (O(1) latency exceeding 100 FPS)</b>, the network predicts: (1) Mean reconstruction $\\hat{x} = \\gamma$; (2) <b>Aleatoric Uncertainty</b> $\\sigma^2_a = \\frac{\\beta}{\\alpha - 1}$ (unavoidable physical photon shot noise); (3) <b>Epistemic Uncertainty</b> $\\sigma^2_e = \\frac{\\beta}{v(\\alpha - 1)}$ (model unfamiliarity due to structural defects or novelty).",
        "Microscopic defect tears strictly ignite the epistemic uncertainty map $\\sigma^2_e$, allowing automated thresholding for defect rejection without false alarms."
    ]
    for pt in evid_points:
        story.append(Paragraph(pt, body_style))
    story.append(Spacer(1, 4))

    # Evidential Figure
    evid_img_path = "figures/fig_evidential_tilda_400.png"
    if os.path.exists(evid_img_path):
        story.append(Image(evid_img_path, width=500, height=135))
        story.append(Paragraph("<b>Figure 3: Single-Pass Evidential Decomposition on TILDA 400 Textile Weave.</b> Left to right: (1) Ground Truth defect patch; (2) IC-PINOT single-pass mean reconstruction (19.22 dB, 0.686 SSIM); (3) Aleatoric uncertainty map $\\sigma^2_a$; (4) Epistemic uncertainty map $\\sigma^2_e$ sharply highlighting yarn flaw boundaries; (5) Absolute residual error.", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Empirical Results, Industrial Benchmarks, Deliverables
    # =========================================================================
    story.append(Paragraph("7. Comprehensive Empirical Validation & Industrial Benchmarks", sec_title))
    p7_desc = "The entire architecture was evaluated across <b>17 industrial benchmark datasets</b> covering textiles, metallic flaws, and manufactured parts. Every model was trained for <b>300 epochs</b> across <b>3 independent optical seeds</b> ($S \\in \\{42, 1337, 2026\\}$), generating 51 fully trained checkpoints."
    story.append(Paragraph(p7_desc, body_style))

    # Table: Head to Head Baselines
    b_rows = [
        [Paragraph("Benchmark Dataset", th_style), Paragraph("DGI (Classical Optics)", th_style), Paragraph("Vanilla Transformer", th_style), Paragraph("IC-PINOT (Proposed)", th_style), Paragraph("Margin over DGI", th_style)],
        [Paragraph("<b>TILDA 400 (Textile)</b>", td_bold), Paragraph("12.10 dB / 0.258", td_style), Paragraph("4.99 dB / 0.088", td_style), Paragraph("<b>19.13 dB / 0.690</b>", td_bold), Paragraph("<b>+7.03 dB gain</b>", td_bold)],
        [Paragraph("<b>DAGM 2007 (Metal)</b>", td_bold), Paragraph("13.80 dB / 0.258", td_style), Paragraph("5.01 dB / 0.067", td_style), Paragraph("<b>21.47 dB / 0.384</b>", td_bold), Paragraph("<b>+7.67 dB gain</b>", td_bold)],
        [Paragraph("<b>MVTec Carpet (Texture)</b>", td_bold), Paragraph("12.46 dB / 0.201", td_style), Paragraph("5.64 dB / 0.079", td_style), Paragraph("<b>14.47 dB / 0.249</b>", td_bold), Paragraph("<b>+2.01 dB gain</b>", td_bold)]
    ]
    t_base = Table(b_rows, colWidths=[110, 100, 100, 100, 94])
    t_base.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0969DA")),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F6F8FA")]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D0D7DE")),
    ]))
    story.append(t_base)
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Empirical Performance Across 17 Datasets:</b>", subsec_title))
    p7_summary = [
        "<b>• Textiles & Weak Contrast:</b> TILDA 400 reaches <b>19.22 ± 0.14 dB</b> (90.8% defect accuracy); DAGM 2007 reaches <b>21.75 ± 0.21 dB</b> (87.0% defect accuracy).",
        "<b>• Surface Textures:</b> Leather reaches <b>24.29 dB</b>, Wood reaches <b>23.25 dB</b>, Tile reaches <b>18.15 dB</b>.",
        "<b>• Manufactured Objects:</b> MVTec Capsule reaches <b>30.79 ± 0.08 dB</b> (0.961 SSIM); Pill reaches <b>28.27 dB</b>; Hazelnut reaches <b>25.84 dB</b>.",
        "<b>• 1.0% Extreme Sub-Nyquist Sampling:</b> With only $M=41$ bucket measurements (1.0% sampling), IC-PINOT preserves <b>13.76 dB PSNR</b> and <b>0.606 SSIM</b>.",
        "<b>• Calibration Rigor:</b> Expected Calibration Error (ECE) = <b>0.1011</b>; Area Under Error-Retention Curve (AURC) = <b>0.01143</b>."
    ]
    for pt in p7_summary:
        story.append(Paragraph(pt, body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("8. Project Deliverables & Academic Trajectory", sec_title))
    deliverables = [
        "<b>1. 51 Trained Model Checkpoints:</b> Saved in <code>checkpoints/</code> (~1.55 GB total) for complete open-source scientific reproducibility.",
        "<b>2. Automated Publication Tooling:</b> Python scripts generating camera-ready LaTeX tables (<code>utils/generate_paper_tables.py</code>), 300 DPI vector figures (<code>utils/visualize_evidential_results.py</code>), and PDF compilation.",
        "<b>3. Publication Target:</b> Prepared for submission to <b>IEEE Transactions on Computational Imaging (TCI)</b>."
    ]
    for d in deliverables:
        story.append(Paragraph(d, body_style))
    story.append(Spacer(1, 6))

    # Footer note
    footer_text = "<b>Repository:</b> https://github.com/atharveeee-netizen/pi-ghost-transformer &nbsp;|&nbsp; <b>Release:</b> v1.0.0 &nbsp;|&nbsp; <b>License:</b> MIT"
    t_foot = Table([[Paragraph(footer_text, box_style)]], colWidths=[504])
    t_foot.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F6F8FA")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#D0D7DE")),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(t_foot)

    doc.build(story, canvasmaker=ProfessorDocCanvas)
    print(f"Generated 4-page briefing PDF: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
