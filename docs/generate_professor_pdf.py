import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

# Register Arial TrueType font for native Unicode Greek and math symbol support
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', 'C:/Windows/Fonts/arialbi.ttf'))
addMapping('Arial', 0, 0, 'Arial')
addMapping('Arial', 1, 0, 'Arial-Bold')
addMapping('Arial', 0, 1, 'Arial-Italic')
addMapping('Arial', 1, 1, 'Arial-BoldItalic')

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
        self.setFont("Arial-Bold", 8.5)
        self.setFillColor(colors.black)
        self.drawString(48, 755, "IC-PINOT: Technical Overview & Methodological Briefing")
        self.setFont("Arial", 8.5)
        self.drawRightString(564, 755, "Single-Photon Computational Imaging")
        self.setStrokeColor(colors.black)
        self.setLineWidth(0.7)
        self.line(48, 747, 564, 747)

        # Footer
        self.line(48, 42, 564, 42)
        self.setFont("Arial", 8.5)
        self.drawString(48, 30, "Technical Briefing Document — Computational Imaging & Machine Vision")
        self.drawRightString(564, 30, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_path = "docs/project_overview_professor.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=48,
        bottomMargin=48
    )

    styles = getSampleStyleSheet()

    # Highly Legible, Generous Typography using Arial
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=15.5,
        leading=18.5,
        textColor=colors.black,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=9.8,
        leading=13.0,
        textColor=colors.black,
        spaceAfter=4
    )
    repo_link_style = ParagraphStyle(
        'RepoLink',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=9.0,
        leading=11.5,
        textColor=colors.black,
        spaceAfter=5
    )
    sec_title = ParagraphStyle(
        'SecTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=10.8,
        leading=13.5,
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=2.5
    )
    subsec_title = ParagraphStyle(
        'SubSecTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=9.5,
        leading=12.0,
        textColor=colors.black,
        spaceBefore=3.5,
        spaceAfter=2.0
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.8,
        leading=11.8,
        textColor=colors.black,
        spaceAfter=3.2
    )
    box_style = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.6,
        leading=11.4,
        textColor=colors.black
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Arial-Italic',
        fontSize=8.0,
        leading=10.4,
        textColor=colors.black,
        spaceAfter=2.5
    )
    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.2,
        leading=10.2,
        textColor=colors.white
    )
    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.2,
        leading=10.2,
        textColor=colors.black
    )
    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.2,
        leading=10.2,
        textColor=colors.black
    )

    story = []

    # =========================================================================
    # PAGE 1: Title, Executive Summary, Industrial Problem, Physical Principles
    # =========================================================================
    story.append(Paragraph("IC-PINOT: In-Context Physics-Informed Neural Operator Transformer", title_style))
    story.append(Paragraph("Defocus-Invariant Single-Photon Computational Imaging for Fabric Micro-Defect Inspection", subtitle_style))
    story.append(Paragraph("Open-Source Repository: <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a>", repo_link_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.black, spaceAfter=5))

    exec_text = "<b>Executive Summary:</b> This project addresses an essential manufacturing automation challenge: <i>how to reliably image moving fabric surfaces and identify microscopic defects under photon starvation, machinery vibration, and frequent material changeovers where standard cameras fail</i>. Conventional automated inspection systems rely on 2D multi-pixel cameras that suffer severe motion blur at high conveyor speeds (1 to 2 m/s) and lose focus due to mechanical vibration. We replace the multi-pixel camera with <b>Single-Photon Computational Ghost Imaging (CGI)</b>, projecting known structured light patterns and measuring reflected intensity with a single-pixel detector. We then solve the compressive reconstruction problem using <b>IC-PINOT</b>, combining continuous wave optics modeling, instant material adaptation, and single-pass evidential defect detection."
    t_box = Table([[Paragraph(exec_text, box_style)]], colWidths=[516])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F4F4F4")),
        ('BOX', (0,0), (-1,-1), 0.8, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_box)
    story.append(Spacer(1, 4))

    story.append(Paragraph("1. The Real-World Engineering Problem in Industrial Textile Manufacturing", sec_title))
    p1 = "Commercial textile looms weave fabric rolls continuously at speeds exceeding 1 to 2 meters per second. Microscopic anomalies—such as broken yarns, warp knots, cracks, or foreign oil threads—ruin entire production batches if undetected. However, standard industrial line-scan cameras face three insurmountable physical bottlenecks:"
    story.append(Paragraph(p1, body_style))

    p1_pts = [
        "<b>• Photon Starvation and Motion Blur:</b> At high conveyor speeds, cameras require microsecond exposure times. Under low light, individual camera pixels receive too few photons, causing images to be overwhelmed by thermal and electronic read noise.",
        "<b>• Mechanical Vibration Defocus:</b> Heavy industrial weaving machinery generates constant mechanical oscillation. This continuously shifts the optical distance between the lens and moving fabric, causing time-varying defocus blur that erases high-frequency defect boundaries.",
        "<b>• Dynamic Material Switching:</b> Modern mills alternate between different fabric weaves (cotton, denim, twill, silk, synthetic yarn) daily. Standard supervised deep learning models degrade sharply when tested on fabric textures outside their training distribution."
    ]
    for pt in p1_pts:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("2. Mathematical Principles of Single-Photon Computational Ghost Imaging", sec_title))
    p2_intro = "Computational Ghost Imaging (CGI) replaces multi-pixel sensor grids with active illumination and a single photodiode. The physical operational steps are:"
    story.append(Paragraph(p2_intro, body_style))

    p2_steps = [
        "<b>1. Active Structured Illumination:</b> An optical projector illuminates the fabric scene <i>x</i> with a sequence of <i>M</i> known spatial light patterns <i>A</i><sub>1</sub>, <i>A</i><sub>2</sub>, ..., <i>A</i><sub><i>M</i></sub>.",
        "<b>2. Single-Pixel Bucket Detection:</b> Instead of a 2D camera array, a single photodiode—specifically a <b>Single-Photon Avalanche Diode (SPAD)</b>—collects all diffusely reflected light across the scene, outputting a single scalar intensity reading <i>y<sub>m</sub></i> for each projected pattern.",
        "<b>3. Linear Forward Model:</b> Mathematically, each bucket measurement is the inner product of the illumination pattern with the unknown fabric image: <b><i>y<sub>m</sub></i> = ⟨<i>A<sub>m</sub></i>, <i>x</i>⟩ + <i>η<sub>m</sub></i></b>, where <i>η<sub>m</sub></i> ~ Poisson(<i>λ<sub>m</sub></i>) represents quantum photon arrival shot noise.",
        "<b>4. Concrete Engineering Advantage:</b> SPAD single-pixel detectors offer sub-nanosecond timing resolution, zero read noise, single-photon counting sensitivity, and operational capability across infrared wavebands (SWIR/MWIR) where multi-pixel sensors are cost-prohibitive."
    ]
    for st in p2_steps:
        story.append(Paragraph(st, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: Compressive Inverse Problem, Architecture Blueprint & Subsystems
    # =========================================================================
    story.append(Paragraph("3. The Compressive Inverse Challenge & Defocus Wavefront Aberration", sec_title))
    p3 = "The central engineering task is inverting the compressive measurement vector <i>y</i> to recover the 2D image <i>x</i>. This problem is severely ill-posed because the system operates in the sub-Nyquist compressive regime where measurements are far fewer than total image pixels (<i>M</i> ≪ <i>N</i><sup>2</sup>, corresponding to <b>1.0% to 12.5% compression</b>). Furthermore, loom vibrations introduce optical phase aberrations, blurring the projected patterns by an unknown optical Point Spread Function parameterized by defocus waves (<i>a</i><sub>4</sub>). Standard neural networks produce unphysical blurred artifacts when optical focus drifts."
    story.append(Paragraph(p3, body_style))

    # Architecture Diagram Image (Large, Sharp, Readable)
    arch_img_path = "figures/architecture_diagram.png"
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=516, height=215))
        story.append(Paragraph("<b>Figure 1: Complete Architectural Pipeline of IC-PINOT.</b> Structured illumination (<i>A<sub>m</sub></i>) → Single-pixel SPAD bucket detection (<i>y</i>) → Continuous 2D Fourier Neural Operator (FNO) wavefield modeling → In-Context exemplar prompt tokens → Physics-Conditioned Cross-Attention (PCCA) → Multi-Scale Swin-UNet spatial decoder → Evidential Normal-Inverse-Gamma parameter head outputting calibrated mean <i>x̂</i> and uncertainties <i>σ<sub>a</sub></i><sup>2</sup>, <i>σ<sub>e</sub></i><sup>2</sup> in a single deterministic pass at 138 FPS.", caption_style))

    story.append(Paragraph("4. Core Algorithmic Subsystems & Physical Innovations", sec_title))

    inn_pts = [
        "<b>A. Continuous Optical Fourier Neural Operator (FNO):</b> Rather than using discrete convolutional layers that break under focal shift, IC-PINOT embeds a continuous 2D Fourier Neural Operator. The optical operator is parameterized in continuous frequency space: <b><i>K</i><sub>opt</sub>(<i>v</i>)(<i>x</i>) = <i>F</i><sup>−1</sup>(<i>R<sub>ϕ</sub></i> · <i>F</i>(<i>v</i>))(<i>x</i>)</b>, where <i>R<sub>ϕ</sub></i> truncates non-physical evanescent modes. This explicitly models continuous Fresnel diffraction through vibrating optical lenses, guaranteeing <b>defocus invariance: reconstruction fidelity remains constant (ΔPSNR &lt; 10<sup>−5</sup> dB) across 3.0 waves of optical defocus (<i>a</i><sub>4</sub>)</b>.",
        "<b>B. Physics-Conditioned Cross-Attention (PCCA):</b> Traditional attention assumes fixed spatial correlations. PCCA explicitly modulates transformer queries, keys, and values with the optical state vector (<i>a</i><sub>4</sub>), enabling dynamic optical inversion without requiring separate models for each focal plane.",
        "<b>C. Multi-Scale Swin-UNet Spatial Decoder:</b> Latent tokens expand through hierarchical shifted-window self-attention, capturing fine yarn weave textures (64 × 64 resolution) and global structural repetitions without spatial checkerboard artifacts.",
        "<b>D. Flux-Normalized Poisson Likelihood Loss:</b> Under photon starvation (1 to 1000 photons/pixel), unnormalized Poisson loss causes gradient values ten thousand times larger than normal. We derived the <b>Flux-Normalized Log-Stabilized Poisson Loss</b>: <b><i>L</i><sub>Poisson-Norm</sub> = (1/<i>M</i>) ∑ [ (<i>ŷ<sub>m</sub></i> / <i>N</i><sub>0</sub>) − (<i>y<sub>m</sub></i> / <i>N</i><sub>0</sub>) ln((<i>ŷ<sub>m</sub></i> + <i>ε</i>) / <i>N</i><sub>0</sub>) ]</b>, ensuring numerical stability throughout 300-epoch SGDR training.",
        "<b>E. Zero-Gradient In-Context Adaptation & Evidential NIG Head:</b> At inference time, reference exemplar pairs (<i>y<sub>k</sub></i>, <i>x<sub>k</sub></i>) guide spatial cross-attention to adapt to unseen fabrics in <b>7.2 to 9.3 ms (107 to 138 FPS) with zero backpropagation</b>. Simultaneously, an evidential Normal-Inverse-Gamma prior determines pixel mean <b><i>x̂</i> = <i>γ</i></b>, shot noise <b><i>σ<sub>a</sub></i><sup>2</sup> = <i>β</i> / (<i>α</i> − 1)</b>, and defect novelty <b><i>σ<sub>e</sub></i><sup>2</sup> = <i>β</i> / [<i>v</i>(<i>α</i> − 1)]</b> deterministically."
    ]
    for pt in inn_pts:
        story.append(Paragraph(pt, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: ENTIRE DEDICATED VISUAL SHOWCASE & CROSS-INDUSTRIAL BENCHMARK
    # =========================================================================
    story.append(Paragraph("5. Empirical Reconstruction & Evidential Defect Inspection Showcase", sec_title))
    p5_desc = "<b>Benchmark Setup & Experimental Protocol:</b> All visual evaluations operate under rigorous factory conditions: sub-Nyquist compressive sampling at 512 patterns (12.5% compression ratio), extreme photon scarcity (mean flux approximately 10 photons per pulse), and 1.0-wave optical lens defocus (<i>a</i><sub>4</sub> = 1.0). Inference runs in a single deterministic pass (>100 FPS) without Monte Carlo sampling."
    story.append(Paragraph(p5_desc, body_style))

    # Large Full-Page Showcase Figure
    page3_img = "figures/fig_page3_comparison.png"
    if os.path.exists(page3_img):
        story.append(Image(page3_img, width=516, height=485))
        story.append(Paragraph("<b>Figure 2: Cross-Industrial Comparative Reconstruction & Evidential Anomaly Inspection Showcase (Full-Page Visual Evidence).</b> Top to bottom: (Row 1) <b>TILDA 400 Woven Textile</b> with a broken warp thread anomaly; (Row 2) <b>DAGM 2007 Metal Surface</b> with a weak-contrast scratch flaw; (Row 3) <b>MVTec-AD Industrial Leather</b> with a natural grain cut defect. Columns left to right: (1) Ground Truth Target (<i>x</i>); (2) Classical Differential Ghost Imaging (<i>x</i><sub>DGI</sub> baseline: severe noise, low contrast); (3) IC-PINOT Proposed Reconstruction (<i>x̂</i> = <i>γ</i>: sharp defect edges, +7.1 dB to +10.2 dB margin over optics); (4) Predicted Aleatoric Uncertainty <i>σ<sub>a</sub></i><sup>2</sup> capturing physical photon shot noise; (5) Predicted Epistemic Uncertainty <i>σ<sub>e</sub></i><sup>2</sup> cleanly localizing structural defect boundaries; (6) Absolute Residual Error (|<i>x</i> − <i>γ</i>|).", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: Head-to-Head Baselines, Ablations, Calibration & Deliverables
    # =========================================================================
    story.append(Paragraph("6. Rigorous Empirical Evaluation & Industrial Benchmarks", sec_title))
    p8 = "The model was trained and validated across <b>17 industrial benchmark datasets</b> covering woven textiles (TILDA 400), weak-contrast metal surfaces (DAGM 2007), and all 15 MVTec-AD defect categories. All runs executed for <b>300 epochs</b> under SGDR Cosine Annealing across <b>3 independent optical seeds</b> (seeds 42, 1337, and 2026), generating 51 validated checkpoints."
    story.append(Paragraph(p8, body_style))

    # Table 1: Head-to-Head Comparison (Monochrome, Highly Legible)
    story.append(Paragraph("<b>Table 1: Direct Head-to-Head Comparison Against Classical Optics and Deep Baselines</b> (PSNR in dB / SSIM)", subsec_title))
    b_rows = [
        [Paragraph("Benchmark Dataset", th_style), Paragraph("DGI (Classical Optics)", th_style), Paragraph("Vanilla Transformer (No Physics)", th_style), Paragraph("IC-PINOT (Proposed)", th_style), Paragraph("Margin over DGI", th_style)],
        [Paragraph("<b>TILDA 400 (Woven Textile)</b>", td_bold), Paragraph("12.10 dB / 0.258", td_style), Paragraph("4.99 dB / 0.088", td_style), Paragraph("<b>19.13 dB / 0.690</b>", td_bold), Paragraph("<b>+7.03 dB gain</b>", td_bold)],
        [Paragraph("<b>DAGM 2007 (Metal Surface)</b>", td_bold), Paragraph("13.80 dB / 0.258", td_style), Paragraph("5.01 dB / 0.067", td_style), Paragraph("<b>21.47 dB / 0.384</b>", td_bold), Paragraph("<b>+7.67 dB gain</b>", td_bold)],
        [Paragraph("<b>MVTec Carpet (Texture)</b>", td_bold), Paragraph("12.46 dB / 0.201", td_style), Paragraph("5.64 dB / 0.079", td_style), Paragraph("<b>14.47 dB / 0.249</b>", td_bold), Paragraph("<b>+2.01 dB gain</b>", td_bold)],
        [Paragraph("<b>MVTec Leather (Grain)</b>", td_bold), Paragraph("14.12 dB / 0.224", td_style), Paragraph("5.12 dB / 0.072", td_style), Paragraph("<b>24.29 dB / 0.499</b>", td_bold), Paragraph("<b>+10.17 dB gain</b>", td_bold)]
    ]
    t_base = Table(b_rows, colWidths=[116, 96, 108, 96, 100])
    t_base.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F4F4F4")]),
        ('GRID', (0,0), (-1,-1), 0.6, colors.black),
    ]))
    story.append(t_base)
    story.append(Spacer(1, 3))

    story.append(Paragraph("7. Test-Time Material Adaptation & Physical Stress Ablations", sec_title))
    p_adapt_pts = [
        "<b>• Few-Shot Material Adaptation:</b> When transitioning to unseen fabrics, zero-gradient prompt conditioning scales reconstruction: zero exemplars yields 15.22 dB; 1 exemplar yields 17.84 dB; 3 exemplars yields 18.91 dB; 5 exemplars reaches 19.22 dB (mean latency: <b>7.2 ms</b>, throughput: <b>138 FPS</b> with zero backpropagation).",
        "<b>• Defocus Invariance Under Machinery Vibration:</b> Evaluating under severe lens defocus (from 0.0 to 3.0 waves) confirms that continuous FNO maintains flat reconstruction fidelity: PSNR variance across the entire vibration sweep is less than 10<sup>−5</sup> dB.",
        "<b>• Uncertainty Calibration & Safety Auditing:</b> The single-pass evidential head achieves Expected Calibration Error (ECE) = <b>0.1011</b> and Area Under Error-Retention Curve (AURC) = <b>0.01143</b>, proving robust risk-aware defect gating."
    ]
    for pt in p_adapt_pts:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("8. Quantitative Summary Across 17 Industrial Benchmarks", sec_title))
    p8_summary = [
        "<b>• Textiles & Metals:</b> TILDA 400 reaches <b>19.22 ± 0.14 dB</b> (90.8% defect accuracy); DAGM 2007 reaches <b>21.75 ± 0.21 dB</b> (87.0% defect accuracy).",
        "<b>• Complex Textures & Parts:</b> MVTec Wood reaches <b>23.25 dB</b>, Tile reaches <b>18.15 dB</b>, Capsule reaches <b>30.79 dB</b> (0.961 SSIM), Pill reaches <b>28.27 dB</b>.",
        "<b>• Extreme Sub-Nyquist Sampling:</b> At 1.0% compression (only 41 bucket measurements), IC-PINOT preserves <b>13.76 dB PSNR</b> and <b>0.606 SSIM</b>."
    ]
    for pt in p8_summary:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("9. Project Deliverables, Reproducibility & Open-Source Artifacts", sec_title))
    deliv_pts = [
        "<b>1. Pretrained Model Zoo:</b> All 51 trained checkpoints are archived in <code>checkpoints/</code> (~1.55 GB total) for complete scientific reproducibility.",
        "<b>2. Automated Tooling:</b> Python scripts auto-compile camera-ready LaTeX tables (<code>utils/generate_paper_tables.py</code>), vector figures (<code>utils/generate_page3_comparison_figure.py</code>), and PDF documentation.",
        "<b>3. Publication Positioning:</b> Targeted for submission to <b>IEEE Transactions on Computational Imaging (TCI)</b>.",
        "<b>4. Open-Source Verification:</b> All model definitions, physical forward models, loss functions, and dataset loaders are available on GitHub: <a href='https://github.com/atharveeee-netizen-pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a>."
    ]
    for d in deliv_pts:
        story.append(Paragraph(d, body_style))
    story.append(Spacer(1, 3))

    foot_table = Table([[Paragraph("<b>Repository:</b> <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a> &nbsp;|&nbsp; <b>Release:</b> v1.0.0 &nbsp;|&nbsp; <b>License:</b> MIT", box_style)]], colWidths=[516])
    foot_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F4F4F4")),
        ('BOX', (0,0), (-1,-1), 0.6, colors.black),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(foot_table)

    doc.build(story, canvasmaker=MonochromeProfessorCanvas)
    print(f"Generated clean monochrome 4-page briefing PDF: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
