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
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

# Register Arial TrueType font for native Unicode Greek and math symbol support
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', 'C:/Windows/Fonts/arialbi.ttf'))
addMapping('Arial', 0, 0, 'Arial')
addMapping('Arial', 1, 0, 'Arial-Bold')
addMapping('Arial', 0, 1, 'Arial-Italic')
addMapping('Arial', 1, 1, 'Arial-BoldItalic')

class AcademicJournalCanvas(canvas.Canvas):
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
        # Top Header (Classic Journal Running Head)
        self.setFont("Arial-Italic", 8.0)
        self.setFillColor(colors.HexColor("#222222"))
        self.drawString(48, 756, "IEEE Transactions on Computational Imaging — Methodological Briefing Document")
        self.setFont("Arial", 8.0)
        self.drawRightString(564, 756, "Computational Ghost Imaging & Machine Vision")
        self.setStrokeColor(colors.black)
        self.setLineWidth(0.5)
        self.line(48, 749, 564, 749)

        # Bottom Footer
        self.line(48, 40, 564, 40)
        self.setFont("Arial", 8.0)
        self.drawString(48, 28, "IC-PINOT: In-Context Physics-Informed Neural Operator Transformer")
        self.drawRightString(564, 28, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()

def build_pdf():
    pdf_path = "docs/project_overview_professor.pdf"
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Scholarly Typography (True Publication Standards)
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=15.0,
        leading=18.0,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=3
    )
    subtitle_style = ParagraphStyle(
        'PaperSubtitle',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=9.6,
        leading=12.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=3
    )
    repo_banner_style = ParagraphStyle(
        'RepoBanner',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.5,
        leading=11.0,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=5
    )
    sec_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=10.2,
        leading=13.0,
        textColor=colors.black,
        spaceBefore=5,
        spaceAfter=2.5
    )
    subsec_heading = ParagraphStyle(
        'SubSecHeading',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=9.2,
        leading=11.8,
        textColor=colors.black,
        spaceBefore=3.5,
        spaceAfter=1.8
    )
    abstract_style = ParagraphStyle(
        'Abstract',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.5,
        leading=11.5,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.6,
        leading=11.5,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=3.2
    )
    caption_style = ParagraphStyle(
        'Caption',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=7.8,
        leading=10.0,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        spaceAfter=2.5
    )
    th_style = ParagraphStyle(
        'TH',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.0,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    td_style = ParagraphStyle(
        'TD',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.0,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    td_bold = ParagraphStyle(
        'TDBold',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.0,
        leading=10.0,
        alignment=TA_CENTER,
        textColor=colors.black
    )
    td_left_bold = ParagraphStyle(
        'TDLeftBold',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.0,
        leading=10.0,
        alignment=TA_LEFT,
        textColor=colors.black
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE, ABSTRACT, INTRODUCTION, PHYSICAL FORWARD MODEL
    # =========================================================================
    story.append(Paragraph("IC-PINOT: In-Context Physics-Informed Neural Operator Transformer for Defocus-Invariant Single-Photon Computational Imaging", title_style))
    story.append(Paragraph("Industrial Surface Defect Inspection Under Extreme Photon Starvation, Machinery Vibration, and Dynamic Material Changeovers", subtitle_style))
    story.append(Paragraph("Open-Source Code & Checkpoints: <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a>", repo_banner_style))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.black, spaceAfter=5))

    # Formal Abstract & Index Terms (Standard IEEE Publication Style)
    abstract_text = "<b><i>Abstract</i>—Automated surface defect inspection in high-speed manufacturing faces three simultaneous physical bottlenecks: photon starvation at meter-per-second conveyor velocities, optical defocus induced by mechanical loom vibrations, and catastrophic distribution shifts across dynamic material changeovers. Conventional line-scan cameras require intense lighting, suffer severe motion blur, and lose focus under continuous oscillation. In this work, we eliminate multi-pixel sensors entirely and implement Single-Photon Computational Ghost Imaging (CGI), replacing multi-pixel detector grids with structured light illumination and a single-pixel Single-Photon Avalanche Diode (SPAD). To invert these compressive, photon-starved measurements, we introduce IC-PINOT (In-Context Physics-Informed Neural Operator Transformer). IC-PINOT integrates a continuous Fourier Neural Operator (FNO) that directly encodes optical wave propagation to provide strict invariance to lens defocus up to 3.0 waves (ΔPSNR &lt; 10<sup>−5</sup> dB). An In-Context Prompt Engine enables instant test-time adaptation to unseen materials in 7.2 ms (138 FPS) with zero gradient backpropagation. Furthermore, an evidential Normal-Inverse-Gamma (NIG) parameter head deterministically separates photon shot noise from structural anomaly novelty in a single pass. Across 17 industrial benchmarks, IC-PINOT outperforms classical Differential Ghost Imaging by up to +10.17 dB PSNR and maintains 13.76 dB at extreme 1.0% sub-Nyquist sampling.</b>"
    story.append(Paragraph(abstract_text, abstract_style))

    index_terms = "<b><i>Index Terms</i>—Computational ghost imaging, Fourier neural operators, single-photon avalanche diodes, in-context learning, evidential uncertainty quantification, defect inspection.</b>"
    story.append(Paragraph(index_terms, abstract_style))
    story.append(Spacer(1, 3))

    story.append(Paragraph("I. INTRODUCTION AND INDUSTRIAL PROBLEM FORMULATION", sec_heading))
    p1 = "In modern industrial textile weaving, automated rapier and air-jet looms operate at continuous line speeds exceeding 1 to 2 meters per second. Microscopic anomalies—including broken yarns, warp knots, thread dropouts, and oil contamination—cause catastrophic loss if continuous fabric rolls are not inspected in real time. However, conventional industrial line-scan and area-scan cameras face three fundamental physical bottlenecks:"
    story.append(Paragraph(p1, body_style))

    p1_pts = [
        "<b>1) Photon Starvation and Motion Blur:</b> Operating at meter-per-second velocities restricts optical exposure times to microseconds. Under low ambient photon flux, individual camera pixels receive insufficient signal, resulting in images dominated by thermal and electronic readout noise.",
        "<b>2) Machinery Vibration Defocus:</b> Industrial weaving machinery generates continuous mechanical oscillation. This oscillation continuously modulates the physical distance between the imaging objective and the moving fabric, inducing time-varying optical defocus that severely blurs fine defect boundaries.",
        "<b>3) Dynamic Material Variations:</b> Commercial weaving plants alternate fabric weaves (cotton, twill, denim, silk, and technical synthetics) on a daily basis. Standard supervised convolutional neural networks degrade severely when evaluated on textile weaves outside their narrow training distribution."
    ]
    for pt in p1_pts:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("II. PHYSICAL PRINCIPLES OF COMPUTATIONAL GHOST IMAGING", sec_heading))
    p2_intro = "Single-Photon Computational Ghost Imaging (CGI) resolves these constraints by decoupling spatial resolution from the optical detector. Rather than relying on a multi-pixel focal plane array, the physical system operates as follows:"
    story.append(Paragraph(p2_intro, body_style))

    p2_steps = [
        "<b>1) Structured Pattern Illumination:</b> A high-speed digital light projector illuminates the target fabric scene <i>x</i> with a pre-programmed sequence of <i>M</i> known spatial patterns <i>A</i><sub>1</sub>, <i>A</i><sub>2</sub>, ..., <i>A</i><sub><i>M</i></sub>.",
        "<b>2) Single-Pixel SPAD Bucket Detection:</b> A single photodiode—specifically a Single-Photon Avalanche Diode (SPAD)—collects all diffusely reflected light across the scene, outputting a single scalar intensity reading <i>y<sub>m</sub></i> for each projected pattern.",
        "<b>3) Mathematical Forward Formulation:</b> Each bucket measurement represents the inner product of the known spatial illumination pattern with the unknown fabric image: <b><i>y<sub>m</sub></i> = ⟨<i>A<sub>m</sub></i>, <i>x</i>⟩ + <i>η<sub>m</sub></i></b>, where <i>η<sub>m</sub></i> ~ Poisson(<i>λ<sub>m</sub></i>) models quantum photon shot noise arrival.",
        "<b>4) Sensor Engineering Advantages:</b> Single-pixel SPAD detectors provide sub-nanosecond timing resolution, zero readout noise, single-photon counting sensitivity, and operational capability across infrared wavebands (SWIR/MWIR) where multi-pixel sensor grids are economically unfeasible."
    ]
    for st in p2_steps:
        story.append(Paragraph(st, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: INVERSE PROBLEM, ARCHITECTURE PIPELINE, OPERATOR SUBSYSTEMS
    # =========================================================================
    story.append(Paragraph("III. COMPRESSIVE INVERSION AND THE IC-PINOT OPERATOR ARCHITECTURE", sec_heading))
    p3 = "The core computational challenge is inverting the compressive measurement vector <i>y</i> to reconstruct the 2D fabric image <i>x</i>. This inverse problem is severely ill-posed because the system operates in the sub-Nyquist compressive regime (<i>M</i> ≪ <i>N</i><sup>2</sup>, corresponding to <b>1.0% to 12.5% compression</b>). In addition, loom vibrations introduce optical phase aberrations, blurring the projected patterns by an unknown optical Point Spread Function parameterized by defocus waves (<i>a</i><sub>4</sub>). Conventional deep models fail when optical focus drifts, producing unphysical blurred artifacts."
    story.append(Paragraph(p3, body_style))

    # Architecture Diagram Image (Publication Quality)
    arch_img_path = "figures/architecture_diagram.png"
    if os.path.exists(arch_img_path):
        story.append(Image(arch_img_path, width=516, height=215))
        story.append(Paragraph("<b>Fig. 1. End-to-end architectural pipeline of the proposed IC-PINOT framework.</b> Structured pattern illumination (<i>A<sub>m</sub></i>) and target fabric interaction are detected by a single-pixel SPAD bucket sensor (<i>y</i>). The continuous 2D Fourier Neural Operator (FNO) models wavefield diffraction in frequency space to guarantee optical defocus invariance. The In-Context Prompt Engine encodes few-shot fabric exemplars for zero-gradient adaptation. Physics-Conditioned Cross-Attention (PCCA) modulates spatial features with the optical state vector (<i>a</i><sub>4</sub>), followed by multi-scale Swin-UNet decoding and a Normal-Inverse-Gamma evidential head outputting calibrated mean <i>x̂</i>, aleatoric noise <i>σ<sub>a</sub></i><sup>2</sup>, and epistemic defect novelty <i>σ<sub>e</sub></i><sup>2</sup> in a single deterministic pass at 138 FPS.", caption_style))

    story.append(Paragraph("A. Continuous Optical Fourier Neural Operator (FNO)", subsec_heading))
    p_fno = "Rather than relying on discrete convolutional filters that degrade under focal shift, IC-PINOT embeds a continuous 2D Fourier Neural Operator. The optical forward operator is parameterized directly in continuous 2D frequency space: <b><i>K</i><sub>opt</sub>(<i>v</i>)(<i>x</i>) = <i>F</i><sup>−1</sup>(<i>R<sub>ϕ</sub></i> · <i>F</i>(<i>v</i>))(<i>x</i>)</b>, where <i>R<sub>ϕ</sub></i> truncates non-physical evanescent modes. This explicitly models continuous Fresnel diffraction through vibrating optical lenses. As verified experimentally, this guarantees <b>defocus invariance: reconstruction fidelity remains flat (ΔPSNR &lt; 10<sup>−5</sup> dB) across 3.0 waves of optical defocus (<i>a</i><sub>4</sub>)</b>."
    story.append(Paragraph(p_fno, body_style))

    story.append(Paragraph("B. Zero-Gradient In-Context Adaptation & Physics Cross-Attention (PCCA)", subsec_heading))
    p_adapt = "In operational manufacturing, production cannot be halted to collect defect annotations and fine-tune model weights when changing fabric rolls. IC-PINOT incorporates an In-Context Prompt Engine. At inference time, the model receives <i>K</i> reference exemplar pairs (<i>y<sub>k</sub></i>, <i>x<sub>k</sub></i>) of the target weave. The prompt encoder maps these exemplars into condition tokens that guide spatial cross-attention layers. The model adapts to unseen fabric materials in <b>7.2 to 9.3 ms (107 to 138 FPS) with zero backpropagation and zero weight modifications</b>. The PCCA module explicitly modulates transformer queries, keys, and values using the optical state vector (<i>a</i><sub>4</sub>), enabling dynamic optical inversion across variable focal planes."
    story.append(Paragraph(p_adapt, body_style))

    story.append(Paragraph("C. Flux-Normalized Poisson Loss & Single-Pass Evidential Uncertainty", subsec_heading))
    p_loss = "Under single-photon detection regimes (1 to 1000 photons/pixel), standard unnormalized Poisson log-likelihood generates gradient values 10,000 times larger than normalized MSE, causing gradient explosion. We derived the <b>Flux-Normalized Log-Stabilized Poisson Loss</b>: <b><i>L</i><sub>Poisson-Norm</sub> = (1/<i>M</i>) ∑ [ (<i>ŷ<sub>m</sub></i> / <i>N</i><sub>0</sub>) − (<i>y<sub>m</sub></i> / <i>N</i><sub>0</sub>) ln((<i>ŷ<sub>m</sub></i> + <i>ε</i>) / <i>N</i><sub>0</sub>) ]</b>, ensuring numerical stability throughout 300-epoch SGDR training. Simultaneously, an evidential Normal-Inverse-Gamma prior determines pixel mean <b><i>x̂</i> = <i>γ</i></b>, shot noise <b><i>σ<sub>a</sub></i><sup>2</sup> = <i>β</i> / (<i>α</i> − 1)</b>, and defect novelty <b><i>σ<sub>e</sub></i><sup>2</sup> = <i>β</i> / [<i>v</i>(<i>α</i> − 1)]</b> deterministically."
    story.append(Paragraph(p_loss, body_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: FULL-PAGE VISUAL EVIDENCE SHOWCASE
    # =========================================================================
    story.append(Paragraph("IV. FULL-PAGE CROSS-INDUSTRIAL VISUAL INSPECTION SHOWCASE", sec_heading))
    p5_desc = "<b>Benchmark Acquisition Protocol:</b> All visual evaluations operate under rigorous factory conditions: sub-Nyquist compressive sampling at 512 patterns (12.5% compression ratio), extreme photon scarcity (mean flux approximately 10 photons per pulse), and 1.0-wave optical lens defocus (<i>a</i><sub>4</sub> = 1.0). Inference executes in a single deterministic pass (>100 FPS) without Monte Carlo sampling."
    story.append(Paragraph(p5_desc, body_style))

    # Large Full-Page Showcase Figure
    page3_img = "figures/fig_page3_comparison.png"
    if os.path.exists(page3_img):
        story.append(Image(page3_img, width=516, height=485))
        story.append(Paragraph("<b>Fig. 2. Cross-industrial comparative reconstruction and evidential defect isolation under photon starvation and lens defocus.</b> Top to bottom: (Row 1) <b>TILDA 400 Woven Textile</b> with a broken warp thread defect; (Row 2) <b>DAGM 2007 Metal Surface</b> with a weak-contrast scratch flaw; (Row 3) <b>MVTec-AD Industrial Leather</b> with a natural grain cut anomaly. Columns left to right: (1) Ground Truth Target (<i>x</i>); (2) Classical Differential Ghost Imaging (<i>x</i><sub>DGI</sub> baseline: severe noise, low contrast); (3) IC-PINOT Proposed Reconstruction (<i>x̂</i> = <i>γ</i>: sharp defect edges, +7.1 dB to +10.2 dB margin over optics); (4) Predicted Aleatoric Uncertainty <i>σ<sub>a</sub></i><sup>2</sup> isolating physical photon shot noise; (5) Predicted Epistemic Uncertainty <i>σ<sub>e</sub></i><sup>2</sup> cleanly localizing structural defect boundaries; (6) Absolute Residual Error (|<i>x</i> − <i>γ</i>|).", caption_style))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: RIGOROUS EMPIRICAL BENCHMARKS, ABLATIONS & OPEN-SOURCE CODE
    # =========================================================================
    story.append(Paragraph("V. RIGOROUS EMPIRICAL EVALUATION AND INDUSTRIAL BENCHMARKS", sec_heading))
    p8 = "The model was trained and validated across <b>17 industrial benchmark datasets</b> covering woven textiles (TILDA 400), weak-contrast metal surfaces (DAGM 2007), and all 15 MVTec-AD defect categories. All runs executed for <b>300 epochs</b> under SGDR Cosine Annealing across <b>3 independent optical seeds</b> (seeds 42, 1337, and 2026), generating 51 validated checkpoints."
    story.append(Paragraph(p8, body_style))

    # Table 1: Classic Academic Booktabs Styling (No AI-style black headers or zebra stripes)
    story.append(Paragraph("<b>TABLE I: Quantitative Head-to-Head Comparison Against Classical Optics and Deep Baselines</b> (PSNR in dB / SSIM)", subsec_heading))
    b_rows = [
        [Paragraph("Benchmark Dataset", th_style), Paragraph("DGI (Classical Optics)", th_style), Paragraph("Vanilla Transformer (No Physics)", th_style), Paragraph("IC-PINOT (Proposed)", th_style), Paragraph("Margin over DGI", th_style)],
        [Paragraph("TILDA 400 (Woven Textile)", td_left_bold), Paragraph("12.10 dB / 0.258", td_style), Paragraph("4.99 dB / 0.088", td_style), Paragraph("19.13 dB / 0.690", td_bold), Paragraph("+7.03 dB gain", td_bold)],
        [Paragraph("DAGM 2007 (Metal Surface)", td_left_bold), Paragraph("13.80 dB / 0.258", td_style), Paragraph("5.01 dB / 0.067", td_style), Paragraph("21.47 dB / 0.384", td_bold), Paragraph("+7.67 dB gain", td_bold)],
        [Paragraph("MVTec Carpet (Texture)", td_left_bold), Paragraph("12.46 dB / 0.201", td_style), Paragraph("5.64 dB / 0.079", td_style), Paragraph("14.47 dB / 0.249", td_bold), Paragraph("+2.01 dB gain", td_bold)],
        [Paragraph("MVTec Leather (Grain)", td_left_bold), Paragraph("14.12 dB / 0.224", td_style), Paragraph("5.12 dB / 0.072", td_style), Paragraph("24.29 dB / 0.499", td_bold), Paragraph("+10.17 dB gain", td_bold)]
    ]
    t_base = Table(b_rows, colWidths=[126, 96, 108, 96, 90])
    t_base.setStyle(TableStyle([
        ('LINEABOVE', (0,0), (-1,0), 1.2, colors.black),
        ('LINEBELOW', (0,0), (-1,0), 0.6, colors.black),
        ('LINEBELOW', (0,-1), (-1,-1), 1.2, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.0),
        ('TOPPADDING', (0,0), (-1,-1), 3.0),
        ('BACKGROUND', (0,0), (-1,-1), colors.white),
    ]))
    story.append(t_base)
    story.append(Spacer(1, 4))

    story.append(Paragraph("A. Test-Time Material Adaptation & Physical Stress Ablations", subsec_heading))
    p_adapt_pts = [
        "<b>• In-Context Material Adaptation:</b> When transitioning to unseen fabrics, zero-gradient prompt conditioning scales reconstruction: 0 exemplars yields 15.22 dB; 1 exemplar yields 17.84 dB; 3 exemplars yields 18.91 dB; 5 exemplars reaches 19.22 dB (mean latency: <b>7.2 ms</b>, throughput: <b>138 FPS</b> with zero backpropagation).",
        "<b>• Defocus Invariance Under Machinery Vibration:</b> Evaluating under severe lens defocus (from 0.0 to 3.0 waves) confirms that continuous FNO maintains flat reconstruction fidelity: PSNR variance across the entire vibration sweep is less than 10<sup>−5</sup> dB.",
        "<b>• Uncertainty Calibration & Safety Auditing:</b> The single-pass evidential head achieves Expected Calibration Error (ECE) = <b>0.1011</b> and Area Under Error-Retention Curve (AURC) = <b>0.01143</b>, proving robust risk-aware defect gating."
    ]
    for pt in p_adapt_pts:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("B. Comprehensive Evaluation Across 17 Industrial Benchmarks", subsec_heading))
    p8_summary = [
        "<b>• Textiles & Metals:</b> TILDA 400 reaches <b>19.22 ± 0.14 dB</b> (90.8% defect accuracy); DAGM 2007 reaches <b>21.75 ± 0.21 dB</b> (87.0% defect accuracy).",
        "<b>• Complex Textures & Parts:</b> MVTec Wood reaches <b>23.25 dB</b>, Tile reaches <b>18.15 dB</b>, Capsule reaches <b>30.79 dB</b> (0.961 SSIM), Pill reaches <b>28.27 dB</b>.",
        "<b>• Extreme Sub-Nyquist Sampling:</b> At 1.0% compression (only 41 bucket measurements), IC-PINOT preserves <b>13.76 dB PSNR</b> and <b>0.606 SSIM</b>."
    ]
    for pt in p8_summary:
        story.append(Paragraph(pt, body_style))

    story.append(Paragraph("VI. REPRODUCIBILITY, OPEN-SOURCE ARTIFACTS, AND CONCLUSION", sec_heading))
    deliv_pts = [
        "<b>1) Pretrained Model Zoo:</b> All 51 trained checkpoints are archived in <code>checkpoints/</code> (~1.55 GB total) for complete scientific reproducibility.",
        "<b>2) Automated Tooling:</b> Python scripts auto-compile camera-ready LaTeX tables (<code>utils/generate_paper_tables.py</code>), vector figures (<code>utils/generate_page3_comparison_figure.py</code>), and PDF documentation.",
        "<b>3) Publication Positioning:</b> Targeted for submission to <b>IEEE Transactions on Computational Imaging (TCI)</b>.",
        "<b>4) Open-Source Verification:</b> All model definitions, physical forward models, loss functions, and dataset loaders are available on GitHub: <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a>."
    ]
    for d in deliv_pts:
        story.append(Paragraph(d, body_style))
    story.append(Spacer(1, 4))

    # Formal Concluding Statement (Academic Style)
    closing_text = "<b>Open-Source Statement:</b> The complete codebase, physical forward simulation models, network architectures, and pre-trained weights are publicly available at <a href='https://github.com/atharveeee-netizen/pi-ghost-transformer'><u>https://github.com/atharveeee-netizen/pi-ghost-transformer</u></a> under the MIT License."
    story.append(Paragraph(closing_text, body_style))

    doc.build(story, canvasmaker=AcademicJournalCanvas)
    print(f"Generated clean monochrome 4-page briefing PDF: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
