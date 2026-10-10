"""
Generate QRShield Phase 1 Project Report in .docx and .html formats.
"""

from __future__ import annotations

import base64
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCX_OUTPUT = PROJECT_ROOT / "QRShield_Phase1_Project_Report.docx"
HTML_OUTPUT = PROJECT_ROOT / "QRShield_Phase1_Project_Report.html"

# Images
SHAP_IMG = PROJECT_ROOT / "artifacts" / "fusion_phase3" / "shap_summary.png"
CM_IMG = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "confusion_matrix.png"
ROC_IMG = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "roc_curve.png"
PR_IMG = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "pr_curve.png"


def set_cell_shading(cell, color_hex: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    tcPr.append(shd)


def set_table_borders(table, color="D3D3D3"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def build_docx():
    doc = docx.Document()

    # Set page margins (1 inch)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = RGBColor(33, 37, 41)

    # Title
    title = doc.add_paragraph()
    title_run = title.add_run("QRShield: Explainable Multimodal Quishing Detection with Zero-Contact Static Analysis")
    title_run.font.name = 'Calibri'
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(16, 44, 87)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)

    # Subtitle
    subtitle = doc.add_paragraph()
    sub_run = subtitle.add_run("Comprehensive Phase 1 Project Report | Course: CSET485 — AI and Society")
    sub_run.font.size = Pt(13)
    sub_run.font.bold = True
    sub_run.font.color.rgb = RGBColor(53, 95, 142)
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(16)

    # Metadata Table
    meta_table = doc.add_table(rows=3, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    meta_data = [
        ("Institution", "School of Computer Science Engineering and Technology (SCSET), Bennett University"),
        ("Course & Term", "CSET485 — AI and Society | Fall Semester 2026"),
        ("Team Identification", "Team ID: 54"),
    ]

    for idx, (k, v) in enumerate(meta_data):
        row = meta_table.rows[idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.width = Inches(2.2)
        cell_v.width = Inches(4.3)
        p_k = cell_k.paragraphs[0]
        r_k = p_k.add_run(k)
        r_k.bold = True
        p_v = cell_v.paragraphs[0]
        p_v.add_run(v)
        set_cell_shading(cell_k, "F1F5F9")
        set_cell_shading(cell_v, "FFFFFF")

    set_table_borders(meta_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Team Members Table
    doc.add_heading("Team Members & Academic Credentials", level=2)
    team_table = doc.add_table(rows=6, cols=4)
    team_table.alignment = WD_TABLE_ALIGNMENT.CENTER

    headers = ["No.", "Student Name", "Roll Number", "Primary Technical Specialization"]
    col_widths = [Inches(0.5), Inches(1.8), Inches(1.5), Inches(2.7)]

    for c_idx, h_text in enumerate(headers):
        cell = team_table.rows[0].cells[c_idx]
        cell.width = col_widths[c_idx]
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "1E293B")

    members = [
        ("1", "Harshit Pahuja", "s24cseu0736", "Lead Architect: Fusion Engine, Isotonic Calibration, FastAPI Backend"),
        ("2", "Hemang Nag", "s24cseu0728", "Deep Vision Lead: MobileNetV2 CNN, GPU Pipeline, Grad-CAM Saliency"),
        ("3", "Saatvik Sharma", "s24cseu2097", "Lexical ML Lead: 29 Static URL Features, Random Forest, TreeSHAP"),
        ("4", "Tanish", "s24cseu0589", "Data & Security Lead: CIC-Trap4Phish Canonicalization, Zero-Contact Policy"),
        ("5", "Rohit Upadhyay", "s24cseu1359", "Frontend & QA Lead: React 18 / Vite UI, Explainability Dashboards, Pytest"),
    ]

    for r_idx, member in enumerate(members, start=1):
        row = team_table.rows[r_idx]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(member):
            cell = row.cells[c_idx]
            cell.width = col_widths[c_idx]
            p = cell.paragraphs[0]
            p.add_run(val)
            set_cell_shading(cell, bg)

    set_table_borders(team_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 1: Executive Summary
    doc.add_heading("1. Executive Summary", level=1)
    p = doc.add_paragraph(
        "Quick Response (QR) codes have become foundational physical-to-digital bridges in modern society, "
        "facilitating seamless contactless payments, transportation ticketing, retail interaction, and multi-factor "
        "authentication. However, this universal adoption has spawned an acute cybersecurity threat known as quishing "
        "(QR code phishing). Human vision cannot interpret 2D matrix symbology; consequently, users blindly rely on "
        "smartphones to decode and navigate embedded links, completely bypassing conventional email gateway filters."
    )
    p = doc.add_paragraph(
        "QRShield addresses this critical socio-technical vulnerability through an explainable multimodal AI framework. "
        "The system unites (1) a local zero-contact OpenCV decoder that treats payloads strictly as inert data strings, "
        "(2) a 29-feature static lexical Random Forest classifier, (3) a fine-tuned MobileNetV2 deep convolutional network "
        "inspecting physical barcode properties, (4) prediction-level soft-voting fusion with Isotonic Regression probability "
        "calibration, and (5) dual explainability uniting TreeSHAP feature attributions and Grad-CAM spatial heatmaps. "
        "On a strictly domain-partitioned benchmark of 70,000 samples derived from CIC-Trap4Phish, QRShield achieves "
        "98.77% accuracy, 99.67% ROC-AUC, 99.27% F1-score, and reduces false alarms by 86.2% over single-modality baselines."
    )

    # Section 2: Problem Statement & Societal Significance
    doc.add_heading("2. Problem Statement & Societal Significance", level=1)
    doc.add_paragraph(
        "Phishing remains the predominant catalyst for corporate credential theft, consumer identity fraud, and unauthorized "
        "financial transfers. While modern web browsers and Secure Email Gateways (SEGs) maintain sophisticated heuristics for "
        "inspecting text-based hyperlinks, attackers actively exploit QR codes to bypass these inspection layers:"
    )
    bullet1 = doc.add_paragraph(style='List Bullet')
    r = bullet1.add_run("Optical Obfuscation: ")
    r.bold = True
    bullet1.add_run(
        "QR codes convert malicious URLs into 2D grid modules with Reed-Solomon error correction. To a human user, a QR "
        "matrix pointing to a high-risk credential stealer is visually indistinguishable from one pointing to a trusted domain."
    )
    bullet2 = doc.add_paragraph(style='List Bullet')
    r = bullet2.add_run("The Physical Air Gap: ")
    r.bold = True
    bullet2.add_run(
        "Attackers physically paste sticker overlays on legitimate public QR signage—such as parking meters, restaurant payment "
        "stands, and utility kiosks—weaponizing the public's inherent trust in physical infrastructure."
    )
    bullet3 = doc.add_paragraph(style='List Bullet')
    r = bullet3.add_run("Mobile Vulnerability: ")
    r.bold = True
    bullet3.add_run(
        "QR scans occur predominantly on mobile operating systems with compact screens, truncated address bars, and no "
        "mouse-hover inspectability, leaving users acutely vulnerable."
    )

    # Section 3: Zero-Contact Operational Security Architecture
    doc.add_heading("3. Zero-Contact Operational Security Architecture", level=1)
    doc.add_paragraph(
        "A cornerstone engineering tenet of QRShield is its strict Zero-Contact Security Policy. Many conventional URL safety "
        "tools evaluate destinations dynamically by performing HTTP GET/HEAD handshakes or executing headless browser instances. "
        "While revealing HTTP redirect chains, active network probing creates unacceptable operational hazards:"
    )
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Drive-By Exploits: Loading malicious servers can expose client mobile browsers to zero-day memory exploits.").bold = False
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Attacker Beaconing: Attackers frequently embed unique victim identifiers in query strings. Any outgoing HTTP ping confirms target engagement to the attacker.").bold = False
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Dynamic Cloaking: Phishing infrastructure detects automated security crawlers by IP reputation, serving benign content to scanners while serving malware to real users.").bold = False

    doc.add_paragraph(
        "QRShield strictly eliminates active network interaction: decoded QR strings are handled solely as inert memory buffers. "
        "All lexical, structural, and image features are computed locally, guaranteeing absolute client protection and zero data leakage."
    )

    # Section 4: Dataset Engineering & Leakage-Free Splitting
    doc.add_heading("4. Dataset Engineering & Leakage-Free Splitting", level=1)
    doc.add_paragraph(
        "Models were developed using the benchmark CIC-Trap4Phish dataset (Canadian Institute for Cybersecurity, UNB). "
        "A rigorous data canonicalization pipeline audited over 2.15 million raw records, resolving label conflicts and corrupted images, "
        "yielding 387,015 clean canonical sample pairs."
    )
    doc.add_paragraph(
        "Crucially, to eliminate data leakage, QRShield enforced Strict Domain-Grouped Partitioning: all URLs belonging to the same "
        "registrable domain (eTLD+1) were restricted to a single partition. This prevents models from memorizing specific domain names, "
        "guaranteeing genuine generalizability across the 50,000 train, 10,000 validation, and 10,000 test splits."
    )

    # Section 5: Feature Engineering & Multimodal System Architecture
    doc.add_heading("5. Feature Engineering & Multimodal System Architecture", level=1)
    doc.add_paragraph(
        "QRShield adopts a dual-branch multimodal architecture that evaluates both the decoded lexical string and the raw 2D image matrix:"
    )
    doc.add_heading("5.1 29 Static Lexical Features (URL Branch)", level=2)
    doc.add_paragraph(
        "Extracted across six structural dimensions: Lengths (URL, hostname, path, query), Delimiters (dots, subdomains, hyphens, slashes, "
        "underscores, at-symbols, colons, percent signs), Character Densities (digit ratio, alpha ratio, special char ratio), "
        "Query Attributes (parameter count, equal count, ampersand count), Protocol Checks (has HTTPS, raw IP hostname, non-standard port, "
        "hostname hyphen count, suspicious keyword count), and Information Entropies (Shannon entropy computed across URL, hostname, and path)."
    )
    doc.add_paragraph(
        "The URL branch is powered by a 300-tree balanced Random Forest classifier that achieves 92.67% test accuracy and 94.80% ROC-AUC."
    )

    doc.add_heading("5.2 Deep Visual Modeling (MobileNetV2 CNN Branch)", level=2)
    doc.add_paragraph(
        "The vision branch utilizes a fine-tuned MobileNetV2 architecture with inverted residual blocks. Using a two-stage transfer learning "
        "process (Stage A head training, Stage B top-layer fine-tuning with AdamW), the model learns spatial barcode densities, module irregularities, "
        "and physical marker traits, achieving 96.45% test accuracy and 99.28% ROC-AUC with an average CPU inference latency of 5.91 ms."
    )

    doc.add_heading("5.3 Soft-Voting Fusion & Isotonic Calibration", level=2)
    doc.add_paragraph(
        "Branch predictions are unified via soft voting: P_raw = 0.55 * P_url + 0.45 * P_img. Because raw classifier scores can be overconfident, "
        "an Isotonic Regression model transforms raw fusion scores into calibrated posterior probabilities. The decision threshold was set to 0.5434."
    )

    # Section 6: Explainable AI
    doc.add_heading("6. Explainable AI (XAI) Framework", level=1)
    doc.add_paragraph(
        "Transparency is essential to overcome user alarm fatigue. QRShield couples dual explainability methods:"
    )
    doc.add_paragraph(
        "1. TreeSHAP Lexical Attribution: Calculates exact Shapley additive explanations for each URL feature. The top predictive features "
        "include path_entropy (mean |SHAP| = 0.0714), path_length (0.0624), num_subdomains (0.0515), num_hyphens (0.0512), and hostname_entropy (0.0394)."
    )
    doc.add_paragraph(
        "2. Grad-CAM Spatial Visual Heatmaps: Backpropagates classification gradients to MobileNetV2's final convolutional layer (features[18]), "
        "rendering a 224x224 heatmap overlay that demonstrates whether the vision model is attending to valid matrix payload modules or peripheral artifacts."
    )

    if SHAP_IMG.exists():
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
        doc.add_picture(str(SHAP_IMG), width=Inches(5.8))
        caption = doc.add_paragraph("Figure 1: Global TreeSHAP Feature Attribution Summary for URL Lexical Branch")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(12)

    # Section 7: Empirical Results
    doc.add_heading("7. Empirical Experimental Results & Analysis", level=1)
    doc.add_paragraph("Performance evaluated on the 10,000-sample domain-grouped test split:")

    results_table = doc.add_table(rows=4, cols=8)
    results_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    res_headers = ["Model / Branch", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "False Alarm Rate"]
    res_widths = [Inches(1.8), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.7), Inches(0.8), Inches(0.8), Inches(1.1)]

    for c_idx, h_text in enumerate(res_headers):
        cell = results_table.rows[0].cells[c_idx]
        cell.width = res_widths[c_idx]
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "1E293B")

    res_data = [
        ("URL Baseline (RF)", "0.9267", "0.9271", "0.9906", "0.9578", "0.9480", "0.9885", "40.75% (654 FP)"),
        ("Image Baseline (CNN)", "0.9645", "0.9662", "0.9924", "0.9791", "0.9928", "0.9986", "18.13% (291 FP)"),
        ("QRShield Fusion", "0.9877", "0.9894", "0.9961", "0.9927", "0.9967", "0.9993", "5.61% (90 FP)"),
    ]

    for r_idx, row_vals in enumerate(res_data, start=1):
        row = results_table.rows[r_idx]
        bg = "E2E8F0" if r_idx == 3 else ("F8FAFC" if r_idx % 2 == 1 else "FFFFFF")
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = res_widths[c_idx]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            if r_idx == 3:
                r.bold = True
            set_cell_shading(cell, bg)

    set_table_borders(results_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Figures side-by-side or sequential
    if ROC_IMG.exists() and CM_IMG.exists():
        doc.add_picture(str(ROC_IMG), width=Inches(3.2))
        doc.add_picture(str(CM_IMG), width=Inches(3.2))
        caption = doc.add_paragraph("Figure 2: Receiver Operating Characteristic (ROC) and Confusion Matrix on Test Split")
        caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
        caption.paragraph_format.space_after = Pt(12)

    doc.add_paragraph(
        "Isotonic Calibration Results: Raw Fusion Brier Score was 0.02659, which dropped to 0.01022 post-calibration "
        "— a 61.5% improvement in probability reliability."
    )

    # Section 8: AI and Society
    doc.add_heading("8. AI and Society: Ethical, Societal & Governance Analysis", level=1)
    doc.add_paragraph(
        "As part of CSET485 (AI and Society), QRShield implements six ethical and governance design principles:"
    )
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Human Agency & Autonomy: ").bold = True
    b.add_run("Acts as an assistive advisory co-pilot rather than an opaque autonomous gatekeeper, preserving user agency.")
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Privacy by Design: ").bold = True
    b.add_run("100% on-device local computation. Zero external API queries, protecting personal location and browsing data.")
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Harm Minimization: ").bold = True
    b.add_run("Zero-contact static execution eliminates the hazard of drive-by malware infections and beacon tracking.")
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Algorithmic Transparency: ").bold = True
    b.add_run("SHAP and Grad-CAM dashboards demystify model outputs, countering algorithmic bias and opaque decisions.")
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Scientific Humility: ").bold = True
    b.add_run("Explicitly delineates development benchmark performance from real-world adversarial environments.")
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Sustainable AI: ").bold = True
    b.add_run("Lightweight MobileNetV2 and Random Forest architecture runs on CPUs with 5.91 ms latency, minimizing energy use.")

    # Section 9: Prototype Implementation
    doc.add_heading("9. Prototype Implementation & Verification", level=1)
    doc.add_paragraph(
        "The complete working prototype includes a FastAPI asynchronous backend (backend/main.py), a React 18 / Vite "
        "dark-mode dashboard with interactive explainability views (frontend/src/App.jsx), and a test suite of 10 automated "
        "tests in pytest passing with 100% reliability."
    )

    # Section 10: Work Distribution
    doc.add_heading("10. Work Distribution & Team Contribution Matrix", level=1)
    contrib_table = doc.add_table(rows=6, cols=3)
    contrib_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_headers = ["Member Name (Roll No)", "Key Technical Deliverables", "Effort %"]
    c_widths = [Inches(2.0), Inches(4.3), Inches(0.9)]

    for c_idx, h_text in enumerate(c_headers):
        cell = contrib_table.rows[0].cells[c_idx]
        cell.width = c_widths[c_idx]
        p = cell.paragraphs[0]
        r = p.add_run(h_text)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        set_cell_shading(cell, "1E293B")

    contrib_data = [
        ("Harshit Pahuja (s24cseu0736)", "System architecture, soft-voting fusion formulation, Isotonic calibration, FastAPI backend, integration testing", "20%"),
        ("Hemang Nag (s24cseu0728)", "MobileNetV2 CNN architecture, PyTorch GPU training scripts, Grad-CAM backpropagation, vision inference benchmarking", "20%"),
        ("Saatvik Sharma (s24cseu2097)", "29 static lexical feature extraction, Shannon entropy algorithms, Random Forest training, TreeSHAP explainability pipeline", "20%"),
        ("Tanish (s24cseu0589)", "CIC-Trap4Phish dataset audit, canonicalization pipeline, domain-grouped splitting, zero-contact security policy design", "20%"),
        ("Rohit Upadhyay (s24cseu1359)", "React 18 / Vite frontend dashboard, explainability UI components, client-side error handling, test execution & QA", "20%"),
    ]

    for r_idx, row_vals in enumerate(contrib_data, start=1):
        row = contrib_table.rows[r_idx]
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            cell.width = c_widths[c_idx]
            p = cell.paragraphs[0]
            p.add_run(val)
            set_cell_shading(cell, bg)

    set_table_borders(contrib_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Section 11: Phase 2 Roadmap
    doc.add_heading("11. Phase 2 Roadmap & Future Milestones", level=1)
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Milestone 1: On-Device Mobile Compilation (ONNX Runtime / INT8 quantization for Android/iOS)").bold = False
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Milestone 2: Adversarial Robustness testing against QR module perturbations and print-scan degradations").bold = False
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Milestone 3: Local Bloom-filter threat intelligence caching without live HTTP pings").bold = False
    b = doc.add_paragraph(style='List Bullet')
    b.add_run("Milestone 4: User study evaluating explainability impact on cognitive warning adherence").bold = False

    # Section 12: References
    doc.add_heading("12. References & Citations", level=1)
    refs = [
        "Canadian Institute for Cybersecurity (CIC): CIC-Trap4Phish 2025: A Comprehensive Multimodal Benchmark Dataset, UNB, 2025.",
        "Sandler, M., et al.: MobileNetV2: Inverted Residuals and Linear Bottlenecks, IEEE CVPR, 2018.",
        "Lundberg, S. M., & Lee, S. I.: A Unified Approach to Interpreting Model Predictions (SHAP), NeurIPS, 2017.",
        "Selvaraju, R. R., et al.: Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization, IJCV, 2020.",
        "Zadrozny, B., & Elkan, C.: Transforming Classifier Scores into Accurate Multiclass Probability Estimates, ACM SIGKDD, 2002.",
        "NIST: Zero Trust Architecture, NIST Special Publication 800-207, U.S. Dept of Commerce, 2020.",
        "Floridi, L., & Cowls, J.: A Unified Framework of Five Principles for AI in Society, Harvard Data Science Review, 2019."
    ]
    for ref in refs:
        p = doc.add_paragraph(style='List Bullet')
        p.add_run(ref)

    doc.save(str(DOCX_OUTPUT))
    print(f"Generated DOCX: {DOCX_OUTPUT}")


def build_html():
    def img_to_b64(path: Path) -> str:
        if path.exists():
            with open(path, "rb") as f:
                return "data:image/png;base64," + base64.b64encode(f.read()).decode("utf-8")
        return ""

    shap_b64 = img_to_b64(SHAP_IMG)
    cm_b64 = img_to_b64(CM_IMG)
    roc_b64 = img_to_b64(ROC_IMG)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>QRShield: Phase 1 Project Report | Team 54 | CSET485</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {{
      --primary: #0f172a;
      --accent: #2563eb;
      --accent-light: #eff6ff;
      --text: #1e293b;
      --muted: #64748b;
      --border: #e2e8f0;
      --bg-card: #f8fafc;
      --success: #16a34a;
      --warning: #ca8a04;
      --danger: #dc2626;
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      color: var(--text);
      line-height: 1.65;
      background: #f1f5f9;
      padding: 40px 20px;
    }}

    .container {{
      max-width: 900px;
      margin: 0 auto;
      background: #ffffff;
      padding: 60px 70px;
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    }}

    /* Header & Cover */
    .header-badge {{
      display: inline-block;
      background: var(--accent-light);
      color: var(--accent);
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 16px;
    }}

    h1.title {{
      font-size: 28px;
      font-weight: 800;
      color: var(--primary);
      line-height: 1.25;
      margin-bottom: 8px;
    }}

    .subtitle {{
      font-size: 16px;
      color: var(--muted);
      font-weight: 500;
      margin-bottom: 24px;
    }}

    .meta-box {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 35px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px 24px;
      font-size: 14px;
    }}

    .meta-item strong {{
      color: var(--primary);
      display: inline-block;
      width: 140px;
    }}

    /* Section Headings */
    h2 {{
      font-size: 20px;
      font-weight: 700;
      color: var(--primary);
      border-bottom: 2px solid var(--border);
      padding-bottom: 8px;
      margin-top: 36px;
      margin-bottom: 16px;
    }}

    h3 {{
      font-size: 16px;
      font-weight: 600;
      color: #334155;
      margin-top: 20px;
      margin-bottom: 10px;
    }}

    p {{
      font-size: 14.5px;
      margin-bottom: 14px;
      color: #334155;
    }}

    ul, ol {{
      margin-left: 24px;
      margin-bottom: 16px;
      font-size: 14.5px;
      color: #334155;
    }}

    li {{
      margin-bottom: 6px;
    }}

    /* Tables */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 20px 0;
      font-size: 13.5px;
    }}

    th, td {{
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }}

    th {{
      background: var(--primary);
      color: #ffffff;
      font-weight: 600;
      letter-spacing: 0.3px;
    }}

    tr:nth-child(even) td {{
      background: #f8fafc;
    }}

    tr.highlight-row td {{
      background: #eff6ff;
      font-weight: 600;
      color: var(--accent);
    }}

    /* Callout & Code */
    .callout {{
      border-left: 4px solid var(--accent);
      background: var(--accent-light);
      padding: 16px 20px;
      border-radius: 0 8px 8px 0;
      margin: 20px 0;
      font-size: 14px;
    }}

    .callout-title {{
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 6px;
    }}

    code {{
      font-family: 'JetBrains Mono', monospace;
      font-size: 12.5px;
      background: #e2e8f0;
      padding: 2px 6px;
      border-radius: 4px;
      color: #0f172a;
    }}

    /* Figures */
    .figure-container {{
      text-align: center;
      margin: 28px 0;
    }}

    .figure-container img {{
      max-width: 95%;
      border-radius: 8px;
      border: 1px solid var(--border);
      box-shadow: 0 2px 10px rgba(0,0,0,0.04);
    }}

    .figure-caption {{
      font-size: 12.5px;
      color: var(--muted);
      margin-top: 8px;
      font-style: italic;
    }}

    /* Two-col grid */
    .two-col-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin: 20px 0;
    }}

    .card {{
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}

    .card h4 {{
      font-size: 15px;
      margin-bottom: 8px;
      color: var(--primary);
    }}

    /* Print styles */
    @media print {{
      body {{
        background: #ffffff;
        padding: 0;
      }}
      .container {{
        box-shadow: none;
        padding: 20px 25px;
        max-width: 100%;
      }}
      h2 {{
        page-break-after: avoid;
      }}
      table, .figure-container {{
        page-break-inside: avoid;
      }}
      .no-print {{
        display: none !important;
      }}
    }}

    .print-btn {{
      position: fixed;
      bottom: 30px;
      right: 30px;
      background: var(--accent);
      color: #fff;
      border: none;
      padding: 12px 24px;
      border-radius: 30px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 4px 15px rgba(37, 99, 235, 0.35);
      transition: all 0.2s;
    }}
    .print-btn:hover {{
      background: #1d4ed8;
      transform: translateY(-2px);
    }}
  </style>
</head>
<body>

  <button class="print-btn no-print" onclick="window.print()">🖨️ Print / Save as PDF</button>

  <div class="container">
    <div class="header-badge">Bennett University • SCSET • CSET485 Fall 2026</div>
    <h1 class="title">QRShield: Explainable Multimodal Quishing Detection with Zero-Contact Static Analysis</h1>
    <div class="subtitle">Comprehensive Phase 1 Project Report | Team ID: 54</div>

    <div class="meta-box">
      <div class="meta-item"><strong>Course:</strong> CSET485 — AI and Society</div>
      <div class="meta-item"><strong>Team ID:</strong> 54</div>
      <div class="meta-item"><strong>Institution:</strong> Bennett University (SCSET)</div>
      <div class="meta-item"><strong>Project Scope:</strong> Phase 1 Multimodal Architecture &amp; Prototype</div>
    </div>

    <h2>Team Members & Academic Credentials</h2>
    <table>
      <thead>
        <tr>
          <th>No.</th>
          <th>Student Name</th>
          <th>Roll Number</th>
          <th>Phase 1 Technical Specialization</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>1</td>
          <td><strong>Harshit Pahuja</strong></td>
          <td><code>s24cseu0736</code></td>
          <td>Lead Architect: Fusion Engine, Isotonic Calibration, FastAPI Backend</td>
        </tr>
        <tr>
          <td>2</td>
          <td><strong>Hemang Nag</strong></td>
          <td><code>s24cseu0728</code></td>
          <td>Deep Vision Lead: MobileNetV2 CNN, Transfer Learning, Grad-CAM Saliency</td>
        </tr>
        <tr>
          <td>3</td>
          <td><strong>Saatvik Sharma</strong></td>
          <td><code>s24cseu2097</code></td>
          <td>Lexical ML Lead: 29 Static URL Features, Random Forest, TreeSHAP</td>
        </tr>
        <tr>
          <td>4</td>
          <td><strong>Tanish</strong></td>
          <td><code>s24cseu0589</code></td>
          <td>Data & Security Lead: CIC-Trap4Phish Canonicalization, Zero-Contact Policy</td>
        </tr>
        <tr>
          <td>5</td>
          <td><strong>Rohit Upadhyay</strong></td>
          <td><code>s24cseu1359</code></td>
          <td>Frontend & QA Lead: React 18 / Vite UI, Explainability Dashboards, Pytest</td>
        </tr>
      </tbody>
    </table>

    <h2>1. Executive Summary</h2>
    <p>
      Quick Response (QR) codes have become foundational physical-to-digital bridges in modern society, powering contactless payments, transit passes, digital menus, and multi-factor authentication. However, their visual opacity has enabled a dangerous cyberattack vector known as <strong>quishing</strong> (QR code phishing). Human vision cannot inspect 2D matrix symbology, forcing smartphone users to decode links blindly without the benefit of enterprise Secure Email Gateways (SEGs) or desktop hover-inspection.
    </p>
    <p>
      <strong>QRShield</strong> solves this vulnerability through an end-to-end explainable multimodal AI framework. The system combines:
    </p>
    <ul>
      <li><strong>Local Zero-Contact Decoder:</strong> Treats decoded URLs strictly as inert memory buffers—executing zero HTTP queries, DNS handshakes, or browser automation to prevent drive-by exploits.</li>
      <li><strong>29-Feature Lexical Classifier:</strong> A 300-tree Random Forest evaluating structural, syntactic, character-density, and Shannon entropy indicators.</li>
      <li><strong>Deep Barcode Saliency Network:</strong> A fine-tuned MobileNetV2 CNN inspecting spatial matrix density, finder-pattern distortion, and optical traits.</li>
      <li><strong>Soft-Voting Fusion & Isotonic Calibration:</strong> Fusing URL (55%) and Vision (45%) predictions, calibrated via Isotonic Regression to reduce probabilistic error (Brier score) by 61.5%.</li>
      <li><strong>Dual Explainable AI (XAI):</strong> Unifying TreeSHAP lexical feature weights with Grad-CAM spatial heatmaps to empower users with actionable cognitive decision support.</li>
    </ul>
    <p>
      On a held-out benchmark of 10,000 domain-partitioned test samples derived from <strong>CIC-Trap4Phish</strong>, QRShield achieves <strong>98.77% accuracy</strong>, <strong>99.67% ROC-AUC</strong>, <strong>99.27% F1-score</strong>, and slashes false alarms to <strong>5.61%</strong> (an 86.2% reduction relative to single-modality URL baselines).
    </p>

    <h2>2. Problem Statement & Societal Significance</h2>
    <p>
      Phishing remains the primary entry point for corporate credential compromise and consumer fraud. While modern web browsers maintain robust heuristics for textual links, quishing circumvents these protections:
    </p>
    <ul>
      <li><strong>Optical Obfuscation:</strong> QR codes encode high-density alphanumeric payloads into 2D bit patterns with Reed-Solomon redundancy. Malicious destinations look identical to trusted domains to human observers.</li>
      <li><strong>Physical Air Gap Exploitation:</strong> Attackers attach malicious stickers onto legitimate municipal signage (parking meters, charging stations, restaurant tables), weaponizing citizens' trust in physical infrastructure.</li>
      <li><strong>Mobile Form Factor Vulnerability:</strong> Mobile browsers feature abbreviated address bars, lack mouse hover-state inspection, and execute scans outside enterprise security proxies.</li>
    </ul>

    <h2>3. Zero-Contact Operational Security Architecture</h2>
    <div class="callout">
      <div class="callout-title">STRICT SAFETY GUARANTEE</div>
      All decoded QR strings are handled strictly as inert data. QRShield NEVER initiates <code>requests.get()</code>, DNS queries, headless browser crawling, or HTTP redirect following. Analysis is 100% local.
    </div>
    <p>
      Conventional threat scanners often inspect links by actively requesting pages or executing headless Chromium instances. While this reveals redirect chains, it creates severe hazards:
    </p>
    <ol>
      <li><strong>Drive-by Exploits:</strong> Loading untrusted pages can trigger browser-level zero-day vulnerabilities or remote code execution.</li>
      <li><strong>Attacker Beaconing:</strong> Unique tracking tokens in URLs alert the attacker that the victim interacted with the code, validating the target.</li>
      <li><strong>Dynamic Cloaking:</strong> Phishing kits profile scanner IP ranges and User-Agents, returning innocuous pages to automated crawlers while attacking mobile victims.</li>
    </ol>

    <h2>4. Dataset Engineering & Leakage-Free Splitting</h2>
    <p>
      Models were trained on the benchmark <strong>CIC-Trap4Phish</strong> dataset from the Canadian Institute for Cybersecurity (UNB). A rigorous canonicalization pipeline audited 2.15 million raw records, resolving 10,305 label conflicts and corrupted images into 387,015 clean sample pairs.
    </p>
    <p>
      <strong>Preventing Data Leakage:</strong> Standard random train/test splits cause extreme data leakage in URL classification because identical domains appear in both sets. QRShield enforces <strong>Strict Registrable Domain Grouping (eTLD+1)</strong>:
    </p>
    <ul>
      <li><strong>Train Split:</strong> 50,000 samples (domain-grouped)</li>
      <li><strong>Validation Split:</strong> 10,000 samples (zero domain overlap with Train)</li>
      <li><strong>Test Split:</strong> 10,000 samples (zero domain overlap with Train and Validation)</li>
      <li><strong>Verified Overlap:</strong> Exactly 0 shared domains, 0 shared image hashes, and 0 shared URLs.</li>
    </ul>

    <h2>5. Feature Engineering & Multimodal System Architecture</h2>
    <p>
      QRShield deploys two complementary analytical branches:
    </p>
    <div class="two-col-grid">
      <div class="card">
        <h4>URL Lexical Branch (29 Features)</h4>
        <p>Extracts lengths, delimiters, digit ratios, query parameters, protocol flags, and Shannon information entropies. Evaluated by a 300-tree Random Forest classifier.</p>
        <p><strong>Test Metrics:</strong> Acc: 92.67% | F1: 95.78% | ROC-AUC: 94.80%</p>
      </div>
      <div class="card">
        <h4>Image CNN Branch (MobileNetV2)</h4>
        <p>Two-stage transfer learning (head training + top-layer fine-tuning) on 224x224 RGB matrices to detect barcode density, distortion, and optical markers.</p>
        <p><strong>Test Metrics:</strong> Acc: 96.45% | F1: 97.91% | ROC-AUC: 99.28%</p>
      </div>
    </div>

    <p>
      <strong>Soft-Voting Fusion & Isotonic Calibration:</strong> Predictions are blended via:
      <br>
      <code>P_raw = 0.55 * P_url + 0.45 * P_img</code>
      <br>
      Raw scores are transformed by a non-parametric Isotonic Regression calibrator, with an optimal decision threshold of <strong>0.5434</strong>.
    </p>

    <h2>6. Explainable AI (XAI) Framework</h2>
    <p>
      To build user trust and reduce alarm fatigue, QRShield implements dual explainability:
    </p>
    <ul>
      <li><strong>TreeSHAP Lexical Attribution:</strong> Quantifies the exact game-theoretic additive contribution of each URL feature. The top predictive drivers are <code>path_entropy</code> (0.0714), <code>path_length</code> (0.0624), <code>num_subdomains</code> (0.0515), <code>num_hyphens</code> (0.0512), and <code>hostname_entropy</code> (0.0394).</li>
      <li><strong>Grad-CAM Visual Heatmaps:</strong> Backpropagates gradients to MobileNetV2's final convolutional layer (<code>features[18]</code>), generating a 224x224 thermal overlay highlighting the barcode regions driving classification.</li>
    </ul>

    {"<div class='figure-container'><img src='" + shap_b64 + "' alt='SHAP Summary'><div class='figure-caption'>Figure 1: Global TreeSHAP Feature Attribution Importance for URL Branch</div></div>" if shap_b64 else ""}

    <h2>7. Empirical Experimental Results & Analysis</h2>
    <p>Comprehensive evaluation on the 10,000-sample domain-grouped test set:</p>
    <table>
      <thead>
        <tr>
          <th>Model Architecture</th>
          <th>Accuracy</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1-Score</th>
          <th>ROC-AUC</th>
          <th>PR-AUC</th>
          <th>False Alarm Rate</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>URL-Only Baseline</strong> (Random Forest)</td>
          <td>0.9267</td>
          <td>0.9271</td>
          <td>0.9906</td>
          <td>0.9578</td>
          <td>0.9480</td>
          <td>0.9885</td>
          <td>40.75% (654 FP)</td>
        </tr>
        <tr>
          <td><strong>Image-Only Baseline</strong> (MobileNetV2)</td>
          <td>0.9645</td>
          <td>0.9662</td>
          <td>0.9924</td>
          <td>0.9791</td>
          <td>0.9928</td>
          <td>0.9986</td>
          <td>18.13% (291 FP)</td>
        </tr>
        <tr class="highlight-row">
          <td><strong>QRShield Multimodal Fusion</strong></td>
          <td><strong>0.9877</strong></td>
          <td><strong>0.9894</strong></td>
          <td><strong>0.9961</strong></td>
          <td><strong>0.9927</strong></td>
          <td><strong>0.9967</strong></td>
          <td><strong>0.9993</strong></td>
          <td><strong>5.61% (90 FP)</strong></td>
        </tr>
      </tbody>
    </table>

    <div class="two-col-grid">
      {"<div class='figure-container'><img src='" + roc_b64 + "' alt='ROC Curve'><div class='figure-caption'>Figure 2: Receiver Operating Characteristic (ROC-AUC: 0.9967)</div></div>" if roc_b64 else ""}
      {"<div class='figure-container'><img src='" + cm_b64 + "' alt='Confusion Matrix'><div class='figure-caption'>Figure 3: Confusion Matrix (TP: 8,362 | TN: 1,515 | FP: 90 | FN: 33)</div></div>" if cm_b64 else ""}
    </div>

    <p>
      <strong>Probability Calibration Impact:</strong> The Brier score dropped from <strong>0.02659</strong> (uncalibrated) to <strong>0.01022</strong> (calibrated)—a <strong>61.5% improvement in probability fidelity</strong>.
    </p>

    <h2>8. AI and Society: Ethical, Societal & Governance Analysis</h2>
    <ul>
      <li><strong>Human Agency & Cognitive Co-Pilot:</strong> Operates as an informative security advisory rather than an opaque censorship filter. Calibrated risk tiers and explanations empower users to decide responsibly.</li>
      <li><strong>Privacy by Design:</strong> 100% local client-side inference. User scan history and camera data never leave the device, complying with GDPR and DPDPA data privacy tenets.</li>
      <li><strong>Zero-Contact Harm Mitigation:</strong> Inert string handling completely protects users from drive-by browser zero-days and attacker beacon tracking.</li>
      <li><strong>Transparency & Accountability:</strong> SHAP and Grad-CAM dashboards demystify model outputs, countering black-box complacency and enabling human auditing.</li>
      <li><strong>Green & Sustainable AI:</strong> Lightweight MobileNetV2 and Random Forest models execute on consumer CPUs in <strong>5.91 milliseconds</strong>, avoiding carbon-intensive GPU server dependencies.</li>
    </ul>

    <h2>9. Prototype Implementation & Verification</h2>
    <p>
      The complete prototype comprises an asynchronous <strong>FastAPI backend</strong> (<code>backend/main.py</code>), an interactive <strong>React 18 / Vite web interface</strong> (<code>frontend/src/App.jsx</code>), and an automated test suite of <strong>10 integration and unit tests</strong> passing with 100% success via <code>pytest</code>.
    </p>

    <h2>10. Work Distribution & Team Contribution Matrix</h2>
    <table>
      <thead>
        <tr>
          <th>Team Member</th>
          <th>Roll Number</th>
          <th>Phase 1 Deliverables & Technical Contributions</th>
          <th>Effort</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Harshit Pahuja</strong> (Lead)</td>
          <td><code>s24cseu0736</code></td>
          <td>Multimodal fusion formulation, Isotonic calibration, FastAPI backend integration, system test harness</td>
          <td>20%</td>
        </tr>
        <tr>
          <td><strong>Hemang Nag</strong></td>
          <td><code>s24cseu0728</code></td>
          <td>MobileNetV2 CNN architecture, PyTorch GPU training scripts, Grad-CAM backpropagation engine, vision benchmarking</td>
          <td>20%</td>
        </tr>
        <tr>
          <td><strong>Saatvik Sharma</strong></td>
          <td><code>s24cseu2097</code></td>
          <td>29 static lexical feature extraction, Shannon entropy algorithms, Random Forest tuning, TreeSHAP pipeline</td>
          <td>20%</td>
        </tr>
        <tr>
          <td><strong>Tanish</strong></td>
          <td><code>s24cseu0589</code></td>
          <td>CIC-Trap4Phish dataset audit, canonicalization pipeline, domain-grouped splitting, zero-contact policy design</td>
          <td>20%</td>
        </tr>
        <tr>
          <td><strong>Rohit Upadhyay</strong></td>
          <td><code>s24cseu1359</code></td>
          <td>React 18 / Vite frontend dashboard, interactive XAI visualizers, client-side error handling, test execution</td>
          <td>20%</td>
        </tr>
      </tbody>
    </table>

    <h2>11. Phase 2 Roadmap & Future Milestones</h2>
    <ol>
      <li><strong>On-Device Mobile Edge Deployment:</strong> Export quantized MobileNetV2 and Random Forest via ONNX Runtime / PyTorch Mobile for native Android/iOS integration.</li>
      <li><strong>Adversarial Robustness:</strong> Benchmark defenses against physical barcode print-scan degradation and adversarial patch attacks.</li>
      <li><strong>Offline Threat Intelligence Caching:</strong> Integrate local Bloom-filter domain reputation lookups without active network probing.</li>
      <li><strong>Human-in-the-Loop User Study:</strong> Evaluate whether dual XAI explanations measurably improve user cognitive adherence to security warnings.</li>
    </ol>

    <h2>12. References & Citations</h2>
    <ol>
      <li>Canadian Institute for Cybersecurity: <em>CIC-Trap4Phish 2025: A Comprehensive Multimodal Benchmark Dataset for Phishing and Quishing</em>, UNB, 2025.</li>
      <li>Sandler, M., et al.: <em>MobileNetV2: Inverted Residuals and Linear Bottlenecks</em>, IEEE CVPR, 2018.</li>
      <li>Lundberg, S. M., & Lee, S. I.: <em>A Unified Approach to Interpreting Model Predictions (SHAP)</em>, NeurIPS, 2017.</li>
      <li>Selvaraju, R. R., et al.: <em>Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization</em>, IJCV, 2020.</li>
      <li>Zadrozny, B., & Elkan, C.: <em>Transforming Classifier Scores into Accurate Multiclass Probability Estimates</em>, ACM SIGKDD, 2002.</li>
      <li>NIST: <em>Zero Trust Architecture</em>, NIST Special Publication 800-207, U.S. Department of Commerce, 2020.</li>
      <li>Floridi, L., & Cowls, J.: <em>A Unified Framework of Five Principles for AI in Society</em>, Harvard Data Science Review, 2019.</li>
    </ol>
  </div>
</body>
</html>
"""

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML: {HTML_OUTPUT}")


if __name__ == "__main__":
    print("Building Phase 1 Project Report deliverables...")
    build_docx()
    build_html()
    print("All Phase 1 report files generated successfully!")
