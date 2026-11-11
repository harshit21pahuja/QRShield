"""
Generate QRShield Viva Preparation Master Guide in .docx and .pdf formats.
Course: CSET485 — AI and Society
Student: Harshit Pahuja (Team Lead & System Architect)
"""

from __future__ import annotations

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCX_OUTPUT = PROJECT_ROOT / "QRShield_Viva_Master_Guide.docx"
PDF_OUTPUT = PROJECT_ROOT / "QRShield_Viva_Master_Guide.pdf"

# Artifact Image paths
SHAP_IMG = PROJECT_ROOT / "artifacts" / "fusion_phase3" / "shap_summary.png"
CM_IMG = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "confusion_matrix.png"
ROC_IMG = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "roc_curve.png"
GRADCAM_IMG = PROJECT_ROOT / "artifacts" / "image_baseline_gpu" / "gradcam" / "gradcam_qr_final_0354435.png"


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


def add_callout(doc, title: str, text: str, border_color="102C57", bg_color="F8FAFC"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_shading(cell, bg_color)

    # Set left border thick, others none
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    r_title = p.add_run(f"■ {title}\n")
    r_title.bold = True
    r_title.font.size = Pt(11)
    r_title.font.color.rgb = RGBColor(16, 44, 87)

    r_text = p.add_run(text)
    r_text.font.size = Pt(10)
    r_text.font.italic = True
    r_text.font.color.rgb = RGBColor(51, 65, 85)

    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_section_heading(doc, text: str, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(14)
    h.paragraph_format.space_after = Pt(4)
    for r in h.runs:
        r.font.name = "Calibri"
        if level == 1:
            r.font.size = Pt(15)
            r.font.bold = True
            r.font.color.rgb = RGBColor(16, 44, 87)
        elif level == 2:
            r.font.size = Pt(12)
            r.font.bold = True
            r.font.color.rgb = RGBColor(53, 95, 142)
    return h


def build_docx():
    doc = docx.Document()

    # Set margins
    for s in doc.sections:
        s.top_margin = Inches(0.8)
        s.bottom_margin = Inches(0.8)
        s.left_margin = Inches(0.8)
        s.right_margin = Inches(0.8)

    # Base Normal Style
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Calibri"
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(33, 37, 41)

    # ─────────────────────────────────────────────────────────────
    # COVER / HEADER
    # ─────────────────────────────────────────────────────────────
    title_p = doc.add_paragraph()
    title_run = title_p.add_run("QRShield: Explainable Multimodal Quishing Detection")
    title_run.font.size = Pt(20)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(16, 44, 87)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_p.paragraph_format.space_after = Pt(2)

    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run("Comprehensive Viva Examination & Defense Handbook | Fall 2026")
    sub_run.font.size = Pt(12)
    sub_run.font.bold = True
    sub_run.font.color.rgb = RGBColor(53, 95, 142)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(12)

    # Metadata Box Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    meta_info = [
        ("Course & Institution", "CSET485 — AI and Society | SCSET, Bennett University, Greater Noida"),
        ("Candidate / Lead", "Harshit Pahuja (Roll No: s24cseu0736) — Lead & System Architect"),
        ("Team Identification", "Team ID: 54 (Hemang Nag, Saatvik Sharma, Tanish, Rohit Upadhyay, Bharat Kumar)"),
        ("Core Architecture", "Zero-Contact Static Analysis, 29-Feature Random Forest, MobileNetV2 CNN, Isotonic Calibration"),
    ]

    for idx, (k, v) in enumerate(meta_info):
        row = meta_table.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.0)
        c1.width = Inches(4.5)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.size = Pt(9.5)
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(v)
        r1.font.size = Pt(9.5)
        set_cell_shading(c0, "F1F5F9")
        set_cell_shading(c1, "FFFFFF")

    set_table_borders(meta_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ─────────────────────────────────────────────────────────────
    # 1. 90-SECOND ELEVATOR PITCH
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "1. Executive 90-Second Opening Pitch (Verbal Script)")

    pitch_text = (
        "\"Good morning/afternoon, respected examiners.\n"
        "Our project is QRShield: Explainable Multimodal Quishing Detection with Zero-Contact Static Analysis "
        "developed for CSET485 (AI and Society).\n\n"
        "Quishing (QR code phishing) is rapidly expanding because human vision cannot decipher 2D barcodes. "
        "Attackers exploit this opacity to bypass corporate email gateways and deceive mobile citizens with physical sticker overlays.\n\n"
        "Traditional detectors either blindly open destinations or perform hazardous dynamic crawling. QRShield eliminates these risks "
        "via a strict Zero-Contact architecture: we decode barcodes purely in memory with OpenCV without making any live network pings or DNS lookups.\n\n"
        "Our system fuses two complementary static modalities:\n"
        "  1. A 29-feature lexical Random Forest model analyzing URL character distributions and Shannon entropy.\n"
        "  2. A fine-tuned MobileNetV2 CNN examining the physical barcode matrix for visual and density anomalies.\n\n"
        "By combining both branches using soft voting (55% URL / 45% Image) and calibrating probabilities with Isotonic Regression, "
        "QRShield achieves 98.77% accuracy and slashes false alarms by 86.2% on 70,000 domain-grouped benchmark samples from CIC-Trap4Phish, "
        "running in just 5.91 milliseconds on CPU with complete TreeSHAP and Grad-CAM explainability.\""
    )
    add_callout(doc, "Verbal Delivery Script (Memorize for Viva Opening)", pitch_text, "102C57", "F0F9FF")

    # ─────────────────────────────────────────────────────────────
    # 2. SOCIETAL & ETHICAL FOUNDATIONS (CSET485)
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "2. AI & Society: Ethical, Societal & Governance Framework")

    p = doc.add_paragraph()
    p.add_run(
        "In direct alignment with CSET485 (AI and Society), QRShield embeds core ethical and legal principles directly into its system architecture:\n"
    )

    ethics_table = doc.add_table(rows=5, cols=3)
    ethics_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Ethical Pillar", "Societal & Security Rationale", "QRShield Technical Implementation"]
    col_w = [Inches(1.5), Inches(2.5), Inches(2.5)]

    for c_i, h in enumerate(headers):
        cell = ethics_table.rows[0].cells[c_i]
        cell.width = col_w[c_i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        set_cell_shading(cell, "1E3A8A")
        r.font.color.rgb = RGBColor(255, 255, 255)

    ethics_data = [
        (
            "Harm Prevention (Zero-Contact)",
            "Dynamic web crawlers risk triggering drive-by malware exploits or notifying attackers via tracking tokens (?victim=123).",
            "Inert memory-only string handling. Strictly 0 network requests, 0 DNS lookups, and 0 headless browser rendering."
        ),
        (
            "Privacy by Design (Anti-Surveillance)",
            "Cloud-based scanners create telemetry logs of everywhere a citizen travels and every transaction they attempt.",
            "100% on-device local execution. Zero telemetry leaves the device, complying with India's DPDPA 2023 & GDPR."
        ),
        (
            "Human Agency & Transparency",
            "Opaque black-box blocks ('Access Denied') cause alarm fatigue and strip users of decision autonomy.",
            "Cognitive Co-Pilot approach: Calibrated risk scores (Low/Mod/High/Crit) backed by TreeSHAP & Grad-CAM justifications."
        ),
        (
            "Sustainable & Green AI",
            "Heavy vision-language models hosted on GPU server farms produce massive operational carbon footprints.",
            "Compact MobileNetV2 (3.4M parameters) + Random Forest operates in 5.91 ms on commodity CPUs with negligible energy."
        )
    ]

    for r_i, row in enumerate(ethics_data, start=1):
        for c_i, val in enumerate(row):
            cell = ethics_table.rows[r_i].cells[c_i]
            cell.width = col_w[c_i]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if c_i == 0:
                r.bold = True
            set_cell_shading(cell, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")

    set_table_borders(ethics_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ─────────────────────────────────────────────────────────────
    # 3. END-TO-END SYSTEM ARCHITECTURE
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "3. Multimodal System Architecture & Operational Flow")

    doc.add_paragraph(
        "The QRShield processing pipeline operates across six distinct, deterministic stages:"
    )

    steps = [
        ("Stage 1: Zero-Contact Ingestion & Matrix Extraction", "The uploaded QR image is ingested into memory as an uncompressed RGB matrix. OpenCV's cv2.QRCodeDetector identifies timing/finder patterns and decodes the embedded string into an inert buffer."),
        ("Stage 2: Lexical URL Feature Extraction (29 Dimensions)", "The inert string buffer is analyzed across 29 static lexical indicators including structural lengths, character ratios, subdomain hierarchies, query parameter counts, protocol flags, and Shannon character entropies."),
        ("Stage 3: Deep Vision Feature Extraction (MobileNetV2)", "In parallel, the raw QR image matrix is resized to 224x224x3, normalized via ImageNet distributions, and passed through an inverted-residual MobileNetV2 convolutional backbone to extract spatial and density representations."),
        ("Stage 4: Prediction-Level Soft Voting Fusion", "The independent class posterior estimates P_url and P_img are combined via optimal validation weights: P_raw = 0.55 * P_url + 0.45 * P_img."),
        ("Stage 5: Non-Parametric Isotonic Calibration", "The raw probability is transformed via a fitted Isotonic Regression step function to align nominal model confidence with empirical Bayesian event likelihood, evaluated against threshold tau* = 0.5434."),
        ("Stage 6: Dual Explainability Generation", "TreeSHAP decomposes the URL prediction into additive feature contributions, while Grad-CAM projects gradient activations onto the final convolutional feature map (features[18]) to generate a spatial thermal saliency map.")
    ]

    for title, desc in steps:
        p = doc.add_paragraph()
        r_t = p.add_run(f"• {title}: ")
        r_t.bold = True
        r_t.font.color.rgb = RGBColor(16, 44, 87)
        p.add_run(desc)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # ─────────────────────────────────────────────────────────────
    # 4. MODALITY DEEP DIVES
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "4. Technical Deep Dive: The Dual Branches")

    add_section_heading(doc, "4.1 Static Lexical URL Feature Extraction (29 Features)", level=2)
    doc.add_paragraph(
        "Phishing URLs exhibit structural and statistical anomalies compared to legitimate web destinations. "
        "QRShield categorizes 29 static features across six distinct mathematical groups:"
    )

    feat_table = doc.add_table(rows=7, cols=3)
    feat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    feat_headers = ["Feature Category", "Specific Variables", "Security & Fraud Indication"]
    f_widths = [Inches(1.5), Inches(2.5), Inches(2.5)]

    for c_i, h in enumerate(feat_headers):
        cell = feat_table.rows[0].cells[c_i]
        cell.width = f_widths[c_i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        set_cell_shading(cell, "334155")
        r.font.color.rgb = RGBColor(255, 255, 255)

    feat_data = [
        ("Structural Lengths (d=4)", "url_length, hostname_length, path_length, query_length", "Phishing URLs are extended to obfuscate targets or embed redirect parameters."),
        ("Delimiters (d=8)", "num_dots, num_subdomains, num_hyphens, num_underscores, num_slashes, num_at, num_percent, num_colon", "@ abuses browser basic auth; excessive hyphens spoof brands (e.g. secure-login-paypal.com)."),
        ("Character Ratios (d=4)", "digit_count, digit_ratio, alpha_ratio, special_ratio", "Machine-generated obfuscation exhibits elevated hex and numeric frequencies."),
        ("Query Structure (d=4)", "num_question, num_equals, num_ampersand, query_param_count", "Quantifies parameter stuffing utilized in credential-stealer phishing kits."),
        ("Protocol & Host Flags (d=6)", "has_https, has_ip_hostname, has_port, hostname_hyphen_count, suspicious_token_count", "Raw IPv4 hostnames (http://192.168.1.1/login) and missing TLS indicate severe malicious risk."),
        ("Shannon Entropies (d=3)", "url_entropy, hostname_entropy, path_entropy", "Measures information disorder: H = -sum(p_i * log2(p_i)). High entropy indicates algorithmic domain generation (DGA).")
    ]

    for r_i, row in enumerate(feat_data, start=1):
        for c_i, val in enumerate(row):
            cell = feat_table.rows[r_i].cells[c_i]
            cell.width = f_widths[c_i]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if c_i == 0:
                r.bold = True
            set_cell_shading(cell, "F8FAFC" if r_i % 2 == 1 else "FFFFFF")

    set_table_borders(feat_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_section_heading(doc, "4.2 Deep Vision Modeling (MobileNetV2 CNN Architecture)", level=2)
    doc.add_paragraph(
        "While lexical features inspect destination syntax, the visual QR grid captures optical properties that "
        "text analysis misses entirely:\n"
        "• High Version Density: Malicious QR codes frequently encode large URLs or obfuscated payloads, producing dense, fine-grained module grids.\n"
        "• Visual Artifacts & Tampering: Physical sticker overlays create misalignment along finder patterns and edge contrast variations.\n"
        "• MobileNetV2 Backbone: Uses inverted residual blocks with depthwise separable convolutions (3.4M parameters), drastically reducing FLOPs compared to ResNet-50.\n"
        "• Two-Stage Transfer Learning:\n"
        "   - Stage A: Feature extractor frozen; custom linear head trained with AdamW (lr = 1e-3, weight decay = 1e-4) for 2 epochs.\n"
        "   - Stage B: Top residual blocks (blocks 14–18) unfrozen and fine-tuned at lr = 1e-4 for 2 epochs."
    )

    # ─────────────────────────────────────────────────────────────
    # 5. MULTIMODAL FUSION & CALIBRATION MATHEMATICS
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "5. Multimodal Fusion & Isotonic Calibration Mathematics")

    math_box = (
        "1. Prediction-Level Soft Voting Fusion:\n"
        "   P_raw = w_url * P_url + w_img * P_img\n"
        "   Validation Grid Search: w_url = 0.55, w_img = 0.45 (Yielded maximum validation F1 = 0.9956)\n\n"
        "2. Isotonic Probability Calibration (PAVA Algorithm):\n"
        "   min_g sum_{i=1}^N (y_i - g(P_raw,i))^2   subject to g(p_a) <= g(p_b) for all p_a <= p_b\n"
        "   Calibrated Probability: P_cal = g_iso(P_raw)\n\n"
        "3. Decision Rule & Optimal Threshold:\n"
        "   y_hat = 1 (MALICIOUS) if P_cal >= 0.5434, else 0 (BENIGN)\n\n"
        "4. Probabilistic Error (Brier Score):\n"
        "   Brier Score = (1/N) * sum_{i=1}^N (P_i - y_i)^2\n"
        "   Raw Fusion Brier: 0.02659  -->  Calibrated Brier: 0.01022  (61.5% Error Reduction)"
    )
    add_callout(doc, "Mathematical Formulations", math_box, "2563EB", "EFF6FF")

    # ─────────────────────────────────────────────────────────────
    # 6. EXPLAINABLE AI (XAI)
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "6. Dual Explainable AI (XAI) Framework")

    doc.add_paragraph(
        "To eliminate black-box distrust, QRShield delivers dual-modality explanations:\n"
        "1. TreeSHAP (Lexical Attribution): Based on cooperative game theory, computes exact additive feature attributions phi_i such that sum(phi_i) = f(x) - E[f(X)].\n"
        "   Top 5 Predictive Signals Identified:\n"
        "     - path_entropy (mean |phi| = 0.0714)\n"
        "     - path_length (mean |phi| = 0.0624)\n"
        "     - num_subdomains (mean |phi| = 0.0515)\n"
        "     - num_hyphens (mean |phi| = 0.0512)\n"
        "     - hostname_entropy (mean |phi| = 0.0394)\n"
        "2. Grad-CAM (Spatial Visual Saliency): Computes gradients at MobileNetV2's final convolutional layer (features[18]) and calculates thermal activation overlays:\n"
        "   alpha_k^c = (1/Z) * sum_i sum_j (d y^c / d A_{i,j}^k),   L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c * A^k)"
    )

    # Embed SHAP and Grad-CAM images if available
    if SHAP_IMG.exists():
        doc.add_paragraph().paragraph_format.space_before = Pt(4)
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.add_run().add_picture(str(SHAP_IMG), width=Inches(5.5))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 1: TreeSHAP Global Feature Importance Summary across 1,000 Test Background Instances.")
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True

    if GRADCAM_IMG.exists():
        p_img2 = doc.add_paragraph()
        p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img2.add_run().add_picture(str(GRADCAM_IMG), width=Inches(3.8))
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap2 = p_cap2.add_run("Figure 2: Grad-CAM Thermal Activation Heatmap Highlighting Suspicious 2D Barcode Modules.")
        r_cap2.font.size = Pt(8.5)
        r_cap2.font.italic = True

    # ─────────────────────────────────────────────────────────────
    # 7. DATASET ENGINEERING & LEAKAGE PREVENTION
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "7. Dataset Engineering & Leakage-Free Splitting")

    doc.add_paragraph(
        "• Source Dataset: CIC-Trap4Phish (Canadian Institute for Cybersecurity, 2025).\n"
        "• Data Hygiene Pipeline: Filtered 2.15 million raw records down to 387,015 clean canonical pairs, resolving 10,305 contradictory duplicate labels.\n"
        "• Benchmark Evaluation Corpus (70,000 samples):\n"
        "   - Training Set: 50,000 samples (2,291 benign, 47,709 malicious)\n"
        "   - Validation Set: 10,000 samples (1,009 benign, 8,991 malicious)\n"
        "   - Test Set: 10,000 samples (1,605 benign, 8,395 malicious)\n\n"
        "• Strict Registrable Domain Grouping (eTLD+1):\n"
        "   In conventional random splits, URLs from the same domain (e.g. fake-bank.com/login and fake-bank.com/verify) leak across train and test sets, "
        "causing the model to memorize domain names rather than learning generalized structural features.\n"
        "   QRShield enforces strict domain-level partitioning using Mozilla's Public Suffix List (tldextract), guaranteeing:\n"
        "   --> Exactly 0 Shared Domains, 0 Shared URLs, and 0 Shared Image Hashes across splits."
    )

    # ─────────────────────────────────────────────────────────────
    # 8. EMPIRICAL RESULTS & KEY NUMBERS
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "8. Empirical Results & Benchmark Performance")

    results_table = doc.add_table(rows=4, cols=8)
    results_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    r_headers = ["Model Architecture", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "FPR", "CPU Latency"]
    r_widths = [Inches(1.8), Inches(0.65), Inches(0.65), Inches(0.65), Inches(0.65), Inches(0.7), Inches(0.75), Inches(0.65)]

    for c_i, h in enumerate(r_headers):
        cell = results_table.rows[0].cells[c_i]
        cell.width = r_widths[c_i]
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(8.5)
        set_cell_shading(cell, "1E3A8A")
        r.font.color.rgb = RGBColor(255, 255, 255)

    r_data = [
        ("URL Branch (Random Forest)", "92.67%", "92.71%", "99.06%", "95.78%", "94.80%", "40.75% (654/1605)", "0.81 ms"),
        ("Image Branch (MobileNetV2)", "96.45%", "96.62%", "99.24%", "97.91%", "99.28%", "18.13% (291/1605)", "5.10 ms"),
        ("QRShield Multimodal Fusion", "98.77%", "98.94%", "99.61%", "99.27%", "99.67%", "5.61% (90/1605)", "5.91 ms")
    ]

    for r_i, row in enumerate(r_data, start=1):
        for c_i, val in enumerate(row):
            cell = results_table.rows[r_i].cells[c_i]
            cell.width = r_widths[c_i]
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if r_i == 3:
                r.bold = True
            set_cell_shading(cell, "FEF3C7" if r_i == 3 else ("F8FAFC" if r_i % 2 == 1 else "FFFFFF"))

    set_table_borders(results_table, "CBD5E1")
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    doc.add_paragraph(
        "Key Numerical Takeaway for Viva: "
        "Multimodal fusion reduced false alarms from 654 (URL model) down to 90—an 86.2% reduction in false positives—while "
        "simultaneously lifting recall to 99.61% and total accuracy to 98.77% in under 6 milliseconds."
    )

    if CM_IMG.exists():
        p_img3 = doc.add_paragraph()
        p_img3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img3.add_run().add_picture(str(CM_IMG), width=Inches(3.8))
        p_cap3 = doc.add_paragraph()
        p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap3 = p_cap3.add_run("Figure 3: Confusion Matrix Benchmark on Held-out 10,000 Test Partition.")
        r_cap3.font.size = Pt(8.5)
        r_cap3.font.italic = True

    # ─────────────────────────────────────────────────────────────
    # 9. LEAD DEFENSE — HARSHIT PAHUJA
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "9. Harshit Pahuja's Personal Contribution Defense")

    doc.add_paragraph(
        "When the examination committee inquires regarding individual technical ownership:\n"
        "• System Architecture & Pipeline Design: Engineered the end-to-end dataflow linking OpenCV in-memory barcode decoding, parallel dual-branch inference, fusion math, and frontend API delivery.\n"
        "• Multimodal Soft Voting Formulation: Designed and conducted grid search optimization yielding the optimal 55% URL / 45% Image weighting.\n"
        "• Isotonic Probability Calibration: Implemented the PAVA non-parametric calibrator, reducing the Brier error score by 61.5% and establishing the tuned decision threshold tau* = 0.5434.\n"
        "• Production FastAPI Microservice: Built the async backend engine (backend/main.py) with structured Pydantic schemas, health checks, memory-cached model deserialization, and CORS security.\n"
        "• Verification & Quality Assurance: Authored the 10-suite automated integration test harness (tests/test_prototype.py), ensuring 100% test pass rate across all edge cases."
    )

    # ─────────────────────────────────────────────────────────────
    # 10. TOP 15 TOUGH VIVA QUESTIONS & MODEL ANSWERS
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "10. Top 15 Tough Viva Questions & Bulletproof Model Answers")

    qa_list = [
        (
            "Q1: Why did you choose a multimodal approach instead of relying purely on URL lexical features?",
            "Sir, lexical features inspect URL syntax but suffer a high 40.75% false positive rate because modern marketing URLs contain long query tokens and high entropy. The vision model inspects the physical 2D barcode matrix. By combining both signals, false alarms dropped from 654 to 90 (an 86.2% reduction) and accuracy rose to 98.77%."
        ),
        (
            "Q2: What is the Zero-Contact Security Policy and why is it critical?",
            "Traditional scanners crawl links with headless browsers. This exposes devices to drive-by browser RCE, triggers attacker tracking beacons (?victim=123) validating victim phone/email targets, and gets bypassed by cloaking filters. QRShield decodes barcodes strictly as inert string buffers in memory—zero socket connections, zero DNS pings, zero HTTP requests."
        ),
        (
            "Q3: How did you prevent data leakage during dataset partitioning?",
            "If URLs are partitioned randomly, multiple URLs from the same malicious domain (e.g. evil-portal.com/login and evil-portal.com/verify) end up in both training and test sets. The model simply memorizes the domain name. We enforced Strict Registrable Domain Grouping (eTLD+1) using Mozilla's Public Suffix List (tldextract), ensuring zero domain, URL, or image hash overlap across splits."
        ),
        (
            "Q4: Why MobileNetV2 rather than ResNet-50 or Vision Transformers (ViT)?",
            "MobileNetV2 uses inverted residual blocks and depthwise separable convolutions, keeping parameters down to 3.4M. ResNet-50 requires ~25M parameters and ViT requires ~86M, which are impractical for low-latency edge mobile processing. MobileNetV2 achieves 96.45% accuracy while inferencing in just 5.1 ms on a standard CPU."
        ),
        (
            "Q5: Why did you apply Isotonic Regression instead of Platt Scaling?",
            "Platt scaling assumes a parametric sigmoid relationship, which underfits when confidence distortions are non-monotonic or multi-modal. Isotonic Regression is non-parametric; it fits a piecewise monotonic step function via the Pool Adjacent Violators Algorithm (PAVA). It reduced our Brier calibration score by 61.5% (0.0266 to 0.0102)."
        ),
        (
            "Q6: What is a Brier Score and why does it matter in quishing detection?",
            "The Brier score measures the mean squared difference between predicted probabilities and actual binary outcomes: (1/N)*sum(P_i - y_i)^2. In security applications, an uncalibrated 90% confidence score might only correspond to a 60% real threat probability. Calibrated Brier score of 0.0102 ensures assigned risk tiers reflect real-world threat frequencies."
        ),
        (
            "Q7: Explain the mathematical intuition of Shannon Entropy in your URL branch.",
            "Shannon entropy H(T) = -sum(p_i * log2(p_i)) quantifies character randomness. Natural English words (e.g. /account/login) have low entropy (~2.5-3.5 bits/char). Algorithmic domain generation (DGAs) or token stuffing (e.g. /x89q2l_k9) have high entropy (>4.5 bits/char). Path and hostname entropy were identified by TreeSHAP as top-tier predictive signals."
        ),
        (
            "Q8: How does TreeSHAP provide local explainability?",
            "TreeSHAP calculates exact Shapley additive explanations from cooperative game theory: f(x) = phi_0 + sum(phi_i). It computes the marginal contribution of each feature across all possible decision tree feature permutations in polynomial time O(TLD^2), showing the user exactly which lexical anomalies (e.g. 4 deceptive hyphens) triggered the threat score."
        ),
        (
            "Q9: How does Grad-CAM work on a QR barcode image?",
            "Grad-CAM computes the partial derivatives of the malicious class score y^c with respect to feature maps A^k in the final convolutional layer (features[18]). These gradients are global-average-pooled into importance weights alpha_k^c and combined via ReLU to highlight the physical barcode modules (finder patterns, timing lines, dense payload matrix) that the CNN focused on."
        ),
        (
            "Q10: Why is the decision threshold set to 0.5434 instead of the default 0.5000?",
            "The default 0.5 threshold assumes symmetric misclassification costs and perfectly balanced data. Because quishing datasets feature natural class imbalance, we tuned the threshold on the validation split specifically to optimize the F1-score, settling on tau* = 0.5434 to achieve an optimal balance between precision (0.9894) and recall (0.9961)."
        ),
        (
            "Q11: What are the failure modes of QRShield (False Positives & False Negatives)?",
            "False Positives (90 samples): Benign marketing URLs featuring affiliate tracking strings and multi-tier hyphenated subdomains. False Negatives (33 samples): Newly registered, ultra-short clean domains that mimic normal landing pages before server-side redirect activation occurs."
        ),
        (
            "Q12: How does QRShield align with India's DPDPA 2023 and GDPR?",
            "Under GDPR and DPDPA Section 6, harvesting user URLs and camera feeds creates unlawful digital tracking footprints. QRShield processes all images and URLs locally on the user's client hardware. No scanned URLs, camera streams, or telemetry logs are ever transmitted to cloud servers."
        ),
        (
            "Q13: Why did you freeze the MobileNetV2 backbone in Stage A of transfer learning?",
            "If random classification head weights are backpropagated into a pre-trained backbone with a large learning rate, large initial gradient updates destroy the pre-trained ImageNet feature representations (catastrophic forgetting). Freezing the backbone stabilizes the head before unfreezing top layers (blocks 14–18) for delicate fine-tuning."
        ),
        (
            "Q14: How does your frontend UI prevent warning fatigue?",
            "Rather than showing generic 'Warning: Malicious Link' popups that users reflexively dismiss, QRShield uses cognitive co-pilot principles: clear color-coded risk tiers (Low/Mod/High/Crit), interactive TreeSHAP attribution badges explaining why it is risky, and a Grad-CAM visualizer with an inert copyable link destination."
        ),
        (
            "Q15: What are the planned Phase 2 extensions for QRShield?",
            "1. Edge Mobile Port: INT8 quantization and ONNX Runtime deployment for native Android/iOS camera integration. 2. Adversarial Defense: Hardening against printed barcode tearing and adversarial sticker patches. 3. Offline Bloom Filter: Local caching of 1M+ known malicious domain hashes for sub-millisecond instant lookup without network connectivity."
        )
    ]

    for q, a in qa_list:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(6)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.size = Pt(10)
        r_q.font.color.rgb = RGBColor(16, 44, 87)

        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_before = Pt(0)
        p_a.paragraph_format.space_after = Pt(4)
        r_a = p_a.add_run(f"Answer: {a}")
        r_a.font.size = Pt(9.5)
        r_a.font.color.rgb = RGBColor(51, 65, 85)

    # ─────────────────────────────────────────────────────────────
    # 11. LIVE DEMO & TEST VERIFICATION CHECKLIST
    # ─────────────────────────────────────────────────────────────
    add_section_heading(doc, "11. Prototype Verification & Live Demo Commands")

    cmd_box = (
        "1. Start FastAPI Backend Engine:\n"
        "   cd \"c:\\Users\\Harshit Pahuja\\Desktop\\New folder\\QRShield\"\n"
        "   .\\.venv313\\Scripts\\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000\n"
        "   --> Accessible at http://127.0.0.1:8000 (Swagger docs at /docs, health probe at /health)\n\n"
        "2. Start React 18 Frontend Dashboard:\n"
        "   cd frontend\n"
        "   npm run dev -- --host\n"
        "   --> Accessible at http://localhost:5173\n\n"
        "3. Run Automated Integration Test Suite (100% Pass Rate):\n"
        "   pytest tests/\n"
        "   --> Executes 10 unit and integration tests verifying OpenCV decode, 29-feature consistency, model deserialization, soft-voting math, and calibration."
    )
    add_callout(doc, "Live Demo Runbook", cmd_box, "16A34A", "F0FDF4")

    # Save DOCX
    doc.save(str(DOCX_OUTPUT))
    print(f"Generated DOCX: {DOCX_OUTPUT}")


def convert_docx_to_pdf():
    import subprocess
    cmd = f"""
    $docxPath = "{DOCX_OUTPUT}"
    $pdfPath = "{PDF_OUTPUT}"
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $doc = $word.Documents.Open($docxPath)
    $doc.SaveAs([ref]$pdfPath, [ref]17)
    $doc.Close()
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    """
    res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    if PDF_OUTPUT.exists():
        print(f"Generated PDF successfully: {PDF_OUTPUT} ({PDF_OUTPUT.stat().st_size} bytes)")
    else:
        print(f"PDF generation failed: {res.stderr}")


if __name__ == "__main__":
    build_docx()
    convert_docx_to_pdf()
