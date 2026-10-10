# QRShield: Explainable Multimodal Quishing Detection
## Phase 1 Project Report — AI and Society (CSET485)

---

### **Project Metadata & Team Credentials**
- **Course Title:** CSET485 — AI and Society
- **Academic Term:** Fall Semester 2026
- **Institution:** School of Computer Science Engineering and Technology (SCSET), Bennett University, Greater Noida, UP, India
- **Team ID:** **54**
- **Project Title:** QRShield: Explainable Multimodal Quishing Detection with Zero-Contact Static Analysis

#### **Team Members & Roll Numbers**
| No. | Student Name | Enrollment / Roll Number | Primary Phase 1 Responsibilities |
| :---: | :--- | :--- | :--- |
| 1 | **Harshit Pahuja** | `s24cseu0736` | **System Architect & Lead**: Multimodal fusion logic, isotonic probability calibration, FastAPI backend engine |
| 2 | **Hemang Nag** | `s24cseu0728` | **Deep Vision Lead**: MobileNetV2 architecture, transfer learning pipeline, Grad-CAM visual explainability |
| 3 | **Saatvik Sharma** | `s24cseu2097` | **Lexical ML Lead**: 29 URL feature extractors, Random Forest modeling, SHAP attribution engine |
| 4 | **Tanish** | `s24cseu0589` | **Data & Security Lead**: Dataset canonicalization (CIC-Trap4Phish), zero-contact security policy, data audit |
| 5 | **Rohit Upadhyay** | `s24cseu1359` | **Frontend & QA Lead**: React 18 / Vite user interface, explainability dashboards, end-to-end integration tests |

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement & Societal Significance](#2-problem-statement--societal-significance)
3. [Zero-Contact Operational Security Architecture](#3-zero-contact-operational-security-architecture)
4. [Dataset Engineering & Data Leakage Prevention](#4-dataset-engineering--data-leakage-prevention)
5. [Feature Engineering & Multimodal System Architecture](#5-feature-engineering--multimodal-system-architecture)
   - 5.1 Static Lexical URL Feature Extraction (29 Features)
   - 5.2 Deep Vision Modeling (MobileNetV2 CNN)
   - 5.3 Prediction-Level Fusion & Isotonic Probability Calibration
6. [Explainable AI (XAI) Framework](#6-explainable-ai-xai-framework)
   - 6.1 Lexical Feature Attribution via TreeSHAP
   - 6.2 Spatial Visual Saliency via Grad-CAM
7. [Empirical Experimental Results & Analysis](#7-empirical-experimental-results--analysis)
   - 7.1 Controlled Test Split Performance
   - 7.2 Probability Calibration Impact
   - 7.3 False Positive & False Negative Error Analysis
8. [AI and Society: Ethical, Societal & Governance Analysis](#8-ai-and-society-ethical-societal--governance-analysis)
9. [Prototype Implementation & Verification](#9-prototype-implementation--verification)
10. [Work Distribution & Team Contribution Matrix](#10-work-distribution--team-contribution-matrix)
11. [Phase 2 Roadmap & Future Milestones](#11-phase-2-roadmap--future-milestones)
12. [References & Citations](#12-references--citations)

---

## 1. Executive Summary

QR codes have become an everyday tool for almost everyone with a smartphone—used for everything from quick UPI/card payments and restaurant menus to metro ticketing and two-factor logins. But this widespread convenience has created a serious blind spot: **quishing** (QR code phishing). Because none of us can read a 2D barcode pattern with our eyes, attackers easily hide dangerous phishing links inside normal-looking QR codes, bypassing corporate email filters and misleading everyday users.

Most existing QR scanners simply decode the link and immediately open it in a browser, or pop up vague warning boxes that users quickly dismiss. In this project, we built **QRShield**, an explainable multimodal system that screens QR codes locally on the device before the user ever accesses them. QRShield brings together:
1. **Zero-Contact Local Decoding:** Decodes the barcode strictly in memory using OpenCV without making any live network calls, browser requests, or DNS queries, keeping the user safe from drive-by malware.
2. **29-Feature Static Lexical Model:** A 300-tree Random Forest classifier that checks link lengths, delimiters, subdomains, and Shannon character entropy.
3. **Deep Barcode Image Model:** A fine-tuned MobileNetV2 CNN that inspects the raw QR code grid for visual anomalies, density differences, and distortion.
4. **Soft-Voting Fusion & Probability Calibration:** We combine the URL model (55% weight) and the image model (45% weight), and apply Isotonic Regression so that the predicted risk score actually reflects true threat probability (reducing Brier calibration error by 61.5%).
5. **Dual Explainability (XAI):** TreeSHAP feature attributions and Grad-CAM visual heatmaps that clearly show users why a link was flagged.

Tested on a 70,000-sample partition of the benchmark **CIC-Trap4Phish** dataset split strictly by domain to prevent data leakage, QRShield achieves **98.77% accuracy**, **99.67% ROC-AUC**, and **99.27% F1-score**. Crucially, combining both modalities dropped our false alarm rate from 40.75% down to just **5.61%** (an 86.2% reduction in false positives) while running in only 5.91 ms on a standard CPU.

---

## 2. Problem Statement & Practical Impact

### 2.1 Why Quishing is Difficult to Stop
Phishing is still one of the most common ways people get their passwords and money stolen. While modern web browsers and email spam filters have gotten quite good at catching suspicious text links, attackers are now using QR codes to sneak past those filters:
- **Visual Opacity:** A QR code is just a grid of black-and-white modules with error correction. To a person, a QR code leading to their official bank looks completely identical to one leading to a fake credential-harvesting clone.
- **Physical Sticker Overlays:** Attackers print sticker replicas and paste them over authentic QR codes on public parking meters, restaurant payment stands, or charging stations, exploiting people's trust in physical objects.
- **Mobile Device Constraints:** People scan QR codes on mobile screens where address bars are small and truncated, and there is no mouse hover preview to inspect the full link before tapping.

### 2.2 Societal & Human-Factor Implications
- **Erosion of Public Digital Trust:** As civic services migrate toward contactless physical interactions, quishing undermines public confidence in digital public infrastructure (DPI).
- **Asymmetric Vulnerability of Non-Technical Populations:** Non-technical citizens and elderly demographics lack the forensic training to scrutinize truncated subdomains, punycode attacks, or token-stuffed URLs.
- **The Blindness of "Black-Box" Warnings:** Generic warning messages (e.g., *"Warning: Suspicious Link"*) suffer from alarm fatigue. Without explicit, interpretable justifications, users frequently bypass security warnings.

---

## 3. Zero-Contact Operational Security Architecture

A paramount contribution of QRShield is its strict **Zero-Contact Security Policy**.

```
[Uploaded QR Image] ──▶ [Local OpenCV QR Engine] ──▶ [Inert String Buffer] ──▶ [Static Feature Pipeline]
                                                                ▲
                                                    [NO Network Requests]
                                                    [NO DNS Lookups]
                                                    [NO Headless Crawling]
                                                    [NO Redirect Following]
```

### Why Dynamic Inspection is Dangerous
Many commercial threat detectors evaluate URLs by executing HTTP `GET`/`HEAD` requests or running headless Chromium instances. While this reveals redirect chains, it exposes the user to severe operational hazards:
1. **Drive-By Exploits:** Loading untrusted code can trigger browser-level remote code execution (RCE) or zero-day memory corruption bugs.
2. **Attacker Beaconing & Tracking:** Attackers often embed unique tracking tokens in phishing URLs (`?victim_id=xyz123`). Performing an HTTP request confirms to the adversary that the victim scanned the code, validating the target's email or phone number.
3. **Cloaking & Evasion:** Sophisticated phishing kits detect automated crawlers via IP reputation or User-Agent profiling, serving harmless pages to scanners while serving phishing landing pages to authentic mobile devices.

### QRShield's Strict Guarantee
QRShield operates exclusively on **local, static data**:
- Decoded QR payloads are treated as inert ASCII/UTF-8 byte sequences.
- No socket connections, DNS requests, HTTP handshakes, or dynamic rendering are ever initiated.
- The entire evaluation pipeline executes locally within memory, ensuring absolute zero data leakage.

---

## 4. Dataset Engineering & Data Leakage Prevention

### 4.1 Benchmark Source: CIC-Trap4Phish
The models were trained and benchmarked against the **CIC-Trap4Phish** dataset (Canadian Institute for Cybersecurity, University of New Brunswick), a state-of-the-art benchmark specifically curated for QR code and mobile phishing detection.

### 4.2 Dataset Canonicalization & Cleaning
The raw metadata corpus contained over **2.15 million raw records**. A systematic data hygiene and auditing pipeline was executed (`scripts/qr_dataset_audit.py`):
1. **Schema Validation & Normalization:** Standardized heterogeneous metadata schemas into unified `[url, label, domain, image_hash]` tuples.
2. **Label Discrepancy Resolution:** Identified and purged 10,305 conflicting duplicate instances where identical URLs had conflicting benign/malicious tags across different crawl timestamps.
3. **Image Integrity Verification:** Computed perceptual and SHA-256 hashes of all QR images, eliminating corrupt, zero-byte, or undecodable image modules.
4. **Valid Canonical Corpus:** Filtered the dataset down to 387,015 clean, canonical image-URL pairs.

### 4.3 Preventing Data Leakage via Domain-Grouped Splitting
A common flaw in academic ML literature is random URL splitting. If `phishing-portal.com/login1` is placed in the training set and `phishing-portal.com/login2` in the test set, the model memorizes the domain name rather than generalizable structural properties, resulting in artificially inflated test scores.

To prevent this data leakage, QRShield enforces **Strict Registrable Domain Grouping** using Mozilla’s Public Suffix List (`tldextract`):
- All URLs sharing an effective top-level domain plus one (eTLD+1) are strictly assigned to either the Train, Validation, or Test partition.
- **Partition Distribution:**
  - **Training Split:** 50,000 samples (Domain-grouped)
  - **Validation Split:** 10,000 samples (Zero domain overlap with Train)
  - **Test Split:** 10,000 samples (Zero domain overlap with Train and Validation)
  - **Verified Set Overlap:** Exactly **0** shared domains, **0** shared image hashes, and **0** shared URLs across splits.

---

## 5. Feature Engineering & Multimodal System Architecture

```
                      Raw QR Code Image (PNG / JPG / WEBP)
                                      │
                                      ▼
                         Local OpenCV QR Decoder
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
               Decoded URL String            Image Tensor (224x224x3)
                        │                           │
                        ▼                           ▼
           29 Static Lexical Features        MobileNetV2 CNN Backbone
                        │                           │
                        ▼                           ▼
             Random Forest (300 Trees)        Linear Classification Head
                        │                           │
                        ▼                           ▼
              P_url (Weight: 0.55)         P_img (Weight: 0.45)
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                     Soft Voting Fusion: 0.55*P_url + 0.45*P_img
                                      │
                                      ▼
                        Isotonic Probability Calibrator
                                      │
                                      ▼
                         Final Calibrated Probability
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
             Binary Verdict (Threshold: 0.5434)  Risk Severity Rating
             [BENIGN / MALICIOUS]                [LOW / MOD / HIGH / CRIT]
                        │                           │
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                      Dual Explainability Dashboards
                      1. TreeSHAP Lexical Feature Contributions
                      2. Grad-CAM Spatial Heatmap Visualization
```

### 5.1 Static Lexical URL Feature Extraction (29 Features)
The URL branch transforms raw URL strings into a 29-dimensional numerical feature vector ($\mathbf{x} \in \mathbb{R}^{29}$) capturing lexical, structural, statistical, and information-theoretic properties:

| Feature Category | Feature Names | Societal & Security Rationale |
| :--- | :--- | :--- |
| **Structural Lengths** | `url_length`, `hostname_length`, `path_length`, `query_length` | Phishing URLs are frequently elongated to obfuscate domain destinations or pack redirect parameters. |
| **Delimiters & Punctuations** | `num_dots`, `num_subdomains`, `num_hyphens`, `num_underscores`, `num_slashes`, `num_at`, `num_percent`, `num_colon` | Excessive dots/subdomains indicate subdomain spoofing (e.g., `paypal.com.account-update.xyz`); `@` symbols abuse browser credential syntax. |
| **Character Densities** | `digit_count`, `digit_ratio`, `alpha_ratio`, `num_special`, `special_ratio` | Machine-generated or obfuscated phishing links exhibit elevated hex encoding and digit frequencies. |
| **Query Parameters** | `num_question`, `num_equals`, `num_ampersand`, `query_param_count` | Quantifies parameter stuffing often utilized in phishing credential-stealer kits. |
| **Protocol & Host Checks** | `has_https`, `has_ip_hostname`, `has_port`, `hostname_hyphen_count`, `suspicious_token_count` | Raw IP hostnames (e.g., `http://192.168.1.5/login`) and brand-mimicking hyphens strongly indicate malicious intent. |
| **Shannon Entropies** | `url_entropy`, `hostname_entropy`, `path_entropy` | Measures the information density and randomness ($H = -\sum p_i \log_2 p_i$). Algorithmic domain generation (DGAs) and random token paths yield anomalously high entropy. |

The URL classification branch uses an optimized **RandomForestClassifier** (300 estimators, balanced class weights, maximum tree depth unconstrained, parallelized inference).

### 5.2 Deep Vision Modeling (MobileNetV2 CNN)
While lexical features inspect the destination, the raw visual matrix contains distinct optical properties:
- **Module Version & Density:** Extremely dense QR codes encode massive URLs or embedded binary payloads.
- **Visual Artifacts:** Steganographic modifications, altered corner finder patterns, watermarked brand logos, and contrast degradations.
- **Model Backbone:** We utilize **MobileNetV2**, an efficient convolutional architecture incorporating inverted residual blocks with depthwise separable convolutions and linear bottlenecks ($3.4\text{M}$ parameters).
- **Two-Stage Transfer Learning:**
  - *Stage A (Feature Extractor Frozen):* Custom dropout ($p=0.2$) and linear classification head trained for 2 epochs using AdamW ($\eta = 10^{-3}$, weight decay $= 10^{-4}$).
  - *Stage B (Fine-Tuning):* Unfroze the top convolutional layers (blocks 14–18) for 2 epochs at a reduced learning rate ($\eta = 10^{-4}$) to adapt feature representations to 2D barcode modules.

### 5.3 Prediction-Level Fusion & Isotonic Probability Calibration
Neither modality alone provides an infallible defense:
- URL features struggle when attackers register clean, short, low-entropy domains.
- Image features struggle when distinct URLs share similar visual matrix layouts.

QRShield combines branch outputs via **Soft-Voting Prediction Fusion**:
$$P_{\text{raw}} = w_{\text{url}} \cdot P_{\text{url}} + w_{\text{img}} \cdot P_{\text{img}}$$

Through exhaustive grid search on the validation set, optimal weights were established at:
$$w_{\text{url}} = 0.55, \quad w_{\text{img}} = 0.45$$

#### Probability Calibration via Isotonic Regression
Raw probabilities output by deep networks and tree ensembles frequently suffer from uncalibrated overconfidence. To transform raw scores into true posterior probabilities, we fitted a non-parametric **Isotonic Regression** model ($g_{\text{iso}}$):
$$P_{\text{calibrated}} = g_{\text{iso}}(P_{\text{raw}})$$

The calibrated decision threshold was optimized on validation F1-score:
$$\tau^* = 0.5434$$

---

## 6. Explainable AI (XAI) Framework

A core tenet of ethical AI is **transparency**. Rather than acting as an inscrutable black box, QRShield provides multimodal, human-interpretable justifications.

### 6.1 Lexical Feature Attribution via TreeSHAP
Using cooperative game theory, TreeSHAP attributes local additive importance values ($\phi_i$) to each of the 29 lexical features:
$$f(x) = \phi_0 + \sum_{i=1}^{29} \phi_i(x)$$

Global feature importance rankings derived from 1,000 test background instances revealed the top predictive signals:
1. `path_entropy` ($\text{mean } |\phi| = 0.0714$) — High character randomness in URL paths.
2. `path_length` ($\text{mean } |\phi| = 0.0624$) — Deeply nested directory trees.
3. `num_subdomains` ($\text{mean } |\phi| = 0.0515$) — Multi-tier subdomain chaining.
4. `num_hyphens` ($\text{mean } |\phi| = 0.0512$) — Hostname brand-spoofing separators.
5. `hostname_entropy` ($\text{mean } |\phi| = 0.0394$) — Obfuscated domain naming.

In the user interface, these raw mathematical weights are dynamically translated into intuitive cognitive indicators (e.g., *"Elevated Risk: Hostname contains 4 deceptive hyphens mimicking genuine brand names"*).

### 6.2 Spatial Visual Saliency via Grad-CAM
For the MobileNetV2 vision branch, we compute **Gradient-weighted Class Activation Mapping (Grad-CAM)** at the final convolutional feature layer (`features[18]`):
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right), \quad \alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$

This computes a $224 \times 224$ normalized thermal saliency overlay highlighting the exact spatial regions (finder patterns, timing patterns, alignment markers, or inner payload modules) driving the vision model’s decision. This visually demonstrates to security analysts whether the network is inspecting legitimate data modules or reacting to watermarked branding and background artifacts.

---

## 7. Empirical Experimental Results & Analysis

### 7.1 Controlled Test Split Performance
All models were evaluated on the held-out 10,000-sample domain-grouped test partition (1,605 benign, 8,395 malicious):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positive Rate (FPR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **URL-Only Baseline** (Random Forest) | 0.9267 | 0.9271 | 0.9906 | 0.9578 | 0.9480 | 0.9885 | 0.4075 (654 / 1,605) |
| **Image-Only Baseline** (MobileNetV2) | 0.9645 | 0.9662 | 0.9924 | 0.9791 | 0.9928 | 0.9986 | 0.1813 (291 / 1,605) |
| **QRShield Fusion** (55/45 Calibrated) | **0.9877** | **0.9894** | **0.9961** | **0.9927** | **0.9967** | **0.9993** | **0.0561** (90 / 1,605) |

#### Confusion Matrix (Multimodal Fusion on 10,000 Test Samples)
- **True Positives (TP):** 8,362 (Malicious correctly detected)
- **True Negatives (TN):** 1,515 (Benign correctly identified)
- **False Positives (FP):** 90 (Benign incorrectly flagged)
- **False Negatives (FN):** 33 (Malicious missed)

**Key Finding:** Multimodal fusion reduced false positives from 654 (URL branch alone) down to just **90**, representing an **86.2% reduction in false alarm rate** while simultaneously boosting recall to **99.61%**.

### 7.2 Probability Calibration Impact
The application of non-parametric Isotonic Regression yielded dramatic improvements in confidence calibration:
- **Brier Score (Raw Fusion):** `0.02659`
- **Brier Score (Isotonic Calibrated):** `0.01022`
- **Calibration Improvement:** **61.5% reduction in probabilistic error**

A calibrated Brier score of 0.0102 indicates that QRShield’s assigned probabilities accurately reflect real-world event frequencies, providing actionable risk scores to end users.

### 7.3 False Positive & False Negative Error Analysis
A rigorous failure audit (`artifacts/fusion_phase3/failure_analysis.json`) revealed the root causes behind edge-case errors:
- **False Positive Drivers (90 samples):**
  - Benign marketing links containing affiliate referral tokens, high query entropy, and tracking parameters (e.g., `utm_source`, session IDs).
  - Legitimate modern web apps featuring multi-tier subdomains and hyphenated services.
- **False Negative Drivers (33 samples):**
  - Ultra-short phishing domains hosted on reputable top-level domains without extended query strings.
  - Phishing links whose lexical features mimic basic homepage structures before server-side redirection takes place.

These failure modes underscore the essential role of dual explainability: by rendering transparent feature breakdowns, human analysts can instantly spot benign tracking parameters and override false alarms.

---

## 8. AI and Society: Ethical, Societal & Governance Analysis

In alignment with the core curriculum of **CSET485 (AI and Society)**, QRShield incorporates ethical, legal, and human-centric governance principles directly into its architectural design:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   ETHICAL & GOVERNANCE PILLARS                         │
├───────────────────┬───────────────────┬────────────────────────────────┤
│ 1. Human Agency   │ 2. Privacy by     │ 3. Harm Prevention             │
│    & Autonomy     │    Design         │    (Zero-Contact)              │
│ Assistive advisory│ 100% local,       │ Inert string handling;         │
│ rather than auto- │ zero external API │ eliminates drive-by malware    │
│ matic censorship. │ or tracking logs. │ & beacon tracking risk.        │
├───────────────────┼───────────────────┼────────────────────────────────┤
│ 4. Algorithmic    │ 5. Scientific     │ 6. Sustainable AI              │
│    Transparency   │    Integrity      │ Lightweight edge models;       │
│ Dual XAI: SHAP +  │ Explicit limits;  │ 5.91 ms latency on CPU;        │
│ Grad-CAM heatmaps.│ no wild claims.   │ minimal carbon footprint.      │
└───────────────────┴───────────────────┴────────────────────────────────┘
```

1. **Human Agency & Decision Empowerment:**
   QRShield avoids paternalistic, unilateral censorship. Instead of silently terminating network requests or forbidding navigation, the system acts as a **cognitive co-pilot**. It presents calibrated threat probabilities, structured risk tiers (Low, Moderate, High, Critical), and granular explainability rationales, enabling users to make informed, autonomous choices.
2. **Privacy Preservation & Anti-Surveillance:**
   Traditional cloud-based link checkers harvest every URL a citizen scans, creating comprehensive behavioral and physical surveillance records. QRShield enforces **100% local inference**. User camera feeds and scanned URLs never leave the client device, preserving privacy rights under GDPR and India's Digital Personal Data Protection Act (DPDPA).
3. **Operational Harm Minimization:**
   By treating decoded QR payloads strictly as inert data strings, QRShield protects users from malicious drive-by downloads and prevents adversaries from verifying victim interaction via beaconing.
4. **Honesty Regarding Generalization Boundaries:**
   Academic AI often suffers from overclaiming. QRShield explicitly acknowledges that its 98.77% accuracy was obtained on a domain-partitioned split of CIC-Trap4Phish. Real-world adversaries employ zero-day cloaking, fast-flux DNS, and novel adversarial matrix perturbations. QRShield documents these limitations transparently.
5. **Environmental & Computational Sustainability:**
   By selecting MobileNetV2 ($3.4\text{M}$ parameters) and a compact 29-feature Random Forest, QRShield operates efficiently on commodity CPUs with an average latency of **5.91 milliseconds** ($7.42\text{ ms}$ P95). This avoids carbon-intensive GPU server farms and allows deployment on resource-constrained mobile hardware.

---

## 9. Prototype Implementation & Verification

The QRShield prototype consists of a modular, production-ready full-stack software implementation:

### 9.1 FastAPI Backend Engine (`backend/main.py`)
- **Asynchronous Microservice:** Exposes high-throughput endpoints (`/health`, `/predict`).
- **Memory-Resident Artifacts:** Loads the 300-tree Random Forest, MobileNetV2 PyTorch weights, and Isotonic calibrator at application startup.
- **Strict Validation:** Rejects corrupted files, non-image formats, and undecodable QR inputs with clear HTTP 400/422 status responses.

### 9.2 Modern React 18 / Vite Frontend (`frontend/src/App.jsx`)
- **Cybersecurity Operations Center (SOC) Aesthetics:** Modern dark-mode interface with accessible typography and clean visual hierarchy.
- **Interactive Inspection Tools:**
  - Drag-and-drop QR image upload zone with live preview and sample test datasets.
  - Interactive tabbed explainability views toggling between SHAP feature attributions and Grad-CAM spatial heatmaps.
  - Safe, inert destination display with one-click clipboard copying.
  - Color-coded threat telemetry badges indicating model confidence.

### 9.3 Comprehensive Automated Verification Suite (`tests/test_prototype.py`)
Ten automated integration and unit test suites were executed with **100% pass rate** via `pytest`:
1. `test_qr_decoding` — Verifies OpenCV matrix detection and string decoding.
2. `test_url_feature_extraction_consistency` — Validates deterministic extraction and ordering of all 29 lexical features.
3. `test_url_model_loading` — Confirms Random Forest deserialization and SHAP attribution calculation.
4. `test_image_model_loading` — Tests MobileNetV2 PyTorch tensor forward pass and Grad-CAM generation.
5. `test_fusion_calculation` — Validates exact 55/45 soft voting arithmetic.
6. `test_calibration_loading` — Confirms Isotonic Regression model integrity and monotonic probability mapping.
7. `test_health_endpoint` — Tests API status reporting and model readiness probes.
8. `test_predict_safe_qr` — End-to-end integration test validating benign QR classification.
9. `test_predict_invalid_image` — Confirms graceful handling of corrupt binary inputs.
10. `test_predict_undecodable_qr` — Validates proper error reporting when an image contains no readable barcode.

---

## 10. Work Distribution & Team Contribution Matrix

The workload for Phase 1 was distributed evenly among all five team members according to technical specialization:

| Team Member | Roll Number | Specific Deliverables & Contributions | Percentage Effort |
| :--- | :---: | :--- | :---: |
| **Harshit Pahuja** *(Lead)* | `s24cseu0736` | Overall system architecture; mathematical formulation of multimodal fusion; Isotonic calibration pipeline; FastAPI backend integration; end-to-end integration testing. | 20% |
| **Hemang Nag** | `s24cseu0728` | MobileNetV2 CNN architecture; two-stage transfer learning; PyTorch GPU training scripts; Grad-CAM gradient backpropagation engine; vision latency benchmarking. | 20% |
| **Saatvik Sharma** | `s24cseu2097` | Feature engineering for 29 static lexical indicators; Shannon entropy implementations; Random Forest model training & hyperparameter tuning; TreeSHAP feature attribution pipeline. | 20% |
| **Tanish** | `s24cseu0589` | CIC-Trap4Phish dataset audit and canonicalization; domain-grouped train/val/test splitting algorithms; zero-contact operational security architecture design; false positive audit. | 20% |
| **Rohit Upadhyay** | `s24cseu1359` | Frontend UI design and implementation (React 18, Tailwind/Vanilla CSS, Lucide icons); interactive SHAP/Grad-CAM visualization dashboards; client-side error handling; user testing. | 20% |

---

## 11. Phase 2 Roadmap & Future Milestones

Building upon the robust Phase 1 multimodal foundation, the following research and engineering milestones are planned for Phase 2:
1. **On-Device Mobile Deployment (Edge AI):** Quantize the MobileNetV2 model to 8-bit integers (INT8) and export the entire multimodal pipeline via ONNX Runtime / PyTorch Mobile for native Android/iOS integration.
2. **Adversarial Robustness & Perturbation Defense:** Evaluate QRShield against adversarial patch attacks, barcode geometric distortions, and subtle optical perturbations designed to mislead convolutional filters.
3. **Local Reputation & Threat Intelligence Caching:** Implement an offline, Bloom-filter-backed local threat intelligence cache of verified malicious domain hashes (updated asynchronously without active link queries).
4. **Multilingual & Internationalized Domain Name (IDN) Defense:** Expand lexical feature extraction to detect homograph and punycode attacks across Cyrillic, Greek, and Latin character sets.
5. **Formal User Experience & Cognitive Study:** Conduct empirical human-in-the-loop experiments measuring whether explainable SHAP and Grad-CAM dashboards reduce user susceptibility to quishing lures compared to standard browser warnings.

---

## 12. References & Citations

1. **Canadian Institute for Cybersecurity (CIC):** *CIC-Trap4Phish 2025: A Comprehensive Multimodal Benchmark Dataset for Phishing and Quishing Detection*, University of New Brunswick, 2025.
2. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C.:** *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
3. **Lundberg, S. M., & Lee, S. I.:** *A Unified Approach to Interpreting Model Predictions (SHAP)*, Advances in Neural Information Processing Systems (NeurIPS), 2017.
4. **Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D.:** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*, International Journal of Computer Vision (IJCV), 2020.
5. **Zadrozny, B., & Elkan, C.:** *Transforming Classifier Scores into Accurate Multiclass Probability Estimates (Isotonic Regression)*, ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2002.
6. **National Institute of Standards and Technology (NIST):** *Zero Trust Architecture*, NIST Special Publication 800-207, U.S. Department of Commerce, 2020.
7. **Floridi, L., & Cowls, J.:** *A Unified Framework of Five Principles for AI in Society*, Harvard Data Science Review, 2019.

---
*End of Phase 1 Project Report — Team ID 54 — CSET485 AI and Society*
