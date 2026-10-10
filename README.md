# QRShield — Explainable Multimodal Quishing Detection

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-teal.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-red.svg)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange.svg)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research-green.svg)](LICENSE)

---

## 1. Project Overview

**QRShield** is an AI-powered security prototype designed to detect **quishing** (QR code phishing) attacks before users access malicious destinations. Traditional QR code scanners blindly open encoded URLs in web browsers or display uninformative link previews. QRShield provides a multi-layered defense combining:
1. **Local, inert QR decoding** (zero-contact URL handling).
2. **Static lexical URL feature extraction** (29 syntactic, structural, and information-theoretic indicators) evaluated with an optimized **RandomForestClassifier**.
3. **Deep visual saliency modeling** of the raw QR code pattern evaluated with a fine-tuned **MobileNetV2** convolutional neural network.
4. **Prediction-level soft-voting fusion** (55% URL / 45% Image) calibrated via **Isotonic Regression** for well-calibrated posterior probabilities.
5. **Dual explainability**: SHAP-grounded feature attribution for the destination URL and **Grad-CAM** saliency heatmaps highlighting spatial module focus on the QR image.

- **Course:** CSET485 — AI and Society
- **Institution:** Bennett University, School of Computer Science Engineering
- **Team Members:**
  - Harshit Pahuja (s24cseu0736)
  - Hemang Nag (s24cseu0728)
  - Saatvik Sharma (s24cseu2097)
  - Tanish (s24cseu0589)
  - Rohit Upadhyay (s24cseu1359)

---

## 2. Zero-Contact Security Policy

```
[Uploaded QR Image] ──▶ [Local OpenCV Decoder] ──▶ [Inert String] ──▶ [Static Analysis Only]
                                                            ▲
                                                 [NO Network Requests]
                                                 [NO Web Crawling]
                                                 [NO Redirect Following]
                                                 [NO DNS Resolution]
```

> [!CAUTION]
> **STRICT SAFETY GUARANTEE:**
> Decoded QR code contents are treated strictly as **inert data strings**. The system **never** performs `requests.get()`, browser automation, HTTP redirects, headless execution, or active network probing against any destination link. All evaluations are performed strictly on local features.

---

## 3. Architecture & User Flow

```
                      QR Code Image Upload (PNG/JPG/WEBP)
                                     │
                                     ▼
                        Local OpenCV QR Decode
                                     │
                       ┌─────────────┴─────────────┐
                       ▼                           ▼
             Decoded URL String            Raw Image (224x224)
                       │                           │
                       ▼                           ▼
          29 Static Lexical Features      MobileNetV2 Deep CNN
                       │                           │
                       ▼                           ▼
            RandomForestClassifier             Logits
                       │                           │
                       ▼                           ▼
             URL Probability (55%)       Image Probability (45%)
                       │                           │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                       Soft Voting: 0.55*P_url + 0.45*P_img
                                     │
                                     ▼
                         Isotonic Calibration
                                     │
                                     ▼
                   Final Calibrated Quishing Probability
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
             BENIGN / MALICIOUS              Risk Level & Score
             (Threshold: 0.5434)             (LOW / MOD / HIGH / CRIT)
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                ┌───────────────────────────────────────────┐
                │ Explainable Evidence Outputs:             │
                │ 1. Top SHAP lexical URL contributions     │
                │ 2. Grad-CAM visual saliency heatmap       │
                │ 3. Inert destination display (with copy)  │
                └───────────────────────────────────────────┘
```

---

## 4. Controlled Development Benchmark Results

The models were evaluated on a controlled, domain-grouped development split (train: 50,000, validation: 10,000, test: 10,000):

| Model / Branch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC | False Positive Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **URL-Only Baseline** (Random Forest) | 0.9267 | 0.9271 | 0.9906 | 0.9578 | 0.9480 | 0.9885 | 0.4075 |
| **Image-Only Baseline** (MobileNetV2) | 0.9645 | 0.9662 | 0.9924 | 0.9791 | 0.9928 | — | — |
| **Multimodal Fusion** (55/45 Soft Voting) | **0.9877** | **0.9894** | **0.9961** | **0.9927** | **0.9967** | **0.9993** | **0.0561** |

### Probability Calibration (Isotonic Regression)
- **Brier Score Before Calibration:** `0.02659`
- **Brier Score After Calibration:** `0.01022` (61.5% improvement in probability calibration)
- **Optimal Decision Threshold:** `0.5434`

> [!IMPORTANT]
> **Dataset Scope & Real-World Generalization Limitations:**
> The metrics shown above were computed on the internal 10,000-sample development test set derived from CIC-Trap4Phish. While domain-grouped splitting was enforced to prevent data leakage, these metrics **do not represent universal real-world performance**. Real-world deployment will encounter zero-day phishing kits, dynamic cloaking, domain hijacking, and novel QR steganography that may alter detection accuracy.

---

## 5. Trained Model Artifacts

All trained models and experimental assets are localized in `artifacts/`:

- **URL Model (`artifacts/url_baseline_phase3/`):**
  - `url_random_forest.joblib`: Trained 300-tree balanced Random Forest classifier.
  - `config.json` & `metrics.json`: Hyperparameters, threshold (`0.3938`), and test metrics.
- **Image Model (`artifacts/image_baseline_gpu/`):**
  - `best_model.pt`: MobileNetV2 checkpoint saved at peak validation F1.
  - `last_model.pt`: Final epoch model weights.
  - `config.json` & `metrics.json`: Stage A + Stage B training parameters, classification report, and ROC/PR curves.
- **Fusion & Explainability (`artifacts/fusion_phase3/`):**
  - `isotonic_calibrator.joblib`: Fitted non-parametric isotonic probability calibrator.
  - `inference_config.json`: Production weights (`url_weight: 0.55`, `image_weight: 0.45`, `threshold: 0.5434`).
  - `shap_feature_importance.csv` & `shap_summary.png`: Global SHAP feature attribution rankings.
  - `false_positives.csv` & `false_negatives.csv`: Error analysis logs.

---

## 6. Installation & Environment Setup

The local prototype uses Python 3.13 in a dedicated virtual environment (`.venv313`) and Node.js for the React UI.

### Prerequisites
- Python 3.13+ installed
- Node.js v18+ and npm installed

### 1. Activate Virtual Environment
```powershell
# In PowerShell (Project Root):
.\.venv313\Scripts\Activate.ps1

# Verify Python version:
python --version
# Expected: Python 3.13.x
```

### 2. Install / Verify Python Dependencies
```powershell
pip install -r requirements.txt
pip install opencv-python-headless pytest httpx
```

### 3. Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

---

## 7. Running the Prototype

### Step 1: Start FastAPI Backend
```powershell
# From project root:
.\.venv313\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API will be accessible at: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### Step 2: Start React Frontend
In a separate terminal window:
```powershell
cd frontend
npm run dev
```
- Web UI will be accessible at: `http://localhost:5173`

---

## 8. API Specification

### `GET /health`
Returns readiness status of all internal models.
```json
{
  "status": "healthy",
  "models_ready": true,
  "components": {
    "url_model": {
      "loaded": true,
      "type": "RandomForestClassifier",
      "features": 29
    },
    "image_model": {
      "loaded": true,
      "type": "MobileNetV2",
      "device": "cpu"
    },
    "fusion": {
      "loaded": true,
      "type": "prediction-level soft voting",
      "weights": { "url": 0.55, "image": 0.45 },
      "calibrator": "IsotonicRegression",
      "threshold": 0.543407
    }
  },
  "safety_guarantee": "No external HTTP connections made to scanned QR destinations."
}
```

### `POST /predict`
Uploads a QR code image (`multipart/form-data` with form key `file`) and returns multimodal predictions.
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@sample_qr.png"
```
**Sample Response:**
```json
{
  "status": "ok",
  "decoded_url": "https://secure-login-account-update.suspicious-domain.com/login?token=xyz",
  "is_http": true,
  "url_probability": 0.988046,
  "image_probability": 0.123483,
  "fusion_probability": 0.598993,
  "calibrated_probability": 0.87234,
  "prediction": "MALICIOUS",
  "risk_level": "CRITICAL",
  "risk_label": "Severe Malicious Threat",
  "threshold": 0.543407,
  "weights": { "url_weight": 0.55, "image_weight": 0.45 },
  "explanation": {
    "url": [
      {
        "feature": "num_hyphens",
        "value": 5.0,
        "importance_weight": 0.0512,
        "impact": "elevated_risk",
        "finding": "Frequent hyphens (5) observed across the URL.",
        "description": "Frequency of hyphens often used to mimic brand names or create deceptive URLs."
      }
    ],
    "image": [
      {
        "component": "Visual Density & Embedding",
        "finding": "Visual features consistent with standard QR codes (probability 12.3%)."
      }
    ],
    "gradcam": {
      "available": true,
      "data_url": "data:image/png;base64,...",
      "focus_region": "Data Matrix / Module Payload"
    }
  },
  "safety_notice": "Static local analysis only. Destination URL was NOT contacted, crawled, or visited."
}
```

---

## 9. Running Tests

Run the comprehensive unit and integration test suite:
```powershell
.\.venv313\Scripts\pytest.exe tests/test_prototype.py -v
```

Tests verified:
1. `test_qr_decoding`: OpenCV QR detection and inert text extraction.
2. `test_url_feature_extraction_consistency`: Strict 29-feature ordering and extraction integrity.
3. `test_url_model_loading`: RandomForestClassifier inference and SHAP evidence generation.
4. `test_image_model_loading`: MobileNetV2 forward pass and Grad-CAM generation.
5. `test_fusion_calculation`: Exact 55/45 soft voting logic.
6. `test_calibration_loading`: IsotonicRegression loading and monotonic transformation.
7. `test_health_endpoint`: Backend `/health` probe.
8. `test_predict_safe_qr`: Full `/predict` pipeline on benign synthetic QR code.
9. `test_predict_invalid_image`: Graceful 400/422 validation on empty or corrupt bytes.
10. `test_predict_undecodable_qr`: Proper 422 error response when an image lacks a readable QR code.

---

## 10. AI and Society / Ethical Considerations

| Dimension | Architectural Implementation |
| :--- | :--- |
| **Transparency & Interpretability** | Combines global SHAP feature importance with local instance-level lexical analysis and Grad-CAM visual heatmaps. Explanations never make absolute causal claims. |
| **Human Agency & Decision Support** | Operates as an assistive security advisory tool rather than an opaque autonomous gatekeeper. |
| **Privacy Preservation** | Completely local inference. No user query logs or uploaded images are sent to third-party cloud services. |
| **Operational Safety** | Scanned URLs are permanently treated as inert strings, preventing drive-by exploits, tracking beacon triggers, or server-side notification to attackers. |
| **Rigorous Evaluation Honesty** | Development benchmark metrics are clearly distinguished from in-the-wild generalization capabilities. |
