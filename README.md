# QRShield — Explainable Multimodal Quishing Detection

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.1.0-teal.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV2-red.svg)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange.svg)](https://scikit-learn.org)
[![React](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Research-green.svg)](LICENSE)

---

## 1. Project Overview & Problem Statement

**QRShield** is an AI-powered cybersecurity system engineered to detect **quishing** (QR code phishing) attacks before users access malicious destinations.

Traditional QR code scanners blindly open encoded URLs in mobile browsers or display uninformative link previews. Attackers exploit this behavior by embedding phishing URLs inside visually standard QR codes, bypassing traditional email and web gateway filters.

QRShield provides an end-to-end, multi-layered defensive pipeline combining:
1. **Zero-Contact Local QR Decoding:** Inert string extraction using OpenCV with zero network requests.
2. **Static Lexical URL Feature Extraction:** 29 syntactic, structural, and information-theoretic indicators evaluated by an optimized **RandomForestClassifier**.
3. **Deep Visual Saliency Modeling:** Raw 224x224 QR matrix pattern analysis evaluated with a fine-tuned **MobileNetV2** deep convolutional neural network.
4. **Prediction-Level Calibrated Soft-Voting Fusion:** 55% URL / 45% Image soft voting calibrated via non-parametric **Isotonic Regression** for well-calibrated posterior probabilities (decision threshold: `0.5434`).
5. **Dual Explainability:** SHAP-grounded feature attribution rankings for URL features and **Grad-CAM** visual saliency heatmaps highlighting spatial module focus on the QR image.

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
> **STRICT ZERO-CONTACT SAFETY GUARANTEE:**
> Scanned QR code payloads are treated strictly as **inert data strings**. The system **never** performs HTTP requests, browser automation, DNS resolution, or active network probing against any destination link. All evaluations are performed strictly on static features.

---

## 3. System Architecture & Request Flow

```mermaid
flowchart TD
    Client[React 19 Frontend / API Client] -->|POST /predict multipart/form-data| Backend[FastAPI Backend - main.py]

    subgraph ZeroContactPerimeter [Zero-Contact Local Security Perimeter]
        Backend -->|Streamed Max 10MB Check| Validator[Payload & Binary Image Validator]
        Validator -->|Valid Image Bytes| Decoder[Local OpenCV QRCodeDetector]

        Decoder -->|Inert URL String| URLBranch[URL Inference Engine]
        Decoder -->|Raw Image Tensor 224x224| ImgBranch[MobileNetV2 Vision Engine]

        URLBranch -->|29 Lexical Features| RF[Random Forest 300 Trees]
        URLBranch -->|Legitimate Domain Baseline| Whitelist[Inert Reputable Domain Filter]
        URLBranch -->|Feature Importance| SHAP[SHAP Lexical Attribution]

        ImgBranch -->|Deep Feature Extraction| CNN[MobileNetV2 Checkpoint]
        ImgBranch -->|Layer features[-1]| GradCAM[Grad-CAM Saliency Generator]

        RF -->|P_url 55%| Fusion[Soft Voting Fusion Engine]
        CNN -->|P_img 45%| Fusion

        Fusion -->|Raw Fusion Score| Calibrator[Isotonic Regression Calibrator]
        Calibrator -->|Threshold 0.5434| Decision[Risk Level: LOW / MOD / HIGH / CRIT]
    end

    Decision --> JSON[Structured Response with Dual Explainability]
    SHAP --> JSON
    GradCAM --> JSON
    JSON --> Client
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
- **Inference Latency:** `~620 ms` on modern multi-core CPU.

> [!IMPORTANT]
> **Dataset Scope & Real-World Generalization Disclosures:**
> The metrics above were computed on an internal 10,000-sample test set derived from CIC-Trap4Phish. Domain-grouped splitting was enforced to eliminate data leakage across splits. These metrics **do not represent universal real-world performance**. Real-world deployment will encounter zero-day phishing kits, dynamic cloaking, domain hijacking, and novel QR steganography that may alter detection accuracy.

---

## 5. Technology Stack

| Layer | Technologies | Role & Implementation Details |
| :--- | :--- | :--- |
| **Backend API** | FastAPI, Uvicorn, Pydantic v2 | High-throughput asynchronous REST API, strict request/response schema validation, readiness probes, and CORS controls. |
| **Machine Learning** | PyTorch, torchvision, scikit-learn, joblib | MobileNetV2 convolutional neural network, 300-tree Random Forest, non-parametric Isotonic Regression probability calibrator. |
| **Computer Vision** | OpenCV (headless), Pillow | Local QR matrix detection, image header validation, and RGB tensor preprocessing. |
| **Explainability** | SHAP, Grad-CAM | Lexical attribution rankings and activation saliency heatmaps overlaid on original QR codes. |
| **Frontend UI** | React 19, Vite, Lucide Icons | Responsive single-page dashboard with drag-and-drop ingestion, interactive risk meters, preset samples, and dual explainability tabs. |
| **Containerization** | Docker, Docker Compose, Nginx | Multi-stage frontend build with Nginx reverse proxy, unprivileged Python backend container with automated healthcheck. |
| **CI/CD** | GitHub Actions | Automated linting (`oxlint`, `compileall`), fast mocked unit tests, security tests, and frontend build verification on push. |

---

## 6. Architectural Decisions: Database & Background Processing

### 6.1 Database Integration Decision
- **Stateless Zero-Contact Design:** QRShield operates as a privacy-preserving analysis tool. The core design principle guarantees: *"Completely local inference. No user query logs or uploaded images are sent to third-party cloud services or persisted."* Storing scanned URLs or images into a persistent database by default would directly contradict this privacy guarantee.
- **Enterprise Extension:** Persistent storage (PostgreSQL/SQLite) is documented as an opt-in audit log feature for enterprise security teams requiring quarantine records, leaving the core open-source engine stateless and private.

### 6.2 Background Processing (Celery / Redis) Decision
- **Sub-Second Latency:** Measured end-to-end inference latency on CPU is **~620 milliseconds** per QR scan.
- **Synchronous Interaction:** Users expect immediate real-time feedback when scanning a QR code (< 1–2 seconds). Adding a Celery broker, worker daemon, Redis database, and polling loop for a 620ms operation would introduce operational overhead and unnecessary points of failure. The architecture remains lean and synchronous for interactive queries.

---

## 7. Environment Configuration (`.env`)

A comprehensive template is available at `.env.example`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_NAME` | `QRShield API` | Service display name in API documentation. |
| `APP_VERSION` | `1.1.0` | Semantic version string. |
| `DEBUG` | `false` | Enable detailed debug logs (disables error sanitization). |
| `HOST` | `127.0.0.1` | Network interface to bind the backend server. |
| `PORT` | `8000` | Port for the backend API. |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated list of allowed web origins. |
| `MAX_UPLOAD_SIZE_BYTES` | `10485760` (10 MB) | Strict maximum upload size limit to prevent DoS memory exhaustion. |
| `LOG_LEVEL` | `INFO` | Logger severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `VITE_API_BASE` | `http://127.0.0.1:8000` | Backend API URL accessed by the React frontend. |

---

## 8. Installation & Local Development Setup

### Prerequisites
- Python 3.11 or Python 3.13
- Node.js v18+ and npm
- (Optional) Docker and Docker Compose for containerized deployment

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/your-username/QRShield.git
cd QRShield

# Create virtual environment
python -m venv .venv
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate
```

### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest httpx
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 9. Running the Application

### Option A: Local Development (Terminal)

**Terminal 1 — FastAPI Backend:**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- Healthcheck: `http://127.0.0.1:8000/health`
- Readiness Probe: `http://127.0.0.1:8000/health/ready`

**Terminal 2 — React Frontend:**
```bash
cd frontend
npm run dev
```
- Dashboard UI: `http://localhost:5173`

---

### Option B: Docker Compose Deployment

> [!NOTE]
> Docker files are verified for path contexts and production readiness. Ensure Docker Desktop or Docker Engine is running on your system.

```bash
# Build and launch both backend and frontend containers
docker compose up --build -d

# Check running status and health probes
docker compose ps

# View live container logs
docker compose logs -f

# Shut down services
docker compose down
```
- Web Dashboard: `http://localhost` (port 80)
- Backend API: `http://localhost:8000`

---

## 10. Automated Testing Suite

The repository includes a multi-tier automated test suite:
1. **Fast Mocked Unit & Security Tests (`tests/test_api_mocked.py`, `tests/test_security.py`):** Runs in `< 2 seconds` without loading 48MB model weights. Validates input validation, upload limits, error sanitization, zero-contact invariance, and CORS policies.
2. **Full End-to-End Integration Tests (`tests/test_prototype.py`):** Tests actual model weights, OpenCV decoding, Random Forest inference, MobileNetV2 forward pass, Grad-CAM generation, and Isotonic calibration.

### Running Fast Unit & Security Tests (CI-Ready):
```bash
pytest tests/test_api_mocked.py tests/test_security.py -v
```

### Running Full Integration Suite:
```bash
pytest tests/test_api_mocked.py tests/test_security.py tests/test_prototype.py -v
```

---

## 11. API Specification & Examples

### `GET /health`
Returns system status and component readiness.
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
Upload a QR image (`multipart/form-data`, key `file`).

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -F "file=@sample_phishing_qr.png"
```

**Response (`200 OK`):**
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

## 12. Model Artifacts & Dataset Requirements

Model artifacts are localized in `artifacts/`:
- **`artifacts/url_baseline_phase3/`**:
  - `url_random_forest.joblib`: 300-tree balanced Random Forest model (48 MB).
  - `config.json` & `metrics.json`: Hyperparameters and test metrics.
- **`artifacts/image_baseline_gpu/`**:
  - `best_model.pt`: MobileNetV2 checkpoint saved at peak validation F1 (9.1 MB).
  - `config.json` & `metrics.json`: Training parameters and ROC/PR evaluation curves.
- **`artifacts/fusion_phase3/`**:
  - `isotonic_calibrator.joblib`: Fitted non-parametric Isotonic Regression calibrator.
  - `inference_config.json`: Optimal soft-voting weights (`url: 0.55`, `image: 0.45`) and threshold (`0.5434`).
  - `shap_feature_importance.csv`: Global SHAP feature attribution rankings.

---

## 13. Known Limitations & Roadmap

1. **Bare Root Domains vs. Deep Phishing Links:** Pure lexical models can show elevated risk on short or newly observed domains. QRShield incorporates an authenticated reputable root domain whitelist baseline to eliminate false alarms on major portals.
2. **Dynamic Cloaking & IP Rotation:** Because QRShield strictly adheres to a zero-contact policy, server-side dynamic redirects that occur only upon HTTP GET requests cannot be observed.
3. **Future Work:**
   - On-device mobile deployment using ONNX Runtime and CoreML.
   - Optional opt-in enterprise audit logging with PostgreSQL and encrypted quarantine storage.
   - Expanded support for multi-frame animated QR codes (dynamic GIF/MP4 quishing).
