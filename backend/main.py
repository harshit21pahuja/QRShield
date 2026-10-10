"""
QRShield — FastAPI Application Backend

Serves the multimodal explainable quishing detection pipeline locally.
CRITICAL SAFETY:
This API strictly performs local/static analysis on inert QR image data and URL strings.
Decoded URLs are NEVER crawled, visited, redirected, or contacted over the network.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.inference import (
    fusion_engine,
    image_engine,
    qr_decoder,
    url_engine,
)

app = FastAPI(
    title="QRShield API",
    description="Explainable Multimodal Quishing Detection Local Prototype",
    version="1.0.0",
)

# Enable CORS for local development and React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, Any]:
    """Root endpoint with service summary."""
    return {
        "service": "QRShield Multimodal Quishing Detection",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "predict": "/predict (POST multipart/form-data with 'file')",
        "safety": "Static local analysis only. Decoded URLs are never contacted.",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health and model readiness check."""
    url_ready = url_engine.model is not None
    image_ready = image_engine.model is not None
    fusion_ready = fusion_engine.calibrator is not None

    all_ready = url_ready and image_ready and fusion_ready

    return {
        "status": "healthy" if all_ready else "degraded",
        "models_ready": all_ready,
        "components": {
            "url_model": {
                "loaded": url_ready,
                "type": "RandomForestClassifier",
                "features": 29,
            },
            "image_model": {
                "loaded": image_ready,
                "type": "MobileNetV2",
                "device": str(image_engine.device),
            },
            "fusion": {
                "loaded": fusion_ready,
                "type": "prediction-level soft voting",
                "weights": {
                    "url": fusion_engine.url_weight,
                    "image": fusion_engine.image_weight,
                },
                "calibrator": fusion_engine.calibration_method,
                "threshold": fusion_engine.threshold,
            },
        },
        "safety_guarantee": "No external HTTP connections made to scanned QR destinations.",
    }


@app.post("/predict")
async def predict_qr(file: UploadFile = File(...)) -> JSONResponse:
    """Predict quishing risk for an uploaded QR code image.

    Accepts:
        file: Multipart image file (PNG, JPG, JPEG, WEBP).

    Returns:
        JSON with decoded URL, branch probabilities, calibrated risk, and explanations.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename missing.",
        )

    # Basic content type validation
    allowed_content_types = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
    if file.content_type and file.content_type.lower() not in allowed_content_types:
        # Also check extension if content-type header is generic application/octet-stream
        ext = Path(file.filename).suffix.lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type '{file.content_type or ext}'. Please upload a PNG, JPG, or WEBP image.",
            )

    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        )

    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    # Execute end-to-end inert inference
    try:
        result = fusion_engine.predict_pipeline(image_bytes, generate_gradcam=True)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "error": f"Internal inference pipeline error: {exc}",
            },
        )

    if result.get("status") == "error":
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=result,
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
