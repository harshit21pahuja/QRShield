"""
QRShield — FastAPI Application Backend

Serves the multimodal explainable quishing detection pipeline locally.

CRITICAL SAFETY POLICY:
This API strictly performs local/static analysis on inert QR image data and URL strings.
Decoded URLs are NEVER crawled, visited, redirected, or contacted over the network.
All input is treated as untrusted and thoroughly validated.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictResponse,
    RootResponse,
)
from src.inference import (
    fusion_engine,
    image_engine,
    qr_decoder,
    url_engine,
)

# Initialize settings and structured logger
settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qrshield.backend")

app = FastAPI(
    title=settings.app_name,
    description=(
        "Explainable Multimodal Quishing Detection Engine combining local OpenCV "
        "QR decoding, static lexical URL Random Forest classification, deep MobileNetV2 "
        "visual saliency modeling, and calibrated soft-voting fusion."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Secure, configurable CORS configuration
allow_all = "*" in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=not allow_all,  # In strict CORS, credentials cannot be used with '*'
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/", response_model=RootResponse, tags=["Discovery"])
def root() -> Dict[str, Any]:
    """Root discovery endpoint providing service status and endpoint links."""
    return {
        "service": "QRShield Multimodal Quishing Detection",
        "version": settings.app_version,
        "documentation": "/docs",
        "health": "/health",
        "predict": "/predict (POST multipart/form-data with 'file')",
        "safety": "Static local analysis only. Decoded URLs are never contacted.",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> Dict[str, Any]:
    """Health and model readiness check.

    Returns component-level loading status and device configuration.
    """
    url_ready = getattr(url_engine, "model", None) is not None
    image_ready = getattr(image_engine, "model", None) is not None
    fusion_ready = getattr(fusion_engine, "calibrator", None) is not None

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
                "device": str(getattr(image_engine, "device", "cpu")),
            },
            "fusion": {
                "loaded": fusion_ready,
                "type": "prediction-level soft voting",
                "weights": {
                    "url": getattr(fusion_engine, "url_weight", 0.55),
                    "image": getattr(fusion_engine, "image_weight", 0.45),
                },
                "calibrator": getattr(fusion_engine, "calibration_method", "IsotonicRegression"),
                "threshold": getattr(fusion_engine, "threshold", 0.543407),
            },
        },
        "safety_guarantee": "No external HTTP connections made to scanned QR destinations.",
    }


@app.get("/health/ready", tags=["Health"])
def readiness_probe() -> Response:
    """Kubernetes / Docker container readiness probe.

    Returns HTTP 200 when all models are loaded and ready to serve inference.
    Returns HTTP 503 Service Unavailable when models are not ready.
    """
    url_ready = getattr(url_engine, "model", None) is not None
    image_ready = getattr(image_engine, "model", None) is not None
    fusion_ready = getattr(fusion_engine, "calibrator", None) is not None

    if url_ready and image_ready and fusion_ready:
        return Response(status_code=status.HTTP_200_OK, content="READY", media_type="text/plain")

    logger.warning("Readiness probe failed: models not fully loaded")
    return Response(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content="MODELS_NOT_READY",
        media_type="text/plain",
    )


def validate_image_payload(image_bytes: bytes) -> None:
    """Perform content validation to ensure payload is a decodable image.

    Raises:
        HTTPException: If file is corrupted, malformed, or not an image.
    """
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            pil_img.verify()
    except (UnidentifiedImageError, ValueError, TypeError) as exc:
        logger.warning("Uploaded payload failed PIL image verification: %s", exc)
        # Note: Handled as 422 for unprocessable image content
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Uploaded file is corrupt or not a recognized image format: {exc}",
        )
    except Exception as exc:
        logger.warning("Unexpected error during image verification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Uploaded file could not be parsed as an image.",
        )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input or empty file"},
        413: {"model": ErrorResponse, "description": "Uploaded file exceeds maximum size limit"},
        422: {"model": ErrorResponse, "description": "Undecodable QR or unprocessable image"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Inference"],
)
async def predict_qr(file: UploadFile = File(...)) -> JSONResponse:
    """Predict quishing risk for an uploaded QR code image.

    Accepts:
        file: Multipart image file (PNG, JPG, JPEG, WEBP).

    Enforces:
        - Strict maximum upload size (configurable via MAX_UPLOAD_SIZE_BYTES, default 10MB)
        - Format and binary validation
        - Zero external network requests (100% offline static analysis)

    Returns:
        JSON with decoded URL, branch probabilities, calibrated risk, and explanations.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file uploaded or filename missing.",
        )

    # 1. Content-Type and extension validation
    filename_ext = Path(file.filename).suffix.lower()
    content_type = (file.content_type or "").lower()

    valid_content_type = content_type in settings.allowed_content_types
    valid_extension = filename_ext in settings.allowed_extensions

    if not valid_content_type and not valid_extension:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{content_type or filename_ext}'. "
                f"Allowed formats: {', '.join(settings.allowed_extensions)}"
            ),
        )

    # 2. Stream chunk-based upload size enforcement to prevent memory exhaustion
    chunk_size = 64 * 1024  # 64 KB chunks
    total_bytes = bytearray()

    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total_bytes.extend(chunk)
            if len(total_bytes) > settings.max_upload_size_bytes:
                max_mb = settings.max_upload_size_bytes / (1024 * 1024)
                logger.warning(
                    "Upload rejected: %s exceeded maximum limit (%s MB)",
                    file.filename,
                    max_mb,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"File exceeds maximum upload size limit of {max_mb:.1f} MB.",
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error reading uploaded file %s: %s", file.filename, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read uploaded file: {exc}",
        )

    image_bytes = bytes(total_bytes)

    # 3. Deep image validation
    try:
        validate_image_payload(image_bytes)
    except HTTPException as exc:
        if exc.status_code == status.HTTP_400_BAD_REQUEST:
            raise
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "error": exc.detail},
        )

    # 4. Execute end-to-end zero-contact inference
    try:
        result = fusion_engine.predict_pipeline(image_bytes, generate_gradcam=True)
    except Exception as exc:
        logger.exception("Inference pipeline failed unexpectedly for %s", file.filename)
        safe_msg = (
            f"Internal inference error: {exc}"
            if settings.debug
            else "Internal error while analyzing QR code. Please verify the input image."
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"status": "error", "error": safe_msg},
        )

    if result.get("status") == "error":
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=result,
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=result,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
