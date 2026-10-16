"""
QRShield — Pydantic Schemas for API Requests and Responses

Defines formal request, response, and component health schemas for OpenAPI documentation,
client contracts, and schema validation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Root & Discovery ─────────────────────────────────────────────────────────

class RootResponse(BaseModel):
    service: str = Field(..., description="Name of the service")
    version: str = Field(..., description="API version")
    documentation: str = Field(..., description="OpenAPI documentation path")
    health: str = Field(..., description="Health check endpoint path")
    predict: str = Field(..., description="Prediction endpoint path and format")
    safety: str = Field(..., description="Zero-contact security statement")


# ── Health & Readiness ───────────────────────────────────────────────────────

class URLComponentStatus(BaseModel):
    loaded: bool
    type: str = "RandomForestClassifier"
    features: int = 29


class ImageComponentStatus(BaseModel):
    loaded: bool
    type: str = "MobileNetV2"
    device: str = "cpu"


class FusionWeights(BaseModel):
    url: float = 0.55
    image: float = 0.45


class FusionComponentStatus(BaseModel):
    loaded: bool
    type: str = "prediction-level soft voting"
    weights: FusionWeights
    calibrator: str = "IsotonicRegression"
    threshold: float = 0.543407


class ComponentsStatus(BaseModel):
    url_model: URLComponentStatus
    image_model: ImageComponentStatus
    fusion: FusionComponentStatus


class HealthResponse(BaseModel):
    status: str = Field(..., description="'healthy' if all models loaded, otherwise 'degraded'")
    models_ready: bool = Field(..., description="True if all models are in memory")
    components: ComponentsStatus
    safety_guarantee: str = Field(..., description="Local inert analysis assurance")


# ── Prediction & Explainability ──────────────────────────────────────────────

class URLExplanationItem(BaseModel):
    feature: str = Field(..., description="Lexical feature identifier")
    value: float = Field(..., description="Extracted numerical feature value")
    importance_weight: float = Field(..., description="Global SHAP importance weight")
    impact: str = Field(..., description="Risk impact category (elevated_risk, neutral, benign_indicator)")
    finding: str = Field(..., description="Human-readable assessment of the feature")
    description: str = Field(..., description="Explanation of what this security indicator means")


class ImageExplanationItem(BaseModel):
    component: str = Field(..., description="QR image visual component analyzed")
    finding: str = Field(..., description="Visual pattern observation")


class GradCAMExplanation(BaseModel):
    available: bool = Field(..., description="True if Grad-CAM saliency map was generated")
    data_url: Optional[str] = Field(None, description="Base64-encoded PNG data URL of the heatmap overlay")
    focus_region: Optional[str] = Field(None, description="Dominant high-activation region in QR code")


class ExplanationResponse(BaseModel):
    url: List[URLExplanationItem]
    image: List[ImageExplanationItem]
    gradcam: Optional[GradCAMExplanation] = None


class PredictWeights(BaseModel):
    url_weight: float = 0.55
    image_weight: float = 0.45


class PredictResponse(BaseModel):
    status: str = Field("ok", description="Status code string ('ok')")
    decoded_url: str = Field(..., description="Inert extracted URL string from QR code")
    is_http: bool = Field(..., description="True if scheme is HTTP or HTTPS")
    url_probability: float = Field(..., ge=0.0, le=1.0, description="Random Forest phishing probability")
    image_probability: float = Field(..., ge=0.0, le=1.0, description="MobileNetV2 visual phishing probability")
    fusion_probability: float = Field(..., ge=0.0, le=1.0, description="Raw 55/45 soft voting score")
    calibrated_probability: float = Field(..., ge=0.0, le=1.0, description="Isotonic calibrated risk probability")
    prediction: str = Field(..., description="Final verdict: 'BENIGN' or 'MALICIOUS'")
    risk_level: str = Field(..., description="Severity category: 'LOW', 'MODERATE', 'HIGH', or 'CRITICAL'")
    risk_label: str = Field(..., description="Human-readable risk category title")
    threshold: float = Field(..., description="Operational decision threshold")
    weights: PredictWeights
    explanation: ExplanationResponse
    safety_notice: str = Field(..., description="Reassurance that destination URL was never contacted")


class ErrorResponse(BaseModel):
    status: str = Field("error", description="Status string")
    error: str = Field(..., description="Error message describing the failure")
    detail: Optional[str] = Field(None, description="Additional context or validation details")
