"""
QRShield — Multimodal Fusion & Calibration Engine

Implements Phase 3 prediction-level soft voting:
  raw_fusion = url_weight * url_prob + image_weight * image_prob
  calibrated_probability = isotonic_calibrator.predict([raw_fusion])[0]

All configurations, weights, and thresholds are loaded directly from the saved
experiment artifacts:
  artifacts/fusion_phase3/inference_config.json
  artifacts/fusion_phase3/isotonic_calibrator.joblib
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np

from src.config import PATHS
from src.inference.image_inference import image_engine
from src.inference.qr_decode import qr_decoder
from src.inference.url_inference import url_engine


class FusionInferenceEngine:
    """Combines URL and QR image probabilities with Isotonic calibration."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        calibrator_path: Optional[Path] = None,
    ) -> None:
        self.config_path = config_path or (
            PATHS.root / "artifacts" / "fusion_phase3" / "inference_config.json"
        )
        self.calibrator_path = calibrator_path or (
            PATHS.root / "artifacts" / "fusion_phase3" / "isotonic_calibrator.joblib"
        )

        self.url_weight: float = 0.55
        self.image_weight: float = 0.45
        self.threshold: float = 0.543407387150166
        self.calibration_method: str = "IsotonicRegression"
        self.calibrator = None

        self._load_resources()

    def _load_resources(self) -> None:
        """Load saved fusion configuration and isotonic regression model."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.url_weight = float(config.get("url_weight", 0.55))
                self.image_weight = float(config.get("image_weight", 0.45))
                self.threshold = float(config.get("fusion_threshold", 0.543407387150166))
                self.calibration_method = config.get("calibration_method", "IsotonicRegression")

        if self.calibrator_path.exists():
            self.calibrator = joblib.load(self.calibrator_path)
        else:
            raise FileNotFoundError(f"Isotonic calibrator not found: {self.calibrator_path}")

    def compute_fusion(
        self, url_probability: float, image_probability: float
    ) -> Dict[str, Any]:
        """Compute 55/45 soft voting and isotonic probability calibration.

        Args:
            url_probability: Malicious probability from URL Random Forest.
            image_probability: Malicious probability from MobileNetV2.

        Returns:
            Dictionary with raw fusion, calibrated probability, prediction, and risk level.
        """
        raw_fusion = (
            self.url_weight * url_probability
            + self.image_weight * image_probability
        )
        raw_fusion = float(np.clip(raw_fusion, 0.0, 1.0))

        if self.calibrator is not None:
            calibrated_arr = self.calibrator.predict([raw_fusion])
            calibrated_prob = float(np.clip(calibrated_arr[0], 0.0, 1.0))
        else:
            calibrated_prob = raw_fusion

        prediction = "MALICIOUS" if calibrated_prob >= self.threshold else "BENIGN"

        # High-Confidence Security Override:
        # In quishing, attackers use standard, clean QR codes to deliver phishing links.
        # A visually clean QR wrapper (low image prob) must NOT mask a confirmed phishing URL (>= 85%).
        if url_probability >= 0.85:
            prediction = "MALICIOUS"
            calibrated_prob = max(calibrated_prob, url_probability)

        # Determine risk level based on calibrated probability & threshold
        if calibrated_prob < 0.20:
            risk_level = "LOW"
            risk_label = "Minimal Threat"
        elif calibrated_prob < self.threshold:
            risk_level = "MODERATE"
            risk_label = "Caution Advised"
        elif calibrated_prob < 0.85:
            risk_level = "HIGH"
            risk_label = "Probable Quishing Threat"
        else:
            risk_level = "CRITICAL"
            risk_label = "Severe Malicious Threat"

        return {
            "url_probability": round(url_probability, 6),
            "image_probability": round(image_probability, 6),
            "fusion_probability": round(raw_fusion, 6),
            "calibrated_probability": round(calibrated_prob, 6),
            "prediction": prediction,
            "risk_level": risk_level,
            "risk_label": risk_label,
            "threshold": round(self.threshold, 6),
            "weights": {
                "url_weight": self.url_weight,
                "image_weight": self.image_weight,
            },
        }

    def predict_pipeline(
        self,
        image_input: Union[bytes, str, Path],
        generate_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """Execute the complete end-to-end QRShield analysis pipeline.

        Pipeline:
            QR image upload -> Local QR decode -> Inert URL string ->
            URL feature extraction (29 features) + MobileNetV2 image inference ->
            55/45 soft voting -> Isotonic calibration -> Risk + explanation.
        """
        # Step 1: Local QR decode (inert string extraction)
        decode_result = qr_decoder.decode(image_input)
        if not decode_result.success:
            return {
                "status": "error",
                "error": decode_result.error,
                "decoded_url": None,
                "url_probability": None,
                "image_probability": None,
                "fusion_probability": None,
                "calibrated_probability": None,
                "prediction": None,
                "risk_level": None,
                "explanation": None,
            }

        decoded_url = decode_result.decoded_url

        # Step 2: URL static feature analysis (zero network requests)
        url_res = url_engine.predict(decoded_url)

        # Step 3: QR Image visual inference + Grad-CAM
        image_res = image_engine.predict(image_input, generate_gradcam=generate_gradcam)

        # Step 4: Multimodal fusion + Isotonic calibration
        fusion_res = self.compute_fusion(
            url_probability=url_res["url_probability"],
            image_probability=image_res["image_probability"],
        )

        return {
            "status": "ok",
            "decoded_url": decoded_url,
            "is_http": decode_result.is_http,
            "url_probability": fusion_res["url_probability"],
            "image_probability": fusion_res["image_probability"],
            "fusion_probability": fusion_res["fusion_probability"],
            "calibrated_probability": fusion_res["calibrated_probability"],
            "prediction": fusion_res["prediction"],
            "risk_level": fusion_res["risk_level"],
            "risk_label": fusion_res["risk_label"],
            "threshold": fusion_res["threshold"],
            "weights": fusion_res["weights"],
            "explanation": {
                "url": url_res["explanation"],
                "image": image_res["explanation"],
                "gradcam": image_res.get("gradcam"),
            },
            "safety_notice": "Static local analysis only. Destination URL was NOT contacted, crawled, or visited.",
        }


# Global singleton instance
fusion_engine = FusionInferenceEngine()
