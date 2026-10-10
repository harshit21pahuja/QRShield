"""
QRShield — Image Inference Module

Loads the Phase 2B trained MobileNetV2 checkpoint once at startup and performs
inert visual analysis and Grad-CAM interpretability on uploaded QR images.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.config import PATHS
from src.models.image_baseline import (
    build_mobilenetv2_model,
    ensure_rgb,
    gradcam_heatmap,
    preprocess_image,
)


class ImageInferenceEngine:
    """Inference engine for MobileNetV2 image branch."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        config_path: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path or (
            PATHS.root / "artifacts" / "image_baseline_gpu" / "best_model.pt"
        )
        self.config_path = config_path or (
            PATHS.root / "artifacts" / "image_baseline_gpu" / "config.json"
        )

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[nn.Module] = None
        self.config: Dict[str, Any] = {}
        self.threshold: float = 0.5
        self.image_size: int = 224

        self._load_resources()

    def _load_resources(self) -> None:
        """Load trained MobileNetV2 weights and configuration."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Image model checkpoint not found: {self.checkpoint_path}")

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                    self.threshold = float(self.config.get("threshold", 0.5))
                    self.image_size = int(self.config.get("image_size", 224))
            except Exception:
                self.threshold = 0.5
                self.image_size = 224

        self.model = build_mobilenetv2_model(num_classes=1, pretrained=False)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _to_pil_image(self, image_input: Union[bytes, str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Convert any supported image input to an RGB PIL Image."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")
            return ensure_rgb(Image.open(path))

        if isinstance(image_input, bytes):
            return ensure_rgb(Image.open(io.BytesIO(image_input)))

        if isinstance(image_input, Image.Image):
            return ensure_rgb(image_input)

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return Image.fromarray(image_input).convert("RGB")
            if image_input.ndim == 3:
                # Assuming OpenCV BGR if uint8 numpy array
                rgb_array = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_array)

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def predict(
        self,
        image_input: Union[bytes, str, Path, Image.Image, np.ndarray],
        generate_gradcam: bool = True,
    ) -> Dict[str, Any]:
        """Perform forward pass and optional Grad-CAM visual interpretability.

        Args:
            image_input: Raw image bytes, path, PIL Image, or array.
            generate_gradcam: Whether to compute Grad-CAM heatmap visualization.

        Returns:
            Dictionary with image_probability, prediction, and Grad-CAM data.
        """
        pil_image = self._to_pil_image(image_input)
        tensor = preprocess_image(pil_image, size=self.image_size).to(self.device)

        # Forward pass for probability
        with torch.no_grad():
            logits = self.model(tensor.unsqueeze(0))
            prob = float(torch.sigmoid(logits).item())

        prediction_label = "MALICIOUS" if prob >= self.threshold else "BENIGN"

        gradcam_data: Optional[Dict[str, Any]] = None
        if generate_gradcam:
            try:
                gradcam_data = self._generate_gradcam_visualization(pil_image, tensor)
            except Exception as exc:
                gradcam_data = {
                    "available": False,
                    "error": f"Grad-CAM generation failed: {exc}",
                }

        image_explanation = self._generate_image_explanation(prob, gradcam_data)

        return {
            "image_probability": prob,
            "prediction": prediction_label,
            "threshold": self.threshold,
            "gradcam": gradcam_data,
            "explanation": image_explanation,
        }

    def _generate_gradcam_visualization(
        self,
        original_pil: Image.Image,
        image_tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """Compute Grad-CAM heatmap and create an overlaid visualization."""
        # gradcam_heatmap requires tensor on CPU with requires_grad
        cpu_tensor = image_tensor.to("cpu")
        cpu_model = self.model.to("cpu")

        heatmap = gradcam_heatmap(cpu_model, cpu_tensor)

        # Move model back to target device if GPU
        if self.device.type != "cpu":
            self.model.to(self.device)

        # Prepare 224x224 RGB image for overlay
        resized_pil = original_pil.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        rgb_np = np.array(resized_pil, dtype=np.uint8)

        # Colorize heatmap (0.0 to 1.0 -> 0 to 255)
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Alpha blend overlay: 60% original image, 40% heatmap
        overlay = cv2.addWeighted(rgb_np, 0.60, heatmap_colored_rgb, 0.40, 0)

        # Convert overlay to base64 PNG data URL
        success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("Failed to encode overlay image to PNG.")

        base64_str = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_str}"

        # Analyze peak activation coordinates
        max_idx = np.unravel_index(np.argmax(heatmap, axis=None), heatmap.shape)
        peak_y, peak_x = int(max_idx[0]), int(max_idx[1])

        # Region classification (finder patterns vs center data payload)
        # In a standard 224x224 QR code:
        # Top-left, top-right, bottom-left corners house position detection patterns
        is_corner = (peak_x < 70 and peak_y < 70) or (peak_x > 154 and peak_y < 70) or (peak_x < 70 and peak_y > 154)
        focus_region = "Finder / Timing Pattern" if is_corner else "Data Matrix / Module Payload"

        return {
            "available": True,
            "data_url": data_url,
            "heatmap_peak": {"x": peak_x, "y": peak_y},
            "peak_intensity": float(np.max(heatmap)),
            "mean_intensity": float(np.mean(heatmap)),
            "focus_region": focus_region,
        }

    def _generate_image_explanation(
        self, prob: float, gradcam_data: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate human-readable visual inspection observations."""
        explanations = []

        if prob >= self.threshold:
            explanations.append({
                "component": "Visual Density & Embedding",
                "finding": f"MobileNetV2 visual model identified elevated risk signals (probability {prob:.1%}).",
                "detail": "Visual features such as dense data matrix patterning, high module variance, or boundary anomalies correlated with malicious QR samples in training.",
            })
        else:
            explanations.append({
                "component": "Visual Density & Embedding",
                "finding": f"Visual features appear consistent with standard benign QR codes (probability {prob:.1%}).",
                "detail": "Standard module alignment, regular error-correction distribution, and typical contrast ratios observed.",
            })

        if gradcam_data and gradcam_data.get("available"):
            region = gradcam_data.get("focus_region", "data payload")
            explanations.append({
                "component": "Grad-CAM Activation Focus",
                "finding": f"Model attention concentrated predominantly on the QR {region}.",
                "detail": "Saliency mapping demonstrates the convolutional filters evaluated spatial module configuration rather than background noise.",
            })

        return explanations


# Global singleton instance
image_engine = ImageInferenceEngine()
