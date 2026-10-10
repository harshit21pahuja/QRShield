"""
QRShield — Local QR Code Decoder

Safely decodes QR codes from uploaded images using OpenCV QRCodeDetector.
CRITICAL SAFETY REQUIREMENT:
Decoded QR URLs must ALWAYS be treated as inert strings.
This module NEVER contacts, fetches, resolves, or opens any decoded URL.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image


class QRDecodeResult:
    def __init__(
        self,
        success: bool,
        decoded_url: Optional[str] = None,
        error: Optional[str] = None,
        points: Optional[np.ndarray] = None,
        is_http: bool = False,
    ):
        self.success = success
        self.decoded_url = decoded_url
        self.error = error
        self.points = points
        self.is_http = is_http

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "decoded_url": self.decoded_url,
            "error": self.error,
            "is_http": self.is_http,
        }


class QRDecoder:
    """Inert local QR Code Decoder using OpenCV."""

    def __init__(self) -> None:
        self.detector = cv2.QRCodeDetector()

    def _to_cv2_image(self, image_input: Union[bytes, str, Path, Image.Image, np.ndarray]) -> np.ndarray:
        """Convert various image input formats to OpenCV BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            image_array = cv2.imread(str(path))
            if image_array is None:
                raise ValueError(f"OpenCV could not read image from {path}")
            return image_array

        if isinstance(image_input, bytes):
            image_stream = np.frombuffer(image_input, np.uint8)
            image_array = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
            if image_array is None:
                raise ValueError("Could not decode image bytes into OpenCV image format.")
            return image_array

        if isinstance(image_input, Image.Image):
            rgb_image = image_input.convert("RGB")
            np_rgb = np.array(rgb_image)
            return cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            if image_input.ndim == 3 and image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_RGBA2BGR)
            return image_input

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def decode(self, image_input: Union[bytes, str, Path, Image.Image, np.ndarray]) -> QRDecodeResult:
        """Decode QR code strictly locally.

        Args:
            image_input: Image file path, raw bytes, PIL Image, or numpy array.

        Returns:
            QRDecodeResult with success flag, decoded inert string, and status.
        """
        try:
            cv_img = self._to_cv2_image(image_input)
        except Exception as exc:
            return QRDecodeResult(
                success=False,
                error=f"Invalid or corrupt image format: {exc}",
            )

        try:
            decoded_text, points, _ = self.detector.detectAndDecode(cv_img)
            decoded_text = (decoded_text or "").strip()

            if not decoded_text:
                # Try multi-detect fallback if standard detect returns empty
                ret, decoded_info, multi_points, _ = self.detector.detectAndDecodeMulti(cv_img)
                if ret and decoded_info:
                    for text in decoded_info:
                        if text and text.strip():
                            decoded_text = text.strip()
                            points = multi_points
                            break

            if not decoded_text:
                return QRDecodeResult(
                    success=False,
                    error="No QR code could be decoded from the provided image. Please ensure the QR code is clearly visible, well-lit, and uncorrupted.",
                )

            import re
            # Extract clean URL if embedded inside metadata/text wrapper
            url_pattern = re.search(r"https?://[^\s\"'<>,]+", decoded_text)
            clean_url = url_pattern.group(0).rstrip(".,;)") if url_pattern else decoded_text

            # Analyze protocol safely
            parsed = urlparse(clean_url)
            scheme = (parsed.scheme or "").lower()
            is_http = scheme in ("http", "https")

            return QRDecodeResult(
                success=True,
                decoded_url=clean_url,
                points=points,
                is_http=is_http,
            )

        except Exception as exc:
            return QRDecodeResult(
                success=False,
                error=f"QR decoding failed: {exc}",
            )


# Global singleton instance
qr_decoder = QRDecoder()
