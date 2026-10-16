"""
QRShield — Backend Configuration Settings

Provides centralized, environment-aware configuration for the FastAPI service,
including CORS policies, file upload limits, host/port settings, and logging.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings:
    """Application settings loaded from environment variables with safe defaults."""

    def __init__(self) -> None:
        self.app_name: str = os.getenv("APP_NAME", "QRShield API")
        self.app_version: str = os.getenv("APP_VERSION", "1.1.0")
        self.debug: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

        # Network binding
        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))

        # CORS Origins — comma-separated string in env, parsed into list
        default_origins = (
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:8000,http://127.0.0.1:8000"
        )
        raw_origins = os.getenv("CORS_ORIGINS", default_origins)
        self.cors_origins: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]

        # Upload limits (Default: 10 MB)
        self.max_upload_size_bytes: int = int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024))
        )

        # Allowed image MIME types and extensions
        self.allowed_content_types: List[str] = [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
        ]
        self.allowed_extensions: List[str] = [".png", ".jpg", ".jpeg", ".webp"]

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # Paths
        self.project_root: Path = PROJECT_ROOT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of Settings."""
    return Settings()
