"""
Configuration module for HealthLens - Person 1 (Document Processing Pipeline).
Centralizes file paths and project constants for clean, maintainable code.
"""

from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHARTS_DIR = PROCESSED_DATA_DIR / "charts"

# Ensure essential data subdirectories exist
SAMPLE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Supported input file extensions
SUPPORTED_DOCUMENT_EXTENSIONS = [".pdf", ".txt"]
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
