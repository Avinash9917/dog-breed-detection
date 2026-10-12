import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"

# Model Filepaths
MODEL_FILE = MODELS_DIR / "dog_breed_model.keras"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"
METRICS_FILE = MODELS_DIR / "metrics.json"
BREEDS_FILE = DATA_DIR / "breeds.json"

# Request & Validation Limits
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

# MobileNetV2 Dog Filter Threshold
# ImageNet classes 151 through 268 (inclusive) correspond to domestic & wild canines.
DOG_PROBABILITY_THRESHOLD = 0.50

# Breed Classifier Confidence Threshold
# If top-1 confidence is below 50%, flag as potential mixed breed.
LOW_CONFIDENCE_THRESHOLD = 0.50

# Rate Limiting Settings
RATE_LIMIT_MAX_REQUESTS = 40  # requests
RATE_LIMIT_WINDOW_SECONDS = 60  # per 60 seconds per client IP

# Grad-CAM Settings
GRAD_CAM_LAYER_NAME = "top_conv"
GRAD_CAM_ALPHA = 0.45
