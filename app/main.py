import os
import re
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Request,
    HTTPException,
    Depends,
    status
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import (
    MAX_UPLOAD_SIZE_BYTES,
    ALLOWED_EXTENSIONS,
    ALLOWED_MIME_TYPES,
    METRICS_FILE,
    STATIC_DIR
)
from app.model_service import model_service
from app.rate_limiter import rate_limit_dependency

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dog_breed_app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load models and artifacts once at application startup.
    Clean up resources gracefully on shutdown.
    """
    logger.info("Application starting up...")
    model_service.initialize()
    logger.info("Application startup complete.")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Dog Breed Detection System",
    description="Fine-grained canine breed classification with trait details and Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def sanitize_filename(filename: Optional[str]) -> str:
    """Strip dangerous characters and path traversals from filenames."""
    if not filename:
        return "upload.jpg"
    clean = os.path.basename(filename)
    clean = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean)
    return clean[:100]


# ------------------------------------------------------------------------------
# Frontend Page Routes (Serves index.html and info.html)
# ------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def serve_home():
    root_index = os.path.join(os.getcwd(), "index.html")
    if os.path.exists(root_index):
        return FileResponse(root_index)
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"message": "Frontend index.html not found."})


@app.get("/info", include_in_schema=False)
@app.get("/info.html", include_in_schema=False)
async def serve_info():
    root_info = os.path.join(os.getcwd(), "info.html")
    if os.path.exists(root_info):
        return FileResponse(root_info)
    info_path = STATIC_DIR / "info.html"
    if info_path.exists():
        return FileResponse(str(info_path))
    return JSONResponse({"message": "Frontend info.html not found."})


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/health")
async def health_check():
    """Health status and model operational state."""
    return {
        "status": "healthy",
        "model_loaded": model_service.model is not None,
        "mobilenet_loaded": model_service.mobilenet is not None,
        "is_demo_mode": model_service.is_demo_mode,
        "classes_count": len(model_service.class_names)
    }


@app.get("/api/info")
async def get_model_info():
    """
    Dynamically loads and returns model evaluation metrics, dataset information,
    and comparative baseline benchmarks from models/metrics.json.
    """
    if not METRICS_FILE.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics file not found."
        )

    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        return metrics_data
    except Exception as e:
        logger.error(f"Error reading metrics.json: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load metrics data."
        )


@app.post("/predict", dependencies=[Depends(rate_limit_dependency)])
async def predict_breed(
    file: UploadFile = File(...),
    gradcam: bool = Form(False)
):
    """
    Identify dog breed from uploaded image.
    - Validates file size (max 5 MB)
    - Validates image content using Pillow
    - Rejects non-canine images using MobileNetV2 ImageNet filter
    - Returns top-3 breeds, confidences, trait cards, and optional Grad-CAM overlay
    - Never saves uploaded files to disk.
    """
    safe_name = sanitize_filename(file.filename)
    
    # Check file extension
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read image bytes into memory (Max 5MB check)
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowable size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # Execute classification pipeline
    try:
        result = model_service.predict(
            file_bytes=contents,
            include_gradcam=gradcam
        )
        return JSONResponse(status_code=status.HTTP_200_OK, content=result)
    except ValueError as ve:
        # Invalid image format or corrupted content
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as exc:
        logger.error(f"Prediction error for {safe_name}: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the image. Please try again."
        )


# Mount Static Files (CSS, JS, assets)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

data_dir = os.path.join(os.getcwd(), "data")
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

models_dir = os.path.join(os.getcwd(), "models")
if os.path.exists(models_dir):
    app.mount("/models", StaticFiles(directory=models_dir), name="models")
