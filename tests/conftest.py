import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import app
from app.rate_limiter import limiter
from app.model_service import model_service

BASE_DIR = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="session", autouse=True)
def init_models():
    """Ensure models are initialized once for test session."""
    if not model_service.initialized:
        model_service.initialize()

@pytest.fixture(autouse=True)
def reset_rate_limiter_each_test():
    """Reset rate limiter state before every single test to prevent interference."""
    limiter.reset()
    yield
    limiter.reset()

@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)

@pytest.fixture
def real_dog_image_bytes():
    """Loads real Chihuahua image from dataset directory."""
    dog_path = BASE_DIR / "dataset" / "n02085620_2650.jpg"
    assert dog_path.exists(), f"Real dog test image not found at {dog_path}"
    with open(dog_path, "rb") as f:
        return f.read()

@pytest.fixture
def real_cat_image_bytes():
    """Loads real cat photo from test_assets directory."""
    cat_path = BASE_DIR / "tests" / "test_assets" / "cat.jpg"
    assert cat_path.exists(), f"Cat test image not found at {cat_path}"
    with open(cat_path, "rb") as f:
        return f.read()

@pytest.fixture
def oversized_image_bytes():
    """Creates a dummy byte payload of 6 MB (exceeding 5 MB limit)."""
    return b"\x00" * (6 * 1024 * 1024)

@pytest.fixture
def wrong_file_type_payload():
    """Returns a text file payload."""
    return b"This is a plain text file, definitely not an image."

@pytest.fixture
def corrupted_image_bytes():
    """Returns corrupted image bytes starting with JPEG SOI but corrupted body."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00corrupted_bytes_that_fail_decoding"
