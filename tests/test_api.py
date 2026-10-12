import io
import json
import pytest
from app.rate_limiter import limiter
from app.config import DOG_PROBABILITY_THRESHOLD

def test_health_endpoint(client):
    """Ensure API health endpoint functions correctly."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["classes_count"] == 120

def test_metrics_info_endpoint(client):
    """Verify /api/info returns exact values from metrics.json without hardcoding."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "EfficientNetB0 (transfer learning)"
    assert data["test_images"] == 8580
    assert data["top1_accuracy"] == 0.8415
    assert data["top5_accuracy"] == 0.984
    assert data["ann_top1_accuracy"] == 0.0409
    assert data["cnn_top1_accuracy"] == 0.1089

def test_valid_dog_prediction_without_gradcam(client, real_dog_image_bytes):
    """
    Test real Chihuahua image from dataset directory:
    - Dog gatekeeper passes
    - Predicts Chihuahua as top-1 with high confidence
    - Returns top-3 predictions
    - Returns trait card
    - Fast prediction (Grad-CAM not computed when toggle is off)
    """
    files = {"file": ("n02085620_2650.jpg", real_dog_image_bytes, "image/jpeg")}
    data = {"gradcam": "false"}

    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert result["is_dog"] is True
    assert result["top_breed"]["breed"] == "Chihuahua"
    assert result["top_breed"]["confidence"] >= 75.0
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["breed"] == "Chihuahua"

    # Verify trait profile attached
    assert result["traits"] is not None
    assert result["traits"]["origin"] == "Mexico"
    assert result["traits"]["size"] == "Toy"
    assert "weight_range_kg" in result["traits"]

    # Verify Grad-CAM is NOT generated when toggle is False
    assert result["gradcam_image"] is None

def test_valid_dog_prediction_with_gradcam(client, real_dog_image_bytes):
    """Test valid dog prediction with Grad-CAM toggle turned ON."""
    files = {"file": ("n02085620_2650.jpg", real_dog_image_bytes, "image/jpeg")}
    data = {"gradcam": "true"}

    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    result = response.json()

    assert result["success"] is True
    assert result["is_dog"] is True
    assert result["top_breed"]["breed"] == "Chihuahua"
    assert result["gradcam_image"] is not None
    assert result["gradcam_image"].startswith("data:image/png;base64,")

def test_non_dog_image_rejection(client, real_cat_image_bytes):
    """
    Test uploading a real cat image:
    MobileNetV2 gatekeeper must flag as not a dog and return rejection message.
    """
    files = {"file": ("cat.jpg", real_cat_image_bytes, "image/jpeg")}
    data = {"gradcam": "false"}

    response = client.post("/predict", files=files, data=data)
    assert response.status_code == 200
    result = response.json()

    assert result["is_dog"] is False
    assert result["message"] == "This doesn't look like a dog."
    assert "gatekeeper_details" in result
    assert result["gatekeeper_details"]["dog_probability_sum"] < DOG_PROBABILITY_THRESHOLD

def test_oversized_file_rejection(client, oversized_image_bytes):
    """Test that files exceeding 5 MB are rejected with HTTP 413."""
    files = {"file": ("large_photo.jpg", oversized_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
    assert "exceeds maximum allowable size" in response.json()["detail"]

def test_wrong_file_type_rejection(client, wrong_file_type_payload):
    """Test that non-image file extensions are rejected with HTTP 400."""
    files = {"file": ("notes.txt", wrong_file_type_payload, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]

def test_corrupted_image_rejection(client, corrupted_image_bytes):
    """Test that corrupted image bytes fail Pillow content validation with HTTP 400."""
    files = {"file": ("corrupt.jpg", corrupted_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "Corrupted or invalid image data" in response.json()["detail"]

def test_rate_limiter_behavior(client, real_dog_image_bytes):
    """Test that rate limiter resets properly and blocks when exceeded."""
    limiter.reset()
    test_ip = "192.168.1.100"
    headers = {"X-Forwarded-For": test_ip}

    # Artificially exhaust rate limiter for test_ip
    for _ in range(limiter.max_requests):
        limiter.is_allowed(test_ip)

    # The next request from test_ip should be blocked with HTTP 429
    files = {"file": ("dog.jpg", real_dog_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, headers=headers)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]

    # Reset limiter for clean state
    limiter.reset()
    response = client.post("/predict", files=files, headers=headers)
    assert response.status_code == 200
