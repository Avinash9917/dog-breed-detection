import io
import json
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import (
    MODEL_FILE,
    CLASS_NAMES_FILE,
    DOG_PROBABILITY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    GRAD_CAM_LAYER_NAME,
    GRAD_CAM_ALPHA
)
from app.traits import trait_service
from app.gradcam import compute_gradcam_heatmap, generate_gradcam_overlay_base64

logger = logging.getLogger("dog_breed_detection")

class ModelService:
    def __init__(self):
        self.model: Optional[tf.keras.Model] = None
        self.mobilenet: Optional[tf.keras.Model] = None
        self.class_names: List[str] = []
        self.is_demo_mode: bool = False
        self.initialized: bool = False

    def initialize(self) -> None:
        """
        Load models and class mappings into memory once during application startup.
        """
        logger.info("[Startup] Initializing ModelService...")
        
        # 1. Load Class Names
        if CLASS_NAMES_FILE.exists():
            with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)
            logger.info(f"[Startup] Loaded {len(self.class_names)} class names.")
        else:
            logger.warning(f"[Startup] Class names file not found at {CLASS_NAMES_FILE}. Running in DEMO MODE.")
            self.is_demo_mode = True

        # 2. Load Dog Breed EfficientNet Model
        if MODEL_FILE.exists() and not self.is_demo_mode:
            try:
                logger.info(f"[Startup] Loading dog breed model from {MODEL_FILE}...")
                self.model = tf.keras.models.load_model(str(MODEL_FILE))
                output_dim = self.model.output_shape[-1]
                assert len(self.class_names) == output_dim, (
                    f"Class names length ({len(self.class_names)}) does not match "
                    f"model output size ({output_dim})."
                )
                logger.info(f"[Startup] Dog breed model loaded successfully. Classes: {output_dim}")
            except Exception as e:
                logger.error(f"[Startup] Failed to load dog breed model: {e}. Falling back to DEMO MODE.")
                self.is_demo_mode = True
        else:
            logger.warning(f"[Startup] Dog breed model file missing at {MODEL_FILE}. Activating DEMO MODE.")
            self.is_demo_mode = True

        # 3. Load MobileNetV2 Gatekeeper Model
        try:
            logger.info("[Startup] Loading MobileNetV2 for dog/non-dog screening...")
            self.mobilenet = tf.keras.applications.MobileNetV2(weights="imagenet")
            logger.info("[Startup] MobileNetV2 gatekeeper initialized.")
        except Exception as e:
            logger.error(f"[Startup] Failed to load MobileNetV2: {e}.")
            self.mobilenet = None

        self.initialized = True
        if self.is_demo_mode:
            logger.warning("[Startup] Running in DEMO MODE. Predictions will be simulated for demonstration.")

    def check_is_dog(self, image_pil: Image.Image) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Runs dog/non-dog gatekeeper check using MobileNetV2 with mobilenet_v2.preprocess_input.
        Computes sum of probabilities over ImageNet classes 151-268 (inclusive).
        """
        # Immediate check for blank / solid-color / featureless images
        arr = np.array(image_pil, dtype=np.float32)
        lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        std_dev = float(np.std(lum))
        if std_dev < 15.0:
            logger.info(f"[Gatekeeper] Rejected blank/solid image (std_dev={std_dev:.2f})")
            return False, 0.0, {
                "dog_probability_sum": 0.0,
                "threshold": DOG_PROBABILITY_THRESHOLD,
                "note": "This image appears blank or has no recognizable features."
            }

        if self.mobilenet is None:
            # If gatekeeper failed to load, pass through to avoid breaking
            return True, 1.0, {"note": "Gatekeeper unavailable, pass through."}

        # MobileNetV2 expects 224x224 RGB image
        mob_img = image_pil.resize((224, 224), Image.Resampling.BILINEAR)
        x = np.array(mob_img.convert("RGB"), dtype=np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        x = np.expand_dims(x, 0)

        preds = self.mobilenet.predict(x, verbose=0)[0]
        
        # ImageNet canine synsets are indexed 151 to 268 inclusive (slice 151:269)
        dog_prob_sum = float(np.sum(preds[151:269]))
        
        is_dog = dog_prob_sum >= DOG_PROBABILITY_THRESHOLD
        logger.info(f"[Gatekeeper] Dog probability sum: {dog_prob_sum:.4f} (Threshold: {DOG_PROBABILITY_THRESHOLD}) -> Is Dog: {is_dog}")

        top_class_idx = int(np.argmax(preds))
        top_prob = float(preds[top_class_idx])

        return is_dog, dog_prob_sum, {
            "dog_probability_sum": round(dog_prob_sum, 4),
            "threshold": DOG_PROBABILITY_THRESHOLD,
            "top_imagenet_class_index": top_class_idx,
            "top_imagenet_probability": round(top_prob, 4)
        }

    def process_and_validate_image(self, file_bytes: bytes) -> Image.Image:
        """
        Validates image content using Pillow, corrects EXIF orientation, and returns PIL Image.
        Raises ValueError if content is corrupted or not a valid image format.
        """
        if not file_bytes:
            raise ValueError("Empty file payload received.")

        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            # Verify image format by reading image structure
            pil_img.verify()
        except Exception as e:
            raise ValueError(f"Corrupted or invalid image data: {str(e)}")

        # Reopen image because verify() closes or invalidates the stream
        pil_img = Image.open(io.BytesIO(file_bytes))

        # Validate format
        fmt = (pil_img.format or "").upper()
        if fmt not in {"JPEG", "JPG", "PNG", "WEBP"}:
            raise ValueError(f"Unsupported image format '{fmt}'. Only JPG, PNG, and WebP are allowed.")

        # EXIF Orientation transpose (crucial for smartphone photos)
        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass  # If EXIF is malformed, proceed with original

        return pil_img.convert("RGB")

    def predict(
        self,
        file_bytes: bytes,
        include_gradcam: bool = False
    ) -> Dict[str, Any]:
        """
        Full prediction pipeline:
        1. In-memory Pillow validation + EXIF orientation
        2. Dog/Non-dog screening via MobileNetV2
        3. EfficientNetB0 breed classification
        4. Grad-CAM generation (only if include_gradcam=True)
        5. Trait lookup from data/breeds.json
        """
        # Step 1: Validate image
        original_pil = self.process_and_validate_image(file_bytes)

        # Step 2: Gatekeeper dog check
        is_dog, dog_prob, gatekeeper_details = self.check_is_dog(original_pil)
        if not is_dog:
            return {
                "success": False,
                "is_dog": False,
                "message": gatekeeper_details.get("note", "This doesn't look like a dog."),
                "gatekeeper_details": gatekeeper_details,
                "is_demo_mode": self.is_demo_mode
            }

        # Step 3: Handle DEMO MODE if model file is missing
        if self.is_demo_mode:
            return self._simulate_demo_prediction(original_pil, include_gradcam)

        # Step 4: EfficientNet Preprocessing (Resize to 224x224 RGB, float32 [0, 255])
        eff_img = original_pil.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.expand_dims(np.array(eff_img, dtype=np.float32), axis=0)

        # Step 5: Breed Inference
        preds = self.model.predict(img_array, verbose=0)[0]
        top3_indices = np.argsort(preds)[-3:][::-1]

        top_predictions = []
        for idx in top3_indices:
            breed_key = self.class_names[idx]
            conf = float(preds[idx])
            top_predictions.append({
                "breed": breed_key,
                "breed_display": breed_key.replace("_", " "),
                "confidence": round(conf * 100.0, 2),
                "confidence_score": round(conf, 4)
            })

        top1_result = top_predictions[0]
        top1_breed = top1_result["breed"]
        top1_confidence = top1_result["confidence"]

        # Step 6: Low Confidence Detection (< 50%)
        is_low_confidence = top1_confidence < (LOW_CONFIDENCE_THRESHOLD * 100.0)
        confidence_message = None
        if is_low_confidence:
            confidence_message = (
                "Not sure, this may be a mixed breed. "
                "Note that this model is trained on purebred dogs from the Stanford Dogs dataset."
            )

        # Step 7: Grad-CAM Explainability (Only computed when toggled on)
        gradcam_overlay_b64 = None
        if include_gradcam:
            try:
                heatmap = compute_gradcam_heatmap(
                    img_array=img_array,
                    model=self.model,
                    last_conv_layer_name=GRAD_CAM_LAYER_NAME,
                    pred_index=int(top3_indices[0])
                )
                gradcam_overlay_b64 = generate_gradcam_overlay_base64(
                    original_pil=original_pil,
                    heatmap=heatmap,
                    alpha=GRAD_CAM_ALPHA
                )
            except Exception as e:
                logger.error(f"[Grad-CAM] Error computing Grad-CAM: {e}")
                gradcam_overlay_b64 = None

        # Step 8: Retrieve Breed Trait Information
        traits = trait_service.get_traits(top1_breed)

        return {
            "success": True,
            "is_dog": True,
            "is_demo_mode": False,
            "top_breed": top1_result,
            "predictions": top_predictions,
            "is_low_confidence": is_low_confidence,
            "confidence_message": confidence_message,
            "traits": traits,
            "gradcam_image": gradcam_overlay_b64,
            "gatekeeper_details": gatekeeper_details
        }

    def _simulate_demo_prediction(self, original_pil: Image.Image, include_gradcam: bool) -> Dict[str, Any]:
        """Simulated prediction for DEMO MODE when model weight files are absent."""
        fallback_breed = "Chihuahua"
        traits = trait_service.get_traits(fallback_breed)
        return {
            "success": True,
            "is_dog": True,
            "is_demo_mode": True,
            "demo_banner": "DEMO MODE ACTIVE: Trained model weight file is missing. Showing simulated prediction.",
            "top_breed": {
                "breed": fallback_breed,
                "breed_display": "Chihuahua",
                "confidence": 84.15,
                "confidence_score": 0.8415
            },
            "predictions": [
                {"breed": "Chihuahua", "breed_display": "Chihuahua", "confidence": 84.15, "confidence_score": 0.8415},
                {"breed": "toy_terrier", "breed_display": "toy terrier", "confidence": 9.42, "confidence_score": 0.0942},
                {"breed": "miniature_pinscher", "breed_display": "miniature pinscher", "confidence": 3.12, "confidence_score": 0.0312}
            ],
            "is_low_confidence": False,
            "confidence_message": None,
            "traits": traits,
            "gradcam_image": None,
            "gatekeeper_details": {"dog_probability_sum": 0.95, "threshold": DOG_PROBABILITY_THRESHOLD}
        }


model_service = ModelService()
