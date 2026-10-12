import io
import base64
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
from PIL import Image

def compute_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "top_conv",
    pred_index: int = None
) -> np.ndarray:
    """
    Computes 2D Grad-CAM heatmap normalized to [0, 1] for target class.
    """
    try:
        # Build grad model from base model or sublayer
        conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Search inside submodels if nested
        conv_layer = None
        for layer in model.layers:
            if hasattr(layer, "get_layer"):
                try:
                    conv_layer = layer.get_layer(last_conv_layer_name)
                    break
                except Exception:
                    pass
        if conv_layer is None:
            raise ValueError(f"Could not locate layer '{last_conv_layer_name}' in model.")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def generate_gradcam_overlay_base64(
    original_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> str:
    """
    Overlays the Grad-CAM heatmap onto the original image and returns a base64 data URI.
    """
    orig_w, orig_h = original_pil.size
    orig_rgb = np.array(original_pil.convert("RGB"), dtype=np.float32) / 255.0

    # Resize heatmap to original image dimensions
    heatmap_tensor = tf.image.resize(heatmap[..., np.newaxis], (orig_h, orig_w)).numpy()
    heatmap_resized = np.squeeze(heatmap_tensor)

    # Colorize heatmap using colormap
    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = cmap(heatmap_resized)[..., :3]

    # Alpha blending
    blended = alpha * colored_heatmap + (1.0 - alpha) * orig_rgb
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    # Convert to PNG in memory
    out_img = Image.fromarray(blended)
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"
