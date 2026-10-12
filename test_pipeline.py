"""
Sanity Verification Test for Dog Breed Classifier Components
"""
import json
import numpy as np
import tensorflow as tf
from train import (
    build_ann_baseline,
    build_custom_cnn,
    build_efficientnet_model,
    compile_model,
    create_callbacks,
    compute_gradcam_heatmap,
    overlay_gradcam,
    INPUT_SHAPE,
    NUM_CLASSES
)

def run_tests():
    print("=" * 60)
    print("RUNNING COMPONENT SANITY TESTS")
    print("=" * 60)

    # 1. Test Model 1: ANN Baseline
    print("[Test 1/5] Building & compiling ANN Baseline...")
    ann = build_ann_baseline(INPUT_SHAPE, NUM_CLASSES)
    ann = compile_model(ann, 1e-3)
    dummy_input = tf.random.uniform((2, 224, 224, 3), minval=0, maxval=255)
    out_ann = ann(dummy_input)
    assert out_ann.shape == (2, 120), f"ANN output shape mismatch: {out_ann.shape}"
    print(f"  ANN output shape: {out_ann.shape} | Parameters: {ann.count_params():,}")

    # 2. Test Model 2: Custom CNN
    print("[Test 2/5] Building & compiling Custom CNN...")
    cnn = build_custom_cnn(INPUT_SHAPE, NUM_CLASSES)
    cnn = compile_model(cnn, 1e-3)
    out_cnn = cnn(dummy_input)
    assert out_cnn.shape == (2, 120), f"CNN output shape mismatch: {out_cnn.shape}"
    print(f"  CNN output shape: {out_cnn.shape} | Parameters: {cnn.count_params():,}")

    # 3. Test Model 3: EfficientNetB0 Transfer Learning
    print("[Test 3/5] Building & compiling EfficientNetB0...")
    effnet, base = build_efficientnet_model(INPUT_SHAPE, NUM_CLASSES)
    base.trainable = False
    effnet = compile_model(effnet, 1e-3)
    out_effnet = effnet(dummy_input)
    assert out_effnet.shape == (2, 120), f"EfficientNet output shape mismatch: {out_effnet.shape}"
    print(f"  EfficientNet output shape: {out_effnet.shape} | Total: {effnet.count_params():,}")

    # Test Phase 2 unfreezing
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    effnet = compile_model(effnet, 1e-5)
    trainable_p2 = sum([tf.keras.backend.count_params(w) for w in effnet.trainable_weights])
    print(f"  Phase 2 Fine-Tuning Trainable Parameters: {trainable_p2:,}")

    # 4. Test Grad-CAM
    print("[Test 4/5] Testing Grad-CAM computation & overlay...")
    single_img = np.random.uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    heatmap = compute_gradcam_heatmap(single_img, effnet, last_conv_layer_name="top_conv", pred_index=0)
    assert heatmap.shape == (7, 7), f"Heatmap shape mismatch: {heatmap.shape}"
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0, f"Heatmap normalization error: min={heatmap.min()}, max={heatmap.max()}"

    overlay = overlay_gradcam(single_img[0], heatmap, alpha=0.4)
    assert overlay.shape == (224, 224, 3), f"Overlay shape mismatch: {overlay.shape}"
    assert overlay.dtype == np.uint8, f"Overlay dtype mismatch: {overlay.dtype}"
    print(f"  Grad-CAM heatmap shape: {heatmap.shape}, overlay shape: {overlay.shape}, dtype: {overlay.dtype}")

    # 5. Test Model Saving and Loading
    print("[Test 5/5] Testing Model Saving and class_names.json...")
    test_model_path = "test_model.keras"
    effnet.save(test_model_path)
    loaded_model = tf.keras.models.load_model(test_model_path)
    loaded_out = loaded_model(dummy_input)
    assert loaded_out.shape == (2, 120), "Loaded model inference failed!"
    import os
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
    print("  Model successfully saved to .keras format and reloaded!")

    print("=" * 60)
    print("ALL 5 COMPONENT SANITY TESTS PASSED PERFECTLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
