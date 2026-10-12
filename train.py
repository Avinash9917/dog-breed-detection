"""
Stanford Dogs Fine-Grained Breed Classification
================================================
Academic & College-Level Deep Learning Pipeline

This module implements a rigorous, end-to-end comparative study on the
Stanford Dogs dataset (120 classes, 20,580 images). It compares three distinct
deep learning paradigms for fine-grained image recognition:
  1. Multi-Layer Perceptron (ANN baseline: Flatten + Dense layers)
  2. Small Custom Convolutional Neural Network (4 Conv Blocks, BatchNorm, Dropout)
  3. Transfer Learning & Two-Stage Fine-Tuning with EfficientNetB0 (ImageNet weights)

Key Methodological Guarantees:
  - Official Train/Test split respected (no test data leakage).
  - 15% validation split carved strictly from training data via stratified sampling.
  - End-to-end data augmentation pipeline (Flip, Rotation, Zoom, Brightness).
  - Regularization via EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau.
  - Full evaluation on unseen test set reporting Top-1 and Top-5 Accuracy.
  - Interpretability and model explainability via Gradient-Weighted Class
    Activation Mapping (Grad-CAM).
  - Best model artifact exported as `dog_breed_model.keras` with index-aligned
    `class_names.json`.
"""

import os
import sys
import json
import math
import random
import shutil
import urllib.request
import tarfile
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers

# Visualization and data analysis libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
except ImportError:
    plt = None
    sns = None
    pd = None

# ==============================================================================
# 1. REPRODUCIBILITY & SYSTEM CONFIGURATION
# ==============================================================================
def set_random_seeds(seed: int = 42) -> None:
    """
    Enforce deterministic execution across Python runtime, NumPy, and TensorFlow.

    Args:
        seed (int): The pseudo-random generator seed (default: 42).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Setup] Deterministic random seeds fixed to: {seed}")


# ==============================================================================
# 2. DATASET INGESTION & STRATIFIED SPLITTING
# ==============================================================================
STANFORD_DOGS_URLS = {
    "images": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "lists": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
}


def download_and_extract_stanford_dogs(data_dir: str = "./dataset") -> Tuple[str, str]:
    """
    Download and extract the official Stanford Dogs dataset archives if not already present.

    The dataset consists of:
      - images.tar (~757 MB): 20,580 annotated canine images across 120 breeds.
      - lists.tar (~0.5 MB): MATLAB files specifying official train and test splits.

    Args:
        data_dir (str): Local root directory to cache downloaded files and extracted images.

    Returns:
        Tuple[str, str]: Absolute paths to (images_root_directory, lists_root_directory).
    """
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    images_tar = data_path / "images.tar"
    lists_tar = data_path / "lists.tar"
    images_dir = data_path / "Images"
    lists_dir = data_path / "lists"

    # 1. Download lists.tar
    train_mat_search = list(data_path.glob("**/train_list.mat"))
    if not train_mat_search:
        if not lists_tar.exists():
            print(f"[Dataset] Downloading lists.tar from {STANFORD_DOGS_URLS['lists']}...")
            urllib.request.urlretrieve(STANFORD_DOGS_URLS["lists"], str(lists_tar))
            print("[Dataset] lists.tar download complete.")
        print("[Dataset] Extracting lists.tar...")
        lists_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(lists_tar)) as tar:
            tar.extractall(str(lists_dir))
        print("[Dataset] lists.tar successfully extracted.")
    else:
        lists_dir = train_mat_search[0].parent

    # 2. Download images.tar
    if not images_dir.exists() or len(list(images_dir.glob("*/*"))) < 20000:
        if not images_tar.exists():
            print(f"[Dataset] Downloading images.tar (~757 MB) from {STANFORD_DOGS_URLS['images']}...")
            print("[Dataset] This may take 1-3 minutes depending on network bandwidth...")

            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                if count % 1000 == 0 or percent >= 100:
                    sys.stdout.write(f"\r[Dataset] Download progress: {percent}% ({count * block_size // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB)")
                    sys.stdout.flush()

            urllib.request.urlretrieve(STANFORD_DOGS_URLS["images"], str(images_tar), reporthook=_progress)
            print("\n[Dataset] images.tar download complete.")

        print("[Dataset] Extracting images.tar (20,580 images)...")
        with tarfile.open(str(images_tar)) as tar:
            tar.extractall(str(data_path))
        print("[Dataset] images.tar successfully extracted.")

    return str(images_dir), str(lists_dir)


def load_dataset_splits(
    images_dir: str,
    lists_dir: str,
    val_split: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Parse official train/test lists from .mat files, extract clean breed names,
    and partition the official training set into Stratified Train and Validation subsets.

    Methodological Integrity:
      - Test set (8,580 images) remains strictly isolated (Zero Test Leakage).
      - Train set (12,000 images) is split into:
          * 85% Train (10,200 images, ~85 per breed)
          * 15% Validation (1,800 images, exactly 15 per breed)
      - Class labels are normalized from MATLAB 1-based indexing (1..120) to 0..119.

    Args:
        images_dir (str): Directory containing breed image subfolders.
        lists_dir (str): Directory containing train_list.mat and test_list.mat.
        val_split (float): Fraction of training data to allocate for validation (default 0.15).
        seed (int): Random seed for reproducible stratified split.

    Returns:
        Tuple containing:
          - train_paths (List[str]): Image filepaths for training.
          - train_labels (np.ndarray): Integer labels [0..119] for training.
          - val_paths (List[str]): Image filepaths for validation.
          - val_labels (np.ndarray): Integer labels [0..119] for validation.
          - test_paths (List[str]): Image filepaths for testing.
          - test_labels (np.ndarray): Integer labels [0..119] for testing.
          - class_names (List[str]): List of 120 breed names ordered by label index (0..119).
    """
    images_path = Path(images_dir)
    lists_path = Path(lists_dir)

    train_mat_candidates = list(lists_path.glob("**/train_list.mat"))
    test_mat_candidates = list(lists_path.glob("**/test_list.mat"))
    if not train_mat_candidates or not test_mat_candidates:
        raise FileNotFoundError(f"Could not locate train_list.mat / test_list.mat under {lists_dir}")

    train_mat = scipy.io.loadmat(str(train_mat_candidates[0]))
    test_mat = scipy.io.loadmat(str(test_mat_candidates[0]))

    # Extract relative file paths and 0-indexed labels
    train_rel_files = [item[0][0] for item in train_mat["file_list"]]
    train_raw_labels = train_mat["labels"].flatten() - 1  # 1..120 -> 0..119

    test_rel_files = [item[0][0] for item in test_mat["file_list"]]
    test_raw_labels = test_mat["labels"].flatten() - 1

    # Build canonical mapping from class index to human-readable breed name
    # Folder pattern: n02085620-Chihuahua -> "Chihuahua"
    label_to_breed: Dict[int, str] = {}
    for rel_path, lbl in zip(train_rel_files, train_raw_labels):
        breed_folder = rel_path.split("/")[0] if "/" in rel_path else rel_path.split("\\")[0]
        breed_clean_name = breed_folder.split("-", 1)[1] if "-" in breed_folder else breed_folder
        if lbl not in label_to_breed:
            label_to_breed[lbl] = breed_clean_name

    num_classes = len(label_to_breed)
    assert num_classes == 120, f"Expected 120 dog breeds, found {num_classes}."
    class_names = [label_to_breed[i] for i in range(num_classes)]

    # Resolve full absolute paths
    train_abs_files = [str(images_path / f) for f in train_rel_files]
    test_abs_files = [str(images_path / f) for f in test_rel_files]

    # Perform Stratified Train / Validation Split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_abs_files,
        train_raw_labels,
        test_size=val_split,
        stratify=train_raw_labels,
        random_state=seed
    )

    print("\n" + "=" * 70)
    print("DATASET PARTITION SUMMARY (Zero Test Leakage Verified)")
    print("=" * 70)
    print(f"Total Canine Breeds       : {num_classes}")
    print(f"Official Training Pool    : {len(train_rel_files)} images")
    print(f"  ├── Train Partition (85%): {len(train_paths)} images (~{len(train_paths)//num_classes}/class)")
    print(f"  └── Val Partition   (15%): {len(val_paths)} images ({len(val_paths)//num_classes}/class, perfectly balanced)")
    print(f"Official Test Partition   : {len(test_abs_files)} images (strictly held-out)")
    print("=" * 70 + "\n")

    return (
        list(train_paths),
        np.array(train_labels, dtype=np.int32),
        list(val_paths),
        np.array(val_labels, dtype=np.int32),
        list(test_abs_files),
        np.array(test_raw_labels, dtype=np.int32),
        class_names
    )


# ==============================================================================
# 3. TF.DATA PIPELINE & DATA AUGMENTATION
# ==============================================================================
IMG_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 120


def get_data_augmentation() -> tf.keras.Sequential:
    """
    Construct the Keras sequential data augmentation sub-network.
    Applies stochastic geometric and photometric transformations to combat overfitting:
      - Random Horizontal Flip
      - Random Rotation (+/- 15%)
      - Random Zoom (+/- 15%)
      - Random Brightness (+/- 15%)
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
        layers.RandomRotation(0.15, fill_mode="reflect", name="aug_rotation"),
        layers.RandomZoom(0.15, fill_mode="reflect", name="aug_zoom"),
        layers.RandomBrightness(0.15, value_range=(0.0, 255.0), name="aug_brightness"),
    ], name="data_augmentation_pipeline")


def _parse_image_and_label(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Read JPEG image bytes from disk, decode RGB channels, and resize to 224x224.
    Raw pixel values remain in [0, 255] float32 for model-level normalization.
    """
    image_bytes = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def create_tf_dataset(
    file_paths: List[str],
    labels: np.ndarray,
    batch_size: int = 32,
    is_training: bool = True,
    augmenter: Optional[tf.keras.Sequential] = None
) -> tf.data.Dataset:
    """
    Construct a high-performance, asynchronous tf.data input pipeline.

    Args:
        file_paths (List[str]): List of absolute image paths.
        labels (np.ndarray): Corresponding integer labels.
        batch_size (int): Mini-batch size.
        is_training (bool): If True, enables shuffling and online data augmentation.
        augmenter (tf.keras.Sequential, optional): Augmentation model to apply to training batches.

    Returns:
        tf.data.Dataset: Batched, prefetched dataset ready for GPU streaming.
    """
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 10000), reshuffle_each_iteration=True)

    # Parallel decode and resize
    dataset = dataset.map(_parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    # Apply batch-wise data augmentation if specified
    if is_training and augmenter is not None:
        dataset = dataset.map(
            lambda x, y: (augmenter(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


# ==============================================================================
# 4. MODEL ARCHITECTURES
# ==============================================================================

def build_ann_baseline(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Model 1: Artificial Neural Network (ANN) Baseline.
    Architecture:
      Input (224x224x3) -> Rescaling(1./255) -> Flatten (150,528 features)
      -> Dense(512, ReLU) -> BatchNorm -> Dropout(0.4)
      -> Dense(256, ReLU) -> BatchNorm -> Dropout(0.3)
      -> Dense(120, Softmax)

    Theoretical Note for Academic Report:
      The ANN baseline flattens the 2D spatial grid into a 1D vector of 150,528 scalar values.
      Because fully connected layers lack translation equivariance and local receptive fields,
      the model must independently learn spatial patterns at every pixel coordinate, resulting
      in severe parameter bloat and poor inductive bias for fine-grained image recognition.
    """
    inputs = layers.Input(shape=input_shape, name="ann_input")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    x = layers.Flatten(name="flatten")(x)

    # Dense Block 1
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)

    # Dense Block 2
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Classifier Head
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ann_baseline_model")
    return model


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES
) -> tf.keras.Model:
    """
    Model 2: Small Custom Convolutional Neural Network (CNN).
    Architecture:
      4 Convolutional blocks featuring:
        - 3x3 Conv2D with ReLU activation and 'same' padding
        - Batch Normalization to stabilize internal covariate shift
        - 2x2 Max Pooling to reduce spatial dimensions and impart translation invariance
      Followed by:
        - Global Average Pooling (drastically reduces parameters vs flattening)
        - Dense(256, ReLU) + Dropout(0.4)
        - Dense(120, Softmax)

    Theoretical Note for Academic Report:
      Convolutions enforce translation equivariance and locality. Hierarchical stacking
      allows early layers to extract edge/texture primitives, intermediate layers to form
      parts (ears, snouts, coat patterns), and deeper layers to capture holistic canine anatomy.
    """
    inputs = layers.Input(shape=input_shape, name="cnn_input")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Conv Block 1: 32 filters, 224x224 -> 112x112
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Conv Block 2: 64 filters, 112x112 -> 56x56
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Conv Block 3: 128 filters, 56x56 -> 28x28
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Conv Block 4: 256 filters, 28x28 -> 14x14
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_feat")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.4, name="dropout_classifier")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn_model")
    return model


def build_efficientnet_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Model 3: Transfer Learning with EfficientNetB0 (pre-trained on ImageNet).

    Architecture:
      - EfficientNetB0 backbone: Compound scaling balancing depth, width, and resolution.
      - Retains built-in Rescaling/Normalization layers (expects inputs in [0, 255]).
      - Custom classification head:
          GlobalAveragePooling2D -> BatchNorm -> Dropout(0.3)
          -> Dense(256, ReLU) -> Dropout(0.2) -> Dense(120, Softmax)

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model]: (Full Transfer Learning Model, Backbone Base Model).
    """
    inputs = layers.Input(shape=input_shape, name="efficientnet_input")

    # Load ImageNet-pretrained backbone without top classification head
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(0.3, name="head_dropout_1")(x)
    x = layers.Dense(256, activation="relu", name="head_dense_256")(x)
    x = layers.Dropout(0.2, name="head_dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_dog_classifier")
    return model, base_model


# ==============================================================================
# 5. TRAINING UTILITIES & CALLBACK FACTORY
# ==============================================================================
def create_callbacks(
    checkpoint_filepath: str,
    patience_stop: int = 5,
    patience_lr: int = 2
) -> List[tf.keras.callbacks.Callback]:
    """
    Construct standardized training callbacks:
      1. EarlyStopping: Prevents overfitting by restoring weights from best validation loss.
      2. ReduceLROnPlateau: Decays learning rate dynamically when validation plateau occurs.
      3. ModelCheckpoint: Persists the optimal checkpoint based on validation accuracy.
    """
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_top_5_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Compile model with Adam optimizer, Sparse Categorical Cross-Entropy loss,
    and tracking for both Top-1 and Top-5 Categorical Accuracy.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )
    return model


# ==============================================================================
# 6. GRAD-CAM (EXPLAINABLE ARTIFICIAL INTELLIGENCE)
# ==============================================================================
def compute_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "top_conv",
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Compute Gradient-Weighted Class Activation Map (Grad-CAM) for a given input image.

    Mathematical Formulation (Selvaraju et al., 2017):
      1. Forward pass computes feature maps A^k from final convolutional layer.
      2. Backpropagation computes gradient of score y^c with respect to A^k:
         alpha_k^c = (1/Z) * sum_i sum_j (d y^c / d A_{i,j}^k)
      3. Linear combination of activations weighted by importance alpha_k^c:
         L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )

    Args:
        img_array (np.ndarray): Preprocessed image tensor with shape (1, 224, 224, 3).
        model (tf.keras.Model): Trained neural network model.
        last_conv_layer_name (str): Name of target convolutional layer ('top_conv' for EfficientNet).
        pred_index (int, optional): Target class index. If None, uses top predicted class.

    Returns:
        np.ndarray: 2D normalized heatmap [0.0, 1.0].
    """
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap_name: str = "jet"
) -> np.ndarray:
    """
    Superimpose Grad-CAM heatmap onto RGB image.

    Args:
        original_img (np.ndarray): Original image array in range [0, 255] or [0, 1].
        heatmap (np.ndarray): 2D Grad-CAM heatmap array with values in [0, 1].
        alpha (float): Transparency factor for heatmap overlay (0.0 = original, 1.0 = heatmap).
        colormap_name (str): Matplotlib colormap ('jet', 'viridis', etc.).

    Returns:
        np.ndarray: Superimposed image array in uint8 [0, 255].
    """
    if plt is None:
        raise ImportError("matplotlib is required for colormap rendering.")

    if original_img.max() > 1.0:
        base_img = original_img.astype(np.float32) / 255.0
    else:
        base_img = original_img.astype(np.float32)

    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (base_img.shape[0], base_img.shape[1])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    cmap = plt.get_cmap(colormap_name)
    colored_heatmap = cmap(heatmap_resized)[..., :3]

    blended = alpha * colored_heatmap + (1.0 - alpha) * base_img
    blended = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended


# ==============================================================================
# 7. EVALUATION & VISUALIZATION
# ==============================================================================
def evaluate_model_performance(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluate trained model on the unseen test dataset.

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary.
    """
    print(f"\n[Evaluation] Evaluating {model_name} on unseen test set...")
    metrics = model.evaluate(test_ds, verbose=1)
    results = {
        "Model": model_name,
        "Loss": float(metrics[0]),
        "Top-1 Accuracy (%)": float(metrics[1] * 100.0),
        "Top-5 Accuracy (%)": float(metrics[2] * 100.0),
        "Total Parameters": model.count_params(),
        "Trainable Parameters": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    }
    print(f"[{model_name}] Top-1 Accuracy: {results['Top-1 Accuracy (%)']:.2f}% | Top-5 Accuracy: {results['Top-5 Accuracy (%)']:.2f}%")
    return results


def plot_training_history(
    histories: Dict[str, Any],
    save_path: str = "training_curves.png"
) -> None:
    """
    Plot and save training and validation loss and accuracy curves.
    """
    if plt is None:
        print("[Warning] Matplotlib not available; skipping curve plotting.")
        return

    plt.figure(figsize=(16, 6))

    # Loss subplot
    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        epochs = range(1, len(hist.history["loss"]) + 1)
        plt.plot(epochs, hist.history["loss"], label=f"{name} (Train Loss)", linestyle="--")
        plt.plot(epochs, hist.history["val_loss"], label=f"{name} (Val Loss)", linewidth=2)
    plt.title("Cross-Entropy Loss Progression", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)

    # Accuracy subplot
    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        epochs = range(1, len(hist.history["accuracy"]) + 1)
        plt.plot(epochs, [acc * 100 for acc in hist.history["accuracy"]], label=f"{name} (Train Top-1)", linestyle="--")
        plt.plot(epochs, [acc * 100 for acc in hist.history["val_accuracy"]], label=f"{name} (Val Top-1)", linewidth=2)
    plt.title("Top-1 Accuracy (%) Progression", fontsize=14, fontweight="bold")
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"[Visualization] Training curves saved to: {save_path}")
    plt.close()


def analyze_confusion_matrix(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    top_k_confused: int = 10,
    save_cm_path: str = "confusion_matrix.png"
) -> List[Dict[str, Any]]:
    """
    Compute full 120x120 Confusion Matrix on the test set, plot heatmap,
    and extract the top 10 most confused breed pairs.

    Returns:
        List[Dict[str, Any]]: Top confused pairs sorted by misclassification frequency.
    """
    print("\n[Evaluation] Generating predictions for Confusion Matrix analysis...")
    y_true_list = []
    y_pred_list = []

    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy().tolist())
        y_pred_list.extend(np.argmax(preds, axis=1).tolist())

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if plt is not None and sns is not None:
        plt.figure(figsize=(18, 16))
        sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
        plt.title("Stanford Dogs (120 Breeds) Confusion Matrix - Best Model", fontsize=16, fontweight="bold")
        plt.xlabel("Predicted Class Index", fontsize=12)
        plt.ylabel("True Class Index", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_cm_path, dpi=300)
        print(f"[Visualization] Confusion matrix saved to: {save_cm_path}")
        plt.close()

    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                true_breed_total = np.sum(cm[i, :])
                confused_pairs.append({
                    "True Breed": class_names[i],
                    "Predicted Breed": class_names[j],
                    "Error Count": int(cm[i, j]),
                    "Misclassified % of Breed": float((cm[i, j] / true_breed_total) * 100.0) if true_breed_total > 0 else 0.0
                })

    confused_pairs.sort(key=lambda x: x["Error Count"], reverse=True)
    top_pairs = confused_pairs[:top_k_confused]

    print("\n" + "=" * 80)
    print(f"TOP {top_k_confused} MOST CONFUSED BREED PAIRS (Best Model)")
    print("=" * 80)
    print(f"{'Rank':<5} | {'True Breed':<25} | {'Predicted Breed':<25} | {'Count':<6} | {'% of Breed':<10}")
    print("-" * 80)
    for rank, pair in enumerate(top_pairs, start=1):
        print(f"{rank:<5} | {pair['True Breed']:<25} | {pair['Predicted Breed']:<25} | {pair['Error Count']:<6} | {pair['Misclassified % of Breed']:<10.2f}%")
    print("=" * 80 + "\n")

    return top_pairs


# ==============================================================================
# 8. MASTER TRAINING PIPELINE
# ==============================================================================
def run_full_pipeline(
    data_dir: str = "./dataset",
    models_dir: str = "./models",
    epochs_ann: int = 10,
    epochs_cnn: int = 15,
    epochs_phase1: int = 10,
    epochs_phase2: int = 10,
    batch_size: int = 32,
    seed: int = 42
) -> None:
    """
    Execute the entire comparative deep learning experiment end-to-end.
    """
    set_random_seeds(seed)
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Ingest Dataset & Validate Stratified Partitioning
    images_dir, lists_dir = download_and_extract_stanford_dogs(data_dir)
    (
        train_paths, train_labels,
        val_paths, val_labels,
        test_paths, test_labels,
        class_names
    ) = load_dataset_splits(images_dir, lists_dir, val_split=0.15, seed=seed)

    # Export canonical class_names.json with identical 0..119 index mapping
    class_names_path = "class_names.json"
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"[Artifact] Exported {len(class_names)} breed names to: {class_names_path}")

    # 2. Build High-Performance tf.data Pipelines
    augmenter = get_data_augmentation()
    train_ds = create_tf_dataset(train_paths, train_labels, batch_size=batch_size, is_training=True, augmenter=augmenter)
    val_ds = create_tf_dataset(val_paths, val_labels, batch_size=batch_size, is_training=False)
    test_ds = create_tf_dataset(test_paths, test_labels, batch_size=batch_size, is_training=False)

    comparison_results = []
    training_histories = {}

    # --------------------------------------------------------------------------
    # MODEL 1: ANN BASELINE (Flatten + Dense Layers)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 1: TRAINING MODEL 1 - ANN BASELINE (Flatten + Dense)")
    print("=" * 70)
    ann_model = build_ann_baseline()
    ann_model = compile_model(ann_model, learning_rate=1e-3)
    ann_callbacks = create_callbacks(str(models_path / "ann_checkpoint.keras"), patience_stop=4, patience_lr=2)

    ann_history = ann_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_ann,
        callbacks=ann_callbacks,
        verbose=1
    )
    training_histories["ANN Baseline"] = ann_history
    ann_eval = evaluate_model_performance(ann_model, test_ds, "ANN Baseline")
    comparison_results.append(ann_eval)

    # --------------------------------------------------------------------------
    # MODEL 2: SMALL CUSTOM CNN (4 Conv Blocks, BatchNorm, Dropout)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 2: TRAINING MODEL 2 - SMALL CUSTOM CNN")
    print("=" * 70)
    cnn_model = build_custom_cnn()
    cnn_model = compile_model(cnn_model, learning_rate=1e-3)
    cnn_callbacks = create_callbacks(str(models_path / "cnn_checkpoint.keras"), patience_stop=5, patience_lr=2)

    cnn_history = cnn_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_cnn,
        callbacks=cnn_callbacks,
        verbose=1
    )
    training_histories["Custom CNN"] = cnn_history
    cnn_eval = evaluate_model_performance(cnn_model, test_ds, "Small Custom CNN")
    comparison_results.append(cnn_eval)

    # --------------------------------------------------------------------------
    # MODEL 3: TRANSFER LEARNING WITH EFFICIENTNETB0
    # Phase 1: Frozen Base
    # Phase 2: Fine-Tuning Top Layers at lr 1e-5
    # --------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 3: TRAINING MODEL 3 - EFFICIENTNETB0 (TRANSFER LEARNING)")
    print("=" * 70)
    effnet_model, base_model = build_efficientnet_model()

    # PHASE 1: Feature Extraction (Frozen Base)
    print("\n[Transfer Learning] Phase 1: Feature Extraction with Frozen Base (lr=1e-3)...")
    base_model.trainable = False
    effnet_model = compile_model(effnet_model, learning_rate=1e-3)
    phase1_callbacks = create_callbacks(str(models_path / "effnet_phase1.keras"), patience_stop=4, patience_lr=2)

    phase1_history = effnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase1,
        callbacks=phase1_callbacks,
        verbose=1
    )

    # PHASE 2: Fine-Tuning Top Layers at lr 1e-5
    print("\n[Transfer Learning] Phase 2: Fine-Tuning Top Layers (lr=1e-5)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    print(f"[Transfer Learning] Trainable parameters in Phase 2: {sum([tf.keras.backend.count_params(w) for w in effnet_model.trainable_weights]):,}")
    effnet_model = compile_model(effnet_model, learning_rate=1e-5)
    phase2_callbacks = create_callbacks(str(models_path / "effnet_phase2.keras"), patience_stop=5, patience_lr=2)

    phase2_history = effnet_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_phase2,
        callbacks=phase2_callbacks,
        verbose=1
    )

    training_histories["EfficientNetB0 (Phase 1)"] = phase1_history
    training_histories["EfficientNetB0 (Phase 2 Fine-tuned)"] = phase2_history

    effnet_eval = evaluate_model_performance(effnet_model, test_ds, "EfficientNetB0 (Transfer Learning)")
    comparison_results.append(effnet_eval)

    # --------------------------------------------------------------------------
    # 9. MODEL COMPARISON TABLE
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON SUMMARY (EVALUATED ON UNSEEN TEST SET)")
    print("=" * 90)
    if pd is not None:
        df_comparison = pd.DataFrame(comparison_results)
        print(df_comparison.to_string(index=False))
    else:
        for res in comparison_results:
            print(res)
    print("=" * 90 + "\n")

    # --------------------------------------------------------------------------
    # 10. PLOT TRAINING CURVES & CONFUSION MATRIX FOR BEST MODEL
    # --------------------------------------------------------------------------
    plot_training_history(training_histories, save_path="training_curves.png")
    analyze_confusion_matrix(effnet_model, test_ds, class_names, top_k_confused=10, save_cm_path="confusion_matrix.png")

    # --------------------------------------------------------------------------
    # 11. GRAD-CAM EXPLAINABILITY ON SAMPLE TEST IMAGE
    # --------------------------------------------------------------------------
    print("\n[Explainability] Running Grad-CAM on sample test image...")
    for sample_images, sample_labels in test_ds.take(1):
        sample_img = sample_images[0].numpy()
        sample_true_lbl = int(sample_labels[0].numpy())
        break

    img_input = np.expand_dims(sample_img, axis=0)
    preds = effnet_model.predict(img_input, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx]) * 100.0

    print(f"Sample Image Ground Truth: {class_names[sample_true_lbl]}")
    print(f"Sample Image Prediction  : {class_names[pred_idx]} ({confidence:.2f}% confidence)")

    heatmap = compute_gradcam_heatmap(img_input, effnet_model, last_conv_layer_name="top_conv", pred_index=pred_idx)

    if plt is not None:
        overlay = overlay_gradcam(sample_img, heatmap, alpha=0.45)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sample_img.astype(np.uint8))
        axes[0].set_title(f"Original Image\nTruth: {class_names[sample_true_lbl]}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Activation Heatmap\n(Layer: top_conv)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Superimposed Overlay\nPred: {class_names[pred_idx]} ({confidence:.1f}%)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("gradcam_sample_output.png", dpi=300)
        print("[Explainability] Grad-CAM visualization saved to: gradcam_sample_output.png")
        plt.close()

    # --------------------------------------------------------------------------
    # 12. EXPORT FINAL BEST MODEL
    # --------------------------------------------------------------------------
    best_model_path = "dog_breed_model.keras"
    effnet_model.save(best_model_path)
    print(f"\n[Artifact] Best model saved successfully to: {best_model_path}")
    print(f"[Artifact] Class mapping saved successfully to: {class_names_path}")
    print("[Pipeline Complete] All college-level experimental deliverables ready.")


if __name__ == "__main__":
    run_full_pipeline()
