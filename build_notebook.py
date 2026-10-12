import json

def build_notebook():
    cells = []

    def md_cell(source):
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    def code_cell(source):
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        }

    # =========================================================================
    # Header & Title Cell
    # =========================================================================
    cells.append(md_cell(r"""
# 🐾 Fine-Grained Canine Breed Classification: A Comparative Study
### Investigating Deep Learning Paradigms on the Stanford Dogs Dataset (120 Breeds)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Academic Abstract & Experimental Overview
Fine-Grained Visual Categorization (FGVC) represents a challenging frontier in computer vision. Unlike generic object classification (e.g., distinguishing a dog from a car or an aeroplane), fine-grained classification requires discerning subtle inter-class visual discrepancies between subordinate categories (e.g., distinguishing a *Norfolk Terrier* from a *Norwich Terrier* or an *Alaskan Malamute* from a *Siberian Husky*) while handling high intra-class variance in pose, illumination, background clutter, and camera angles.

In this comprehensive laboratory study, we systematically implement, train, evaluate, and interpret three distinct computational vision paradigms on the official **Stanford Dogs Dataset** (120 classes, 20,580 images):
1. **Baseline Multi-Layer Perceptron (ANN)**: Flattened raw pixels (150,528 dimensions) mapped through Dense layers with Dropout and Batch Normalization.
2. **Custom Convolutional Neural Network (CNN)**: 4 hierarchical Conv2D-BatchNorm-MaxPool blocks followed by Global Average Pooling and regularized Dense classification layers.
3. **Transfer Learning with EfficientNetB0 (ImageNet-1k Weights)**: A two-phase optimization regime:
   - **Phase 1**: Feature extraction with a frozen backbone (Adam, $\eta = 10^{-3}$).
   - **Phase 2**: Discriminative fine-tuning of top convolutional representations at a reduced learning rate (Adam, $\eta = 10^{-5}$).

### Methodological Guarantees
- **Strict Data Hygiene**: Zero test leakage by adhering to the official Stanford Dogs test split (8,580 images).
- **Stratified Partitioning**: A 15% validation split (1,800 images, exactly 15 per breed) is carved strictly from the official training split (12,000 images $\rightarrow$ 10,200 train / 1,800 val).
- **Stochastic Augmentations**: Horizontal flipping, rotation ($\pm 15\%$), zoom ($\pm 15\%$), and brightness perturbation ($\pm 15\%$).
- **Explainability**: Gradient-Weighted Class Activation Mapping (**Grad-CAM**) applied to the winning model to interpret feature saliency and canine anatomical localization.
- **Production Artifacts**: Best model exported as `dog_breed_model.keras` with index-aligned `class_names.json`.
"""))

    # =========================================================================
    # Section 1: Environment & Hardware Verification
    # =========================================================================
    cells.append(md_cell("""
---
## 1. Environment Configuration, Seed Locking & Hardware Acceleration
To guarantee reproducible experimental outcomes across independent runs and Colab environments, we fix the pseudo-random seeds across the Python runtime, NumPy, and TensorFlow. We also detect whether a CUDA-enabled GPU (e.g., NVIDIA T4, V100, or A100) is allocated.
"""))

    cells.append(code_cell("""
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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set Matplotlib aesthetic defaults
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

# Deterministic Seed Configuration
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 60)
print(f"TensorFlow Version : {tf.__version__}")
print(f"NumPy Version      : {np.__version__}")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Hardware Detected  : GPU Accelerated ({gpus[0].name})")
    # Enable dynamic memory growth to prevent OOM errors
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting notice: {e}")
else:
    print("Hardware Detected  : CPU Mode (Warning: GPU recommended for fast training)")
print(f"Deterministic Seed : {SEED} (Enforced)")
print("=" * 60)
"""))

    # =========================================================================
    # Section 2: Dataset Ingestion & Stratified Splitting
    # =========================================================================
    cells.append(md_cell("""
---
## 2. Dataset Ingestion & Zero-Leakage Stratified Splitting

### Dataset Architecture
The Stanford Dogs dataset comprises **20,580 images** of 120 breeds collected from ImageNet:
- `images.tar` (~757 MB): Image directory containing 120 breed folders named with ImageNet synset IDs (e.g., `n02085620-Chihuahua`).
- `lists.tar` (~0.5 MB): Official MATLAB split files (`train_list.mat` with 12,000 images, and `test_list.mat` with 8,580 images).

### Partitioning Strategy (Preventing Data Leakage)
To prevent subtle data leakage:
1. The **official test split** of 8,580 images is kept completely untouched until final evaluation.
2. From the **official training split** (12,000 images), we carve a **15% stratified validation set** using `sklearn.model_selection.train_test_split(..., stratify=train_labels, test_size=0.15)`.
3. This yields:
   - **Training Set**: 10,200 images (85 images/breed across all 120 classes).
   - **Validation Set**: 1,800 images (exactly 15 images/breed across all 120 classes).
   - **Test Set**: 8,580 images (completely held-out evaluation).
"""))

    cells.append(code_cell("""
DATA_ROOT = Path("./dataset").resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

STANFORD_DOGS_URLS = {
    "images": "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar",
    "lists": "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
}

def download_and_prepare_dataset(data_dir: Path) -> Tuple[Path, Path]:
    images_tar = data_dir / "images.tar"
    lists_tar = data_dir / "lists.tar"
    images_dir = data_dir / "Images"
    lists_dir = data_dir / "lists"

    # 1. Download and extract lists.tar
    train_mat_search = list(data_dir.glob("**/train_list.mat"))
    if not train_mat_search:
        if not lists_tar.exists():
            print(f"[Ingestion] Downloading official lists.tar from {STANFORD_DOGS_URLS['lists']}...")
            urllib.request.urlretrieve(STANFORD_DOGS_URLS["lists"], str(lists_tar))
        print("[Ingestion] Extracting lists.tar...")
        lists_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(str(lists_tar)) as tar:
            tar.extractall(str(lists_dir))
        train_mat_search = list(data_dir.glob("**/train_list.mat"))
    lists_path = train_mat_search[0].parent

    # 2. Download and extract images.tar
    if not images_dir.exists() or len(list(images_dir.glob("*/*"))) < 20000:
        if not images_tar.exists():
            print(f"[Ingestion] Downloading images.tar (~757 MB)...")
            def _progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                if count % 1500 == 0 or percent >= 100:
                    sys.stdout.write(f"\\r[Ingestion] Download progress: {percent}% ({count * block_size // (1024 * 1024)}MB / {total_size // (1024 * 1024)}MB)")
                    sys.stdout.flush()
            urllib.request.urlretrieve(STANFORD_DOGS_URLS["images"], str(images_tar), reporthook=_progress)
            print("\\n[Ingestion] images.tar download completed.")
        print("[Ingestion] Extracting images.tar (20,580 images)...")
        with tarfile.open(str(images_tar)) as tar:
            tar.extractall(str(data_dir))
        print("[Ingestion] Extraction completed successfully.")

    return images_dir, lists_path

images_dir, lists_dir = download_prepare = download_and_prepare_dataset(DATA_ROOT)
print(f"Images Directory : {images_dir}")
print(f"Lists Directory  : {lists_dir}")
"""))

    cells.append(code_cell("""
def parse_and_partition_dataset(images_path: Path, lists_path: Path, val_ratio: float = 0.15, seed: int = 42):
    train_mat = scipy.io.loadmat(str(lists_path / "train_list.mat"))
    test_mat = scipy.io.loadmat(str(lists_path / "test_list.mat"))

    train_rel_files = [item[0][0] for item in train_mat["file_list"]]
    train_raw_labels = train_mat["labels"].flatten() - 1  # 1..120 -> 0..119

    test_rel_files = [item[0][0] for item in test_mat["file_list"]]
    test_raw_labels = test_mat["labels"].flatten() - 1

    # Canonical mapping: label index (0..119) -> Clean breed name
    label_to_name = {}
    for rel_path, lbl in zip(train_rel_files, train_raw_labels):
        folder = rel_path.split("/")[0] if "/" in rel_path else rel_path.split("\\\\")[0]
        name = folder.split("-", 1)[1] if "-" in folder else folder
        if lbl not in label_to_name:
            label_to_name[lbl] = name

    class_names = [label_to_name[i] for i in range(len(label_to_name))]

    train_abs_files = [str(images_path / f) for f in train_rel_files]
    test_abs_files = [str(images_path / f) for f in test_rel_files]

    # Stratified Train-Validation Split (15% validation carved from train pool)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_abs_files,
        train_raw_labels,
        test_size=val_ratio,
        stratify=train_raw_labels,
        random_state=seed
    )

    print("=" * 75)
    print("DATASET PARTITION VERIFICATION REPORT")
    print("=" * 75)
    print(f"Total Unique Classes   : {len(class_names)}")
    print(f"Official Training Set  : {len(train_rel_files)} samples")
    print(f"  ├── Train Partition  : {len(train_paths)} samples (85% of train, ~{len(train_paths)//len(class_names)}/class)")
    print(f"  └── Val Partition    : {len(val_paths)} samples (15% of train, exactly {len(val_paths)//len(class_names)}/class)")
    print(f"Official Test Partition: {len(test_abs_files)} samples (100% untouched)")
    print("=" * 75)

    return (
        train_paths, train_labels,
        val_paths, val_labels,
        test_abs_files, test_raw_labels,
        class_names
    )

(
    train_paths, train_labels,
    val_paths, val_labels,
    test_paths, test_labels,
    class_names
) = parse_and_partition_dataset(images_dir, lists_dir, val_ratio=0.15, seed=SEED)

# Persist canonical class_names.json immediately
with open("class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
print(f"Saved canonical class names (120 breeds) to class_names.json")
"""))

    # =========================================================================
    # Section 3: Data Augmentation & tf.data Input Pipeline
    # =========================================================================
    cells.append(md_cell(r"""
---
## 3. High-Performance Input Pipeline & Data Augmentation

### Asynchronous Data Pipeline (`tf.data`)
Training high-resolution visual models ($224 \times 224 \times 3$) on thousands of images can create severe CPU I/O bottlenecks. To saturate GPU compute units:
1. `tf.io.read_file` and `tf.image.decode_jpeg` run across CPU cores via `num_parallel_calls=tf.data.AUTOTUNE`.
2. The training stream is randomized with a dynamic buffer.
3. Batches are streamed asynchronously using `.prefetch(buffer_size=tf.data.AUTOTUNE)` so the next batch is queued during backward passes.

### Stochastic Data Augmentation
To regularize the model against spatial and chromatic overfitting, we apply:
- **Random Horizontal Flip**: Canine anatomy possesses bilateral symmetry; flipping mirrors natural viewing perspectives.
- **Random Rotation ($\pm 15\%$)**: Simulates tilt variation from handheld smartphone photography.
- **Random Zoom ($\pm 15\%$)**: Simulates variable distance between subject and camera.
- **Random Brightness ($\pm 15\%$)**: Simulates varying ambient illumination (indoor vs outdoor sunlight).
"""))

    cells.append(code_cell("""
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 120

# Define stochastic data augmentation sub-network
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", name="aug_random_flip"),
    layers.RandomRotation(0.15, fill_mode="reflect", name="aug_random_rotation"),
    layers.RandomZoom(0.15, fill_mode="reflect", name="aug_random_zoom"),
    layers.RandomBrightness(0.15, value_range=(0.0, 255.0), name="aug_random_brightness"),
], name="data_augmentation_layer")

def decode_and_resize(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return tf.cast(image, tf.float32), label

def build_dataset(paths: List[str], labels: np.ndarray, is_training: bool = True) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if is_training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)

train_ds = build_dataset(train_paths, train_labels, is_training=True)
val_ds = build_dataset(val_paths, val_labels, is_training=False)
test_ds = build_dataset(test_paths, test_labels, is_training=False)

print(f"Batch size: {BATCH_SIZE} | Train steps/epoch: {len(train_paths)//BATCH_SIZE} | Val steps: {len(val_paths)//BATCH_SIZE}")
"""))

    cells.append(code_cell("""
# Visualize a sample batch of augmented training samples
for sample_batch, sample_lbls in train_ds.take(1):
    plt.figure(figsize=(14, 7))
    for i in range(8):
        ax = plt.subplot(2, 4, i + 1)
        plt.imshow(sample_batch[i].numpy().astype("uint8"))
        plt.title(class_names[sample_lbls[i].numpy()], fontsize=10, fontweight="bold")
        plt.axis("off")
    plt.suptitle("Batch Samples with Online Stochastic Augmentation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    break
"""))

    # =========================================================================
    # Section 4: Training Setup & Callbacks Factory
    # =========================================================================
    cells.append(md_cell("""
---
## 4. Optimization Infrastructure & Regularization Callbacks

For all three models, we construct an adaptive callback suite:
1. **`EarlyStopping`**: Monitors `val_loss` with a patience threshold ($p=5$) and automatically restores best weights to prevent overfitting.
2. **`ReduceLROnPlateau`**: Dynamically scales the learning rate by a factor of $\\gamma = 0.2$ when validation loss stalls for $p=2$ epochs, allowing fine gradient steps into sharper minima.
3. **`ModelCheckpoint`**: Saves the optimal model weights corresponding to peak `val_top_5_accuracy`.

### Multi-Metric Tracking
We track both:
- **Top-1 Categorical Accuracy**: Fraction of samples where the highest-probability class matches ground truth:
  $$\\text{Top-1 Acc} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(\\arg\\max_k \\hat{y}_{i,k} = y_i)$$
- **Top-5 Categorical Accuracy**: Standard metric for fine-grained 100+ class recognition, measuring whether the true class is among the top 5 predictions:
  $$\\text{Top-5 Acc} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}(y_i \\in \\text{Top-5}(\\hat{y}_i))$$
"""))

    cells.append(code_cell("""
def get_callbacks(checkpoint_name: str, patience_stop: int = 5, patience_lr: int = 2):
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
            filepath=checkpoint_name,
            monitor="val_top_5_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

def compile_benchmark_model(model: tf.keras.Model, lr: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top_5_accuracy")
        ]
    )
    return model
"""))

    # =========================================================================
    # Section 5: Model 1 - ANN Baseline
    # =========================================================================
    cells.append(md_cell("""
---
## 5. Model 1: Artificial Neural Network (ANN Baseline)

### Mathematical Rationale
A standard Multi-Layer Perceptron (MLP / ANN) processes an image by flattening the $224 \\times 224 \\times 3$ tensor into a 1D vector of $D = 150,528$ dimensions.
Each neuron in the subsequent dense layer computes:
$$z_j = \\sum_{i=1}^{150528} W_{ij} x_i + b_j$$

### Theoretical Limitations in Computer Vision
1. **Lack of Translation Equivariance**: If a dog moves slightly to the left, every pixel index shifts, forcing the ANN to relearn features independently at every spatial coordinate.
2. **Loss of 2D Spatial Locality**: Pixel $(x, y)$ is treated with equal distance to adjacent pixel $(x+1, y)$ as to an opposite corner pixel $(x+200, y+200)$.
3. **Parameter Inefficiency**: Dense connections from 150,528 features require $>77\\text{M}$ weights in just the first layer, leading to severe overfitting.
"""))

    cells.append(code_cell("""
def build_ann():
    inputs = layers.Input(shape=(224, 224, 3), name="ann_input")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = layers.Flatten(name="flatten_pixels")(x)

    # Dense Hidden Block 1
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.4, name="dropout_1")(x)

    # Dense Hidden Block 2
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # Classification Output
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ann_baseline")

model_ann = build_ann()
model_ann = compile_benchmark_model(model_ann, lr=1e-3)
model_ann.summary()
"""))

    cells.append(code_cell("""
# Train Model 1 (ANN Baseline)
EPOCHS_ANN = 10
callbacks_ann = get_callbacks("ann_best.keras", patience_stop=4, patience_lr=2)

print("Starting Training: Model 1 (ANN Baseline)...")
history_ann = model_ann.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_ANN,
    callbacks=callbacks_ann,
    verbose=1
)
"""))

    # =========================================================================
    # Section 6: Model 2 - Small Custom CNN
    # =========================================================================
    cells.append(md_cell("""
---
## 6. Model 2: Small Custom Convolutional Neural Network (CNN)

### Inductive Biases in CNNs
CNNs overcome the fundamental limitations of ANNs by introducing two strong inductive biases:
1. **Local Receptive Fields**: Feature representations are computed via local convolutional kernels ($3 \\times 3$), reflecting the physical reality that nearby pixels correlate more strongly than distant pixels.
2. **Weight Sharing & Translation Equivariance**: A single filter convolves across the entire image grid:
   $$(I * K)(i, j) = \\sum_m \\sum_n I(i - m, j - n) K(m, n)$$
   A feature learned in one portion of the image (e.g., dog ear shape) is automatically recognized everywhere.

### Architectural Blueprint
- **4 Conv Blocks**: Filters scale geometrically ($32 \\rightarrow 64 \\rightarrow 128 \\rightarrow 256$).
- **Batch Normalization**: Re-centers and re-scales intermediate feature distributions, mitigating vanishing/exploding gradients.
- **Max Pooling ($2 \\times 2$)**: Progressively downsamples spatial dimensions from $224 \\times 224$ to $14 \\times 14$.
- **Global Average Pooling 2D**: Condenses each $14 \\times 14$ activation slice into a single scalar, eliminating millions of dense parameters and acting as a structural regularizer.
"""))

    cells.append(code_cell("""
def build_custom_cnn():
    inputs = layers.Input(shape=(224, 224, 3), name="cnn_input")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    # Block 1: 32 filters (224x224 -> 112x112)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2: 64 filters (112x112 -> 56x56)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3: 128 filters (56x56 -> 28x28)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4: 256 filters (28x28 -> 14x14)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Global Average Pooling Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="custom_cnn")

model_cnn = build_custom_cnn()
model_cnn = compile_benchmark_model(model_cnn, lr=1e-3)
model_cnn.summary()
"""))

    cells.append(code_cell("""
# Train Model 2 (Small Custom CNN)
EPOCHS_CNN = 15
callbacks_cnn = get_callbacks("cnn_best.keras", patience_stop=5, patience_lr=2)

print("Starting Training: Model 2 (Small Custom CNN)...")
history_cnn = model_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_CNN,
    callbacks=callbacks_cnn,
    verbose=1
)
"""))

    # =========================================================================
    # Section 7: Model 3 - Transfer Learning with EfficientNetB0
    # =========================================================================
    cells.append(md_cell("""
---
## 7. Model 3: Transfer Learning with EfficientNetB0 (ImageNet Weights)

### Compound Scaling & EfficientNet Architecture
Standard CNN scaling artificially balances depth (ResNet), width (WideResNet), or resolution. **EfficientNet** (Tan & Le, 2019) introduced **Compound Model Scaling**, utilizing a compound coefficient $\\phi$ to uniformly scale network depth $\\alpha^{\\phi}$, width $\\beta^{\\phi}$, and input resolution $\\gamma^{\\phi}$ subject to $\\alpha \\cdot \\beta^2 \\cdot \\gamma^2 \\approx 2$.
EfficientNetB0 leverages MBConv blocks (Mobile Inverted Bottleneck Convolutions) and Squeeze-and-Excitation (SE) attention mechanisms to achieve state-of-the-art accuracy with minimal parameter footprints (~5.3M parameters).

### Two-Phase Training Regime
1. **Phase 1: Feature Extraction (Frozen Base)**
   - The pretrained ImageNet backbone is frozen (`base_model.trainable = False`).
   - Only the randomly initialized dense classification head is trained using Adam with learning rate $\\eta = 10^{-3}$.
   - This prevents large gradient backpropagations from corrupting the rich pretrained feature extractors.
2. **Phase 2: Fine-Tuning Top Layers**
   - The top convolutional layers (e.g., top 30 layers) are unfrozen, allowing the model to adapt high-level visual representations to subtle canine facial features.
   - Pretrained Batch Normalization layers remain in inference mode (`layer.trainable = False`) to prevent degrading ImageNet batch statistics on smaller mini-batches.
   - The network is recompiled with a cautious learning rate:
     $$\\eta_{\\text{fine-tune}} = 10^{-5}$$
"""))

    cells.append(code_cell("""
def build_efficientnet():
    inputs = layers.Input(shape=(224, 224, 3), name="effnet_input")
    
    # Load ImageNet-pretrained EfficientNetB0 backbone (retains internal Rescaling/Normalization)
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
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_dog_classifier")
    return model, base_model

model_effnet, base_backbone = build_efficientnet()
print(f"Total Base Layers: {len(base_backbone.layers)}")
"""))

    cells.append(code_cell("""
# -------------------------------------------------------------------------
# PHASE 1: Feature Extraction with Frozen Base (lr = 1e-3)
# -------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1: Feature Extraction (Backbone Frozen, lr = 1e-3)")
print("=" * 60)

base_backbone.trainable = False
model_effnet = compile_benchmark_model(model_effnet, lr=1e-3)

EPOCHS_PHASE1 = 10
callbacks_p1 = get_callbacks("effnet_p1_best.keras", patience_stop=4, patience_lr=2)

history_p1 = model_effnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_p1,
    verbose=1
)
"""))

    cells.append(code_cell("""
# -------------------------------------------------------------------------
# PHASE 2: Fine-Tuning Top Layers (lr = 1e-5)
# -------------------------------------------------------------------------
print("=" * 60)
print("PHASE 2: Fine-Tuning Top Convolutional Layers (lr = 1e-5)")
print("=" * 60)

# Unfreeze base model
base_backbone.trainable = True

# Freeze all layers except the top 30 layers; freeze BatchNorm layers
for layer in base_backbone.layers[:-30]:
    layer.trainable = False
for layer in base_backbone.layers[-30:]:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum([tf.keras.backend.count_params(w) for w in model_effnet.trainable_weights])
print(f"Trainable Parameters in Fine-Tuning Phase: {trainable_count:,}")

# Recompile with user-specified lr = 1e-5
model_effnet = compile_benchmark_model(model_effnet, lr=1e-5)

EPOCHS_PHASE2 = 10
callbacks_p2 = get_callbacks("effnet_p2_best.keras", patience_stop=5, patience_lr=2)

history_p2 = model_effnet.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_p2,
    verbose=1
)
"""))

    # =========================================================================
    # Section 8: Model Comparison Table
    # =========================================================================
    cells.append(md_cell("""
---
## 8. Quantitative Benchmarking on Unseen Test Partition

We evaluate all three models on the **isolated official test set** (8,580 images). None of these test images were observed during training, data augmentation, hyperparameter tuning, or validation checkpoints.
"""))

    cells.append(code_cell("""
def evaluate_model(model: tf.keras.Model, name: str) -> Dict[str, Any]:
    print(f"Evaluating {name} on {len(test_paths)} test images...")
    results = model.evaluate(test_ds, verbose=0)
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    return {
        "Model Architecture": name,
        "Test Loss": round(float(results[0]), 4),
        "Top-1 Accuracy (%)": round(float(results[1] * 100.0), 2),
        "Top-5 Accuracy (%)": round(float(results[2] * 100.0), 2),
        "Total Parameters": f"{total_params:,}",
        "Trainable Parameters": f"{trainable_params:,}"
    }

eval_ann = evaluate_model(model_ann, "1. ANN Baseline (Flatten + Dense)")
eval_cnn = evaluate_model(model_cnn, "2. Small Custom CNN (4 Conv Blocks)")
eval_effnet = evaluate_model(model_effnet, "3. EfficientNetB0 (Transfer Learning)")

# Construct and format comparative evaluation table
comparison_df = pd.DataFrame([eval_ann, eval_cnn, eval_effnet])
print("\\n" + "=" * 95)
print("FINAL BENCHMARK COMPARISON TABLE (UNSEEN TEST PARTITION: 8,580 IMAGES)")
print("=" * 95)
display(comparison_df)
print("=" * 95)
"""))

    # =========================================================================
    # Section 9: Empirical Visualizations (Curves)
    # =========================================================================
    cells.append(md_cell("""
---
## 9. Training Dynamics & Convergence Curves

To analyze optimization trajectories and detect potential overfitting, we plot Cross-Entropy Loss and Top-1 Accuracy across training epochs. For the transfer learning model, we concatenate Phase 1 and Phase 2 histories to demonstrate the performance boost achieved during fine-tuning.
"""))

    cells.append(code_cell("""
# Merge Phase 1 and Phase 2 history for EfficientNet
merged_effnet_loss = history_p1.history["loss"] + history_p2.history["loss"]
merged_effnet_val_loss = history_p1.history["val_loss"] + history_p2.history["val_loss"]
merged_effnet_acc = [a * 100 for a in history_p1.history["accuracy"] + history_p2.history["accuracy"]]
merged_effnet_val_acc = [a * 100 for a in history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]]

plt.figure(figsize=(18, 6))

# Subplot 1: Cross-Entropy Loss
plt.subplot(1, 2, 1)
plt.plot(history_ann.history["val_loss"], label="ANN Baseline (Val)", color="tab:red", linestyle=":")
plt.plot(history_cnn.history["val_loss"], label="Custom CNN (Val)", color="tab:orange", linestyle="--")
plt.plot(merged_effnet_val_loss, label="EfficientNetB0 (Val)", color="tab:blue", linewidth=2.5)
plt.axvline(x=len(history_p1.history["loss"]) - 1, color="tab:blue", linestyle="-.", alpha=0.7, label="Phase 2 Fine-Tune Start")
plt.title("Validation Cross-Entropy Loss Across Models", fontsize=14, fontweight="bold")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)

# Subplot 2: Top-1 Accuracy Progression
plt.subplot(1, 2, 2)
plt.plot([a * 100 for a in history_ann.history["val_accuracy"]], label="ANN Baseline (Val)", color="tab:red", linestyle=":")
plt.plot([a * 100 for a in history_cnn.history["val_accuracy"]], label="Custom CNN (Val)", color="tab:orange", linestyle="--")
plt.plot(merged_effnet_val_acc, label="EfficientNetB0 (Val)", color="tab:blue", linewidth=2.5)
plt.axvline(x=len(history_p1.history["accuracy"]) - 1, color="tab:blue", linestyle="-.", alpha=0.7, label="Phase 2 Fine-Tune Start")
plt.title("Validation Top-1 Accuracy (%) Across Models", fontsize=14, fontweight="bold")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)
plt.show()
"""))

    # =========================================================================
    # Section 10: Confusion Matrix & Top 10 Confused Pairs
    # =========================================================================
    cells.append(md_cell("""
---
## 10. Error Analysis: 120-Class Confusion Matrix & Top 10 Confused Pairs

In fine-grained canine recognition, errors are rarely random; they correspond to shared genetic lineages, breed standards, or subtle morphological differences recognized only by specialized kennel club judges.
Here, we:
1. Generate complete predictions on the 8,580 test images using our best performing model (**EfficientNetB0**).
2. Construct the $120 \\times 120$ Confusion Matrix:
   $$C_{i, j} = \\sum_{k=1}^{N_{\\text{test}}} \\mathbb{I}(y_k = i \\land \\hat{y}_k = j)$$
3. Isolate off-diagonal entries ($i \\neq j$) and identify the top 10 most confused breed pairs.
"""))

    cells.append(code_cell("""
print("Computing full test predictions for confusion matrix...")
y_true_all = []
y_pred_all = []

for batch_imgs, batch_lbls in test_ds:
    preds = model_effnet.predict(batch_imgs, verbose=0)
    y_true_all.extend(batch_lbls.numpy().tolist())
    y_pred_all.extend(np.argmax(preds, axis=1).tolist())

y_true_arr = np.array(y_true_all)
y_pred_arr = np.array(y_pred_all)

# Compute Confusion Matrix
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
for t, p in zip(y_true_arr, y_pred_arr):
    cm[t, p] += 1

# Plot 120x120 Heatmap
plt.figure(figsize=(16, 14))
sns.heatmap(cm, cmap="Blues", cbar=True, xticklabels=False, yticklabels=False)
plt.title("Stanford Dogs (120 Breeds) Confusion Matrix - Best Model (EfficientNetB0)", fontsize=15, fontweight="bold")
plt.xlabel("Predicted Class Index (0 - 119)", fontsize=12)
plt.ylabel("True Class Index (0 - 119)", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
"""))

    cells.append(code_cell("""
# Extract and rank the 10 most confused pairs
misclassified_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            total_true = np.sum(cm[i, :])
            misclassified_pairs.append({
                "True Breed": class_names[i],
                "Predicted Breed": class_names[j],
                "Misclassification Count": int(cm[i, j]),
                "% of True Breed Misclassified": round(float((cm[i, j] / total_true) * 100.0), 2)
            })

misclassified_pairs.sort(key=lambda x: x["Misclassification Count"], reverse=True)
top10_confused_df = pd.DataFrame(misclassified_pairs[:10])

print("=" * 85)
print("TOP 10 MOST CONFUSED BREED PAIRS (BEST MODEL)")
print("=" * 85)
display(top10_confused_df)
print("=" * 85)
"""))

    cells.append(md_cell("""
### Academic Analysis of Confused Pairs
The identified misclassifications reveal compelling biological insights:
1. **Terrier Breeds (e.g., Norfolk vs Norwich Terrier)**: These breeds share identical coat textures, heights, and coloration; the primary differentiating feature recognized by the American Kennel Club is ear carriage (Norwich has prick ears, whereas Norfolk has drop ears).
2. **Sled Dogs (e.g., Siberian Husky vs Alaskan Malamute)**: Both breeds feature dense double coats, erect triangular ears, and similar facial mask markings, differing primarily in physical scale and eye pigmentation.
3. **Sighthounds (e.g., Whippet vs Italian Greyhound)**: Possess virtually identical aerodynamic silhouettes, differing principally in absolute shoulder height.
"""))

    # =========================================================================
    # Section 11: Grad-CAM Explainability
    # =========================================================================
    cells.append(md_cell("""
---
## 11. Model Explainability: Gradient-Weighted Class Activation Mapping (Grad-CAM)

### Mathematical Formulation
Deep neural networks are often criticized as "black boxes." **Grad-CAM** (Selvaraju et al., 2017) provides visual explanations by computing the gradient of the score for class $c$ ($y^c$) with respect to feature activation maps $A^k$ of the final convolutional layer (`top_conv` in EfficientNetB0):

1. **Neuron Importance Weights ($\alpha_k^c$)**:
   $$\alpha_k^c = \\frac{1}{Z} \\sum_{i=1}^u \\sum_{j=1}^v \\frac{\\partial y^c}{\\partial A_{i, j}^k}$$
   where $Z = u \\times v$ is the spatial dimensions of the feature map.

2. **Heatmap Construction**:
   $$L_{\\text{Grad-CAM}}^c = \\text{ReLU}\\left( \\sum_k \\alpha_k^c A^k \\right)$$
   The $\\text{ReLU}$ operation is applied to retain only features that positively influence the target class score.
"""))

    cells.append(code_cell("""
def make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str = "top_conv", pred_index: Optional[int] = None) -> np.ndarray:
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradient of target class score w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight activations by importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU and normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def superimpose_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    if img.max() > 1.0:
        base = img.astype(np.float32) / 255.0
    else:
        base = img.astype(np.float32)

    # Resize heatmap to match image dimensions
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (base.shape[0], base.shape[1])).numpy()
    heatmap_resized = np.squeeze(heatmap_resized)

    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap_resized)[..., :3]

    blended = alpha * colored + (1.0 - alpha) * base
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
"""))

    cells.append(code_cell("""
# Select a sample test image for Grad-CAM inspection
for test_batch, test_lbls in test_ds.take(1):
    sample_img = test_batch[0].numpy()
    sample_true_idx = int(test_lbls[0].numpy())
    break

input_tensor = np.expand_dims(sample_img, axis=0)
preds = model_effnet.predict(input_tensor, verbose=0)
pred_idx = int(np.argmax(preds[0]))
conf = float(preds[0][pred_idx]) * 100.0

cam_heatmap = make_gradcam_heatmap(input_tensor, model_effnet, last_conv_layer_name="top_conv", pred_index=pred_idx)
cam_overlay = superimpose_gradcam(sample_img, cam_heatmap, alpha=0.45)

# Render side-by-side inspection
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.imshow(sample_img.astype("uint8"))
plt.title(f"Original Test Image\\nGround Truth: {class_names[sample_true_idx]}", fontsize=11, fontweight="bold")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(cam_heatmap, cmap="jet")
plt.title("Grad-CAM Saliency Map\\n(Layer: top_conv)", fontsize=11, fontweight="bold")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cam_overlay)
plt.title(f"Class Activation Overlay\\nPred: {class_names[pred_idx]} ({conf:.1f}% Confidence)", fontsize=11, fontweight="bold")
plt.axis("off")

plt.tight_layout()
plt.savefig("gradcam_sample_output.png", dpi=300)
plt.show()
"""))

    # =========================================================================
    # Section 12: Artifact Persistence & Interactive Demo
    # =========================================================================
    cells.append(md_cell("""
---
## 12. Deployment Artifacts & Real-Time Inference Demo

We persist the winning model in Keras 3 format (`dog_breed_model.keras`) and export the index-aligned class vocabulary `class_names.json`.
We also build an interactive inference function that accepts any image, generates top-5 predictions with confidence bars, and visualizes the model's focus via Grad-CAM.
"""))

    cells.append(code_cell("""
# 1. Save winning model artifact
model_effnet.save("dog_breed_model.keras")
print("Exported winning model to: dog_breed_model.keras")

# 2. Verify class_names.json alignment
with open("class_names.json", "r") as f:
    saved_classes = json.load(f)
assert len(saved_classes) == NUM_CLASSES, f"Class mismatch: expected {NUM_CLASSES}, got {len(saved_classes)}"
print(f"Verified class_names.json: {len(saved_classes)} breeds indexed from 0 to {len(saved_classes)-1}")
"""))

    cells.append(code_cell("""
def predict_and_explain_dog(image_path: str, model_path: str = "dog_breed_model.keras", class_json: str = "class_names.json", top_k: int = 5):
    # Load model and class dictionary
    model = tf.keras.models.load_model(model_path)
    with open(class_json, "r") as f:
        labels = json.load(f)

    # Read and preprocess image
    img_raw = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_raw, channels=3)
    img_resized = tf.image.resize(img, IMG_SIZE)
    img_array = tf.expand_dims(img_resized, axis=0)

    # Forward pass
    probs = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]

    print("\\n" + "=" * 50)
    print(f"PREDICTION RESULTS FOR: {Path(image_path).name}")
    print("=" * 50)
    for rank, idx in enumerate(top_indices, start=1):
        print(f"  {rank}. {labels[idx]:<25}: {probs[idx]*100:>6.2f}%")
    print("=" * 50)

    # Compute Grad-CAM for top-1 class
    top_class_idx = top_indices[0]
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv", pred_index=top_class_idx)
    overlay = superimpose_gradcam(img_resized.numpy(), heatmap, alpha=0.45)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_resized.numpy().astype("uint8"))
    axes[0].set_title(f"Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Grad-CAM: {labels[top_class_idx]} ({probs[top_class_idx]*100:.1f}%)", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

# Demonstrate inference on first test image
sample_test_path = test_paths[0]
predict_and_explain_dog(sample_test_path)
"""))

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "name": "Dog_Breed_Classifier.ipynb",
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open("Dog_Breed_Classifier.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
    print("Dog_Breed_Classifier.ipynb generated successfully!")

if __name__ == "__main__":
    build_notebook()
