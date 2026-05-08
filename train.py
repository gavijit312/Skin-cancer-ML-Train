import os
# CPU only — set before importing TensorFlow so it takes effect
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import tensorflow as tf
import numpy as np


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    label: int


# ------------------ SEED ------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ------------------ LOAD DATA ------------------
def one_hot_to_index(row, class_names):
    active = [i for i, c in enumerate(class_names) if float(row[c]) > 0.5]
    return active[0]


def load_samples(dataset_root: Path):
    csv_path   = dataset_root / "GroundTruth.csv"
    images_dir = dataset_root / "images"
    masks_dir  = dataset_root / "masks"

    samples = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        class_names = [c for c in reader.fieldnames if c != "image"]

        for row in reader:
            img_id = row["image"]
            label  = one_hot_to_index(row, class_names)

            img_path  = images_dir / f"{img_id}.jpg"
            mask_path = masks_dir  / f"{img_id}_segmentation.png"

            if img_path.exists():
                samples.append(Sample(img_path, mask_path, label))

    return samples, class_names


# ------------------ SPLIT ------------------
def stratified_split(samples, train_ratio=0.7, val_ratio=0.15, seed=42):
    by_class = {}
    for s in samples:
        by_class.setdefault(s.label, []).append(s)

    train, val, test = [], [], []
    rng = random.Random(seed)

    for cls_samples in by_class.values():
        rng.shuffle(cls_samples)
        n = len(cls_samples)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)

        train += cls_samples[:n_train]
        val   += cls_samples[n_train : n_train + n_val]
        test  += cls_samples[n_train + n_val :]

    return train, val, test


# ------------------ CLASS WEIGHTS ------------------
def compute_class_weights(samples, num_classes):
    """Inverse-frequency weighting so minority classes are not ignored."""
    counts = Counter(s.label for s in samples)
    total  = sum(counts.values())
    weights = {
        cls: total / (num_classes * count)
        for cls, count in counts.items()
    }
    print("Class weights:", {k: f"{v:.3f}" for k, v in sorted(weights.items())})
    return weights


# ------------------ IMAGE + MASK ------------------
def load_image_with_mask(img_path, mask_path, size, training=False):
    """
    Correct order:
      1. Load raw image  →  [0, 255]
      2. Apply mask      →  still [0, 255]
      3. Augment         →  still [0, 255]   (only when training=True)
      4. preprocess_input → normalised range expected by EfficientNet

    Previously preprocess_input was called before augmentation, so colour-jitter
    ops that expect [0, 255] were running on already-normalised values and
    producing garbage inputs.
    """
    # 1. Load raw image (float32, [0, 255])
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (size, size))      # float32, [0, 255]

    # 2. Apply segmentation mask on raw pixel values
    def load_mask():
        m = tf.io.read_file(mask_path)
        m = tf.io.decode_png(m, channels=1)
        m = tf.image.resize(m, (size, size), method="nearest")
        return tf.cast(m, tf.float32) / 255.0     # [0, 1]

    def default_mask():
        return tf.ones((size, size, 1), dtype=tf.float32)

    mask = tf.cond(mask_path != "", load_mask, default_mask)
    img  = tf.cast(img, tf.float32) * mask        # background zeroed, [0, 255]

    # 3. Augment BEFORE preprocess_input (values still in [0, 255])
    if training:
        img = augment(img)

    # 4. Normalise last — EfficientNet expects values in [-1, 1]
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img


def augment(img):
    """
    Dermoscopy-specific augmentation.
    Called on raw [0, 255] float32 tensors — do NOT clip to [-1, 1] here.
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Random 90-degree rotations
    k   = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)

    # Colour jitter — these ops expect [0, 255] range
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
    img = tf.image.random_hue(img, max_delta=0.05)

    # Random zoom via crop-and-resize
    size      = tf.shape(img)[0]
    crop_size = tf.cast(
        tf.cast(size, tf.float32) * tf.random.uniform([], 0.85, 1.0),
        tf.int32
    )
    img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
    img = tf.image.resize(img, [size, size])

    # Keep in a valid pixel range before preprocess_input
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img


def make_dataset(samples, size=224, batch=32, training=True):
    img_paths   = [str(s.image_path)       for s in samples]
    mask_paths  = [str(s.mask_path)        for s in samples]
    mask_exists = [s.mask_path.exists()    for s in samples]
    labels      = [s.label                 for s in samples]

    ds = tf.data.Dataset.from_tensor_slices(
        (img_paths, mask_paths, mask_exists, labels)
    )

    if training:
        ds = ds.shuffle(len(samples), reshuffle_each_iteration=True)

    def process(img_p, mask_p, has_mask, label):
        mask_p = tf.cond(has_mask, lambda: mask_p, lambda: "")
        # Pass training flag so augmentation only happens on train set
        img = load_image_with_mask(img_p, mask_p, size, training=training)
        return img, label

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------ MODEL ------------------
def build_model(num_classes, size=224):
    """
    Key fixes vs. original:
      - No augmentation layers inside the model (they ran on preprocessed data).
      - Correct layer order: BatchNorm BEFORE Dropout (not after).
        BN needs a stable distribution to normalise; Dropout should follow.
      - Two dense blocks with L2 regularisation.
      - Reasonable starting capacity — not over-parameterised for small datasets.
    """
    inputs = tf.keras.Input(shape=(size, size, 3))

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )
    base.trainable = False

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Block 1 — BatchNorm → Dropout → Dense
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    # Block 2 — BatchNorm → Dropout → Dense
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


# ------------------ TRAIN ------------------
def main():
    set_seed(42)

    root = Path("dataset")
    samples, class_names = load_samples(root)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    train, val, test = stratified_split(samples)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    class_weights = compute_class_weights(train, num_classes)

    train_ds = make_dataset(train, training=True)
    val_ds   = make_dataset(val,   training=False)
    test_ds  = make_dataset(test,  training=False)

    model, base = build_model(num_classes)
    # model.summary()

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_fp = str(
        checkpoints_dir / "ckpt-{epoch:02d}-val_loss-{val_loss:.4f}.keras"
    )

    def make_callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(
                patience=7,
                restore_best_weights=True,
                monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=3,
                factor=0.5,
                min_lr=1e-7,
                monitor="val_loss",
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                "best_model.keras",
                save_best_only=True,
                monitor="val_loss",
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_fp,
                save_weights_only=False,
                save_freq="epoch",
                monitor="val_loss",
                verbose=0
            ),
        ]

    # ── Phase 1: Train head only (base frozen) ────────────────────────────────
    print("\n🔥 Phase 1 — Head Training (base frozen)")
    model.compile(
        # Lower LR than original (1e-3 was too aggressive with frozen BN layers)
        optimizer=tf.keras.optimizers.Adam(3e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=make_callbacks(),
        class_weight=class_weights
    )

    # ── Phase 2: Fine-tune last 60 base layers ────────────────────────────────
    print("\n🔥 Phase 2 — Fine-Tuning (last 60 base layers unfrozen)")
    base.trainable = True

    # Unfreeze last 60 layers
    for layer in base.layers[:-60]:
        layer.trainable = False

    # Keep ALL BatchNorm layers frozen to preserve ImageNet statistics
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),   # small LR for fine-tuning
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=make_callbacks(),
        class_weight=class_weights
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n📊 Loading best checkpoint for final evaluation...")
    model = tf.keras.models.load_model("best_model.keras")
    loss, acc = model.evaluate(test_ds)
    print(f"\n✅ FINAL TEST ACCURACY: {acc:.4f}  |  LOSS: {loss:.4f}")


if __name__ == "__main__":
    main()