import os
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

# CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    label: int


# ------------------ SEED ------------------
def set_seed(seed=42):
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ------------------ LOAD DATA ------------------
def one_hot_to_index(row, class_names):
    active = [i for i, c in enumerate(class_names) if float(row[c]) > 0.5]
    return active[0]


def load_samples(dataset_root: Path):
    csv_path = dataset_root / "GroundTruth.csv"
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"

    samples = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        class_names = [c for c in reader.fieldnames if c != "image"]

        for row in reader:
            img_id = row["image"]
            label = one_hot_to_index(row, class_names)

            img_path = images_dir / f"{img_id}.jpg"
            mask_path = masks_dir / f"{img_id}_segmentation.png"

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
        n_val = int(n * val_ratio)

        train += cls_samples[:n_train]
        val += cls_samples[n_train:n_train+n_val]
        test += cls_samples[n_train+n_val:]

    return train, val, test


# ------------------ IMAGE + MASK ------------------
def load_image_with_mask(img_path, mask_path, size):
    # Image
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (size, size))
    img = tf.cast(img, tf.float32)

    # Mask (safe)
    def load_mask():
        m = tf.io.read_file(mask_path)
        m = tf.io.decode_png(m, channels=1)
        m = tf.image.resize(m, (size, size))
        return tf.cast(m, tf.float32) / 255.0

    def default_mask():
        return tf.ones((size, size, 1), dtype=tf.float32)

    mask = tf.cond(mask_path != "", load_mask, default_mask)

    # Apply mask
    img = img * mask

    # Preprocess
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    return img


def make_dataset(samples, size=224, batch=32, training=True):
    img_paths = [str(s.image_path) for s in samples]
    mask_paths = [str(s.mask_path) for s in samples]
    mask_exists = [s.mask_path.exists() for s in samples]
    labels = [s.label for s in samples]

    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths, mask_exists, labels))

    if training:
        ds = ds.shuffle(len(samples))

    def process(img_p, mask_p, has_mask, label):
        mask_p = tf.cond(has_mask, lambda: mask_p, lambda: "")
        img = load_image_with_mask(img_p, mask_p, size)
        return img, label

    ds = ds.map(process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return ds


# ------------------ MODEL ------------------
def build_model(num_classes, size=224):
    inputs = tf.keras.Input(shape=(size, size, 3))

    # Light augmentation
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)

    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )

    base.trainable = False

    x = base.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    return model, base


# ------------------ TRAIN ------------------
def main():
    set_seed(42)

    root = Path("archive (1)")
    samples, class_names = load_samples(root)

    train, val, test = stratified_split(samples)

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    train_ds = make_dataset(train)
    val_ds = make_dataset(val, training=False)
    test_ds = make_dataset(test, training=False)

    model, base = build_model(len(class_names))

    # Phase 1
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True)
    ]

    print("\n🔥 Phase 1 Training")
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # Phase 2 (fine-tuning)
    print("\n🔥 Phase 2 Fine-Tuning")

    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # Test
    model = tf.keras.models.load_model("best_model.keras")
    loss, acc = model.evaluate(test_ds)

    print(f"\n✅ FINAL TEST ACCURACY: {acc:.4f}")


if __name__ == "__main__":
    main()