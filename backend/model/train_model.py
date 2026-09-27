"""
Train a lightweight olive-leaf classifier (frozen MobileNetV2 plus a small head).

The local Dataset folder uses Turkish names and stores each photo as a top half
and a bottom half. Those two halves stay in the same split so the accuracy
number is not inflated. After training this runs fully offline.

Usage:
    python train_model.py --data_dir ../../Dataset --epochs 12
"""

import argparse
import json
import os
import random
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

FOLDER_ALIASES = {
    "sağlam": "healthy",
    "saglam": "healthy",
    "healthy": "healthy",
    "hastalıklı": "diseased",
    "hastaliklı": "diseased",
    "diseased": "diseased",
    "peacock_spot": "peacock_spot",
    "olive_peacock_spot": "peacock_spot",
    "aculus_olearius": "aculus_olearius",
    "olive_knot": "olive_knot",
    "verticillium_wilt": "verticillium_wilt",
    "anthracnose": "anthracnose",
}


def class_name_for(folder_name):
    return FOLDER_ALIASES.get(folder_name, folder_name)


def pair_key(filename):
    name = filename
    for prefix in ("tophalf_", "bottomhalf_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return name


def collect_groups(data_dir):
    groups = defaultdict(list)
    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        label = class_name_for(folder)
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(filename)[1].lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            groups[(label, pair_key(filename))].append(path)
    return groups


def split_groups(groups, seed=42):
    by_class = defaultdict(list)
    for (label, key), paths in groups.items():
        by_class[label].append(paths)
    rng = random.Random(seed)
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    class_names = sorted(by_class)
    index = {name: i for i, name in enumerate(class_names)}
    for label, bundles in by_class.items():
        rng.shuffle(bundles)
        cut = max(1, int(round(len(bundles) * 0.8)))
        if cut >= len(bundles):
            cut = len(bundles) - 1
        for bundle in bundles[:cut]:
            for path in bundle:
                train_paths.append(path)
                train_labels.append(index[label])
        for bundle in bundles[cut:]:
            for path in bundle:
                val_paths.append(path)
                val_labels.append(index[label])
    return class_names, train_paths, train_labels, val_paths, val_labels


def make_dataset(paths, labels, num_classes, training):
    labels = tf.one_hot(tf.constant(labels, dtype=tf.int32), num_classes)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)

    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image = tf.image.resize(image, IMG_SIZE)
        image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        return image, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        augment = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        ds = ds.map(lambda image, label: (augment(image, training=True), label), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to class-folder dataset")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--output_dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    groups = collect_groups(args.data_dir)
    class_names, train_paths, train_labels, val_paths, val_labels = split_groups(groups)
    print(f"Detected classes: {class_names}")
    print(f"Train images: {len(train_paths)} | Validation images: {len(val_paths)}")
    print(f"Photo groups: {len(groups)} (top and bottom halves stay together)")

    train_ds = make_dataset(train_paths, train_labels, len(class_names), training=True)
    val_ds = make_dataset(val_paths, val_labels, len(class_names), training=False)
    model = build_model(num_classes=len(class_names))
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "olive_classifier.keras")
    model.save(model_path)
    with open(os.path.join(args.output_dir, "class_indices.json"), "w", encoding="utf-8") as handle:
        json.dump({i: name for i, name in enumerate(class_names)}, handle, ensure_ascii=False, indent=2)

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nSaved model to {model_path}")
    print(f"Validation accuracy: {val_acc:.3f}")
    print("This score is sick-versus-healthy on the local leaf photos, not a named-disease diagnosis.")


if __name__ == "__main__":
    main()
