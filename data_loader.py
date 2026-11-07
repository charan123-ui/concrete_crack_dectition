"""
Data Loading, Preprocessing & Augmentation Module
=================================================
Loads images from directory splits, resizes them to 224x224,
applies data augmentation, and sets up high-performance tf.data pipelines.
"""

import os
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Set image constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

def get_data_augmentation():
    """
    Creates an explicit data augmentation sequential layer
    to improve generalization and prevent overfitting on concrete textures.
    """
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", name="random_flip"),
        layers.RandomRotation(0.12, name="random_rotation"),
        layers.RandomZoom(0.1, name="random_zoom"),
        layers.RandomContrast(0.1, name="random_contrast"),
    ], name="data_augmentation")
    return augmentation

def load_datasets(dataset_dir="dataset", batch_size=BATCH_SIZE, img_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Loads train, validation, and optional test datasets from subdirectories.
    
    Folder structure expected:
      dataset/
        train/
          Negative/
          Positive/
        val/
          Negative/
          Positive/
        test/ (optional)
          Negative/
          Positive/
          
    Returns:
      train_ds, val_ds, test_ds, class_names
    """
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset train or val directory missing inside '{dataset_dir}'!")

    print(f"[*] Loading training data from: {train_dir}")
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",            # 0 for Negative, 1 for Positive
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=42
    )

    print(f"[*] Loading validation data from: {val_dir}")
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False
    )
    
    test_ds = None
    if os.path.exists(test_dir):
        print(f"[*] Loading testing data from: {test_dir}")
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False
        )

    class_names = train_ds.class_names
    print(f"[+] Detected Classes: {class_names} (0: {class_names[0]}, 1: {class_names[1]})")

    # Configure dataset for performance using buffering & prefetch
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

if __name__ == "__main__":
    t_ds, v_ds, te_ds, classes = load_datasets()
    for images, labels in t_ds.take(1):
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
