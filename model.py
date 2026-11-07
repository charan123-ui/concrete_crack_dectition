"""
Convolutional Neural Network (CNN) Model Architecture
=====================================================
A Deep Convolutional Neural Network designed for binary
surface crack detection on structural concrete surfaces.
"""

import os
import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_concrete_crack_cnn(input_shape=(224, 224, 3), num_classes=2, use_augmentation=True):
    """
    Builds the CNN model architecture.
    
    Structure:
      - Input Layer (224 x 224 x 3)
      - Data Augmentation (Flip, Rotation, Zoom) [Active during training]
      - Normalization / Rescaling (1./255)
      - Conv Block 1: Conv2D(32, 3x3, relu) + MaxPool(2x2)
      - Conv Block 2: Conv2D(64, 3x3, relu) + MaxPool(2x2)
      - Conv Block 3: Conv2D(128, 3x3, relu) + MaxPool(2x2)
      - Conv Block 4: Conv2D(128, 3x3, relu) + MaxPool(2x2)
      - Feature Pooling: GlobalMaxPooling2D (Captures localized crack signals)
      - Classifier Head: Dense(64, relu) + Dropout(0.3) + Output Softmax
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    
    x = inputs
    if use_augmentation:
        x = layers.RandomFlip("horizontal_and_vertical", name="aug_flip")(x)
        x = layers.RandomRotation(0.1, name="aug_rotation")(x)
        x = layers.RandomZoom(0.08, name="aug_zoom")(x)
        
    # Rescale pixel intensities from [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(x)
    
    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    
    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    
    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    
    # Block 4
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    
    # Global Max Pooling extracts maximum crack feature activations across the spatial grid
    x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ConcreteCrackCNN")
    return model

if __name__ == "__main__":
    cnn = build_concrete_crack_cnn()
    cnn.summary()
