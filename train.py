"""
Model Training Pipeline for Concrete Crack Detection
===================================================
Trains the Convolutional Neural Network on concrete surface images,
saves the best trained model checkpoint as '.keras', and generates
actual training & validation accuracy and loss performance graphs.
"""

import os
import sys
import json
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "outputs"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

# Ensure local imports work regardless of current working directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_datasets, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
from model import build_concrete_crack_cnn

def train_model(
    dataset_dir="dataset",
    models_dir="models",
    outputs_dir="outputs",
    epochs=10,
    batch_size=BATCH_SIZE,
    learning_rate=0.0005
):
    """
    Executes end-to-end model training, validation, checkpointing, and graph generation.
    """
    # Resolve relative paths relative to script dir
    dataset_path = os.path.join(SCRIPT_DIR, dataset_dir) if not os.path.isabs(dataset_dir) else dataset_dir
    models_path = os.path.join(SCRIPT_DIR, models_dir) if not os.path.isabs(models_dir) else models_dir
    outputs_path = os.path.join(SCRIPT_DIR, outputs_dir) if not os.path.isabs(outputs_dir) else outputs_dir

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(outputs_path, exist_ok=True)
    
    tf.random.set_seed(42)
    
    print("\n=======================================================")
    print(" STEP 1: Loading Dataset for Concrete Crack Classifier")
    print("=======================================================")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        dataset_dir=dataset_path,
        batch_size=batch_size,
        img_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    
    print("\n=======================================================")
    print(" STEP 2: Instantiating CNN Architecture")
    print("=======================================================")
    model = build_concrete_crack_cnn(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        num_classes=len(class_names),
        use_augmentation=True
    )
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    model_save_path = os.path.join(models_path, "concrete_crack_cnn.keras")
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]
    
    print("\n=======================================================")
    print(" STEP 3: Training CNN Model")
    print(f" Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {learning_rate}")
    print("=======================================================")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Ensure final model is saved
    if not os.path.exists(model_save_path):
        model.save(model_save_path)
        print(f"[+] Model saved to {model_save_path}")
        
    print("\n=======================================================")
    print(" STEP 4: Saving Actual Training History & Performance Graphs")
    print("=======================================================")
    hist_dict = {
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
        "loss": [float(v) for v in history.history.get("loss", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
    }
    
    history_json_path = os.path.join(outputs_path, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(hist_dict, f, indent=4)
    print(f"[+] History data saved to: {history_json_path}")
    
    # Generate actual accuracy and loss graphs
    plot_training_curves(hist_dict, outputs_path)
    
    print("\n[OK] Training Pipeline Finished Successfully!")
    print(f"    Saved Model:  {model_save_path}")
    print(f"    Output Plots: {outputs_path}/training_history.png")
    return model, hist_dict

def plot_training_curves(history_data, outputs_dir):
    """Plots actual training & validation accuracy and loss graphs."""
    acc = history_data["accuracy"]
    val_acc = history_data["val_accuracy"]
    loss = history_data["loss"]
    val_loss = history_data["val_loss"]
    epochs_range = range(1, len(acc) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Accuracy Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, marker="o", color="#1f77b4", linewidth=2.2, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, marker="s", color="#ff7f0e", linewidth=2.2, linestyle="--", label="Validation Accuracy")
    plt.title("Model Accuracy per Epoch", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy", fontsize=11)
    plt.ylim([0.45, 1.02])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True)
    
    # Loss Subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, marker="o", color="#d62728", linewidth=2.2, label="Training Loss")
    plt.plot(epochs_range, val_loss, marker="s", color="#9467bd", linewidth=2.2, linestyle="--", label="Validation Loss")
    plt.title("Model Loss per Epoch (Cross-Entropy)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    chart_path = os.path.join(outputs_dir, "training_history.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Performance chart saved to: {chart_path}")

if __name__ == "__main__":
    train_model(epochs=8)
