"""
Model Evaluation Module for Concrete Crack Detection
====================================================
Evaluates the trained '.keras' CNN classifier on unseen concrete test/val data.
Calculates Accuracy, Precision, Recall, F1-Score, and generates Confusion Matrix heatmap.
"""

import os
import sys
import json
import matplotlib
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), "outputs"))
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from data_loader import load_datasets, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE

def evaluate_model(
    model_path="models/concrete_crack_cnn.keras",
    dataset_dir="dataset",
    outputs_dir="outputs"
):
    """
    Loads saved model, executes predictions across evaluation set,
    and calculates complete scikit-learn metrics.
    """
    model_full_path = os.path.join(SCRIPT_DIR, model_path) if not os.path.isabs(model_path) else model_path
    dataset_full_path = os.path.join(SCRIPT_DIR, dataset_dir) if not os.path.isabs(dataset_dir) else dataset_dir
    outputs_full_path = os.path.join(SCRIPT_DIR, outputs_dir) if not os.path.isabs(outputs_dir) else outputs_dir

    os.makedirs(outputs_full_path, exist_ok=True)
    
    if not os.path.exists(model_full_path):
        raise FileNotFoundError(f"Model file not found at '{model_full_path}'. Please run train.py first!")

    print(f"[*] Loading trained CNN model from: {model_full_path}")
    model = keras.models.load_model(model_full_path)
    
    _, val_ds, test_ds, class_names = load_datasets(
        dataset_dir=dataset_full_path,
        batch_size=BATCH_SIZE,
        img_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    
    eval_ds = test_ds if test_ds is not None else val_ds
    eval_name = "Test" if test_ds is not None else "Validation"
    print(f"[*] Evaluating on {eval_name} dataset...")

    y_true = []
    y_pred_probs = []

    for images, labels in eval_ds:
        probs = model.predict(images, verbose=0)
        y_pred_probs.extend(probs)
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate scikit-learn metrics
    accuracy = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    prec_class, rec_class, f1_class, sup_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred)
    clr = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    print("\n=======================================================")
    print(f" EVALUATION RESULTS ({eval_name} Set)")
    print("=======================================================")
    print(f" Accuracy:           {accuracy * 100:.2f}%")
    print(f" Weighted Precision: {precision * 100:.2f}%")
    print(f" Weighted Recall:    {recall * 100:.2f}%")
    print(f" Weighted F1-Score:  {f1 * 100:.2f}%")
    print("-------------------------------------------------------")
    print(" Class Breakdown:")
    for i, name in enumerate(class_names):
        print(f"   - {name:<10}: Precision={prec_class[i]*100:.2f}%, Recall={rec_class[i]*100:.2f}%, F1={f1_class[i]*100:.2f}% (Count={sup_class[i]})")
    print("=======================================================\n")

    metrics_summary = {
        "dataset_split": eval_name,
        "total_samples": int(len(y_true)),
        "accuracy": round(accuracy, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": clr
    }

    report_json_path = os.path.join(outputs_full_path, "evaluation_report.json")
    with open(report_json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"[+] Evaluation summary saved to: {report_json_path}")

    plot_confusion_matrix(cm, class_names, outputs_full_path, eval_name)
    print("[OK] Evaluation finished successfully!")
    return metrics_summary

def plot_confusion_matrix(cm, class_names, outputs_dir, split_name="Test"):
    """Generates and saves a high-quality Confusion Matrix heatmap."""
    plt.figure(figsize=(7, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 16, "fontweight": "bold"}
    )
    
    plt.title(f"Confusion Matrix ({split_name} Set)", fontsize=14, fontweight="bold", pad=14)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("Actual True Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    cm_path = os.path.join(outputs_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {cm_path}")

if __name__ == "__main__":
    evaluate_model()
