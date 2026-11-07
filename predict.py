"""
Inference & Prediction Module for Concrete Crack Detection
==========================================================
Loads the trained '.keras' model, runs predictions on single or multiple images,
computes confidence scores, assesses crack severity, and highlights crack areas.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageOps

import tensorflow as tf
from tensorflow import keras

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ["Negative", "Positive"]  # 0: Non-Cracked, 1: Cracked

def load_inference_model(model_path="models/concrete_crack_cnn.keras"):
    """Loads and caches the compiled Keras model."""
    full_path = os.path.join(SCRIPT_DIR, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file '{full_path}' not found! Run train.py first.")
    model = keras.models.load_model(full_path)
    return model

def preprocess_image(image_input, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Accepts a PIL Image or file path and prepares it as a batch tensor (1, 224, 224, 3).
    """
    if isinstance(image_input, str):
        full_img_path = os.path.join(SCRIPT_DIR, image_input) if not os.path.isabs(image_input) else image_input
        img = Image.open(full_img_path).convert("RGB")
    else:
        img = image_input.convert("RGB")
        
    img_resized = img.resize(target_size, Image.Resampling.BILINEAR)
    img_array = keras.utils.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    return img, img_batch

def generate_crack_heatmap_overlay(original_pil_img):
    """
    Generates a visual inspection overlay highlighting dark fracture stress zones
    on the concrete surface to provide clear visual feedback.
    """
    img_rgb = original_pil_img.convert("RGB")
    gray = img_rgb.convert("L")
    
    gray_inv = ImageOps.invert(gray)
    edges = gray_inv.filter(ImageFilter.FIND_EDGES)
    
    np_orig = np.array(img_rgb).astype(np.float32)
    np_edges = np.array(edges).astype(np.float32)
    
    mask = (np_edges > 65).astype(np.float32)
    
    overlay = np_orig.copy()
    overlay[..., 0] = np.clip(overlay[..., 0] * (1 - 0.7 * mask) + 245 * mask * 0.7, 0, 255)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1 - 0.7 * mask), 0, 255)
    overlay[..., 2] = np.clip(overlay[..., 2] * (1 - 0.7 * mask), 0, 255)
    
    return Image.fromarray(overlay.astype(np.uint8))

def predict_single_image(model, image_input):
    """
    Runs prediction on a single concrete image.
    """
    orig_img, img_batch = preprocess_image(image_input)
    
    probabilities = model.predict(img_batch, verbose=0)[0]
    prob_neg = float(probabilities[0])
    prob_pos = float(probabilities[1])
    
    predicted_idx = int(np.argmax(probabilities))
    is_cracked = (predicted_idx == 1)
    confidence = prob_pos if is_cracked else prob_neg
    
    if is_cracked:
        label = "Cracked Concrete"
        if confidence > 0.90:
            severity = "Severe Structural Fracture"
            recommendation = "Immediate structural audit required. Epoxy pressure injection or carbon fiber reinforcement advised."
        elif confidence > 0.75:
            severity = "Moderate Surface Crack"
            recommendation = "Monitor propagation width. Seal with polyurethane sealant to prevent moisture infiltration."
        else:
            severity = "Minor Hairline Crack"
            recommendation = "Periodic monitoring. Non-structural cosmetic sealant recommended."
    else:
        label = "Non-Cracked (Sound Concrete)"
        severity = "Normal / Undamaged"
        recommendation = "Concrete surface shows structural integrity. Routine scheduled maintenance inspection."
        
    overlay_img = generate_crack_heatmap_overlay(orig_img) if is_cracked else orig_img
    
    return {
        "label": label,
        "is_cracked": is_cracked,
        "confidence": confidence,
        "confidence_percent": f"{confidence * 100:.2f}%",
        "prob_negative": prob_neg,
        "prob_positive": prob_pos,
        "severity": severity,
        "recommendation": recommendation,
        "original_image": orig_img,
        "overlay_image": overlay_img
    }

def main():
    parser = argparse.ArgumentParser(description="Predict concrete cracks using trained CNN model.")
    parser.add_argument("--image", type=str, help="Path to single concrete image file")
    parser.add_argument("--dir", type=str, help="Directory containing concrete images to batch predict")
    parser.add_argument("--model", type=str, default="models/concrete_crack_cnn.keras", help="Path to model file")
    
    args = parser.parse_args()
    
    model = load_inference_model(args.model)
    
    if args.image:
        print(f"\n[*] Inspecting image: {args.image}")
        res = predict_single_image(model, args.image)
        print("-------------------------------------------------------")
        print(f" Result:         {res['label']}")
        print(f" Confidence:     {res['confidence_percent']}")
        print(f" Severity Level: {res['severity']}")
        print(f" Advice:         {res['recommendation']}")
        print("-------------------------------------------------------\n")
    elif args.dir:
        dir_path = os.path.join(SCRIPT_DIR, args.dir) if not os.path.isabs(args.dir) else args.dir
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"\n[*] Inspecting {len(files)} images in directory: {dir_path}")
        print("=" * 70)
        print(f"{'Filename':<25} | {'Prediction':<20} | {'Confidence':<12} | {'Severity'}")
        print("-" * 70)
        for fp in files:
            res = predict_single_image(model, fp)
            fn = os.path.basename(fp)
            print(f"{fn:<25} | {res['label']:<20} | {res['confidence_percent']:<12} | {res['severity']}")
        print("=" * 70 + "\n")
    else:
        demo_dir = os.path.join(SCRIPT_DIR, "test_samples")
        if os.path.exists(demo_dir):
            files = [os.path.join(demo_dir, f) for f in os.listdir(demo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"\n[*] Running default inference on {len(files)} sample test images:")
            print("=" * 75)
            print(f"{'Filename':<22} | {'Prediction':<22} | {'Confidence':<10} | {'Severity'}")
            print("-" * 75)
            for fp in files:
                res = predict_single_image(model, fp)
                fn = os.path.basename(fp)
                print(f"{fn:<22} | {res['label']:<22} | {res['confidence_percent']:<10} | {res['severity']}")
            print("=" * 75 + "\n")
        else:
            print("Please specify an image with --image <path> or folder with --dir <path>")

if __name__ == "__main__":
    main()
