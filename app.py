"""
Streamlit Web Application: Concrete Crack Detection in Buildings
================================================================
Final Year Mini Project:
  Title: A CNN-Based Image Classifier for Concrete Crack Detection in Buildings
  Student Name: KOTTE Guru Ramcharan
  Roll Number:  23ME1A5489
  Project Guide: Mrs. Pavani
  Academic Year: 2026–2027
  Domain: Deep Learning and Computer Vision
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tensorflow as tf
from tensorflow import keras

from predict import load_inference_model, predict_single_image, generate_crack_heatmap_overlay

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Concrete Crack Detector | CivilInspect AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for attractive, professional academic look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    .main-header h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .main-header p {
        color: #e0e8f5;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }
    .meta-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 20px;
        font-size: 0.95rem;
    }
    .result-card-cracked {
        background: #fff5f5;
        border: 2px solid #e53e3e;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
    .result-card-sound {
        background: #f0fff4;
        border: 2px solid #38a169;
        border-radius: 10px;
        padding: 20px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Project Header
st.markdown("""
<div class="main-header">
    <h1>🏢 Concrete Crack Detection in Buildings</h1>
    <p>A CNN-Based Deep Learning Computer Vision Inspection System for Structural Health Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with Student Information & Settings
with st.sidebar:
    st.image("https://img.icons8.com/color/96/structural.png", width=70)
    st.title("Project Details")
    
    st.markdown("""
    **Student:** KOTTE Guru Ramcharan  
    **Roll No:** `23ME1A5489`  
    **Project Guide:** Mrs. Pavani  
    **Academic Year:** 2026–2027  
    **Domain:** Deep Learning & Computer Vision  
    """)
    st.divider()
    
    navigation = st.radio(
        "Select Operation Mode:",
        ["🔍 Single Image Inspection", "📁 Batch Images Inspection", "📊 Model Performance & Metrics", "📖 Project Abstract & Info"],
        index=0
    )
    st.divider()
    
    model_path = os.path.join(SCRIPT_DIR, "models", "concrete_crack_cnn.keras")
    if os.path.exists(model_path):
        st.success(" Model Status: Ready (`.keras` loaded)")
    else:
        st.warning("⚠️ Model not found! Please run `python train.py` to train.")

# Cached model loader
@st.cache_resource
def get_model():
    if os.path.exists(model_path):
        return load_inference_model(model_path)
    return None

model = get_model()

# ==============================================================================
# MODE 1: SINGLE IMAGE INSPECTION
# ==============================================================================
if navigation == "🔍 Single Image Inspection":
    st.subheader("Single Concrete Tile Inspection")
    st.write("Upload a photograph of a building concrete wall, column, slab, or pavement to inspect for surface cracks.")
    
    col_input, col_preset = st.columns([2, 1])
    
    selected_image = None
    image_source = None
    
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Concrete Image (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Choose an image from your computer to inspect"
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file)
            image_source = uploaded_file.name

    with col_preset:
        st.write("**Or select from test demo samples:**")
        sample_dir = os.path.join(SCRIPT_DIR, "test_samples")
        sample_options = ["None"]
        if os.path.exists(sample_dir):
            samples = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.jpg', '.png'))]
            sample_options += samples
            
        chosen_sample = st.selectbox("Quick Demo Samples:", sample_options, index=1 if len(sample_options) > 1 else 0)
        if chosen_sample != "None" and uploaded_file is None:
            sample_path = os.path.join(sample_dir, chosen_sample)
            selected_image = Image.open(sample_path)
            image_source = chosen_sample

    if selected_image is not None:
        st.markdown(f"**Loaded Inspection Target:** `{image_source}`")
        c1, c2 = st.columns(2)
        with c1:
            st.image(selected_image, caption="Concrete Surface Preview", use_container_width=True)
            
        with c2:
            st.write("### Run CNN Inference")
            if model is None:
                st.error("Model file `models/concrete_crack_cnn.keras` not found. Please execute `python train.py` first.")
            else:
                predict_button = st.button("🚀 Inspect Concrete Image", type="primary", use_container_width=True)
                
                if predict_button:
                    with st.spinner("Analyzing concrete micro-texture with CNN layers..."):
                        result = predict_single_image(model, selected_image)
                        
                        # Display Results
                        if result["is_cracked"]:
                            st.markdown(f"""
                            <div class="result-card-cracked">
                                <h3 style="color:#c53030; margin-top:0;">⚠️ DEFECT DETECTED: Cracked Concrete</h3>
                                <p><strong>Confidence Score:</strong> <span style="font-size:1.3rem; color:#c53030; font-weight:700;">{result['confidence_percent']}</span></p>
                                <p><strong>Estimated Severity:</strong> <span style="color:#e53e3e; font-weight:600;">{result['severity']}</span></p>
                                <p><strong>Civil Engineering Advice:</strong> {result['recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write("#### Crack Localization / Stress Mask")
                            st.image(result["overlay_image"], caption="Crack Localization Overlay (Red Warning Mask)", use_container_width=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card-sound">
                                <h3 style="color:#2f855a; margin-top:0;">✅ SOUND CONCRETE: Non-Cracked</h3>
                                <p><strong>Confidence Score:</strong> <span style="font-size:1.3rem; color:#2f855a; font-weight:700;">{result['confidence_percent']}</span></p>
                                <p><strong>Condition:</strong> <span style="color:#38a169; font-weight:600;">Structural surface intact</span></p>
                                <p><strong>Maintenance Advice:</strong> {result['recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)

                        # Display Class Probabilities
                        st.write("#### Class Probabilities")
                        prob_df = pd.DataFrame({
                            "Class": ["Negative (Sound Concrete)", "Positive (Cracked Concrete)"],
                            "Probability": [result["prob_negative"], result["prob_positive"]]
                        })
                        st.bar_chart(prob_df.set_index("Class"))

# ==============================================================================
# MODE 2: BATCH IMAGES INSPECTION
# ==============================================================================
elif navigation == "📁 Batch Images Inspection":
    st.subheader("Batch Inspection of Multiple Concrete Images")
    st.write("Upload a batch of concrete photos (e.g. from a drone or camera survey) to generate an automated inspection summary.")
    
    batch_files = st.file_uploader(
        "Upload Multiple Concrete Images:",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    demo_button = st.button("Run Batch Test on Built-in Test Samples")
    
    files_to_process = []
    if batch_files:
        files_to_process = [(f.name, Image.open(f)) for f in batch_files]
    elif demo_button:
        sample_dir = os.path.join(SCRIPT_DIR, "test_samples")
        if os.path.exists(sample_dir):
            for fn in os.listdir(sample_dir):
                if fn.lower().endswith(('.jpg', '.png')):
                    files_to_process.append((fn, Image.open(os.path.join(sample_dir, fn))))
                    
    if files_to_process:
        if model is None:
            st.error("Trained model not found! Please train the model first.")
        else:
            st.write(f"Processing **{len(files_to_process)}** images...")
            results = []
            
            progress_bar = st.progress(0)
            for idx, (fname, img) in enumerate(files_to_process):
                res = predict_single_image(model, img)
                results.append({
                    "Filename": fname,
                    "Verdict": res["label"],
                    "Confidence": res["confidence_percent"],
                    "Severity": res["severity"],
                    "Advice": res["recommendation"],
                    "is_cracked": res["is_cracked"],
                    "img": img
                })
                progress_bar.progress((idx + 1) / len(files_to_process))
                
            # Summary Metrics
            total = len(results)
            cracked_cnt = sum(1 for r in results if r["is_cracked"])
            sound_cnt = total - cracked_cnt
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Inspected Images", total)
            m2.metric("Cracks Detected (Defective)", cracked_cnt, delta=f"{(cracked_cnt/total)*100:.1f}% defect rate", delta_color="inverse")
            m3.metric("Sound Concrete (Safe)", sound_cnt)
            
            # Tabular Display
            df_summary = pd.DataFrame([{
                "Filename": r["Filename"],
                "Prediction": r["Verdict"],
                "Confidence": r["Confidence"],
                "Severity": r["Severity"],
                "Action": r["Advice"]
            } for r in results])
            st.dataframe(df_summary, use_container_width=True)
            
            # Visual Gallery Grid
            st.write("#### Visual Inspection Results Gallery")
            cols = st.columns(3)
            for i, r in enumerate(results):
                with cols[i % 3]:
                    st.image(r["img"], caption=f"{r['Filename']}\n{r['Verdict']} ({r['Confidence']})", use_container_width=True)

# ==============================================================================
# MODE 3: MODEL PERFORMANCE & METRICS
# ==============================================================================
elif navigation == "📊 Model Performance & Metrics":
    st.subheader("Model Performance Evaluation & Training History")
    st.write("Genuine evaluation results generated directly from model training and validation/test datasets.")
    
    outputs_dir = os.path.join(SCRIPT_DIR, "outputs")
    report_path = os.path.join(outputs_dir, "evaluation_report.json")
    history_img_path = os.path.join(outputs_dir, "training_history.png")
    cm_img_path = os.path.join(outputs_dir, "confusion_matrix.png")
    
    if os.path.exists(report_path):
        with open(report_path, "r") as f:
            metrics = json.load(f)
            
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
        col2.metric("Precision", f"{metrics['precision']*100:.2f}%")
        col3.metric("Recall", f"{metrics['recall']*100:.2f}%")
        col4.metric("F1-Score", f"{metrics['f1_score']*100:.2f}%")
        st.divider()
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.write("### Training & Validation Curves")
            if os.path.exists(history_img_path):
                st.image(history_img_path, caption="Actual Training & Validation Accuracy and Loss Curves", use_container_width=True)
            else:
                st.info("Run `python train.py` to generate training history graphs.")
                
        with c_right:
            st.write("### Confusion Matrix")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix on Test/Validation Split", use_container_width=True)
            else:
                st.info("Run `python evaluate.py` to generate confusion matrix.")
                
        st.write("### Detailed Per-Class Classification Report")
        if "classification_report" in metrics:
            cr_df = pd.DataFrame(metrics["classification_report"]).transpose()
            st.dataframe(cr_df.style.format(precision=3), use_container_width=True)
    else:
        st.warning("⚠️ No evaluation report found! Please run `python train.py` and `python evaluate.py` to produce genuine metrics.")

# ==============================================================================
# MODE 4: PROJECT ABSTRACT & ARCHITECTURE INFO
# ==============================================================================
elif navigation == "📖 Project Abstract & Info":
    st.subheader("Academic Project Overview")
    
    st.markdown("""
    <div class="meta-box">
        <strong>PROJECT TITLE:</strong> A CNN-Based Image Classifier for Concrete Crack Detection in Buildings<br>
        <strong>STUDENT NAME:</strong> KOTTE Guru Ramcharan &nbsp;|&nbsp; <strong>ROLL NUMBER:</strong> 23ME1A5489<br>
        <strong>PROJECT GUIDE:</strong> Mrs. Pavani &nbsp;|&nbsp; <strong>ACADEMIC YEAR:</strong> 2026–2027<br>
        <strong>DOMAIN:</strong> Deep Learning and Computer Vision
    </div>
    """, unsafe_allow_html=True)
    
    st.write("### Abstract")
    st.write("""
    This study designs a CNN-based image classifier to detect surface cracks in structural images,
    addressing the problem of concrete crack detection in buildings through a data-driven approach.
    Data collected from civil-inspection photograph datasets is cleaned, transformed, and used to build
    and evaluate the underlying model. The outcome is delivered as a structural-health inspection aid,
    enabling practical adoption beyond the classroom.
    """)
    
    st.write("### Keywords")
    st.info("Concrete, Crack, Detection, Buildings, CNN, Image, Computer Vision, Structural Health Monitoring.")
    
    st.write("### CNN Architecture Workflow")
    st.markdown("""
    1. **Input Layer:** Receives concrete surface image tiles resized to `224 × 224 × 3`.
    2. **Data Augmentation:** Random Flip, Rotation, and Zoom to ensure invariance to lighting and orientation.
    3. **Normalization:** Rescaling pixel values to `[0, 1]` via `1./255`.
    4. **4-Stage Feature Extraction:**
       - **Block 1:** 32 filters (3×3) + ReLU + Max Pooling (2×2)
       - **Block 2:** 64 filters (3×3) + ReLU + Max Pooling (2×2)
       - **Block 3:** 128 filters (3×3) + ReLU + Max Pooling (2×2)
       - **Block 4:** 128 filters (3×3) + ReLU + Max Pooling (2×2)
    5. **Classification Head:** Global Max Pooling + Dense(64, ReLU) + Dropout(0.3) + Output Softmax (2 classes: Negative vs Positive).
    """)

# Footer
st.divider()
st.markdown("<p style='text-align:center; color:#718096; font-size:0.88rem;'>Final-Year Mini Project &copy; 2026–2027 | KOTTE Guru Ramcharan (23ME1A5489) | Supervised by Mrs. Pavani</p>", unsafe_allow_html=True)
