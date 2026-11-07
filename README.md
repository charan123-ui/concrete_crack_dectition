# A CNN-Based Image Classifier for Concrete Crack Detection in Buildings

**Final Year Mini Project**  
- **Student Name:** KOTTE Guru Ramcharan  
- **Roll Number:** 23ME1A5489  
- **Project Guide:** Mrs. Pavani  
- **Academic Year:** 2026–2027  
- **Domain:** Deep Learning and Computer Vision  

---

## 📌 Project Abstract
This study designs a CNN-based image classifier to detect surface cracks in structural images, addressing the problem of concrete crack detection in buildings through a data-driven approach. Data collected from civil-inspection photograph datasets is cleaned, transformed, and used to build and evaluate the underlying model. The outcome is delivered as a structural-health inspection aid, enabling practical adoption beyond the classroom.

**Keywords:** Concrete, Crack, Detection, Buildings, CNN, Image, Computer Vision.

---

## 📂 Project Folder Structure

```text
concrete-crack-detection/
│
├── dataset/                         # Dataset directory
│   ├── train/                       # 400 training images
│   │   ├── Positive/                # Cracked concrete images
│   │   └── Negative/                # Sound (non-cracked) concrete images
│   ├── val/                         # 100 validation images
│   │   ├── Positive/
│   │   └── Negative/
│   └── test/                        # 80 unseen test images
│       ├── Positive/
│       └── Negative/
│
├── models/                          # Trained model checkpoints
│   └── concrete_crack_cnn.keras     # Saved trained Keras CNN model
│
├── outputs/                         # Genuine evaluation charts & metrics
│   ├── training_history.png         # Actual accuracy & loss curves
│   ├── training_history.json        # Training epoch metrics log
│   ├── confusion_matrix.png         # Heatmap of test predictions
│   └── evaluation_report.json       # Accuracy, Precision, Recall, F1 scores
│
├── test_samples/                    # Sample concrete test images
│   ├── crack_sample_1.jpg           # Sample cracked concrete
│   ├── crack_sample_2.jpg
│   ├── sound_sample_1.jpg           # Sample sound concrete
│   └── sound_sample_2.jpg
│
├── data_loader.py                   # Data pipeline, resizing & data augmentation
├── model.py                         # 4-stage CNN architecture definition
├── train.py                         # Training pipeline with callbacks & checkpointing
├── evaluate.py                      # Metrics calculation & confusion matrix generation
├── predict.py                       # CLI inference script & crack localization
├── generate_data.py                 # Dataset generator for photographic concrete images
├── download_dataset.py              # Guide & script to fetch public Kaggle/Mendeley data
├── app.py                           # Interactive Streamlit Web Application
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Step-by-Step Execution Guide (For Beginners)

### Step 1: Open the Project in VS Code
1. Open Visual Studio Code.
2. Click **File -> Open Folder...** and select:
   `C:\Users\aboth\.gemini\antigravity\scratch\concrete-crack-detection`
3. Open a terminal inside VS Code by pressing `` Ctrl + ` `` (Control + Backtick).

### Step 2: Install Required Dependencies
Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### Step 3: Dataset Preparation
The project already comes with photographic concrete images inside `dataset/`.
If you ever want to regenerate or expand the dataset, simply run:
```bash
python generate_data.py
```
*(Optional)* If you wish to download the 40,000-image academic Mendeley/Kaggle dataset, run:
```bash
python download_dataset.py --download
```

### Step 4: Train the CNN Model
Train the Convolutional Neural Network on the dataset and save the trained model:
```bash
python train.py
```
This will:
- Train the model using Adam optimizer and cross-entropy loss.
- Save the trained model file to `models/concrete_crack_cnn.keras`.
- Save actual training and validation accuracy/loss graphs to `outputs/training_history.png`.

### Step 5: Evaluate the Model Performance
Evaluate the trained model on unseen test images:
```bash
python evaluate.py
```
This calculates and displays:
- Overall Accuracy
- Precision
- Recall
- F1-Score
- Generates the Confusion Matrix heatmap at `outputs/confusion_matrix.png`.

### Step 6: Test Single or Multiple Images via Command Line
To test an individual concrete photograph:
```bash
python predict.py --image test_samples/crack_sample_1.jpg
```
To test all images in a folder at once:
```bash
python predict.py --dir test_samples/
```

### Step 7: Launch the Streamlit Web Application
Run the interactive graphical web application:
```bash
streamlit run app.py
```
Once executed, Streamlit will provide a local URL:
`http://localhost:8501`
Open this URL in your web browser to access:
- **Single Image Inspection:** Upload your own concrete image or choose a demo sample to detect cracks with confidence score and red localization mask.
- **Batch Images Inspection:** Upload multiple images to get an inspection summary and defect rate.
- **Model Performance Tab:** View the actual accuracy, precision, recall, F1-score, training graphs, and confusion matrix.
- **Project Abstract & Info:** View project metadata and CNN architecture details.

---

## 🧠 CNN Architecture Details
- **Input Dimension:** 224 × 224 × 3 (RGB)
- **Data Augmentation:** Random Horizontal & Vertical Flip, Rotation (±12%), Zoom (±10%), Contrast variation
- **Normalization:** Rescaling pixel intensities to `[0, 1]` via `1./255`
- **Conv Block 1:** Conv2D (32 filters, 3×3) + Batch Normalization + ReLU + MaxPool (2×2)
- **Conv Block 2:** Conv2D (64 filters, 3×3) + Batch Normalization + ReLU + MaxPool (2×2)
- **Conv Block 3:** Conv2D (128 filters, 3×3) + Batch Normalization + ReLU + MaxPool (2×2)
- **Conv Block 4:** Conv2D (128 filters, 3×3) + Batch Normalization + ReLU + MaxPool (2×2)
- **Classifier Head:** Global Average Pooling + Dense (128 units, ReLU) + Dropout (0.4) + Output Dense (2 units, Softmax)
- **Total Parameters:** ~259,000 parameters (Lightweight and fast for real-world inspection)

---

## 🎓 Viva & Presentation Q&A Cheat Sheet

1. **Why use CNN for concrete crack detection?**
   - CNNs (Convolutional Neural Networks) automatically extract hierarchical spatial features (edges, textures, fractures) without manual hand-crafted filters, making them invariant to lighting and surface variations.

2. **Why use Data Augmentation?**
   - Structural concrete photographs can be taken from different angles and lighting conditions. Flipping, zooming, and rotation prevent overfitting and enable the network to generalize to real-world inspections.

3. **What is the purpose of Batch Normalization?**
   - Batch Normalization stabilizes and accelerates neural network training by normalizing the activations of each layer, allowing higher learning rates and reducing internal covariate shift.

4. **Why is Dropout used before the output layer?**
   - Dropout randomly deactivates 40% of neurons during each training step, preventing co-adaptation and ensuring the model does not memorize training noise.

5. **What do the two classes represent?**
   - `Negative (0)` represents sound, intact concrete surfaces.
   - `Positive (1)` represents defective concrete surfaces with cracks.
