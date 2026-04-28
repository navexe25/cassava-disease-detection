# 🌿 Cassava Leaf Disease Detection using ResNet-50 and LLM-Based Explainability

An AI-powered web application for detecting cassava leaf diseases from leaf images using Deep Learning (**ResNet-50**) with intelligent disease explanations generated using **LLaMA 3.3 70B** via Groq.

This project combines **Computer Vision + Transfer Learning + Explainable AI + Web UI** to help identify crop diseases quickly and accurately.

---

# 📌 Project Overview

Cassava is a major food crop in many countries, but leaf diseases can significantly reduce yield and quality. Manual diagnosis is slow, requires expertise, and may not always be available to farmers.

This system automates disease detection by:

* Accepting an uploaded cassava leaf image
* Classifying the disease using a trained ResNet-50 model
* Displaying confidence scores
* Generating AI-based disease explanations
* Providing treatment and prevention guidance

---

# 🚀 Features

✅ Upload cassava leaf images
✅ Real-time disease prediction
✅ Confidence score visualization
✅ Interactive web UI using Gradio
✅ AI-generated disease explanation
✅ Treatment and prevention suggestions
✅ Clean and deployable project structure

---

# 🧠 Model Architecture

## Base Model

* **ResNet-50**
* Pretrained on ImageNet
* Fine-tuned on cassava leaf disease dataset

## Why ResNet-50?

ResNet-50 uses **Residual Connections (Skip Connections)** that solve vanishing gradient problems in deep neural networks, allowing more accurate training.

Benefits:

* Strong image classification performance
* Faster convergence
* Robust feature extraction
* Excellent transfer learning backbone

---

# 📂 Disease Classes

The model predicts the following 5 classes:

1. Cassava Bacterial Blight
2. Cassava Brown Streak Disease
3. Cassava Green Mottle
4. Healthy Leaf
5. Cassava Mosaic Disease

---

# 🏋️ Training Strategy

## Data Preprocessing

Applied image augmentations:

* Resize (224 × 224)
* Random Crop
* Rotation
* Horizontal Flip
* Vertical Flip
* Perspective Transform
* Affine Transform
* Color Jitter
* Gaussian Blur
* Random Erasing
* Normalization

## Loss Function

Used custom **Focal Loss** with Label Smoothing.

Why?

* Handles class imbalance
* Focuses on difficult samples
* Improves generalization

## Optimizer

* AdamW

## Learning Rate Scheduler

* OneCycleLR

## Mixed Precision Training

Used:

* autocast()
* GradScaler()

For faster GPU training and reduced memory usage.

---

# 📊 Performance

Example metrics (update with your real values):

* Validation Accuracy: **97.58%**
* Training Epochs: **50**
* Early Stopping Enabled

---

# 🖥️ Web Application (UI)

Built using **Gradio**.

The interface includes:

* Image Upload
* Disease Prediction
* Confidence Chart
* AI Explanation Panel
* Clean Dashboard Layout

---

# 🤖 LLM Explainability Module

After prediction, the app sends the result to **LLaMA 3.3 70B** through Groq to generate farmer-friendly explanations.

Generated output includes:

* Disease Overview
* Causes
* Symptoms
* Treatment
* Prevention
* Economic Impact

This improves trust and usability beyond raw predictions.

---

# 📁 Project Structure

```text
cassava-disease-detection/
│── app.py
│── training.py
│── best_resnet50_cassava.pth
│── requirements.txt
│── README.md
│── sample_images/
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/cassava-disease-detection.git
cd cassava-disease-detection
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Application

```bash
python app.py
```

Then open the local Gradio URL in browser.

---

# 🏋️ Train Model

To retrain:

```bash
python train.py
```

---

# 🔑 API Key Setup (Groq)

Create environment variable:

## Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_api_key"
```

## Linux / Mac

```bash
export GROQ_API_KEY="your_api_key"
```

---

# 📸 Sample Workflow

1. Upload cassava leaf image
2. Click Analyze
3. View predicted disease
4. Check confidence chart
5. Read AI explanation
6. Apply treatment recommendations

---

# 🌍 Real-World Applications

* Smart Agriculture
* Farmer Advisory Systems
* Mobile Crop Diagnostics
* Agricultural Research
* Rural Disease Monitoring
* Precision Farming

---

# 🔮 Future Improvements

* Mobile App Deployment
* Multilingual Support
* Leaf Segmentation
* Disease Severity Estimation
* Real-time Camera Detection
* Cloud Deployment
* More Crop Support

---

# 🛠️ Tech Stack

* Python
* PyTorch
* Torchvision
* NumPy
* Gradio
* Plotly
* Pillow
* Scikit-learn
* Groq API
* LLaMA 3.3 70B

---


Developed by **Naveen Kumar**


