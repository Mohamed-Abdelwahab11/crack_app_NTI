# 🧱 Surface Crack Detection App

> A deep learning-powered web application that detects cracks in concrete surfaces from uploaded images — built with Streamlit and deployed on Streamlit Cloud.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![ONNX](https://img.shields.io/badge/Model-ONNX-orange?logo=onnx)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project was developed as part of the **NTI (National Telecommunication Institute)** training program. It uses a **Convolutional Neural Network (CNN)** trained on concrete surface images to classify whether a surface contains a crack (**Positive**) or not (**Negative**).

The model was originally trained using **TensorFlow/Keras** and then exported to the **ONNX format** for lightweight, cross-platform deployment — allowing it to run on any Python version without TensorFlow dependencies.

---

## 🚀 Live Demo

👉 **[Open the App on Streamlit Cloud](https://share.streamlit.io)**

---

## 🧠 How It Works

```
User uploads image
       ↓
Image resized to 224×224 pixels
       ↓
Normalized to [0, 1] range
       ↓
Passed through CNN (ONNX inference)
       ↓
Sigmoid output → threshold at 0.5
       ↓
✅ No Crack  or  ⚠️ Crack Detected
```

---

## 🏗️ Project Structure

```
crack_app_NTI/
├── app.py                  # Main Streamlit application
├── crack_model_v1.onnx     # Converted CNN model (ONNX format, 43MB)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for Streamlit Cloud
└── README.md               # This file
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Frontend / UI** | Streamlit |
| **Model Format** | ONNX (converted from Keras) |
| **Inference Engine** | ONNX Runtime |
| **Language** | Python 3.11+ |
| **Deployment** | Streamlit Cloud |
| **Version Control** | GitHub |

---

## 🔧 Model Details

| Property | Value |
|----------|-------|
| Architecture | CNN (Conv2D + MaxPool + Dense) |
| Input Shape | `(224, 224, 3)` — RGB image |
| Output | Sigmoid score `[0, 1]` |
| Threshold | `0.5` |
| Original Format | Keras `.keras` (~128 MB) |
| Deployed Format | ONNX `.onnx` (~43 MB) |
| Training Framework | TensorFlow / Keras |

---

## 💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Mohamed-Abdelwahab11/crack_app_NTI.git
cd crack_app_NTI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Dependencies

```
streamlit
numpy
Pillow
onnxruntime
```

---

## 🚢 Deployment Notes

The app is deployed on **Streamlit Cloud** with the following considerations:

- **TensorFlow** was dropped in favor of **ONNX Runtime** because Streamlit Cloud runs **Python 3.14**, which TensorFlow does not support yet.
- The original Keras model (`.keras`) was converted to ONNX using `tf2onnx`, preserving full model accuracy.
- The ONNX model is committed directly to the repository (43MB, within GitHub's limit).

---

## 👨‍💻 Author

**Mohamed Abdelwahab**
NTI Training Program — Deep Learning Track

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
