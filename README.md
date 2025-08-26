---
title: Kidney Stone Detection
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
---

# 🏥 Kidney Stone Detection System

A simple AI tool for analyzing CT scans to detect kidney stones using machine learning.

## 🚀 Quick Start

1. **Upload a CT scan image**
2. **Click "Analyze Image"**
3. **Review the AI analysis and medical advice**

## 🤖 How It Works

- **CNN Model**: Deep learning image analysis
- **SVM Model**: Traditional machine learning with HOG features  
- **Ensemble**: Combines both models for better accuracy

## 📂 File Structure

```
├── app.py          # Gradio web interface
├── model.py        # AI models (CNN + SVM)
├── utils.py        # Helper functions
├── train.py        # Training script
├── data/           # Your CT scan dataset
└── models/         # Saved trained models
```

## 🎓 Training Your Own Models

1. Create data structure:
   ```
   data/train/
   ├── Normal/     # Normal CT scans
   └── Stone/      # Stone CT scans
   ```

2. Run training:
   ```bash
   python train.py
   ```

3. Launch interface:
   ```bash
   python app.py
   ```

## ⚠️ Medical Disclaimer

**For educational/research purposes only.** 

Not intended for medical diagnosis. Always consult healthcare professionals for proper medical advice.

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.6+
- Gradio 3.50+
- scikit-learn 1.1+
