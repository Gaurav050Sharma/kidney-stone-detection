# Kidney Stone Detection System 🏥

An AI-powered medical imaging system for detecting kidney stones in CT scans using CNN+SVM ensemble learning.

## 🚀 Quick Start

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_USERNAME/kidney-stone-detection)

## ✨ Features

- **AI-Powered Detection**: CNN + SVM ensemble achieving 100% training accuracy
- **Real-time Analysis**: Results in under 2 seconds
- **Medical Interface**: Professional advice and confidence scoring
- **Educational Tool**: For learning medical image analysis

## 🧠 Model Architecture

- **CNN Model**: Deep learning for complex pattern recognition
- **SVM Model**: Traditional ML for texture analysis using HOG features
- **Ensemble Prediction**: Combined approach for higher accuracy
- **Training Data**: 4,540 CT scan images (3,565 Normal, 975 Stone)

## 📁 Project Structure

```
├── app.py                 # Main Gradio interface
├── model.py              # AI models (CNN + SVM)
├── train.py              # Training script
├── utils.py              # Helper functions
└── models/               # Trained model files
    ├── kidney_stone_cnn.h5
    └── kidney_stone_svm.pkl
```

## 🛠️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/kidney-stone-detection.git
cd kidney-stone-detection
pip install -r requirements.txt
python app.py
```

## 📊 Performance

- **Training Accuracy**: 100%
- **Model Types**: CNN + SVM Ensemble
- **Input Size**: 150x150 RGB images
- **Processing Time**: <2 seconds per image

## ⚠️ Medical Disclaimer

This system is for **educational purposes only**. Always consult qualified healthcare professionals for medical diagnosis and treatment. AI analysis should never replace professional medical examination.

## 🔬 Technical Details

### CNN Architecture
- 3 Convolutional layers with BatchNormalization
- Global Average Pooling
- Dense layers with Dropout
- Sigmoid activation for binary classification

### SVM Features
- HOG (Histogram of Oriented Gradients) feature extraction
- RBF kernel with probability estimates
- 324 feature dimensions

### Ensemble Method
- Simple average of CNN and SVM probabilities
- Threshold: 0.5 for stone detection
- Confidence scoring for medical assessment

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## 📧 Contact

For questions or collaboration: [Your Email]

---

**Built with ❤️ for medical education and AI learning**
