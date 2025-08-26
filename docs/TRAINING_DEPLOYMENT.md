# 🧠 Training Guide

## Complete Model Training Pipeline

### 1. Environment Setup

```bash
# Activate virtual environment
kidney_stone_env\Scripts\activate

# Verify installations
python -c "import tensorflow; print(f'TensorFlow: {tensorflow.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
```

### 2. Dataset Preparation

#### Required Structure:
```
data/
├── train/
│   ├── Normal/     # Normal CT scans (at least 1000 images)
│   └── Stone/      # Stone CT scans (at least 1000 images)
└── test/
    ├── Normal/     # Test normal images (at least 200 images)
    └── Stone/      # Test stone images (at least 200 images)
```

#### Data Requirements:
- **Format**: JPEG, PNG, or DICOM
- **Size**: Minimum 150x150 pixels
- **Quality**: High-resolution medical images
- **Labeling**: Verified by medical professionals

### 3. Training Execution

#### Automatic Training:
```bash
python -m src.training.train_models
```

#### Manual Training Steps:
```python
from src.training.train_models import TrainingPipeline

# Initialize training pipeline
trainer = TrainingPipeline()

# Run complete training
trainer.run_complete_training()
```

### 4. Training Parameters

#### CNN Configuration:
```python
MODEL_CONFIG = {
    'cnn': {
        'input_shape': (150, 150, 3),
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'validation_split': 0.2,
        'early_stopping_patience': 10
    }
}
```

#### SVM Configuration:
```python
MODEL_CONFIG = {
    'svm': {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'random_state': 42
    }
}
```

### 5. Training Monitoring

Monitor training progress through:
- **Console Output**: Real-time metrics
- **TensorBoard**: Visualization (if enabled)
- **Model Checkpoints**: Best model saving
- **Validation Metrics**: Performance tracking

### 6. Model Evaluation

After training completion:
```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.evaluate_all_models()
```

### 7. Troubleshooting

#### Common Issues:
- **Memory Error**: Reduce batch size
- **Convergence Issues**: Adjust learning rate
- **Overfitting**: Increase dropout, add regularization
- **Data Imbalance**: Use class weights

---

# 🚀 Deployment Guide

## Hugging Face Spaces Deployment

### 1. Repository Setup

```bash
# Clone Hugging Face repository
git clone https://huggingface.co/spaces/your-username/kidney-stone-detector
cd kidney-stone-detector

# Copy project files
cp -r /path/to/project/* .
```

### 2. Hugging Face Configuration

File: `README.md` (Header)
```yaml
---
title: Kidney Stone Detection
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.50.0
app_file: app.py
pinned: false
license: mit
---
```

### 3. Deployment Steps

```bash
# Add files
git add .

# Commit changes
git commit -m "Deploy kidney stone detection system"

# Push to Hugging Face
git push
```

### 4. Verification

- Check deployment status on Hugging Face Spaces
- Test web interface functionality
- Verify model loading and predictions

## GitHub Integration

### 1. Repository Creation

```bash
git init
git remote add origin https://github.com/your-username/kidney-stone-detector.git
```

### 2. Documentation

Ensure these files are included:
- `README.md`: Project overview
- `docs/DOCUMENTATION.md`: Detailed documentation
- `requirements.txt`: Dependencies
- `LICENSE`: MIT license

### 3. GitHub Actions (Optional)

Create `.github/workflows/deploy.yml` for automated deployment.

## Local Deployment

### 1. Production Setup

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn app:app -b 0.0.0.0:8000
```

### 2. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 7860

CMD ["python", "app.py"]
```

```bash
docker build -t kidney-stone-detector .
docker run -p 7860:7860 kidney-stone-detector
```

---

# 🔧 Configuration Options

## Environment Variables

```bash
# Model paths
export MODEL_CNN_PATH="models/kidney_stone_cnn.h5"
export MODEL_SVM_PATH="models/kidney_stone_svm.pkl"

# Data paths
export DATA_PATH="data/"
export UPLOAD_PATH="uploads/"

# Application settings
export GRADIO_SHARE=false
export GRADIO_DEBUG=false
```

## Custom Configuration

Edit `config/config.py` for:
- Model parameters
- Data paths
- Medical advice templates
- UI customization

---

# 📊 Monitoring & Maintenance

## Performance Monitoring

- **Response Time**: Track prediction latency
- **Accuracy**: Monitor prediction accuracy
- **Usage**: Track user interactions
- **Errors**: Log and monitor errors

## Model Updates

- **Retraining**: Regular model retraining
- **Version Control**: Model versioning
- **A/B Testing**: Compare model versions
- **Rollback**: Ability to revert changes

## Security Considerations

- **Input Validation**: Validate uploaded images
- **Rate Limiting**: Prevent abuse
- **Data Privacy**: Secure image handling
- **HIPAA Compliance**: Medical data protection

---

*For additional support, refer to the main documentation or create an issue on GitHub.*
