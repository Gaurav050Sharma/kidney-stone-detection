"""
Utility functions for kidney stone detection
Simple helper functions consolidated
"""

import os
import cv2
import numpy as np
from PIL import Image
import tempfile

def preprocess_uploaded_image(uploaded_file):
    """Process uploaded image for prediction"""
    try:
        # Handle different input types
        if hasattr(uploaded_file, 'name'):
            # Gradio NamedString object
            image_path = uploaded_file.name
            image = cv2.imread(image_path)
        else:
            # File path string
            image = cv2.imread(uploaded_file)
        
        if image is None:
            return None
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        image = cv2.resize(image, (150, 150))
        
        return image
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def save_temp_image(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        if uploaded_file is None:
            return None
            
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            # Handle PIL Image
            if isinstance(uploaded_file, Image.Image):
                uploaded_file.save(tmp.name)
                return tmp.name
            # Handle file path
            elif isinstance(uploaded_file, str):
                return uploaded_file
            # Handle Gradio file
            else:
                return uploaded_file.name
                
    except Exception as e:
        print(f"Error saving temporary image: {e}")
        return None

def format_prediction_result(result):
    """Format prediction result for display"""
    if result is None:
        return "❌ Error processing image"
    
    prediction = result['prediction']
    confidence = result['confidence']
    
    if prediction == "Stone Detected":
        emoji = "⚠️"
        color = "red"
    else:
        emoji = "✅"
        color = "green"
    
    formatted_result = f"""
    {emoji} **{prediction}**
    
    **Confidence Score:** {confidence:.1%}
    
    **Individual Model Scores:**
    • CNN Model: {result['cnn_confidence']:.1%}
    • SVM Model: {result['svm_confidence']:.1%}
    """
    
    return formatted_result

def get_medical_advice(prediction):
    """Get medical advice based on prediction"""
    from model import MEDICAL_ADVICE
    
    if prediction == "Stone Detected":
        return MEDICAL_ADVICE['stone_detected']
    else:
        return MEDICAL_ADVICE['normal']

def validate_data_structure(data_path):
    """Validate that data directory has correct structure"""
    required_dirs = ['Normal', 'Stone']
    
    if not os.path.exists(data_path):
        return False, f"Data directory not found: {data_path}"
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        return False, f"Missing directories: {missing_dirs}"
    
    # Check if directories have images
    total_images = 0
    for dir_name in required_dirs:
        dir_path = os.path.join(data_path, dir_name)
        image_count = len([f for f in os.listdir(dir_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        total_images += image_count
        
        if image_count == 0:
            return False, f"No images found in {dir_name} directory"
    
    return True, f"Data structure valid. Found {total_images} total images."

def create_data_directories():
    """Create required data directory structure"""
    dirs_to_create = [
        'data/train/Normal',
        'data/train/Stone',
        'data/test/Normal', 
        'data/test/Stone',
        'models'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Created data directory structure:")
    print("data/")
    print("├── train/")
    print("│   ├── Normal/")
    print("│   └── Stone/")
    print("└── test/")
    print("    ├── Normal/")
    print("    └── Stone/")
    print("\nPlace your CT scan images in the appropriate folders.")

def check_models_exist():
    """Check if trained models exist"""
    cnn_path = 'models/kidney_stone_cnn.h5'
    svm_path = 'models/kidney_stone_svm.pkl'
    
    cnn_exists = os.path.exists(cnn_path)
    svm_exists = os.path.exists(svm_path)
    
    return cnn_exists and svm_exists, cnn_exists, svm_exists
