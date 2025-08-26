"""
Kidney Stone Detection Models
Simple consolidated model file with CNN and SVM implementations
"""

import os
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

class KidneyStoneDetector:
    """Main class for kidney stone detection using CNN and SVM models"""
    
    def __init__(self):
        self.cnn_model = None
        self.svm_model = None
        self.image_size = (150, 150)
        
    def build_cnn_model(self):
        """Build CNN model architecture"""
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.cnn_model = model
        return model
    
    def extract_hog_features(self, image):
        """Extract HOG features for SVM using scikit-image HOG"""
        try:
            from skimage.feature import hog
            from skimage import exposure
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Resize image
            resized = cv2.resize(gray, (64, 64))
            
            # Calculate HOG features using scikit-image
            features = hog(resized, 
                          orientations=9, 
                          pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), 
                          visualize=False,
                          feature_vector=True)
            
            return features
        except Exception as e:
            print(f"Error extracting HOG features: {e}")
            # Return zero vector as fallback
            return np.zeros(324)  # 9*2*2*9 = 324 features
    
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess training data"""
        images = []
        labels = []
        hog_features = []
        
        # Load Normal images
        normal_dir = os.path.join(data_dir, 'Normal')
        if os.path.exists(normal_dir):
            for filename in os.listdir(normal_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(normal_dir, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img, self.image_size)
                        
                        images.append(img_resized)
                        labels.append(0)  # 0 for Normal
                        hog_features.append(self.extract_hog_features(img_resized))
        
        # Load Stone images
        stone_dir = os.path.join(data_dir, 'Stone')
        if os.path.exists(stone_dir):
            for filename in os.listdir(stone_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(stone_dir, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img, self.image_size)
                        
                        images.append(img_resized)
                        labels.append(1)  # 1 for Stone
                        hog_features.append(self.extract_hog_features(img_resized))
        
        return np.array(images), np.array(labels), np.array(hog_features)
    
    def train_models(self, data_dir):
        """Train both CNN and SVM models"""
        print("Loading and preprocessing data...")
        images, labels, hog_features = self.load_and_preprocess_data(data_dir)
        
        if len(images) == 0:
            print("No images found! Please check your data directory structure.")
            return
        
        print(f"Loaded {len(images)} images")
        print(f"Normal: {np.sum(labels == 0)}, Stone: {np.sum(labels == 1)}")
        
        # Normalize images for CNN
        images_normalized = images.astype('float32') / 255.0
        
        # Split data
        X_train_cnn, X_test_cnn, y_train, y_test = train_test_split(
            images_normalized, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        X_train_svm, X_test_svm, _, _ = train_test_split(
            hog_features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train CNN
        print("\nTraining CNN model...")
        if self.cnn_model is None:
            self.build_cnn_model()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )
        
        history = self.cnn_model.fit(
            X_train_cnn, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate CNN
        cnn_predictions = (self.cnn_model.predict(X_test_cnn) > 0.5).astype(int)
        print("\nCNN Performance:")
        print(classification_report(y_test, cnn_predictions))
        
        # Train SVM
        print("\nTraining SVM model...")
        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(X_train_svm, y_train)
        
        # Evaluate SVM
        svm_predictions = self.svm_model.predict(X_test_svm)
        print("\nSVM Performance:")
        print(classification_report(y_test, svm_predictions))
        
        # Save models
        print("\nSaving models...")
        os.makedirs('models', exist_ok=True)
        self.cnn_model.save('models/kidney_stone_cnn.h5')
        
        with open('models/kidney_stone_svm.pkl', 'wb') as f:
            pickle.dump(self.svm_model, f)
        
        print("Training completed and models saved!")
        return history
    
    def load_models(self):
        """Load trained models"""
        try:
            self.cnn_model = keras.models.load_model('models/kidney_stone_cnn.h5')
            with open('models/kidney_stone_svm.pkl', 'rb') as f:
                self.svm_model = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_image(self, image_path):
        """Predict kidney stone presence in image"""
        if self.cnn_model is None or self.svm_model is None:
            if not self.load_models():
                return None
        
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, self.image_size)
            
            # CNN prediction
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            cnn_prob = self.cnn_model.predict(img_batch)[0][0]
            
            # SVM prediction
            hog_features = self.extract_hog_features(img_resized).reshape(1, -1)
            svm_prob = self.svm_model.predict_proba(hog_features)[0][1]
            
            # Ensemble prediction
            ensemble_prob = (cnn_prob + svm_prob) / 2
            prediction = "Stone Detected" if ensemble_prob > 0.5 else "Normal"
            
            return {
                'prediction': prediction,
                'confidence': float(ensemble_prob),
                'cnn_confidence': float(cnn_prob),
                'svm_confidence': float(svm_prob)
            }
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None

# Medical advice templates
MEDICAL_ADVICE = {
    'stone_detected': """
    ⚠️ **STONE DETECTED - SEEK MEDICAL ATTENTION**
    
    **Immediate Actions:**
    • Contact your healthcare provider immediately
    • Stay hydrated (drink plenty of water)
    • Avoid foods high in oxalates temporarily
    
    **Recommended Next Steps:**
    • Schedule appointment with urologist
    • Request complete urinalysis
    • Consider CT scan for stone size/location
    
    **Pain Management:**
    • Over-the-counter pain relievers may help
    • Apply heat to affected area
    • Gentle movement may assist stone passage
    
    **Emergency Signs - Go to ER if you experience:**
    • Severe, unbearable pain
    • Blood in urine
    • Fever or chills
    • Inability to urinate
    
    ⚕️ **This is an AI analysis. Professional medical diagnosis is required.**
    """,
    
    'normal': """
    ✅ **NORMAL SCAN - No Stones Detected**
    
    **Preventive Measures:**
    • Maintain adequate hydration (8-10 glasses water daily)
    • Balanced diet with moderate calcium intake
    • Limit sodium and animal protein
    • Regular exercise
    
    **Continue Monitoring:**
    • Annual check-ups with healthcare provider
    • Report any urinary symptoms promptly
    • Maintain healthy lifestyle habits
    
    **When to Consult Doctor:**
    • Persistent back or side pain
    • Changes in urination patterns
    • Blood in urine
    • Recurring UTIs
    
    ⚕️ **This AI analysis doesn't replace regular medical check-ups.**
    """
}
