"""
Prediction utilities and medical advice generation
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import random
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.model_architectures import CNNModel, SVMModel, EnsemblePredictor
from config.config import MODEL_CONFIG, MEDICAL_ADVICE

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handle image preprocessing for predictions"""
    
    def __init__(self, target_size=(150, 150)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target image size for model input
        """
        self.target_size = target_size
    
    def preprocess_image(self, image):
        """
        Preprocess image for model prediction
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Preprocessed image array ready for prediction
        """
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image is in RGB format
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:  # Already RGB or BGR
                    # Convert BGR to RGB if needed (OpenCV default is BGR)
                    if np.mean(image[:,:,0]) > np.mean(image[:,:,2]):  # Simple heuristic
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize image
            image = cv2.resize(image, self.target_size)
            
            # Normalize pixel values to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None


class MedicalAdvisor:
    """Generate medical advice based on predictions"""
    
    def __init__(self):
        """Initialize medical advisor with predefined advice"""
        self.advice = MEDICAL_ADVICE
    
    def estimate_stone_size(self):
        """
        Estimate stone size (simulated for demo purposes)
        
        Returns:
            Estimated stone size in cm
        """
        # In a real application, this would be based on image analysis
        stone_size = round(random.uniform(0.3, 4.5), 1)
        return stone_size
    
    def get_risk_level(self, confidence, stone_size=None):
        """
        Determine risk level based on confidence and stone size
        
        Args:
            confidence: Prediction confidence
            stone_size: Estimated stone size
            
        Returns:
            Risk level string
        """
        if confidence > 0.8:
            if stone_size and stone_size > 3.0:
                return "High Risk"
            else:
                return "Moderate Risk"
        elif confidence > 0.6:
            return "Moderate Risk"
        else:
            return "Low Risk"
    
    def generate_positive_advice(self, confidence, stone_size=None):
        """
        Generate advice for positive (stone detected) cases
        
        Args:
            confidence: Prediction confidence
            stone_size: Estimated stone size
            
        Returns:
            Formatted medical advice string
        """
        risk_level = self.get_risk_level(confidence, stone_size)
        
        advice_parts = []
        
        # Header
        advice_parts.append("🔴 KIDNEY STONE DETECTED")
        advice_parts.append(f"Confidence Level: {confidence*100:.1f}%")
        advice_parts.append(f"Risk Level: {risk_level}")
        
        if stone_size:
            advice_parts.append(f"Estimated Stone Size: {stone_size} cm")
        
        advice_parts.append("\n" + "="*40)
        
        # Immediate care instructions
        advice_parts.append("🚨 IMMEDIATE CARE INSTRUCTIONS:")
        for instruction in self.advice['positive_advice']['immediate_care']:
            advice_parts.append(f"• {instruction}")
        
        advice_parts.append("\n📋 DIETARY RECOMMENDATIONS:")
        for recommendation in self.advice['positive_advice']['dietary_recommendations']:
            advice_parts.append(f"• {recommendation}")
        
        # Warning signs
        advice_parts.append("\n⚠️ SEEK EMERGENCY CARE IF:")
        for warning in self.advice['positive_advice']['warning_signs']:
            advice_parts.append(f"• {warning}")
        
        # Disclaimer
        advice_parts.append("\n" + "="*40)
        advice_parts.append("⚕️ MEDICAL DISCLAIMER:")
        advice_parts.append(self.advice['disclaimer'])
        
        return "\n".join(advice_parts)
    
    def generate_negative_advice(self, confidence):
        """
        Generate advice for negative (no stone detected) cases
        
        Args:
            confidence: Prediction confidence
            
        Returns:
            Formatted medical advice string
        """
        advice_parts = []
        
        # Header
        advice_parts.append("✅ NO KIDNEY STONE DETECTED")
        advice_parts.append(f"Confidence Level: {confidence*100:.1f}%")
        advice_parts.append("\n" + "="*40)
        
        # Prevention advice
        advice_parts.append("🛡️ PREVENTION RECOMMENDATIONS:")
        for prevention in self.advice['negative_advice']['prevention']:
            advice_parts.append(f"• {prevention}")
        
        advice_parts.append("\n💡 GENERAL HEALTH TIPS:")
        for tip in self.advice['negative_advice']['general_tips']:
            advice_parts.append(f"• {tip}")
        
        # Disclaimer
        advice_parts.append("\n" + "="*40)
        advice_parts.append("⚕️ MEDICAL DISCLAIMER:")
        advice_parts.append(self.advice['disclaimer'])
        
        return "\n".join(advice_parts)


class KidneyStonePredictor:
    """Main prediction class combining all components"""
    
    def __init__(self, model_dir="saved_models"):
        """
        Initialize predictor with model paths
        
        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir
        self.preprocessor = ImagePreprocessor()
        self.medical_advisor = MedicalAdvisor()
        
        # Models (will be loaded on first prediction)
        self.cnn_model = None
        self.svm_model = None
        self.ensemble_predictor = None
        self._models_loaded = False
    
    def load_models(self):
        """Load trained models"""
        try:
            # Load CNN model
            cnn_path = os.path.join(self.model_dir, "cnn_model.h5")
            if os.path.exists(cnn_path):
                self.cnn_model = CNNModel(MODEL_CONFIG)
                self.cnn_model.load_model(cnn_path)
                logger.info("CNN model loaded successfully")
            else:
                logger.error(f"CNN model not found at {cnn_path}")
                return False
            
            # Load SVM model
            svm_path = os.path.join(self.model_dir, "svm_model.pkl")
            if os.path.exists(svm_path):
                self.svm_model = SVMModel(MODEL_CONFIG)
                self.svm_model.load_model(svm_path)
                logger.info("SVM model loaded successfully")
            else:
                logger.error(f"SVM model not found at {svm_path}")
                return False
            
            # Create ensemble predictor
            self.ensemble_predictor = EnsemblePredictor(
                self.cnn_model, 
                self.svm_model
            )
            
            self._models_loaded = True
            logger.info("All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def predict(self, image):
        """
        Make prediction on input image
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with prediction results and medical advice
        """
        # Load models if not already loaded
        if not self._models_loaded:
            if not self.load_models():
                return {
                    'error': 'Failed to load models. Please ensure models are trained and saved.'
                }
        
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image)
        if processed_image is None:
            return {
                'error': 'Failed to preprocess image. Please ensure image is valid.'
            }
        
        try:
            # Make ensemble prediction
            prediction_results = self.ensemble_predictor.predict_combined(processed_image)
            
            # Extract results
            is_stone_detected = prediction_results['prediction']
            confidence = prediction_results['confidence']
            
            # Generate medical advice
            if is_stone_detected:
                stone_size = self.medical_advisor.estimate_stone_size()
                medical_advice = self.medical_advisor.generate_positive_advice(
                    confidence, stone_size
                )
                diagnosis = "Positive - Kidney Stone Detected"
            else:
                stone_size = None
                medical_advice = self.medical_advisor.generate_negative_advice(confidence)
                diagnosis = "Negative - No Kidney Stone Detected"
            
            return {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'stone_detected': is_stone_detected,
                'stone_size': stone_size,
                'medical_advice': medical_advice,
                'cnn_prediction': prediction_results['cnn_prediction'],
                'svm_prediction': prediction_results['svm_prediction'],
                'cnn_confidence': prediction_results['cnn_confidence']
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'error': f'Prediction failed: {str(e)}'
            }
