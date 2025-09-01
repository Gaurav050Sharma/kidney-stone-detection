"""
Kidney Stone Detection System - Hugging Face Space
Production version with trained CNN+SVM models
"""

import gradio as gr
import numpy as np
from PIL import Image
import os
import cv2

# Try to import the model, with fallback if it fails
try:
    from model import KidneyStoneDetector
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: model.py not available, using fallback prediction")

# Try to import additional dependencies
try:
    import tensorflow as tf
    import pickle
    from skimage.feature import hog
    TF_AVAILABLE = True
    SKIMAGE_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    SKIMAGE_AVAILABLE = False
    print("Warning: TensorFlow or scikit-image not available")

def load_models():
    """Load trained models with error handling"""
    cnn_model = None
    svm_model = None
    
    try:
        if os.path.exists("kidney_stone_cnn.h5") and TF_AVAILABLE:
            cnn_model = tf.keras.models.load_model("kidney_stone_cnn.h5")
            print("✅ CNN model loaded successfully")
        else:
            print("❌ CNN model file not found or TensorFlow not available")
    except Exception as e:
        print(f"❌ Error loading CNN model: {e}")
    
    try:
        if os.path.exists("kidney_stone_svm.pkl"):
            with open("kidney_stone_svm.pkl", 'rb') as f:
                svm_model = pickle.load(f)
            print("✅ SVM model loaded successfully")
        else:
            print("❌ SVM model file not found")
    except Exception as e:
        print(f"❌ Error loading SVM model: {e}")
    
    return cnn_model, svm_model

# Load models at startup
cnn_model, svm_model = load_models()

def extract_hog_features(image):
    """Extract HOG features for SVM prediction"""
    try:
        if not SKIMAGE_AVAILABLE:
            print("scikit-image not available for HOG features")
            return np.zeros(324)
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (image * 255).astype(np.uint8)
        
        # Resize to the same size used during training
        resized = cv2.resize(gray, (64, 64))
        
        # Calculate HOG features (same parameters as training)
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

def preprocess_image(image, target_size=(150, 150)):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None
    """Preprocess image for model prediction"""
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        
        # Resize image
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_kidney_stone(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "❌ Please upload an image", "Please upload a CT scan image to analyze."
    
    try:
        # Check if models are loaded
        if cnn_model is None and svm_model is None:
            return "❌ Models not loaded", "Trained models could not be loaded. Using fallback prediction."
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return "❌ Image processing failed", "Could not process the uploaded image."
        
        # Prepare image for prediction
        img_batch = np.expand_dims(processed_image, axis=0)
        
        # Get predictions from available models
        cnn_confidence = 0.0
        svm_confidence = 0.0
        prediction = "No Stone"
        
        # CNN prediction
        if cnn_model is not None:
            try:
                cnn_pred = cnn_model.predict(img_batch, verbose=0)
                cnn_confidence = float(cnn_pred[0][0]) * 100
                if cnn_confidence > 50:
                    prediction = "Stone Detected"
            except Exception as e:
                print(f"CNN prediction error: {e}")
                cnn_confidence = 0.0
        
        # SVM prediction (using HOG features)
        if svm_model is not None:
            try:
                # Extract HOG features for SVM (same as training)
                hog_features = extract_hog_features(processed_image)
                hog_features = hog_features.reshape(1, -1)  # Reshape for prediction
                
                svm_pred_proba = svm_model.predict_proba(hog_features)
                svm_confidence = float(svm_pred_proba[0][1]) * 100  # Probability of stone class
                
                # Also get the actual prediction
                svm_pred = svm_model.predict(hog_features)
                if svm_pred[0] == 1:  # Stone detected
                    prediction = "Stone Detected"
                    
            except Exception as e:
                print(f"SVM prediction error: {e}")
                svm_confidence = 0.0
        
        # Ensemble prediction (average of both models)
        if cnn_model is not None and svm_model is not None:
            ensemble_confidence = (cnn_confidence + svm_confidence) / 2
            prediction = "Stone Detected" if ensemble_confidence > 50 else "No Stone"
        elif cnn_model is not None:
            ensemble_confidence = cnn_confidence
        elif svm_model is not None:
            ensemble_confidence = svm_confidence
        else:
            # Fallback: simple image analysis
            gray = cv2.cvtColor((processed_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            mean_intensity = np.mean(gray)
            ensemble_confidence = 75.0 if mean_intensity < 100 else 25.0
            prediction = "Stone Detected" if mean_intensity < 100 else "No Stone"
        
        if prediction == "Stone Detected":
            status = "🔴 KIDNEY STONE DETECTED"
            advice = """
⚠️ **IMPORTANT MEDICAL ADVICE:**
- Consult a urologist immediately for proper diagnosis
- Stay well hydrated (drink plenty of water)
- Avoid foods high in oxalates (spinach, nuts, chocolate)
- Consider pain management options
- Follow up with imaging studies as recommended
            """
        else:
            status = "✅ NO KIDNEY STONES DETECTED"
            advice = """
✅ **PREVENTIVE RECOMMENDATIONS:**
- Maintain adequate hydration (2-3 liters of water daily)
- Follow a balanced, low-sodium diet
- Regular exercise and healthy lifestyle
- Monitor for symptoms and regular check-ups
- Continue preventive care measures
            """
        
        # Determine model status
        model_status = []
        if cnn_model is not None:
            model_status.append("CNN")
        if svm_model is not None:
            model_status.append("SVM")
        
        model_info = "+".join(model_status) if model_status else "Fallback"
        
        prediction_text = f"""
{status}

**Prediction Details:**
- Overall Confidence: {ensemble_confidence:.1f}%
- CNN Model Confidence: {cnn_confidence:.1f}%
- SVM Model Confidence: {svm_confidence:.1f}%
- Status: Production Mode (Trained Models)
- Model: {model_info} Ensemble
        """
        
        return prediction_text, advice
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "An error occurred during analysis. Please try again with a different image."

# Create Gradio interface
with gr.Blocks(title="Kidney Stone Detection System") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🏥 Kidney Stone Detection System</h1>
        <p>Upload a CT scan image to detect kidney stones using AI</p>
        <p><strong>🤖 AI-Powered:</strong> CNN+SVM Ensemble Model for Medical Image Analysis</p>
        <p><strong>⚠️ DISCLAIMER:</strong> This is for educational purposes only. Always consult medical professionals.</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload CT Scan Image",
                type="pil"
            )
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary")
            
        with gr.Column():
            prediction_output = gr.Textbox(
                label="Prediction Results",
                lines=8,
                interactive=False
            )
            advice_output = gr.Textbox(
                label="Medical Advice",
                lines=10,
                interactive=False
            )
    
    predict_btn.click(
        fn=predict_kidney_stone,
        inputs=[image_input],
        outputs=[prediction_output, advice_output]
    )
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd;">
        <p><strong>Important:</strong> This system is for demonstration purposes only.</p>
        <p>Always seek professional medical advice for accurate diagnosis and treatment.</p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
