"""
Kidney Stone Detection - Working Gradio Interface
"""

import gradio as gr
import os
import tempfile
import numpy as np
from PIL import Image
from model import KidneyStoneDetector

# Initialize detector
detector = KidneyStoneDetector()

def predict_kidney_stone(image):
    """Main prediction function"""
    if image is None:
        return "Please upload an image", "Please upload a CT scan image to analyze."
    
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            temp_path = tmp_file.name
        
        # Make prediction
        result = detector.predict_image(temp_path)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if result is None:
            return "Prediction Error", "Could not analyze the image."
        
        # Format results
        prediction = result['prediction']
        confidence = result['confidence']
        cnn_confidence = result['cnn_confidence']
        svm_confidence = result['svm_confidence']
        
        if prediction == "Stone Detected":
            status = "🔴 KIDNEY STONE DETECTED"
            advice = """⚠️ IMPORTANT MEDICAL ADVICE:
- Consult a urologist immediately
- Stay well hydrated
- Avoid high-oxalate foods
- Consider pain management
- Follow up with imaging studies"""
        else:
            status = "✅ NO KIDNEY STONES DETECTED"
            advice = """✅ PREVENTIVE RECOMMENDATIONS:
- Maintain adequate hydration
- Follow balanced, low-sodium diet
- Regular exercise
- Monitor for symptoms
- Continue preventive care"""
        
        prediction_text = f"""{status}

Overall Confidence: {confidence * 100:.2f}%
CNN Model: {cnn_confidence * 100:.2f}%
SVM Model: {svm_confidence * 100:.2f}%"""
        
        return prediction_text, advice
        
    except Exception as e:
        return f"Error: {str(e)}", "An error occurred during analysis."

# Create interface
demo = gr.Interface(
    fn=predict_kidney_stone,
    inputs=gr.Image(type="pil", label="Upload CT Scan Image"),
    outputs=[
        gr.Textbox(label="🔍 Prediction Result", lines=5),
        gr.Textbox(label="⚕️ Medical Advice", lines=8)
    ],
    title="🏥 Kidney Stone Detection System",
    description="AI-powered medical image analysis for CT scans. Upload an image to detect kidney stones using our trained CNN+SVM ensemble model.",
    article="⚠️ **Disclaimer**: This is for educational purposes only. Always consult healthcare professionals for medical diagnosis.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
