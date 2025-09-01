"""
Kidney Stone Detection - Hugging Face Spaces Deployment
Optimized for cloud deployment
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
    """Main prediction function for Hugging Face deployment"""
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
• Consult a urologist immediately
• Stay well hydrated (8-10 glasses water daily)
• Avoid high-oxalate foods temporarily
• Consider pain management options
• Schedule follow-up imaging studies"""
        else:
            status = "✅ NO KIDNEY STONES DETECTED"
            advice = """✅ PREVENTIVE RECOMMENDATIONS:
• Maintain adequate hydration
• Follow balanced, low-sodium diet
• Regular exercise routine
• Monitor for any symptoms
• Continue preventive healthcare"""
        
        prediction_text = f"""{status}

Overall Confidence: {confidence * 100:.1f}%
CNN Model: {cnn_confidence * 100:.1f}%
SVM Model: {svm_confidence * 100:.1f}%"""
        
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
    description="AI-powered medical image analysis for CT scans. Upload an image to detect kidney stones using our trained CNN+SVM ensemble model achieving 100% training accuracy.",
    article="⚠️ **Medical Disclaimer**: This is for educational purposes only. Always consult healthcare professionals for medical diagnosis. Developed by Gaurav Sharma.",
    theme="soft",
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
