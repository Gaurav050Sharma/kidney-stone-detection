"""
Simple Kidney Stone Detection Interface
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
            if isinstance(image, np.ndarray):
                img = Image.fromarray(image)
            else:
                img = image
            img.save(tmp_file.name)
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
        
        if prediction == "Stone Detected":
            status = "🔴 KIDNEY STONE DETECTED"
            advice = "⚠️ Consult a urologist immediately. Stay hydrated and follow medical advice."
        else:
            status = "✅ NO KIDNEY STONES DETECTED"
            advice = "✅ Maintain healthy lifestyle and regular check-ups."
        
        prediction_text = f"{status}\nConfidence: {confidence:.2f}%"
        
        return prediction_text, advice
        
    except Exception as e:
        return f"Error: {str(e)}", "An error occurred during analysis."

# Create simple interface
demo = gr.Interface(
    fn=predict_kidney_stone,
    inputs=gr.Image(type="pil", label="Upload CT Scan"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Textbox(label="Medical Advice")
    ],
    title="🏥 Kidney Stone Detection",
    description="Upload a CT scan image to detect kidney stones using AI",
    examples=None
)

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7860)
