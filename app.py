"""
Kidney Stone Detection System - Hugging Face Space
Production version with trained CNN+SVM models
"""

import gradio as gr
import numpy as np
from PIL import Image
import os
from model import KidneyStoneDetector

# Initialize the detector with trained models
detector = KidneyStoneDetector()

def predict_kidney_stone(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "❌ Please upload an image", "Please upload a CT scan image to analyze."
    
    try:
        # Check if models are available
        if not os.path.exists("kidney_stone_cnn.h5") or not os.path.exists("kidney_stone_svm.pkl"):
            return "❌ Models not found", "Trained models are not available. Please ensure models are deployed."
        
        # Process the image with actual trained models
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Save image temporarily for processing
        temp_path = "temp_image.jpg"
        img.save(temp_path)
        
        # Use the actual trained models for prediction
        result = detector.predict_image(temp_path)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if result is None:
            return "❌ Prediction Error", "Could not analyze the image. Please try again."
        
        # Extract results from the trained models
        prediction = result['prediction']
        confidence = result['confidence']
        cnn_confidence = result.get('cnn_confidence', confidence)
        svm_confidence = result.get('svm_confidence', confidence)
        
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
        
        prediction_text = f"""
{status}

**Prediction Details:**
- Overall Confidence: {confidence:.1f}%
- CNN Model Confidence: {cnn_confidence:.1f}%
- SVM Model Confidence: {svm_confidence:.1f}%
- Status: Production Mode (Trained Models)
- Model: CNN+SVM Ensemble
        """
        
        return prediction_text, advice
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "An error occurred during analysis. Please try again."

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
