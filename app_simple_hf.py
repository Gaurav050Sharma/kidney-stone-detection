"""
Kidney Stone Detection System - Hugging Face Space
Simple demo version for deployment
"""

import gradio as gr
import numpy as np
from PIL import Image

def predict_kidney_stone(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "❌ Please upload an image", "Please upload a CT scan image to analyze."
    
    try:
        # For demo purposes, simulate a prediction
        # In production, this would use your trained CNN+SVM models
        
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Simple simulation based on image properties
        img_array = np.array(img.convert('L'))  # Convert to grayscale
        mean_intensity = np.mean(img_array)
        
        # Simulate prediction logic
        if mean_intensity < 100:
            prediction = "Stone Detected"
            confidence = 85.6
        else:
            prediction = "No Stone"
            confidence = 92.3
        
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
- Status: Demo Mode (Simulated Results)
- Note: This is a demonstration version
        """
        
        return prediction_text, advice
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "An error occurred during analysis."

# Create Gradio interface
with gr.Blocks(title="Kidney Stone Detection System") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🏥 Kidney Stone Detection System</h1>
        <p>Upload a CT scan image to detect kidney stones using AI</p>
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
