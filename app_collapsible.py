"""
Kidney Stone Detection - Interface with Collapsible Results
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
        return "Please upload an image", "Please upload a CT scan image to analyze.", gr.update(visible=False)
    
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
            return "Prediction Error", "Could not analyze the image.", gr.update(visible=True)
        
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

Overall Confidence: {confidence:.2f}%
CNN Model: {cnn_confidence:.2f}%
SVM Model: {svm_confidence:.2f}%"""
        
        return prediction_text, advice, gr.update(visible=True)
        
    except Exception as e:
        return f"Error: {str(e)}", "An error occurred during analysis.", gr.update(visible=True)

# Create custom CSS for better styling
css = """
.upload-area {
    border: 2px dashed #007acc !important;
    border-radius: 10px !important;
    padding: 20px !important;
    text-align: center !important;
}

.results-container {
    border: 1px solid #e0e0e0 !important;
    border-radius: 10px !important;
    padding: 15px !important;
    margin-top: 20px !important;
    background-color: #f9f9f9 !important;
}

.prediction-box {
    background-color: #e8f4fd !important;
    padding: 15px !important;
    border-radius: 8px !important;
    margin-bottom: 10px !important;
}

.advice-box {
    background-color: #fff3cd !important;
    padding: 15px !important;
    border-radius: 8px !important;
}
"""

# Create Blocks interface
with gr.Blocks(css=css, title="Kidney Stone Detection") as demo:
    
    gr.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1 style='margin: 0; font-size: 2.5em;'>🏥 Kidney Stone Detection System</h1>
            <p style='margin: 10px 0 0 0; font-size: 1.2em;'>AI-powered medical image analysis for CT scans</p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload CT Scan Image")
            image_input = gr.Image(
                type="pil", 
                label="", 
                elem_classes="upload-area"
            )
            
            analyze_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary", 
                size="lg",
                elem_id="analyze-button"
            )
            
            gr.Markdown("""
            **Instructions:**
            1. Upload a CT scan image (JPG, PNG)
            2. Click 'Analyze Image' to get results
            3. View detailed predictions and advice below
            """)
        
        with gr.Column(scale=1):
            # Results section - initially hidden
            with gr.Group(visible=False) as results_group:
                gr.Markdown("### 📊 Analysis Results")
                
                prediction_output = gr.Textbox(
                    label="🔍 Prediction Result",
                    lines=6,
                    elem_classes="prediction-box"
                )
                
                advice_output = gr.Textbox(
                    label="⚕️ Medical Advice", 
                    lines=8,
                    elem_classes="advice-box"
                )
                
                gr.Markdown("""
                **Note:** Results are automatically revealed after analysis.
                """)
    
    # Footer disclaimer
    gr.HTML("""
        <div style='text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 8px; margin-top: 20px; border-left: 4px solid #dc3545;'>
            <p style='margin: 0; color: #721c24; font-weight: bold;'>
                ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
                Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.
            </p>
        </div>
    """)
    
    # Event handler
    analyze_btn.click(
        fn=predict_kidney_stone,
        inputs=[image_input],
        outputs=[prediction_output, advice_output, results_group]
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="127.0.0.1", server_port=7861)
