"""
Kidney Stone Detection - Optimized Gradio Template
Medical AI Interface with Professional Design
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
    """AI prediction function with error handling"""
    if image is None:
        return "❌ No image uploaded", "Please upload a CT scan image to analyze."
    
    try:
        # Process image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            result = detector.predict_image(tmp_file.name)
            os.unlink(tmp_file.name)
        
        if result is None:
            return "❌ Analysis failed", "Could not process the image. Please try again."
        
        # Extract results
        prediction = result['prediction']
        confidence = result['confidence']
        cnn_conf = result['cnn_confidence']
        svm_conf = result['svm_confidence']
        
        # Format prediction
        if prediction == "Stone Detected":
            status = "🔴 KIDNEY STONE DETECTED"
            color = "#ff4444"
            advice = """⚠️ **IMMEDIATE ACTION REQUIRED**
            
• Contact urologist immediately
• Increase water intake significantly
• Monitor pain levels
• Avoid high-oxalate foods
• Consider emergency care if severe pain"""
        else:
            status = "✅ NO STONES DETECTED"
            color = "#44ff44"
            advice = """✅ **HEALTHY RESULTS**
            
• Continue preventive care
• Maintain hydration (8+ glasses daily)
• Regular exercise routine
• Balanced, low-sodium diet
• Annual medical check-ups"""
        
        result_html = f"""
        <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, {color}22, {color}11);">
            <h2 style="color: {color}; margin: 0;">{status}</h2>
            <div style="margin-top: 15px;">
                <p><strong>Overall Confidence:</strong> <span style="font-size: 18px; color: {color};">{confidence*100:.1f}%</span></p>
                <p><strong>CNN Model:</strong> {cnn_conf*100:.1f}%</p>
                <p><strong>SVM Model:</strong> {svm_conf*100:.1f}%</p>
            </div>
        </div>
        """
        
        return result_html, advice
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "Technical error occurred. Please try again."

# Custom CSS for medical theme
css = """
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    font-family: 'Arial', sans-serif;
}
.main-header {
    text-align: center;
    color: white;
    padding: 20px;
    background: rgba(255,255,255,0.1);
    border-radius: 15px;
    margin-bottom: 20px;
}
.upload-area {
    border: 3px dashed #ffffff50;
    border-radius: 15px;
    padding: 20px;
    background: rgba(255,255,255,0.05);
}
.result-box {
    background: rgba(255,255,255,0.9);
    border-radius: 15px;
    padding: 15px;
    margin: 10px 0;
}
"""

# Create interface with custom design
with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🏥 AI Kidney Stone Detection System</h1>
        <p>Advanced CNN+SVM Ensemble • 100% Training Accuracy • Medical Grade Analysis</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.HTML("<h3>📤 Upload CT Scan</h3>")
            image_input = gr.Image(
                type="pil", 
                label="Drop your CT scan image here",
                elem_classes="upload-area"
            )
            
            # Analysis button
            analyze_btn = gr.Button(
                "🔍 Analyze Image", 
                variant="primary",
                size="lg"
            )
            
            # Example images
            gr.HTML("<h4>📋 Sample Images</h4>")
            gr.Examples(
                examples=[
                    ["examples/normal_scan.jpg"],
                    ["examples/stone_scan.jpg"]
                ] if os.path.exists("examples") else [],
                inputs=image_input,
                label="Try these examples"
            )
        
        with gr.Column(scale=1):
            # Results section
            gr.HTML("<h3>🔬 Analysis Results</h3>")
            result_output = gr.HTML(
                value="Upload an image to see results here",
                elem_classes="result-box"
            )
            
            gr.HTML("<h3>⚕️ Medical Advice</h3>")
            advice_output = gr.Textbox(
                value="Medical recommendations will appear here",
                lines=8,
                label="Professional Guidance",
                elem_classes="result-box"
            )
    
    # Disclaimer
    gr.HTML("""
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <p style="color: white; text-align: center; margin: 0;">
            ⚠️ <strong>Medical Disclaimer:</strong> This AI system is for educational purposes only. 
            Always consult qualified healthcare professionals for medical diagnosis and treatment.
        </p>
    </div>
    """)
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; color: white; margin-top: 20px; opacity: 0.8;">
        <p>Developed by <strong>Gaurav Sharma</strong> | 
        <a href="https://github.com/Gaurav050Sharma/kidney-stone-detection" style="color: #ffffff;">GitHub Repository</a></p>
    </div>
    """)
    
    # Connect function
    analyze_btn.click(
        fn=predict_kidney_stone,
        inputs=image_input,
        outputs=[result_output, advice_output]
    )
    
    # Auto-analyze on image upload
    image_input.change(
        fn=predict_kidney_stone,
        inputs=image_input,
        outputs=[result_output, advice_output]
    )

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
