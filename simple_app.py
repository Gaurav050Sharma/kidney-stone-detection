"""
Kidney Stone Detection System - Simple Gradio Interface
"""

import gradio as gr
import os
from model import KidneyStoneDetector, MEDICAL_ADVICE
from utils import save_temp_image, format_prediction_result, get_medical_advice, check_models_exist

# Initialize detector
detector = KidneyStoneDetector()

def predict_kidney_stone(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "❌ Please upload an image", "Please upload a CT scan image to analyze."
    
    try:
        # Save uploaded image temporarily
        temp_path = save_temp_image(image)
        if temp_path is None:
            return "❌ Error processing image", "Could not process the uploaded image."
        
        # Make prediction
        result = detector.predict_image(temp_path)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
        
        if result is None:
            return "❌ Prediction Error", "Could not analyze the image. Please ensure models are trained."
        
        # Format results
        prediction_text = format_prediction_result(result)
        medical_advice = get_medical_advice(result['prediction'])
        
        return prediction_text, medical_advice
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "An error occurred during analysis."

def train_models():
    """Train models if data is available"""
    data_path = "data/train"
    if not os.path.exists(data_path):
        return "❌ No training data found. Please add images to data/train/Normal and data/train/Stone folders."
    
    try:
        detector.train_models(data_path)
        return "✅ Models trained successfully!"
    except Exception as e:
        return f"❌ Training failed: {str(e)}"

def check_system_status():
    """Check if models are available"""
    models_exist, cnn_exists, svm_exists = check_models_exist()
    
    if models_exist:
        return "✅ System ready - Models loaded successfully"
    else:
        status = "⚠️ Models not found:\n"
        if not cnn_exists:
            status += "• CNN model missing\n"
        if not svm_exists:
            status += "• SVM model missing\n"
        status += "\nPlease train models first or add your dataset."
        return status

# Custom CSS for medical interface
css = """
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.medical-header {
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 20px;
    text-align: center;
    border-radius: 10px;
    margin-bottom: 20px;
}

.result-positive {
    background-color: #fef2f2;
    border-left: 4px solid #ef4444;
    padding: 15px;
    border-radius: 5px;
}

.result-negative {
    background-color: #f0fdf4;
    border-left: 4px solid #22c55e;
    padding: 15px;
    border-radius: 5px;
}

.medical-advice {
    background-color: #fafafa;
    border: 1px solid #e5e7eb;
    padding: 20px;
    border-radius: 8px;
    font-size: 14px;
    line-height: 1.6;
}
"""

# Create Gradio interface
with gr.Blocks(css=css, title="Kidney Stone Detection") as app:
    gr.HTML("""
        <div class="medical-header">
            <h1>🏥 Kidney Stone Detection System</h1>
            <p>AI-powered medical image analysis for CT scans</p>
        </div>
    """)
    
    with gr.Tab("🔬 Image Analysis"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="filepath",
                    label="Upload CT Scan Image",
                    height=300
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Image",
                    variant="primary",
                    size="lg"
                )
                
                status_btn = gr.Button(
                    "📊 Check System Status",
                    variant="secondary"
                )
            
            with gr.Column():
                prediction_output = gr.Textbox(
                    label="📋 Analysis Result",
                    lines=6,
                    max_lines=10
                )
                
                advice_output = gr.Textbox(
                    label="⚕️ Medical Recommendations",
                    lines=15,
                    max_lines=20
                )
    
    with gr.Tab("🎓 Training"):
        with gr.Column():
            gr.Markdown("""
            ### Model Training
            
            To train the models, you need to prepare your dataset:
            
            1. Create folders: `data/train/Normal/` and `data/train/Stone/`
            2. Add CT scan images to respective folders
            3. Click the training button below
            """)
            
            train_btn = gr.Button(
                "🚀 Train Models",
                variant="primary"
            )
            
            training_output = gr.Textbox(
                label="Training Status",
                lines=5
            )
    
    with gr.Tab("ℹ️ Information"):
        gr.Markdown("""
        ### About This System
        
        This AI system combines two machine learning approaches:
        - **CNN (Convolutional Neural Network)**: Deep learning for image analysis
        - **SVM (Support Vector Machine)**: Traditional ML with HOG features
        
        ### How It Works
        1. Upload a CT scan image
        2. Both models analyze the image
        3. Results are combined for final prediction
        4. Medical advice is generated based on findings
        
        ### ⚠️ Important Medical Disclaimer
        
        **This tool is for educational and research purposes only.**
        
        - Not intended for medical diagnosis
        - Always consult healthcare professionals
        - Results should not replace professional medical advice
        - Emergency symptoms require immediate medical attention
        
        ### System Requirements
        - High-quality CT scan images
        - Trained models (see Training tab)
        - Proper dataset structure
        """)
    
    # Event handlers
    analyze_btn.click(
        fn=predict_kidney_stone,
        inputs=[image_input],
        outputs=[prediction_output, advice_output]
    )
    
    train_btn.click(
        fn=train_models,
        outputs=[training_output]
    )
    
    status_btn.click(
        fn=check_system_status,
        outputs=[prediction_output]
    )

# Launch configuration
if __name__ == "__main__":
    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860
    )
