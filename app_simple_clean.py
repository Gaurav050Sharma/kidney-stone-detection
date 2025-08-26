import gradio as gr
import tempfile
import os
from model import KidneyStoneDetector

detector = KidneyStoneDetector()

def predict(image):
    if not image:
        return "Please upload an image", ""
    
    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image.save(tmp.name)
        result = detector.predict_image(tmp.name)
        os.unlink(tmp.name)
    
    if not result:
        return "Error analyzing image", ""
    
    prediction = result['prediction']
    confidence = result['confidence'] * 100
    
    status = f"🔴 {prediction}" if prediction == "Stone Detected" else f"✅ {prediction}"
    result_text = f"{status}\nConfidence: {confidence:.1f}%"
    
    advice = "Consult a doctor immediately" if prediction == "Stone Detected" else "Continue healthy lifestyle"
    
    return result_text, advice

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Textbox(label="Result"), gr.Textbox(label="Advice")],
    title="Kidney Stone Detection"
)

demo.launch(share=True)
