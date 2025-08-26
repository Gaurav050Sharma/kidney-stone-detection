"""
Test script to verify the trained models work correctly
"""

import os
from model import KidneyStoneDetector
from utils import check_models_exist

def test_models():
    print("🧪 Testing Kidney Stone Detection Models")
    print("=" * 50)
    
    # Check if models exist
    models_exist, cnn_exists, svm_exists = check_models_exist()
    
    print(f"CNN Model exists: {'✅' if cnn_exists else '❌'}")
    print(f"SVM Model exists: {'✅' if svm_exists else '❌'}")
    
    if not models_exist:
        print("\n❌ Models not found! Please run training first:")
        print("   python train.py")
        return
    
    # Initialize detector
    detector = KidneyStoneDetector()
    
    # Test loading models
    print("\n📥 Loading models...")
    success = detector.load_models()
    
    if success:
        print("✅ Models loaded successfully!")
        
        # Test with a sample image from test set
        test_image_paths = []
        
        # Look for test images
        normal_test_dir = "data/test/Normal"
        stone_test_dir = "data/test/Stone"
        
        if os.path.exists(normal_test_dir):
            normal_files = [f for f in os.listdir(normal_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if normal_files:
                test_image_paths.append((os.path.join(normal_test_dir, normal_files[0]), "Normal"))
        
        if os.path.exists(stone_test_dir):
            stone_files = [f for f in os.listdir(stone_test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if stone_files:
                test_image_paths.append((os.path.join(stone_test_dir, stone_files[0]), "Stone"))
        
        # Test predictions
        if test_image_paths:
            print(f"\n🔬 Testing predictions on {len(test_image_paths)} sample images...")
            
            for image_path, expected in test_image_paths:
                print(f"\nTesting: {os.path.basename(image_path)} (Expected: {expected})")
                
                result = detector.predict_image(image_path)
                
                if result:
                    prediction = result['prediction']
                    confidence = result['confidence']
                    cnn_conf = result['cnn_confidence']
                    svm_conf = result['svm_confidence']
                    
                    print(f"  Prediction: {prediction}")
                    print(f"  Confidence: {confidence:.2%}")
                    print(f"  CNN: {cnn_conf:.2%}, SVM: {svm_conf:.2%}")
                    
                    # Check if prediction matches expected
                    expected_pred = "Stone Detected" if expected == "Stone" else "Normal"
                    status = "✅ CORRECT" if prediction == expected_pred else "❌ INCORRECT"
                    print(f"  Status: {status}")
                else:
                    print("  ❌ Prediction failed")
        else:
            print("\n⚠️  No test images found in data/test/ directories")
            
    else:
        print("❌ Failed to load models")
    
    print("\n🎉 Model testing completed!")

if __name__ == "__main__":
    test_models()
