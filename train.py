# Simple Training Script
# Run this to train your models quickly

from model import KidneyStoneDetector
from utils import validate_data_structure, create_data_directories
import os

def main():
    print("🏥 Kidney Stone Detection - Training Script")
    print("=" * 50)
    
    # Check if data directories exist
    data_path = "data/train"
    
    if not os.path.exists(data_path):
        print("Creating data directory structure...")
        create_data_directories()
        print("\n⚠️  Please add your CT scan images to:")
        print("   • data/train/Normal/ (for normal scans)")
        print("   • data/train/Stone/ (for stone scans)")
        print("\nThen run this script again.")
        return
    
    # Validate data structure
    is_valid, message = validate_data_structure(data_path)
    if not is_valid:
        print(f"❌ Data validation failed: {message}")
        return
    
    print(f"✅ {message}")
    
    # Initialize detector and train
    detector = KidneyStoneDetector()
    
    print("\n🚀 Starting training process...")
    detector.train_models(data_path)
    
    print("\n🎉 Training completed!")
    print("You can now run the web interface: python simple_app.py")

if __name__ == "__main__":
    main()
