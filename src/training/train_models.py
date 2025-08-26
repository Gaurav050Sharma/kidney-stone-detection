"""
Training pipeline for kidney stone detection models
"""

import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.data_loader import DataLoader, SVMFeatureExtractor
from src.models.model_architectures import CNNModel, SVMModel
from src.evaluation.evaluator import ModelEvaluator
from config.config import (
    DATA_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, 
    AUGMENTATION_CONFIG, SVM_CONFIG
)

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Complete training pipeline for both CNN and SVM models"""
    
    def __init__(self):
        """Initialize training pipeline"""
        self.data_loader = DataLoader(DATA_CONFIG)
        self.svm_extractor = SVMFeatureExtractor(SVM_CONFIG)
        self.evaluator = ModelEvaluator(MODEL_CONFIG['class_names'])
        
        # Models
        self.cnn_model = CNNModel(MODEL_CONFIG)
        self.svm_model = SVMModel(SVM_CONFIG)
        
        # Data
        self.train_df = None
        self.validate_df = None
        self.test_df = None
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        
    def prepare_data(self):
        """Prepare all data for training"""
        logger.info("Starting data preparation...")
        
        # Load training data
        data_frame = self.data_loader.load_training_data()
        if data_frame is None:
            raise ValueError("Failed to load training data")
        
        # Split train/validation
        self.train_df, self.validate_df = self.data_loader.split_train_validation(data_frame)
        
        # Create data generators
        self.train_generator, self.validation_generator = self.data_loader.create_data_generators(
            self.train_df, self.validate_df, AUGMENTATION_CONFIG
        )
        
        # Load test data
        self.test_df = self.data_loader.load_test_data()
        if self.test_df is not None:
            self.test_generator = self.data_loader.create_test_generator(self.test_df)
        
        logger.info("Data preparation completed")
    
    def train_cnn_model(self):
        """Train CNN model"""
        logger.info("Starting CNN model training...")
        
        # Build and compile model
        self.cnn_model.build_model()
        self.cnn_model.compile_model()
        
        # Display model summary
        print("\nCNN Model Architecture:")
        print("="*50)
        self.cnn_model.model.summary()
        
        # Train model
        history = self.cnn_model.train(
            self.train_generator,
            self.validation_generator,
            TRAINING_CONFIG
        )
        
        # Plot training history
        self.evaluator.plot_training_history(history, "CNN Training History")
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_CONFIG['cnn_model_path']), exist_ok=True)
        self.cnn_model.save_model(MODEL_CONFIG['cnn_model_path'])
        
        logger.info("CNN model training completed")
        return history
    
    def train_svm_model(self):
        """Train SVM model"""
        logger.info("Starting SVM model training...")
        
        # Prepare SVM data
        (all_training_features, training_labels, 
         all_testing_features, testing_labels) = self.svm_extractor.prepare_svm_data(
            DATA_CONFIG['train_path'], 
            DATA_CONFIG['test_path']
        )
        
        # Split training data for validation
        features_training, features_validation, labels_training, labels_validation = train_test_split(
            all_training_features, training_labels, 
            test_size=0.2, 
            random_state=DATA_CONFIG['random_state']
        )
        
        # Train SVM model
        self.svm_model.train(features_training, labels_training)
        
        # Evaluate on validation set
        validation_results = self.evaluator.evaluate_svm_model(
            self.svm_model, features_validation, labels_validation
        )
        
        # Generate evaluation report
        self.evaluator.generate_evaluation_report(validation_results, "SVM Validation")
        
        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(
            validation_results['true_labels'],
            validation_results['predictions'],
            "SVM Validation Confusion Matrix"
        )
        
        # Evaluate on test set
        if all_testing_features:
            test_results = self.evaluator.evaluate_svm_model(
                self.svm_model, all_testing_features, testing_labels
            )
            
            self.evaluator.generate_evaluation_report(test_results, "SVM Test")
            self.evaluator.plot_confusion_matrix(
                test_results['true_labels'],
                test_results['predictions'],
                "SVM Test Confusion Matrix"
            )
        
        # Save model
        os.makedirs(os.path.dirname(MODEL_CONFIG['svm_model_path']), exist_ok=True)
        self.svm_model.save_model(MODEL_CONFIG['svm_model_path'])
        
        logger.info("SVM model training completed")
        return validation_results
    
    def evaluate_cnn_model(self):
        """Evaluate trained CNN model"""
        if self.test_generator is None:
            logger.warning("No test data available for CNN evaluation")
            return None
        
        logger.info("Evaluating CNN model...")
        
        # Evaluate model
        results = self.evaluator.evaluate_cnn_model(
            self.cnn_model.model, 
            self.test_generator
        )
        
        # Generate evaluation report
        self.evaluator.generate_evaluation_report(results, "CNN Test")
        
        # Plot confusion matrix
        self.evaluator.plot_confusion_matrix(
            results['true_labels'],
            results['predictions'],
            "CNN Test Confusion Matrix"
        )
        
        # Plot prediction comparison
        self.evaluator.plot_prediction_comparison(
            results['true_labels'],
            results['predictions'],
            "CNN Prediction vs Actual"
        )
        
        return results
    
    def run_complete_training(self):
        """Run complete training pipeline"""
        logger.info("Starting complete training pipeline...")
        
        try:
            # Prepare data
            self.prepare_data()
            
            # Train CNN model
            cnn_history = self.train_cnn_model()
            
            # Evaluate CNN model
            cnn_results = self.evaluate_cnn_model()
            
            # Train SVM model
            svm_results = self.train_svm_model()
            
            logger.info("Complete training pipeline finished successfully!")
            
            # Print summary
            print("\n" + "="*60)
            print("TRAINING PIPELINE SUMMARY")
            print("="*60)
            
            if cnn_results:
                print(f"CNN Test Accuracy: {cnn_results['accuracy']:.4f}")
            
            if svm_results:
                print(f"SVM Validation Accuracy: {svm_results['accuracy']:.4f}")
            
            print(f"Models saved:")
            print(f"  CNN: {MODEL_CONFIG['cnn_model_path']}")
            print(f"  SVM: {MODEL_CONFIG['svm_model_path']}")
            print("="*60)
            
            return {
                'cnn_history': cnn_history,
                'cnn_results': cnn_results,
                'svm_results': svm_results
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training pipeline
    pipeline = TrainingPipeline()
    results = pipeline.run_complete_training()
    
    return results


if __name__ == "__main__":
    main()
