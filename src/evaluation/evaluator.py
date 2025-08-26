"""
Model evaluation utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate model performance and generate visualizations"""
    
    def __init__(self, class_names):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
    
    def evaluate_cnn_model(self, model, test_generator):
        """
        Evaluate CNN model performance
        
        Args:
            model: Trained CNN model
            test_generator: Test data generator
            
        Returns:
            Dictionary with evaluation results
        """
        # Make predictions
        steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
        predictions = model.predict(test_generator, steps=steps)
        
        # Get true labels
        test_generator.reset()
        true_labels = test_generator.classes
        
        # Convert predictions to class labels
        predicted_labels = np.argmax(predictions, axis=-1)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Generate classification report
        report = classification_report(
            true_labels, 
            predicted_labels, 
            target_names=self.class_names,
            output_dict=True
        )
        
        logger.info(f"CNN Model Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predicted_labels,
            'true_labels': true_labels,
            'classification_report': report,
            'prediction_probabilities': predictions
        }
    
    def evaluate_svm_model(self, model, test_features, test_labels):
        """
        Evaluate SVM model performance
        
        Args:
            model: Trained SVM model
            test_features: Test features
            test_labels: Test labels
            
        Returns:
            Dictionary with evaluation results
        """
        # Make predictions
        predictions = model.predict(test_features)
        
        # Convert string labels to numeric
        true_labels_numeric = [1 if label == 'Stone' else 0 for label in test_labels]
        predicted_labels_numeric = [1 if pred == 'Stone' else 0 for pred in predictions]
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels_numeric, predicted_labels_numeric)
        
        # Generate classification report
        report = classification_report(
            test_labels,
            predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        logger.info(f"SVM Model Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'true_labels': test_labels,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, true_labels, predicted_labels, title="Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            true_labels: True labels
            predicted_labels: Predicted labels
            title: Plot title
        """
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='BuPu',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(title)
        plt.tight_layout()
        plt.show()
    
    def plot_prediction_comparison(self, true_labels, predicted_labels, title="Prediction Comparison"):
        """
        Plot comparison between predicted and actual categories
        
        Args:
            true_labels: True labels
            predicted_labels: Predicted labels
            title: Plot title
        """
        # Create DataFrame for easier plotting
        if isinstance(true_labels[0], (int, np.integer)):
            true_categories = [self.class_names[label] for label in true_labels]
        else:
            true_categories = true_labels
            
        if isinstance(predicted_labels[0], (int, np.integer)):
            predicted_categories = [self.class_names[label] for label in predicted_labels]
        else:
            predicted_categories = predicted_labels
        
        df = pd.DataFrame({
            'True': true_categories,
            'Predicted': predicted_categories
        })
        
        # Count occurrences
        predicted_counts = df['Predicted'].value_counts()
        actual_counts = df['True'].value_counts()
        
        # Ensure all categories are present
        all_categories = sorted(list(set(predicted_counts.index) | set(actual_counts.index)))
        predicted_values = [predicted_counts.get(cat, 0) for cat in all_categories]
        actual_values = [actual_counts.get(cat, 0) for cat in all_categories]
        
        # Create bar plot
        bar_width = 0.35
        r1 = np.arange(len(all_categories))
        r2 = [x + bar_width for x in r1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(r1, predicted_values, color='blue', width=bar_width, 
                edgecolor='grey', label='Predicted')
        plt.bar(r2, actual_values, color='green', width=bar_width, 
                edgecolor='grey', label='Actual')
        
        plt.xlabel('Category', fontweight='bold')
        plt.ylabel('Count', fontweight='bold')
        plt.title(title)
        plt.xticks([r + bar_width/2 for r in range(len(all_categories))], all_categories)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self, history, title="Training History"):
        """
        Plot training history
        
        Args:
            history: Training history object
            title: Plot title
        """
        plt.figure(figsize=(12, 8))
        
        # Plot accuracy
        plt.subplot(2, 1, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Plot loss
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, results, model_name):
        """
        Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
        """
        print(f"\n{'='*50}")
        print(f"{model_name.upper()} MODEL EVALUATION REPORT")
        print(f"{'='*50}")
        
        print(f"Overall Accuracy: {results['accuracy']:.4f}")
        print(f"\nDetailed Classification Report:")
        
        # Print classification report
        report = results['classification_report']
        for class_name in self.class_names:
            class_metrics = report[class_name]
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall: {class_metrics['recall']:.4f}")
            print(f"  F1-Score: {class_metrics['f1-score']:.4f}")
            print(f"  Support: {class_metrics['support']}")
        
        # Overall metrics
        print(f"\nMacro Average:")
        print(f"  Precision: {report['macro avg']['precision']:.4f}")
        print(f"  Recall: {report['macro avg']['recall']:.4f}")
        print(f"  F1-Score: {report['macro avg']['f1-score']:.4f}")
        
        print(f"\nWeighted Average:")
        print(f"  Precision: {report['weighted avg']['precision']:.4f}")
        print(f"  Recall: {report['weighted avg']['recall']:.4f}")
        print(f"  F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        print(f"{'='*50}\n")
