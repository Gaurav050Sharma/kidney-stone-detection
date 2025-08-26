"""
Kidney Stone Detection Package
Professional implementation for medical image analysis
"""

__version__ = "1.0.0"
__author__ = "AI Medical Research Team"
__description__ = "AI-powered kidney stone detection using CNN and SVM models"

# Import main components
from .data.data_loader import DataLoader, SVMFeatureExtractor
from .models.model_architectures import CNNModel, SVMModel, EnsemblePredictor
from .utils.predictor import KidneyStonePredictor
from .training.train_models import TrainingPipeline
from .evaluation.evaluator import ModelEvaluator

__all__ = [
    'DataLoader',
    'SVMFeatureExtractor', 
    'CNNModel',
    'SVMModel',
    'EnsemblePredictor',
    'KidneyStonePredictor',
    'TrainingPipeline',
    'ModelEvaluator'
]
