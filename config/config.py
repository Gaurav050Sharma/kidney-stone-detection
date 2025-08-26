"""
Configuration settings for Kidney Stone Detection Project
"""

import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Data Configuration
DATA_CONFIG = {
    'train_path': 'OneDrive/Documents/kidnee/CT_SCAN/Train',
    'test_path': 'OneDrive/Documents/kidnee/CT_SCAN/Test',
    'image_size': (150, 150),
    'batch_size': 15,
    'validation_split': 0.2,
    'random_state': 42
}

# Model Configuration
MODEL_CONFIG = {
    'cnn_model_path': 'saved_models/cnn_model.h5',
    'svm_model_path': 'saved_models/svm_model.pkl',
    'input_shape': (150, 150, 3),
    'num_classes': 2,
    'class_names': ['Normal', 'Stone']
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 50,
    'patience': 10,
    'learning_rate_reduction_factor': 0.5,
    'learning_rate_reduction_patience': 2,
    'min_learning_rate': 0.00001
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'rescale': 1./255,
    'shear_range': 0.1,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1
}

# SVM Configuration
SVM_CONFIG = {
    'hog_orientations': 8,
    'hog_pixels_per_cell': (16, 16),
    'hog_cells_per_block': (1, 1),
    'svm_kernel': 'rbf',
    'svm_C': 1,
    'svm_gamma': 'auto'
}

# Medical Advice Configuration
MEDICAL_ADVICE = {
    'positive_advice': {
        'immediate_care': [
            "Consult a urologist immediately for proper diagnosis and treatment plan.",
            "Drink plenty of water (2.5-3 liters per day) to help flush the stone.",
            "Take prescribed pain relievers (ibuprofen, acetaminophen) as needed.",
            "Consider alpha blockers (tamsulosin) if prescribed by doctor."
        ],
        'dietary_recommendations': [
            "Reduce sodium intake to less than 2,000mg per day.",
            "Limit foods high in oxalate (spinach, chocolate, nuts).",
            "Increase citrus fruits intake (lemons, oranges).",
            "Maintain adequate calcium intake from food sources."
        ],
        'warning_signs': [
            "Seek immediate medical attention if you experience:",
            "- Severe pain that prevents sitting still",
            "- Nausea and vomiting",
            "- Fever and chills",
            "- Blood in urine",
            "- Difficulty urinating"
        ]
    },
    'negative_advice': {
        'prevention': [
            "Continue maintaining good kidney health.",
            "Stay well hydrated (8-10 glasses of water daily).",
            "Maintain a balanced diet low in sodium and sugar.",
            "Regular medical check-ups are recommended."
        ],
        'general_tips': [
            "Exercise regularly to maintain overall health.",
            "Monitor your weight and blood pressure.",
            "Limit processed foods and sugary drinks.",
            "Include calcium-rich foods in your diet."
        ]
    },
    'disclaimer': "This AI prediction is for educational purposes only. Always consult with a qualified healthcare professional for proper medical diagnosis and treatment."
}
