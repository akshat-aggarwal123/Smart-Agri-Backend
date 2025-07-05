# config.py
import os
import torch

# Model paths - update these to match your actual model files
MODEL_PATHS = {
    "crop": "models/crop_recommender_triplet.pt",
    "sustainability": "models/sustainability_predictor.pt", 
    "yield": "models/yield_predictor.pt"
}

# Feature configurations - UPDATED to match your trained models
# Your crop model expects 17 features based on the error message
CROP_FEATURES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall',
    # Add the remaining 10 features that your model was trained on
    # These might include encoded categorical features, derived features, etc.
    'feature8', 'feature9', 'feature10', 'feature11', 'feature12',
    'feature13', 'feature14', 'feature15', 'feature16', 'feature17'
]

SUSTAINABILITY_FEATURES = [
    'temperature', 'humidity', 'ph', 'rainfall'  # Adjust based on your preprocessing
]

YIELD_FEATURES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'crop_type'  # Adjust based on your preprocessing
]

# Other configurations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG_LEVEL = "INFO"