# config.py
import os
import torch

# Model paths - update these to match your actual model files
MODEL_PATHS = {
    "crop": "models/crop_recommender_triplet.pt",
    "sustainability": "models/sustainability_predictor.pt", 
    "yield": "models/yield_predictor.pt"
}

# Feature configurations - adjust these based on your actual preprocessing
CROP_FEATURES = [
    'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'
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