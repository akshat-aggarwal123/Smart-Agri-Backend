import torch
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.utils import to_tensor

# Crop embeddings (replace with your actual embeddings)
CROP_EMBEDDINGS = {
    "rice": np.random.rand(128),
    "wheat": np.random.rand(128),
    "corn": np.random.rand(128),
    # Add more crops as needed
}

def predict_crop(model, input_data: dict) -> str:
    """Predict best crop using triplet network"""
    # Preprocess and normalize input
    features = DataPreprocessor.normalize_crop_input(input_data)
    
    # Convert to tensor and predict
    with torch.no_grad():
        input_tensor = to_tensor(features).unsqueeze(0)
        embedding = model(input_tensor).numpy().flatten()
    
    # Find closest crop embedding
    best_crop, min_dist = None, float('inf')
    for crop, crop_emb in CROP_EMBEDDINGS.items():
        dist = np.linalg.norm(embedding - crop_emb)
        if dist < min_dist:
            min_dist = dist
            best_crop = crop
    
    return best_crop

# CORRECTED FUNCTION NAME: Changed from predict_sustainability_score to predict_sustainability
def predict_sustainability(model, input_data: dict) -> float:
    """Predict sustainability score (0-1)"""
    # Preprocess and normalize input
    features = DataPreprocessor.normalize_sustainability_input(input_data)
    
    # Convert to tensor and predict
    with torch.no_grad():
        input_tensor = to_tensor(features).unsqueeze(0)
        prediction = model(input_tensor).item()
    
    return round(prediction, 4)

def predict_yield(model, input_data: dict) -> float:
    """Predict crop yield in kg/hectare"""
    # Preprocess and normalize input
    features = DataPreprocessor.normalize_yield_input(input_data)
    
    # Convert to tensor and predict
    with torch.no_grad():
        input_tensor = to_tensor(features).unsqueeze(0)
        prediction = model(input_tensor).item()
    
    return round(prediction, 2)