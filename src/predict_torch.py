import torch
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.utils import to_tensor
from config import CROP_EMBEDDINGS

# ------------------------------------------------------------------ helpers
def _as_device_batch(features: np.ndarray, model: torch.nn.Module) -> torch.Tensor:
    """
    Convert feature array -> 1×N torch tensor on the model's device.
    """
    device = next(model.parameters()).device
    return to_tensor(features).unsqueeze(0).to(device)

# ------------------------------------------------------------------ crop
def predict_crop(model, input_data) -> str:
    """
    Return the crop whose reference embedding is closest to the sample embedding.
    """
    try:
        # Debug: Print what we received
        print(f"predict_crop received: {type(input_data)} = {input_data}")
        
        # Ensure input_data is a dictionary
        if not isinstance(input_data, dict):
            raise ValueError(f"Expected dict, got {type(input_data)}: {input_data}")
        
        # Check required fields
        required_fields = ['n', 'p', 'k', 'temperature_c', 'humidity_pct', 
                          'soil_ph', 'rainfall_mm', 'soil_moisture_pct', 
                          'fertilizer_usage_kg', 'pesticide_usage_kg']
        
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        features = DataPreprocessor.normalize_crop_input(input_data)
        print(f"Normalized features shape: {features.shape}")
        
        with torch.no_grad():
            x = _as_device_batch(features, model)
            embedding = model(x).cpu().numpy().flatten()
        
        print(f"Generated embedding shape: {embedding.shape}")
        
        best_crop, min_dist = None, float("inf")
        for crop, ref in CROP_EMBEDDINGS.items():
            dist = np.linalg.norm(embedding - ref)
            if dist < min_dist:
                best_crop, min_dist = crop, dist
        
        return best_crop if best_crop else "other"
        
    except Exception as e:
        print(f"Error in predict_crop: {e}")
        print(f"Input data type: {type(input_data)}")
        print(f"Input data: {input_data}")
        raise e  # Re-raise to see full traceback

# ------------------------------------------------------------------ sustainability
def predict_sustainability(model, input_data) -> float:
    """
    Predict sustainability score.
    """
    try:
        # Debug: Print what we received
        print(f"predict_sustainability received: {type(input_data)} = {input_data}")
        
        # Ensure input_data is a dictionary
        if not isinstance(input_data, dict):
            raise ValueError(f"Expected dict, got {type(input_data)}: {input_data}")
        
        # Check required fields
        required_fields = ['temperature_c', 'humidity_pct', 'soil_ph', 'rainfall_mm']
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        features = DataPreprocessor.normalize_sustainability_input(input_data)
        print(f"Normalized features shape: {features.shape}")
        
        with torch.no_grad():
            x = _as_device_batch(features, model)
            prediction = model(x).item()
        
        return round(prediction, 4)
        
    except Exception as e:
        print(f"Error in predict_sustainability: {e}")
        print(f"Input data type: {type(input_data)}")
        print(f"Input data: {input_data}")
        raise e  # Re-raise to see full traceback

# ------------------------------------------------------------------ yield
def predict_yield(model, input_data) -> float:
    """
    Predict crop yield.
    """
    try:
        # Debug: Print what we received
        print(f"predict_yield received: {type(input_data)} = {input_data}")
        
        # Ensure input_data is a dictionary
        if not isinstance(input_data, dict):
            raise ValueError(f"Expected dict, got {type(input_data)}: {input_data}")
        
        # Check required fields
        required_fields = ['soil_ph', 'soil_moisture_pct', 'temperature_c', 
                          'rainfall_mm', 'fertilizer_usage_kg', 'pesticide_usage_kg']
        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        features = DataPreprocessor.normalize_yield_input(input_data)
        print(f"Normalized features shape: {features.shape}")
        
        with torch.no_grad():
            x = _as_device_batch(features, model)
            prediction = model(x).item()
        
        return round(prediction, 2)
        
    except Exception as e:
        print(f"Error in predict_yield: {e}")
        print(f"Input data type: {type(input_data)}")
        print(f"Input data: {input_data}")
        raise e  # Re-raise to see full traceback