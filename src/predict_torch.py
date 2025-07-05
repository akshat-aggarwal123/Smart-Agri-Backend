# src/predict_torch.py
import torch
import numpy as np
from src.data_preprocessing import DataPreprocessor
from src.utils import to_tensor

# ------------------------------------------------------------------ helpers
def _as_device_batch(features: np.ndarray, model: torch.nn.Module) -> torch.Tensor:
    """
    Convert feature array -> 1×N torch tensor on the model's device.
    """
    device = next(model.parameters()).device
    return to_tensor(features).unsqueeze(0).to(device)

# ------------------------------------------------------------------ crop
def predict_crop(model, input_data: dict) -> str:
    """
    Return the crop whose reference embedding is closest to the sample embedding.
    """
    features = DataPreprocessor.normalize_crop_input(input_data)
    with torch.no_grad():
        x = _as_device_batch(features, model)
        embedding = model(x).cpu().numpy().flatten()   # ← .cpu() before .numpy()

    best_crop, min_dist = None, float("inf")
    for crop, ref in CROP_EMBEDDINGS.items():
        dist = np.linalg.norm(embedding - ref)
        if dist < min_dist:
            best_crop, min_dist = crop, dist
    return best_crop

# ------------------------------------------------------------------ sustainability
def predict_sustainability(model, input_data: dict) -> float:
    features = DataPreprocessor.normalize_sustainability_input(input_data)
    with torch.no_grad():
        x = _as_device_batch(features, model)
        prediction = model(x).item()                  # scalar already on CPU
    return round(prediction, 4)

# ------------------------------------------------------------------ yield
def predict_yield(model, input_data: dict) -> float:
    features = DataPreprocessor.normalize_yield_input(input_data)
    with torch.no_grad():
        x = _as_device_batch(features, model)
        prediction = model(x).item()
    return round(prediction, 2)