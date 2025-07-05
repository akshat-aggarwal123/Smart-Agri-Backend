import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: str) -> None:
    """Save data to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def to_tensor(data: np.ndarray, device: str = None) -> torch.Tensor:
    """Convert numpy array to torch tensor"""
    tensor = torch.from_numpy(data).float()
    if device:
        tensor = tensor.to(device)
    return tensor

def get_device() -> str:
    """Get the best available device (GPU if available, else CPU)"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():  # For Apple Silicon Macs
        return "mps"
    else:
        return "cpu"

def get_current_season() -> str:
    """Get current growing season based on month"""
    month = datetime.now().month
    if 3 <= month <= 5:
        return "spring"
    elif 6 <= month <= 8:
        return "summer"
    elif 9 <= month <= 11:
        return "fall"
    else:
        return "winter"

def validate_input(data: dict, required_fields: list) -> bool:
    """Validate input data contains all required fields"""
    return all(field in data for field in required_fields)

def log_prediction(endpoint: str, input_data: dict, prediction: Any) -> None:
    """Log prediction details (in real app, would write to database)"""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {endpoint} prediction - Input: {input_data}, Result: {prediction}")