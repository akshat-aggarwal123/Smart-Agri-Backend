# src/model_loader.py
import torch
from config import MODEL_PATHS
from src.model_definitions import CropRecommender, SustainabilityPredictor, YieldPredictor, CropEmbeddingModel

class ModelLoader:
    _models = {}
   
    @classmethod
    def get_model_input_size(cls, model_path):
        """Helper method to determine input size from saved model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            # Check the first layer's input size
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # Look for the first linear layer
                for key, value in state_dict.items():
                    if 'fc1.weight' in key:
                        return value.shape[1]  # Input size is the second dimension
            return None
        except Exception as e:
            print(f"Error determining input size for {model_path}: {e}")
            return None

    @classmethod
    def load_models(cls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        if not cls._models:
            # Load crop model
            try:
                crop_checkpoint = torch.load(
                    MODEL_PATHS["crop"],
                    map_location=device,
                    weights_only=False
                )
                
                # Auto-detect input size from saved model
                input_size = cls.get_model_input_size(MODEL_PATHS["crop"])
                if input_size is None:
                    raise ValueError("Could not determine input size for crop model")
                
                print(f"Crop model input size: {input_size}")
                
                embedding_size = crop_checkpoint.get('embedding_size', 64)
                
                crop_model = CropEmbeddingModel(input_size=input_size, embedding_size=embedding_size)
                crop_model.load_state_dict(crop_checkpoint['model_state_dict'])
                crop_model.eval()
                crop_model.to(device)
                
                print("Crop model loaded successfully")
                cls._models["crop"] = crop_model
                
            except Exception as e:
                print(f"Error loading crop model: {e}")
                cls._models["crop"] = None
           
            # Load sustainability model
            try:
                sust_checkpoint = torch.load(
                    MODEL_PATHS["sustainability"],
                    map_location=device,
                    weights_only=False
                )
                
                # Auto-detect input size from saved model
                input_size = cls.get_model_input_size(MODEL_PATHS["sustainability"])
                if input_size is None:
                    raise ValueError("Could not determine input size for sustainability model")
                
                print(f"Sustainability model input size: {input_size}")
                
                sust_model = SustainabilityPredictor(input_size=input_size)
                sust_model.load_state_dict(sust_checkpoint['model_state_dict'])
                sust_model.eval()
                sust_model.to(device)
                
                print("Sustainability model loaded successfully")
                cls._models["sustainability"] = sust_model
                
            except Exception as e:
                print(f"Error loading sustainability model: {e}")
                cls._models["sustainability"] = None
           
            # Load yield model
            try:
                yield_checkpoint = torch.load(
                    MODEL_PATHS["yield"],
                    map_location=device,
                    weights_only=False
                )
                
                # Auto-detect input size from saved model
                input_size = cls.get_model_input_size(MODEL_PATHS["yield"])
                if input_size is None:
                    raise ValueError("Could not determine input size for yield model")
                
                print(f"Yield model input size: {input_size}")
                
                yield_model = YieldPredictor(input_size=input_size)
                yield_model.load_state_dict(yield_checkpoint['model_state_dict'])
                yield_model.eval()
                yield_model.to(device)
                
                print("Yield model loaded successfully")
                cls._models["yield"] = yield_model
                
            except Exception as e:
                print(f"Error loading yield model: {e}")
                cls._models["yield"] = None
       
        return cls._models