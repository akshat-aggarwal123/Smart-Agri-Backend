# src/model_loader.py
import torch
from config import MODEL_PATHS
from src.model_definitions import CropRecommender, SustainabilityPredictor, YieldPredictor, CropEmbeddingModel

class ModelLoader:
    _models = {}
   
    @classmethod
    def load_models(cls):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        if not cls._models:
            try:
                # Load crop model (using CropEmbeddingModel since that's what was trained)
                crop_checkpoint = torch.load(
                    MODEL_PATHS["crop"],  # This should point to crop_recommender_triplet.pt
                    map_location=device,
                    weights_only=False
                )
                
                # Get the input size from the checkpoint or use default
                # You might need to adjust this based on your actual input size
                input_size = 7  # Adjust this based on your actual feature count
                embedding_size = crop_checkpoint.get('embedding_size', 64)
                
                crop_model = CropEmbeddingModel(input_size=input_size, embedding_size=embedding_size)
                crop_model.load_state_dict(crop_checkpoint['model_state_dict'])
                crop_model.eval()
                
                print("Crop model loaded successfully")
                
            except Exception as e:
                print(f"Error loading crop model: {e}")
                raise
           
            try:
                # Load sustainability model
                sust_checkpoint = torch.load(
                    MODEL_PATHS["sustainability"],  # This should point to sustainability_predictor.pt
                    map_location=device,
                    weights_only=False
                )
                
                # Get input size from your preprocessing - adjust as needed
                input_size = 4  # Adjust this based on your actual feature count
                
                sust_model = SustainabilityPredictor(input_size=input_size)
                sust_model.load_state_dict(sust_checkpoint['model_state_dict'])
                sust_model.eval()
                
                print("Sustainability model loaded successfully")
                
            except Exception as e:
                print(f"Error loading sustainability model: {e}")
                raise
           
            try:
                # Load yield model
                yield_checkpoint = torch.load(
                    MODEL_PATHS["yield"],  # This should point to yield_predictor.pt
                    map_location=device,
                    weights_only=False
                )
                
                # Get input size from your preprocessing - adjust as needed
                input_size = 8  # Adjust this based on your actual feature count
                
                yield_model = YieldPredictor(input_size=input_size)
                yield_model.load_state_dict(yield_checkpoint['model_state_dict'])
                yield_model.eval()
                
                print("Yield model loaded successfully")
                
            except Exception as e:
                print(f"Error loading yield model: {e}")
                raise
           
            cls._models = {
                "crop": crop_model,
                "sustainability": sust_model,
                "yield": yield_model
            }
       
        return cls._models