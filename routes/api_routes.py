# routes/api_routes.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from src.model_loader import ModelLoader
from src.predict_torch import predict_crop, predict_sustainability, predict_yield
from src.utils import validate_input, log_prediction
from config import CROP_FEATURES, SUSTAINABILITY_FEATURES, YIELD_FEATURES

api_router = APIRouter()
models = ModelLoader.load_models()

# Pydantic models for request validation
class CropPredictionRequest(BaseModel):
    # Define your crop prediction fields based on CROP_FEATURES
    # Example fields (replace with actual features from your config):
    temperature: float
    humidity: float
    soil_ph: float
    rainfall: float
    # Add other fields as needed

class SustainabilityPredictionRequest(BaseModel):
    # Define your sustainability prediction fields based on SUSTAINABILITY_FEATURES
    # Example fields (replace with actual features from your config):
    water_usage: float
    pesticide_usage: float
    soil_quality: float
    # Add other fields as needed

class YieldPredictionRequest(BaseModel):
    # Define your yield prediction fields based on YIELD_FEATURES
    # Example fields (replace with actual features from your config):
    crop_type: str
    area_hectares: float
    fertilizer_amount: float
    # Add other fields as needed

# Response models
class CropPredictionResponse(BaseModel):
    recommended_crop: str

class SustainabilityPredictionResponse(BaseModel):
    sustainability_score: float

class YieldPredictionResponse(BaseModel):
    predicted_yield_kg_per_hectare: float

@api_router.post("/predict/crop", response_model=CropPredictionResponse)
async def crop_prediction(request: CropPredictionRequest):
    data = request.dict()
    
    # Validate input
    if not validate_input(data, CROP_FEATURES):
        raise HTTPException(status_code=400, detail="Missing required crop prediction parameters")
    
    try:
        prediction = predict_crop(models["crop"], data)
        log_prediction('crop', data, prediction)
        return CropPredictionResponse(recommended_crop=prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict/sustainability", response_model=SustainabilityPredictionResponse)
async def sustainability_prediction(request: SustainabilityPredictionRequest):
    data = request.dict()
    
    # Validate input
    if not validate_input(data, SUSTAINABILITY_FEATURES):
        raise HTTPException(status_code=400, detail="Missing required sustainability prediction parameters")
    
    try:
        score = predict_sustainability(models["sustainability"], data)
        log_prediction('sustainability', data, score)
        return SustainabilityPredictionResponse(sustainability_score=score)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict/yield", response_model=YieldPredictionResponse)
async def yield_prediction(request: YieldPredictionRequest):
    data = request.dict()
    
    # Validate input
    if not validate_input(data, YIELD_FEATURES):
        raise HTTPException(status_code=400, detail="Missing required yield prediction parameters")
    
    try:
        yield_kg = predict_yield(models["yield"], data)
        log_prediction('yield', data, yield_kg)
        return YieldPredictionResponse(predicted_yield_kg_per_hectare=yield_kg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")