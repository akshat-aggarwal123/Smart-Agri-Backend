from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Literal, Dict, Callable
from src.model_loader import ModelLoader  # loads & returns torch models
from src.predict_torch import (
    predict_crop,
    predict_sustainability,
    predict_yield,
)
from src.utils import log_prediction

# --------------------------------------------------------------------------
# FastAPI router
# --------------------------------------------------------------------------
api_router = APIRouter(prefix="/predict", tags=["prediction"])


# --------------------------------------------------------------------------
# Pydantic request/response models
# --------------------------------------------------------------------------
class CropPredictionRequest(BaseModel):
    n: float
    p: float
    k: float
    temperature_c: float
    humidity_pct: float
    soil_ph: float
    rainfall_mm: float
    soil_moisture_pct: float
    fertilizer_usage_kg: float
    pesticide_usage_kg: float
    crop_type: Literal[
        "rice", "wheat", "corn", "sugarcane", "pulses", "cotton", "other"
    ] = Field(default="other", description="Plain string; one‑hot happens server‑side")


class SustainabilityPredictionRequest(BaseModel):
    temperature_c: float
    humidity_pct: float
    soil_ph: float
    rainfall_mm: float


class YieldPredictionRequest(BaseModel):
    soil_ph: float
    soil_moisture_pct: float
    temperature_c: float
    rainfall_mm: float
    fertilizer_usage_kg: float
    pesticide_usage_kg: float
    crop_type: Literal[
        "rice", "wheat", "corn", "sugarcane", "pulses", "cotton", "other"
    ] = Field(default="other")


class CropPredictionResponse(BaseModel):
    recommended_crop: str


class SustainabilityPredictionResponse(BaseModel):
    sustainability_score: float


class YieldPredictionResponse(BaseModel):
    predicted_yield_kg_per_hectare: float


# --------------------------------------------------------------------------
# Dependency that loads the Torch models exactly once
# --------------------------------------------------------------------------
def get_models() -> Dict[str, object]:
    # FastAPI caches dependency results by default (scope="singleton")
    return ModelLoader.load_models()


# --------------------------------------------------------------------------
# Key‑alias maps – clean API → training feature names
# --------------------------------------------------------------------------
CROP_ALIAS = {
    "n": "n",  # Keep as 'n' to match preprocessing
    "p": "p",  # Keep as 'p' to match preprocessing  
    "k": "k",  # Keep as 'k' to match preprocessing
    "temperature_c": "temperature_c",
    "humidity_pct": "humidity_pct",
    "rainfall_mm": "rainfall_mm",
    "soil_ph": "soil_ph",
    # No aliasing needed - keep field names as they are
}

YIELD_ALIAS = {
    "temperature_c": "temperature_c",
    "rainfall_mm": "rainfall_mm",
    "soil_ph": "soil_ph",
    # No aliasing needed - keep field names as they are
}

# For sustainability - no aliasing needed
SUS_ALIAS = {
    "temperature_c": "temperature_c",
    "humidity_pct": "humidity_pct",
    "rainfall_mm": "rainfall_mm",
    "soil_ph": "soil_ph",
}


def _apply_alias(payload: Dict, alias: Dict[str, str]) -> Dict:
    """Rename clean request keys → training keys."""
    return {alias.get(k, k): v for k, v in payload.items()}


# --------------------------------------------------------------------------
# Generic prediction helper
# --------------------------------------------------------------------------
async def _predict(
    *,
    request_data: BaseModel,
    model_key: str,
    raw_predict_fn: Callable,
    alias_map: Dict[str, str] | None,
    response_model: BaseModel,
    response_field: str,
    models: Dict[str, object],
):
    try:
        # Convert request to dict
        data = request_data.dict()
        
        # Apply field name aliases if provided
        if alias_map:
            data = _apply_alias(data, alias_map)

        # Get the model
        model = models.get(model_key)
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"{model_key} model not available"
            )

        # Make prediction
        prediction = raw_predict_fn(model, data)
        
        # Log the prediction
        log_prediction(model_key, data, prediction)
        
        # Create response with the correct field name
        return response_model(**{response_field: prediction})

    except Exception as exc:
        print(f"Prediction error in {model_key}: {exc}")
        print(f"Input data: {data if 'data' in locals() else request_data.dict()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {exc}",
        ) from exc


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@api_router.post(
    "/crop",
    response_model=CropPredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def crop_endpoint(
    req: CropPredictionRequest, models=Depends(get_models)
):
    return await _predict(
        request_data=req,
        model_key="crop",
        raw_predict_fn=predict_crop,
        alias_map=CROP_ALIAS,
        response_model=CropPredictionResponse,
        response_field="recommended_crop",
        models=models,
    )


@api_router.post(
    "/sustainability",
    response_model=SustainabilityPredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def sustainability_endpoint(
    req: SustainabilityPredictionRequest, models=Depends(get_models)
):
    return await _predict(
        request_data=req,
        model_key="sustainability",
        raw_predict_fn=predict_sustainability,
        alias_map=SUS_ALIAS,
        response_model=SustainabilityPredictionResponse,
        response_field="sustainability_score",
        models=models,
    )


@api_router.post(
    "/yield",
    response_model=YieldPredictionResponse,
    status_code=status.HTTP_200_OK,
)
async def yield_endpoint(
    req: YieldPredictionRequest, models=Depends(get_models)
):
    return await _predict(
        request_data=req,
        model_key="yield",
        raw_predict_fn=predict_yield,
        alias_map=YIELD_ALIAS,
        response_model=YieldPredictionResponse,
        response_field="predicted_yield_kg_per_hectare",
        models=models,
    )


# Health check endpoint
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Agricultural Prediction API is running"}