# routes/api_routes.py
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
    ] = Field(..., description="Plain string; one‑hot happens server‑side")


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
    ]


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
    "n": "N",
    "p": "P",
    "k": "K",
    "temperature_c": "temperature",
    "humidity_pct": "humidity",
    "rainfall_mm": "rainfall",
    "soil_ph": "ph",            # ← ADD THIS LINE
    # soil_moisture_pct already matches training name?
}

YIELD_ALIAS = {
    "temperature_c": "temperature",
    "rainfall_mm": "rainfall",
    "soil_ph": "ph",            # ← ADD THIS TOO
    # add others only if training names differ
}

# For sustainability we left alias_map=None, so add a map
SUS_ALIAS = {
    "temperature_c": "temperature",
    "humidity_pct": "humidity",
    "rainfall_mm": "rainfall",
    "soil_ph": "ph",
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
    models: Dict[str, object],
):
    try:
        data = request_data.dict()
        if alias_map:
            data = _apply_alias(data, alias_map)

        prediction = raw_predict_fn(models[model_key], data)
        log_prediction(model_key, data, prediction)
        return response_model({response_model.model_fields.keys()[0]: prediction})

    except Exception as exc:
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
        alias_map=None,
        response_model=SustainabilityPredictionResponse,
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
        models=models,
    )