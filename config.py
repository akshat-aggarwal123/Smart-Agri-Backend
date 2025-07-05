"""
config.py
---------
Keep this file in sync with:
  • routes/api_routes.py  – field names in request models
  • whatever preprocessing you used at training time
"""

from pathlib import Path
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_PATHS = {
    "crop":           MODEL_DIR / "crop_recommender_triplet.pt",
    "sustainability": MODEL_DIR / "sustainability_predictor.pt",
    "yield":          MODEL_DIR / "yield_predictor.pt",
}

# ---------------------------------------------------------------------------
# Feature lists (exact order used when fitting the models)
# ---------------------------------------------------------------------------
# Crop recommender (17 inputs = 10 numeric + 7 one‑hot crop flags)
CROP_FEATURES = [
    "n", "p", "k",
    "temperature_c", "humidity_pct", "soil_ph", "rainfall_mm",
    "soil_moisture_pct", "fertilizer_usage_kg", "pesticide_usage_kg",
    # one‑hot flags derived from crop_type
    "crop_rice", "crop_wheat", "crop_corn", "crop_sugarcane",
    "crop_pulses", "crop_cotton", "crop_other",
]

# Sustainability score (4 numeric inputs)
SUSTAINABILITY_FEATURES = [
    "temperature_c", "humidity_pct", "soil_ph", "rainfall_mm",
]

# Yield predictor (6 numeric + 7 one‑hot crop flags = 13 total)
YIELD_FEATURES = [
    "soil_ph", "soil_moisture_pct", "temperature_c", "rainfall_mm",
    "fertilizer_usage_kg", "pesticide_usage_kg",
    # one‑hot crop flags
    "crop_rice", "crop_wheat", "crop_corn", "crop_sugarcane",
    "crop_pulses", "crop_cotton", "crop_other",
]

# ---------------------------------------------------------------------------
# Runtime flags
# ---------------------------------------------------------------------------
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
LOG_LEVEL = "INFO"