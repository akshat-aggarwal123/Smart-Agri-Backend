# routes/api_routes.py
from flask import Blueprint, request, jsonify
from src.model_loader import ModelLoader
from src.predict_torch import predict_crop, predict_sustainability, predict_yield  # Corrected import
from src.utils import validate_input, log_prediction, error_response
from config import CROP_FEATURES, SUSTAINABILITY_FEATURES, YIELD_FEATURES

# ... rest of the code remains the same ...

api_bp = Blueprint('api', __name__)
models = ModelLoader.load_models()

@api_bp.route('/predict/crop', methods=['POST'])
def crop_prediction():
    data = request.json
    
    # Validate input
    if not validate_input(data, CROP_FEATURES):
        return error_response("Missing required crop prediction parameters")
    
    try:
        prediction = predict_crop(models["crop"], data)
        log_prediction('crop', data, prediction)
        return jsonify({"recommended_crop": prediction})
    except Exception as e:
        return error_response(f"Prediction failed: {str(e)}", 500)

@api_bp.route('/predict/sustainability', methods=['POST'])
def sustainability_prediction():
    data = request.json
    
    # Validate input
    if not validate_input(data, SUSTAINABILITY_FEATURES):
        return error_response("Missing required sustainability prediction parameters")
    
    try:
        score = predict_sustainability(models["sustainability"], data)
        log_prediction('sustainability', data, score)
        return jsonify({"sustainability_score": score})
    except Exception as e:
        return error_response(f"Prediction failed: {str(e)}", 500)

@api_bp.route('/predict/yield', methods=['POST'])
def yield_prediction():
    data = request.json
    
    # Validate input
    if not validate_input(data, YIELD_FEATURES):
        return error_response("Missing required yield prediction parameters")
    
    try:
        yield_kg = predict_yield(models["yield"], data)
        log_prediction('yield', data, yield_kg)
        return jsonify({"predicted_yield_kg_per_hectare": yield_kg})
    except Exception as e:
        return error_response(f"Prediction failed: {str(e)}", 500)