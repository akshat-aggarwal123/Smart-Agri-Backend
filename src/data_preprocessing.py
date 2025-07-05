import numpy as np
from config import CROP_FEATURES, SUSTAINABILITY_FEATURES, YIELD_FEATURES

class DataPreprocessor:
    # Normalization parameters (should match training preprocessing)
    # These would typically be saved/loaded from files, but hardcoded for simplicity
    # @TODO For 17 crop features (10 numeric + 7 one-hot)
    CROP_MEAN = np.array([
            50.55, 42.36, 48.15, 25.62, 71.48, 6.47, 103.46, # 7 original features
            65.0, 15.0, 8.0,    # soil_moisture_pct, fertilizer_usage_kg, pesticide_usage_kg
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5   # one-hot crop flags (mostly 0, with "other" being common)
        ])
    CROP_STD = np.array([
            36.92, 50.65, 49.96, 5.06, 22.26, 0.77, 54.86,
            15.0, 10.0, 5.0,  # soil_moisture_pct, fertilizer_usage_kg, pesticide_usage_kg
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5  # one-hot crop flags
        ])
    
    # For 4 sustainability features
    SUSTAINABILITY_MEAN = np.array([25.0, 70.0, 6.5, 100.0])     # temp, humidity, ph, rainfall
    SUSTAINABILITY_STD = np.array([5.0, 20.0, 1.0, 50.0])
    
    # @TODO For 13 yield features (6 numeric + 7 hot-one)
    YIELD_MEAN = np.array([
            6.5, 65.0, 25.0, 100.0, 15.0, 8.0,  # 6 numeric features
            0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5   # 7 one-hot crop flags
        ])
    YIELD_STD = np.array([
            1.0, 15.0, 5.0, 50.0, 10.0, 5.0,  # 6 numeric features
            0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5  # 7 one-hot crop flags
        ])
    
    # Crop type mapping (should match training data)
    CROP_TYPES = [
        'rice', 'wheat', 'corn', 'sugarcane', 'pulses', 'cotton', 'other'
    ]
    
    @staticmethod
    def normalize_crop_input(data: dict) -> np.ndarray:
        """Normalize crop recommendation input features"""
        features = np.array([
            data['n'], data['p'], data['k'],
            data['temperature_c'], data['humidity_pct'],
            data['soil_ph'], data['rainfall_mm'],
            data['soil_moisture_pct'], data['fertilizer_usage_kg'],
            data['pesticide_usage_kg']
        ], dtype=np.float32)
        
        return (features - DataPreprocessor.CROP_MEAN) / DataPreprocessor.CROP_STD
    
    @staticmethod
    def normalize_sustainability_input(data: dict) -> np.ndarray:
        """Normalize sustainability prediction input features"""
        features = np.array([
            data['water_usage'], data['energy_consumption'],
            data['co2_emissions'], data['land_usage']
        ], dtype=np.float32)
        
        return (features - DataPreprocessor.SUSTAINABILITY_MEAN) / DataPreprocessor.SUSTAINABILITY_STD
    
    @staticmethod
    def normalize_yield_input(data: dict) -> np.ndarray:
        """Normalize yield prediction input features"""
        # Convert categorical features to numerical
        soil_type = DataPreprocessor._encode_soil_type(data['soil_type'])
        crop_variety = DataPreprocessor._encode_crop_variety(data['crop_variety'])
        
        # Convert dates to day of year
        planting_doy = DataPreprocessor._date_to_doy(data['planting_date'])
        harvest_doy = DataPreprocessor._date_to_doy(data['harvest_date'])
        
        features = np.array([
            data['location_lat'], data['location_lon'],
            soil_type, crop_variety,
            planting_doy, harvest_doy,
            data['fertilizer_amount'], data['rainfall']
        ], dtype=np.float32)
        
        return (features - DataPreprocessor.YIELD_MEAN) / DataPreprocessor.YIELD_STD
    
    @staticmethod
    def _encode_soil_type(soil_type: str) -> int:
        """Encode soil type to numerical value"""
        soil_mapping = {
            'sandy': 0, 'loamy': 1, 'clay': 2, 
            'silty': 3, 'peaty': 4, 'chalky': 5
        }
        return soil_mapping.get(soil_type.lower(), 1)  # Default to loamy
    
    @staticmethod
    def _encode_crop_variety(crop: str) -> int:
        """Encode crop variety to numerical value"""
        try:
            return DataPreprocessor.CROP_TYPES.index(crop.lower())
        except ValueError:
            return 0  # Default to first crop type
    
    @staticmethod
    def _date_to_doy(date_str: str) -> int:
        """Convert date string (YYYY-MM-DD) to day of year"""
        try:
            year, month, day = map(int, date_str.split('-'))
            # Simplified calculation (for demo purposes)
            # In production, use datetime module
            month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
            return sum(month_days[:month]) + day
        except:
            return 150  # Default to day 150