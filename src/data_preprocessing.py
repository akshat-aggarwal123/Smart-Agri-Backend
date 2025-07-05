import numpy as np
from config import CROP_FEATURES, SUSTAINABILITY_FEATURES, YIELD_FEATURES

class DataPreprocessor:
    # Normalization parameters (should match training preprocessing)
    # These would typically be saved/loaded from files, but hardcoded for simplicity
    CROP_MEAN = np.array([50.55, 42.36, 48.15, 25.62, 71.48, 6.47, 103.46])
    CROP_STD = np.array([36.92, 50.65, 49.96, 5.06, 22.26, 0.77, 54.86])
    
    SUSTAINABILITY_MEAN = np.array([500.0, 250.0, 150.0, 10.0])
    SUSTAINABILITY_STD = np.array([150.0, 75.0, 50.0, 5.0])
    
    YIELD_MEAN = np.array([40.0, -100.0, 3.0, 5.0, 150.0, 250.0, 100.0, 500.0])
    YIELD_STD = np.array([10.0, 20.0, 2.0, 2.0, 30.0, 50.0, 50.0, 200.0])
    
    # Crop type mapping (should match training data)
    CROP_TYPES = [
        'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
        'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
        'banana', 'mango', 'grapes', 'watermelon', 'muskmelon',
        'apple', 'orange', 'papaya', 'coconut', 'cotton',
        'jute', 'coffee'
    ]
    
    @staticmethod
    def normalize_crop_input(data: dict) -> np.ndarray:
        """Normalize crop recommendation input features"""
        features = np.array([
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
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