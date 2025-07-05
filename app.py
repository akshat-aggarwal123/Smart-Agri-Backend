from flask import Flask
from routes.api_routes import api_bp
from src.model_loader import ModelLoader

app = Flask(__name__)

# Load models during startup
ModelLoader.load_models()

# Register blueprint
app.register_blueprint(api_bp, url_prefix='/api')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)