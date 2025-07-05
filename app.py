# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes.api_routes import api_router
from src.model_loader import ModelLoader
import uvicorn

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models during startup
    ModelLoader.load_models()
    yield
    # Cleanup on shutdown (if needed)

app = FastAPI(
    title="Agricultural Prediction API",
    description="API for crop, sustainability, and yield predictions",
    version="1.0.0",
    lifespan=lifespan
)

# Include router
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)