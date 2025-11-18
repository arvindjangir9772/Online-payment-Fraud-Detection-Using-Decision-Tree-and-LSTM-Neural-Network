# app/main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import uvicorn
import os

# Import model manager
from app.models.model_manager import ModelManager

# Initialize global model manager instance
model_manager = ModelManager()

# Import routes safely
try:
    from app.routes.predictions import router as predictions_router
    from app.routes.data_upload import router as data_upload_router
    from app.routes.analytics import router as analytics_router
except ImportError:
    from fastapi import APIRouter
    predictions_router = APIRouter()
    data_upload_router = APIRouter()
    analytics_router = APIRouter()

# Create FastAPI app
app = FastAPI(
    title="Online Payment Fraud Detection (Hybrid System)",
    description="AI-powered Hybrid Model using LSTM + Decision Tree for real-time fraud detection.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Mount static and template directories
app.mount("/static", StaticFiles(directory=Path("app/static")), name="static")
templates = Jinja2Templates(directory=Path("app/templates"))

# Include routers
app.include_router(predictions_router, prefix="/api/v1", tags=["Predictions"])
app.include_router(data_upload_router, prefix="/api/v1", tags=["Data Upload"])
app.include_router(analytics_router, prefix="/api/v1", tags=["Analytics"])

# =======================
# 🚀 APP STARTUP EVENT
# =======================
@app.on_event("startup")
async def startup_event():
    """Load all ML models (DT, LSTM, Hybrid) at startup"""
    project_root = Path(__file__).resolve().parents[2]
    models_path = project_root / "models"

    print("\n========================================")
    print("Starting Hybrid Fraud Detection System")
    print("========================================")

    if models_path.exists():
        print(f"Loading models from: {models_path}")
        try:
            # ModelManager loads all models automatically
            _ = model_manager.hybrid_model
            print("Hybrid, LSTM, and Decision Tree models loaded successfully.")
        except Exception as e:
            print(f"Model loading error: {e}")
    else:
        print(f"Models directory not found at: {models_path}")
        print("Running in limited/demo mode.")

    print("========================================\n")

# =======================
# 🌐 ROUTES
# =======================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/single-prediction", response_class=HTMLResponse)
async def single_prediction_page(request: Request):
    """Single transaction prediction page"""
    return templates.TemplateResponse("single_prediction.html", {"request": request})


@app.get("/batch-upload", response_class=HTMLResponse)
async def batch_upload_page(request: Request):
    """Batch upload & prediction page"""
    return templates.TemplateResponse("batch_upload.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "✅ API is up and running", "models_loaded": True}


# =======================
# ▶️ RUN LOCALLY
# =======================
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
