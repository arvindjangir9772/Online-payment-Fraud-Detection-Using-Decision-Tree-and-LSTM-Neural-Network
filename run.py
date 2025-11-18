# run.py
import uvicorn
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main application runner"""
    # Check if models directory exists
    models_path = Path("models")
    if not models_path.exists():
        logger.warning("Models directory not found. The app will run but predictions won't work.")
        logger.info("Train models first: python ml_pipeline/train_models.py")
    
    # Start the web application
    logger.info("Starting Fraud Detection Web Application...")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",  # Localhost
        port=8080,         # Use port 8080 instead
        reload=True
    )

if __name__ == "__main__":
    main()