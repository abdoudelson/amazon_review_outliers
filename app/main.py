from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.models import load_latest_model, predict_outliers
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the latest model at startup
try:
    model = load_latest_model()
    logger.info("Model loaded successfully at startup.")
except Exception as e:
    logger.error("Failed to load the model at startup: %s", e)
    model = None  # Set model to None so we can handle it later in predict endpoint


class ReviewData(BaseModel):
    review_text: str


class PredictionResponse(BaseModel):
    is_outlier: bool


@app.post("/predict", response_model=PredictionResponse)
async def predict(data: ReviewData):
    # Check if the model is loaded
    if model is None:
        logger.error("Model is not loaded; cannot make predictions.")
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Attempt to make a prediction
        logger.info("Received prediction request for text: %s", data.review_text)
        is_outlier = predict_outliers(model, data.review_text)
        logger.info("Prediction successful: %s", is_outlier)
        return {"is_outlier": is_outlier}
    except Exception as e:
        # Log the error and return a 500 HTTP error response
        logger.error("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail="Error during prediction")
