import mlflow
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from colorlog import ColoredFormatter
import os
import json
from typing import Dict, Union
from functools import wraps
import time

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        args[0].logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

class PredictPreprocessor:
    def __init__(self, mlflow_tracking_uri: str = "http://mlflow:5050"):
        """
        Initialize the PredictPreprocessor with MLflow configuration and required models.
        
        Parameters:
        - mlflow_tracking_uri: The URI for MLflow tracking server
        """
        # Initialize environment variables
        self._setup_environment()
        
        # Set up MLflow
        self._setup_mlflow(mlflow_tracking_uri)
        
        # Set up logging
        self.logger = self.setup_logger()
        
        # Load models and transformers
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.scaler = self.load_artifact("StandardScaler")
            self.label_encoder = self.load_artifact("LabelEncoder")
        except Exception as e:
            self.logger.critical(f"Failed to initialize preprocessor: {str(e)}")
            raise

        # Define expected input schema
        self.input_schema = {
            'text': str,
            'rating': (int, float),
            'verified_purchase': bool
        }

    @staticmethod
    def _setup_environment():
        """Set up environment variables for MLflow and AWS credentials."""
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_TRACKING_URI", "http://minio:9000"
        )
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

    @staticmethod
    def _setup_mlflow(tracking_uri: str):
        """Set up MLflow tracking."""
        mlflow.set_tracking_uri(tracking_uri)

    def setup_logger(self) -> logging.Logger:
        """Set up colored logging with proper formatting."""
        formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "white",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
        
        logger = logging.getLogger(__name__)
        # Clear existing handlers to avoid duplicate logging
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        return logger

    def load_artifact(self, artifact_name: str):
        """
        Load a model artifact from MLflow with retries and error handling.
        """
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Loading {artifact_name} from MLflow (attempt {attempt + 1}/{max_retries})...")
                model_uri = f"models:/{artifact_name}/latest"
                artifact = mlflow.sklearn.load_model(model_uri)
                self.logger.info(f"{artifact_name} loaded successfully.")
                return artifact
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    self.logger.error(f"Failed to load {artifact_name} after {max_retries} attempts: {str(e)}")
                    raise

    def validate_input(self, input_data: Dict) -> None:
        """Validate input data against schema."""
        missing_fields = set(self.input_schema.keys()) - set(input_data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        for field, expected_type in self.input_schema.items():
            value = input_data[field]
            if isinstance(expected_type, tuple):
                if not isinstance(value, expected_type):
                    try:
                        # Attempt conversion to first valid type
                        input_data[field] = expected_type[0](value)
                    except (ValueError, TypeError):
                        raise ValueError(f"Field '{field}' has invalid type: {type(value)}")
            elif not isinstance(value, expected_type):
                raise ValueError(f"Field '{field}' has invalid type: {type(value)}")
    
    @log_execution_time
    def preprocess_input(self, input_data: Union[Dict, str]) -> pd.DataFrame:
        """
        Preprocess incoming data for predictions with comprehensive error handling and logging.
        
        Parameters:
        - input_data (dict or str): Input data with keys 'text', 'rating', 'verified_purchase'.
        
        Returns:
        - pd.DataFrame: A DataFrame with preprocessed features ready for model prediction.
        """
        try:
            self.logger.info("Starting preprocessing pipeline...")
            self.logger.info(f"Input string at position 91-93: {input_data[90:93]}")

            self.logger.debug(f"Raw input data: {input_data} of type {type(input_data)}")

            # Parse JSON if needed
            if isinstance(input_data, str):
                try:
                    input_data = json.loads(input_data.replace("'", '"'))
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parsing failed: {str(e)}")
                    raise ValueError(f"Invalid JSON format: {str(e)}")

            # Validate input
            self.validate_input(input_data)

            # Create DataFrame
            try:
                input_df = pd.DataFrame({key: [value] for key, value in input_data.items()})
                self.logger.info(input_df.head())
            except Exception as e:
                self.logger.error(f"Failed to create DataFrame: {str(e)}")
                raise ValueError(f"DataFrame creation failed: {str(e)}")

            # Scale numerical features
            try:
                input_df["rating"] = self.scaler.transform(input_df[["rating"]])
            except Exception as e:
                self.logger.error(f"Scaling failed: {str(e)}")
                raise ValueError(f"Failed to scale rating: {str(e)}")

            # Encode categorical features
            try:
                input_df["verified_purchase"] = self.label_encoder.transform(
                    input_df["verified_purchase"].astype(str)
                )
            except Exception as e:
                self.logger.error(f"Encoding failed: {str(e)}")
                raise ValueError(f"Failed to encode verified_purchase: {str(e)}")

            # Generate embeddings
            try:
                text_embedding = self.model.encode(
                    input_df["text"].tolist(),
                    show_progress_bar=False
                )
                embedded_df = pd.DataFrame(text_embedding)
            except Exception as e:
                self.logger.error(f"Embedding generation failed: {str(e)}")
                raise ValueError(f"Failed to generate text embeddings: {str(e)}")

            # Combine features
            try:
                processed_df = pd.concat(
                    [
                        embedded_df,
                        input_df[["rating", "verified_purchase"]].reset_index(drop=True),
                    ],
                    axis=1,
                )
            except Exception as e:
                self.logger.error(f"Feature combination failed: {str(e)}")
                raise ValueError(f"Failed to combine features: {str(e)}")

            self.logger.info("Preprocessing completed successfully.")
            return processed_df

        except Exception as e:
            self.logger.error(f"Preprocessing pipeline failed: {str(e)}")
            raise

