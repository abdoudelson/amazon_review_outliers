import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer
import logging
from colorlog import ColoredFormatter


class PredictPreprocessor:
    def __init__(self):
        self.logger = self.setup_logger()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.scaler = self.load_artifact("scaler")
        self.label_encoder = self.load_artifact("label_encoder")

    def setup_logger(self):
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
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger

    def load_artifact(self, artifact_name: str):
        """
        Load a model artifact from MLflow.
        """
        try:
            self.logger.info(f"Loading {artifact_name} from MLflow...")
            model_uri = f"models:/{artifact_name}/Production"
            artifact = mlflow.pyfunc.load_model(model_uri)
            self.logger.info(f"{artifact_name} loaded successfully.")
            return artifact
        except Exception as e:
            self.logger.error(f"Error loading {artifact_name} from MLflow: {e}")
            raise

    def preprocess_input(self, input_data: dict) -> pd.DataFrame:
        """
        Preprocess incoming data for predictions.

        Parameters:
        - input_data (dict): Input data with keys 'text', 'rating', 'verified_purchase'.

        Returns:
        - pd.DataFrame: A DataFrame with preprocessed features ready for model prediction.
        """
        try:
            self.logger.info("Preprocessing input data for prediction...")

            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])

            # Process numerical features
            self.logger.info("Scaling numerical features...")
            input_df["rating"] = self.scaler.transform(input_df[["rating"]])

            # Encode categorical features
            self.logger.info("Encoding categorical features...")
            input_df["verified_purchase"] = self.label_encoder.transform(
                input_df["verified_purchase"].astype(str)
            )

            # Generate text embeddings
            self.logger.info("Generating text embeddings...")
            text_embedding = self.model.encode(
                input_df["text"].tolist(), show_progress_bar=False
            )
            embedded_df = pd.DataFrame(text_embedding)

            # Combine embeddings with scaled and encoded features
            processed_df = pd.concat(
                [
                    embedded_df,
                    input_df[["rating", "verified_purchase"]].reset_index(drop=True),
                ],
                axis=1,
            )

            self.logger.info("Input data preprocessed successfully.")
            return processed_df
        except Exception as e:
            self.logger.error(f"Error during input preprocessing: {e}")
            raise


# Example usage
if __name__ == "__main__":
    preprocessor = PredictPreprocessor()

    # Sample input data for prediction
    sample_input = {
        "text": "Great product, really enjoyed it!",
        "rating": 5,
        "verified_purchase": True,
    }

    # Preprocess the sample input
    processed_data = preprocessor.preprocess_input(sample_input)
    print(processed_data)
