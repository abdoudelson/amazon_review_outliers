import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
import boto3
import os
from io import StringIO
import logging
from mlflow.models.signature import infer_signature

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_TRACKING_URI", "http://minio:9000"
)
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")
mlflow.set_tracking_uri("http://mlflow:5050")

ARTIFACT_LOCATION = os.getenv("ARTIFACT_BUCKET", "")


def read_csv_from_minio(bucket_name, file_key):
    """
    Read a CSV file from MinIO and return it as a Pandas DataFrame.
    """
    # Initialize Boto3 client for MinIO
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
    )

    try:
        # Fetch the CSV file from MinIO and load it into a Pandas DataFrame
        logger.info(f"Downloading {file_key} from MinIO bucket {bucket_name}...")
        csv_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        csv_string = csv_obj["Body"].read().decode("utf-8")
        logger.info(f"{file_key} successfully downloaded.")
        return pd.read_csv(StringIO(csv_string)).iloc[:, 1:]
    except Exception as e:
        logger.error(f"Error reading {file_key} from MinIO: {e}")
        raise


def train_and_log_models():
    """
    Train models on data from MinIO, log them to MLflow, and register them in the model registry.
    """
    # Set MLflow tracking URI to the remote server

    # Configure MLflow to use MinIO for artifact storage

    # Set the correct bucket and artifact storage path for the experiment
    experiment_name = "Outlier Detection Models"

    # Create or retrieve the experiment with the specified artifact location
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            name=experiment_name, artifact_location=ARTIFACT_LOCATION
        )
    mlflow.set_experiment(experiment_name)

    # Define MinIO bucket and file paths
    bucket_name = "databucket"
    train_file_key = "data/embeddings/amazon_reviews_embeddings/train.csv"

    # Read preprocessed training data directly from MinIO
    logger.info("Reading training data from MinIO...")
    X_train = read_csv_from_minio(bucket_name, train_file_key)

    # Define models to train
    models = {
        "IsolationForest": IsolationForest(contamination=0.1, random_state=42),
        "LocalOutlierFactor": LocalOutlierFactor(
            n_neighbors=20, contamination=0.1, novelty=True
        ),
    }

    # Prepare an input example for model registration
    input_example = X_train.sample(1)  # Example input of 5 samples
    signature = infer_signature(X_train, input_example)

    # Train and log each model with MLflow
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")
            model.fit(X_train.values)
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"{model_name}_model",
                registered_model_name=f"{model_name}",
                signature=signature,
                input_example=input_example,
            )
            mlflow.log_params({"contamination": 0.1, "model": model_name})
            logger.info(
                f"{model_name} trained and logged to MLflow at mlflow:5050 with artifacts in "
            )


# Run the training and logging function
if __name__ == "__main__":
    train_and_log_models()
