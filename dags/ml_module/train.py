import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import boto3
import os
from io import StringIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_csv_from_minio(bucket_name, file_key):
    """
    Read a CSV file from MinIO and return it as a Pandas DataFrame.
    
    Parameters:
    - bucket_name (str): Name of the MinIO bucket.
    - file_key (str): Path to the file within the bucket.

    Returns:
    - pd.DataFrame: The loaded DataFrame.
    """
    # Initialize Boto3 client for MinIO
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", ""),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    )

    try:
        # Fetch the CSV file from MinIO and load it into a Pandas DataFrame
        logger.info(f"Downloading {file_key} from MinIO bucket {bucket_name}...")
        csv_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        csv_string = csv_obj['Body'].read().decode('utf-8')
        logger.info(f"{file_key} successfully downloaded.")
        return pd.read_csv(StringIO(csv_string))
    except Exception as e:
        logger.error(f"Error reading {file_key} from MinIO: {e}")
        raise

def train_and_log_models():
    """
    Train models on data from MinIO and log them to MLflow.
    """
    # Set MLflow tracking URI to the remote server
    mlflow.set_tracking_uri("http://mlflow:5050")

    # Configure MLflow to use MinIO for artifact storage
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("http://minio:9000","")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "")
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "")

    # Set the correct bucket and artifact storage path for the experiment
    artifact_location = os.getenv('ARTIFACT_BUCKET',"")
    experiment_name = "Outlier Detection Models"
    
    # Create or retrieve the experiment with the specified artifact location
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
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
        "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    }

    # Train and log each model with MLflow
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            logger.info(f"Training {model_name}...")
            model.fit(X_train)
            mlflow.sklearn.log_model(model, f"{model_name}_model")
            mlflow.log_params({"contamination": 0.1, "model": model_name})
            logger.info(f"{model_name} trained and logged to MLflow at mlflow:5050 with artifacts in {artifact_location}")

# Run the training and logging function
if __name__ == "__main__":
    train_and_log_models()
