# dags/review_processing_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from ml_module.predictpreprocess import PredictPreprocessor
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, Any
import json
import os
import boto3
import mlflow
import logging
from io import StringIO
import numpy as np
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_TRACKING_URI", "http://minio:9000"
    )
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minio")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
prediction_model = os.getenv(
        "PREDICTION_MODEL", "IsolationForest"
    )
# Boto3 client configuration for MinIO
s3_client = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",  # MinIO endpoint
    aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minio"),
    aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minio123"),
)

# MLflow configuration
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5050"))


def preprocess_text(text: str) -> str:
    """Preprocess review text"""
    # Add your text preprocessing logic here
    return text.lower().strip()


def preprocess_review_task(review_data: Dict[str, Any], task_id: str) -> str:
    """Preprocess the review data and save it for prediction"""
    try:
        preprocessor = PredictPreprocessor()
        # Preprocess text

        processed_input = preprocessor.preprocess_input(review_data)

        # Save processed data to MinIO (S3-compatible) for the prediction task
        csv_buffer = StringIO()
        processed_input.to_csv(csv_buffer, index=False)
        processed_key = f"processed_reviews/{task_id}.csv"
        s3_client.put_object(
            Bucket="databucket",
            Key=processed_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv"
        )

        logger.info(f"Preprocessed data saved to MinIO at {processed_key}")
        return processed_key

    except Exception as e:
        logger.error(f"Error in preprocessing review: {str(e)}")
        raise


def predict_review_task(task_id: str) -> Dict[str, Any]:
    """Load preprocessed review data and make a prediction"""
    try:
        # Load the preprocessed review data from MinIO (S3-compatible)
        processed_key = f"processed_reviews/{task_id}.csv"
        response = s3_client.get_object(Bucket="databucket", Key=processed_key)
        csv_content = response["Body"].read().decode("utf-8")
        
        # Load the CSV content into a pandas DataFrame
        csv_buffer = StringIO(csv_content)
        processed_data = pd.read_csv(csv_buffer)
        # Load the latest model from MLflow
        latest_model = mlflow.sklearn.load_model(
            model_uri = f"models:/IsolationForest/latest"
        )
        result_key=f"predictions/output/prediction.json"
        # Make prediction
        logger.info(processed_data)
        prediction = latest_model.predict(processed_data.values)
        s3_client.put_object(
            Bucket="databucket",
            Key=result_key,
            Body=json.dumps(
                {
                    "task_id": task_id,
                    "prediction": prediction,
                    "model_version": latest_model.metadata.version,
                }
            ),
            ContentType="application/json",
        )
        return {
            "task_id": task_id,
            "prediction": prediction,
            "model_version": latest_model.metadata.version,
        }

    except Exception as e:
        logger.error(f"Error in predicting review: {str(e)}")
        raise


# DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "review_processing_pipeline",
    default_args=default_args,
    description="Preprocess review data and make predictions",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "review_processing", "prediction"],
)


def preprocess_review(**context):
    """Airflow task to preprocess review"""
    dag_conf = context["dag_run"].conf
    review_data = dag_conf["review_data"]
    task_id = dag_conf["task_id"]
    return preprocess_review_task(review_data, task_id)


def predict_review(**context):
    """Airflow task to predict review"""
    task_id = context["dag_run"].conf["task_id"]
    return predict_review_task(task_id)


# Define the start and end tasks
begin = EmptyOperator(task_id="begin", dag=dag)
end = EmptyOperator(task_id="end", dag=dag)

# Define tasks
preprocess_task = PythonOperator(
    task_id="preprocess_review",
    python_callable=preprocess_review,
    provide_context=True,
    dag=dag,
)

predict_task = PythonOperator(
    task_id="predict_review",
    python_callable=predict_review,
    provide_context=True,
    dag=dag,
)

# Set task dependencies
begin >> preprocess_task >> predict_task >> end
