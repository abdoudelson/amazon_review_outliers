from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from ml_module.preprocess import preprocess
from ml_module.train import train_and_log_models
from ml_module.create_minio_bucket import create_minio_bucket
from airflow.operators.empty import EmptyOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from datetime import timedelta
## AWS S3 parameters
AWS_CONN_ID = "aws_default"
DATA_BUCKET_NAME = "data_bucket"
MLFLOW_ARTIFACT_BUCKET = "ml_artifact_bucket"

## MLFlow parameters
MLFLOW_CONN_ID = "mlflow_default"
EXPERIMENT_NAME = "Possum_tails"
MAX_RESULTS_MLFLOW_LIST_EXPERIMENTS = 1000


XCOM_BUCKET = "localxcom"


# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

# Define the DAG
with DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='ML pipeline DAG for preprocessing, training, and saving the model',
    schedule_interval=None,
    start_date=days_ago(1),
    dagrun_timeout=timedelta(minutes=60),
) as dag:

    # Task to create the MinIO bucket for MLflow artifacts
    create_minio_bucket = PythonOperator(
        task_id='create_minio_bucket',
        python_callable=create_minio_bucket
    )

    # Task for data preprocessing
    preprocess_task = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess
    )

    # Task for training models
    train_task = PythonOperator(
        task_id='train_models',
        python_callable=train_and_log_models
    )


    # Define task dependencies
    create_minio_bucket >> preprocess_task >> train_task