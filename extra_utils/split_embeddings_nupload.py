import pandas as pd
from sklearn.model_selection import train_test_split
import boto3
import os
from io import StringIO


def download_from_minio(s3_client, bucket_name, s3_key):
    """Download a CSV file from MinIO and return it as a pandas DataFrame."""
    try:
        print(f"Downloading {s3_key} from MinIO bucket {bucket_name}...")
        csv_obj = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        data = pd.read_csv(csv_obj["Body"])
        print("Download and data load completed.")
        return data
    except Exception as e:
        print(f"Error downloading from MinIO: {e}")
        raise


def upload_to_minio(s3_client, bucket_name, s3_key, data):
    """Upload a DataFrame to MinIO as a CSV file."""
    try:
        print(f"Uploading {s3_key} to MinIO bucket {bucket_name}...")
        csv_buffer = StringIO()
        data.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
        print("Upload completed.")
    except Exception as e:
        print(f"Error uploading to MinIO: {e}")
        raise


def train_test_val_split_and_save(
    s3_client,
    bucket_name,
    input_key,
    output_dir,
    test_size=0.3,
    val_size=0.3,
    random_state=42,
):
    """
    Downloads data from MinIO, performs a train-test-validation split, and uploads the split data to MinIO.

    Parameters:
    - s3_client: Boto3 S3 client configured for MinIO.
    - bucket_name (str): Name of the MinIO bucket.
    - input_key (str): Key for the input CSV file in MinIO.
    - output_dir (str): Directory path in MinIO to save train, test, and validation CSV files.
    - test_size (float): Proportion of the data to use as the test set (final size).
    - val_size (float): Proportion of the data to use as the validation set (final size).
    - random_state (int): Random state for reproducibility.
    """
    # Download the input file from MinIO
    data = download_from_minio(s3_client, bucket_name, input_key)

    # Perform initial train + (test + validation) split
    print("Performing initial train-test_val split...")
    train_data, test_val_data = train_test_split(
        data, test_size=(test_size + val_size), random_state=random_state
    )

    # Further split test_val_data into test and validation sets
    test_ratio = test_size / (test_size + val_size)
    print("Splitting test_val into test and validation sets...")
    test_data, val_data = train_test_split(
        test_val_data, test_size=test_ratio, random_state=random_state
    )
    print("Train-test-validation split completed.")

    # Define S3 keys for the train, test, and validation sets
    train_key = os.path.join(output_dir, "train.csv")
    test_key = os.path.join(output_dir, "test.csv")
    val_key = os.path.join(output_dir, "val.csv")

    # Upload train, test, and validation sets to MinIO
    upload_to_minio(s3_client, bucket_name, train_key, train_data)
    upload_to_minio(s3_client, bucket_name, test_key, test_data)
    upload_to_minio(s3_client, bucket_name, val_key, val_data)


# Configure MinIO connection
s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
)

# Define MinIO bucket and paths
bucket_name = "databucket"
input_key = "data/embeddings/data_for_training_combined.csv"
output_directory = "data/embeddings/amazon_reviews_embeddings/"

# Run the train-test-validation split and save function
train_test_val_split_and_save(s3_client, bucket_name, input_key, output_directory)
print("END !!!")
