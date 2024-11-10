import mlflow
import mlflow.sklearn
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sentence_transformers import SentenceTransformer
import boto3
import os
import logging
from colorlog import ColoredFormatter

# Silence Git warnings
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

class DataPreprocessor:
    def __init__(self):
        self.logger = self.setup_logger()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.s3_client = self.initialize_s3_client()
        self.bucket_name = "databucket"
        self.s3_key_train = "data/embeddings/amazon_reviews_embeddings/train.csv"
        self.s3_key_test = "data/embeddings/amazon_reviews_embeddings/test.csv"
        self.artifact_location = os.getenv("ARTIFACT_BUCKET", "")

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

    def initialize_s3_client(self):
        self.logger.info("Initializing S3 client...")
        return boto3.client(
            "s3",
            endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

    def check_data_in_s3(self):
        try:
            self.logger.info("Checking if train.csv and test.csv are present in S3...")
            self.s3_client.head_object(Bucket=self.bucket_name, Key=self.s3_key_train)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=self.s3_key_test)
            self.logger.info(
                "Both train.csv and test.csv found in S3. Skipping preprocessing."
            )
            return True
        except self.s3_client.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.logger.info(
                    "One or both files not found in S3. Starting preprocessing."
                )
                return False
            else:
                self.logger.error("Error checking files in S3: %s", e)
                raise

    def load_data(self):
        try:
            self.logger.info("Starting to load Amazon Reviews dataset...")
            reviews = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                "raw_review_All_Beauty",
                trust_remote_code=True,
            ).get("full")
            metadata = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                "raw_meta_All_Beauty",
                split="full",
                trust_remote_code=True,
            )
            self.logger.info("Data loaded successfully!")
            return pd.DataFrame(metadata), pd.DataFrame(reviews)
        except Exception as e:
            self.logger.error("Failed to load data: %s", e)
            raise

    def preprocess_data(self, metadata, reviews):
        try:
            self.logger.info("Merging reviews and metadata on parent_asin...")
            combined_df = pd.merge(metadata, reviews, on="parent_asin", how="inner")
            self.logger.info("Data merged successfully!")

            # Standardize numerical and encode categorical features
            numerical_features = ["rating", "helpful_vote"]
            categorical_features = ["verified_purchase"]

            self.logger.info("Starting feature scaling for numerical features...")
            scaler = StandardScaler()
            combined_df[numerical_features] = scaler.fit_transform(
                combined_df[numerical_features]
            )
            self.logger.info("Numerical features scaled.")

            self.logger.info("Encoding categorical features...")
            label_encoder = LabelEncoder()
            for col in categorical_features:
                combined_df[col] = label_encoder.fit_transform(
                    combined_df[col].astype(str)
                )
            self.logger.info("Categorical features encoded.")

            # Log scaler and label encoder to MLflow
            experiment_name = "Data Preprocessing Experiment"
            if mlflow.get_experiment_by_name(experiment_name) is None:
                mlflow.create_experiment(
                    name=experiment_name, artifact_location=self.artifact_location
                )
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run(run_name="Data Preprocessing Artifacts"):
                # Define input examples for model signature inference
                input_example_scaler = combined_df[numerical_features].iloc[[0]].values
                input_example_label_encoder = [1]

                # Register and log the scaler model
                mlflow.sklearn.log_model(
                    scaler,
                    artifact_path="scaler",
                    registered_model_name="StandardScaler",
                    input_example=input_example_scaler
                )
                self.logger.info("Scaler model registered and logged to MLflow.")

                # Register and log the label encoder model
                mlflow.sklearn.log_model(
                    label_encoder,
                    artifact_path="label_encoder",
                    registered_model_name="LabelEncoder",
                    input_example=input_example_label_encoder
                )
                self.logger.info("LabelEncoder model registered and logged to MLflow.")

            return combined_df, numerical_features, categorical_features
        except Exception as e:
            self.logger.error("Error during preprocessing: %s", e)
            raise

    def generate_text_embeddings(
        self, combined_df, numerical_features, categorical_features
    ):
        """
        Generate text embeddings and combine them with other features.
        """
        self.logger.info("Generating text embeddings...")
        text_embeddings = self.model.encode(
            combined_df["text"].tolist(), show_progress_bar=True
        )
        embedded_df = pd.DataFrame(text_embeddings)
        other_features = combined_df[
            numerical_features + categorical_features
        ].reset_index(drop=True)
        processed_df = pd.concat([embedded_df, other_features], axis=1)
        self.logger.info("Text embeddings generated and combined with other features.")
        return processed_df

    def split_and_save_data(self, combined_df):
        try:
            self.logger.info("Splitting data into training and testing sets...")
            X_train, X_test = train_test_split(
                combined_df, test_size=0.2, random_state=42
            )
            self.logger.info("Data split successfully.")

            local_dir = "data"
            os.makedirs(local_dir, exist_ok=True)
            train_file_path = os.path.join(local_dir, "train.csv")
            test_file_path = os.path.join(local_dir, "test.csv")

            self.logger.info("Saving training and testing data locally...")
            X_train.to_csv(train_file_path, index=False)
            X_test.to_csv(test_file_path, index=False)
            self.logger.info(
                "Data saved locally at %s and %s", train_file_path, test_file_path
            )
            return train_file_path, test_file_path
        except Exception as e:
            self.logger.error("Error saving data: %s", e)
            raise

    def upload_to_s3(self, file_paths):
        try:
            self.logger.info("Uploading data to S3...")
            for file_path, s3_key in zip(
                file_paths, [self.s3_key_train, self.s3_key_test]
            ):
                with open(file_path, "rb") as data:
                    self.s3_client.put_object(
                        Bucket=self.bucket_name, Key=s3_key, Body=data
                    )
                    self.logger.info(f"Data {s3_key} uploaded to S3 successfully!")
        except Exception as e:
            self.logger.error("Error uploading to S3: %s", e)
            raise

    def run(self):
        metadata, reviews = self.load_data()
        combined_df, numerical_features, categorical_features = self.preprocess_data(
            metadata, reviews
        )
        if not self.check_data_in_s3():
            processed_df = self.generate_text_embeddings(
                combined_df, numerical_features, categorical_features
            )
            file_paths = self.split_and_save_data(processed_df)
            self.upload_to_s3(file_paths)
            self.logger.info("Pipeline completed successfully.")
        else:
            self.logger.info(
                "Skipping preprocessing as the train.csv and test.csv files already exist in S3."
            )


# Initialize and run the pipeline
def preprocess():
    data_preprocessor = DataPreprocessor()
    data_preprocessor.run()
