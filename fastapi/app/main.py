from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
from enum import Enum
import requests
import json
import os
import time
from datetime import datetime
import logging
import boto3
from requests.auth import HTTPBasicAuth
from functools import lru_cache

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseModel):
    """Application settings with validation"""
    AIRFLOW_API_URL: str = Field(default="http://airflow-webserver:8080/api/v1")
    AIRFLOW_USERNAME: str = Field(default="airflow")
    AIRFLOW_PASSWORD: str = Field(default="airflow")
    MINIO_ENDPOINT: str = Field(default="http://minio:9000")
    MINIO_ACCESS_KEY: str = Field(default="minio")
    MINIO_SECRET_KEY: str = Field(default="minio123")
    MINIO_BUCKET: str = Field(default="databucket")
    DAG_TIMEOUT: int = Field(default=300)
    RESULT_KEY: str = Field(default="predictions/output/prediction.json")

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    """Cached settings to avoid repeated environment variable lookups"""
    return Settings()

class DagStatus(str, Enum):
    """Enum for all possible Airflow DAG run states"""
    SUCCESS = "success"
    FAILED = "failed"
    RUNNING = "running"
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    NONE = "none"
    UP_FOR_RETRY = "up_for_retry"
    UP_FOR_RESCHEDULE = "up_for_reschedule"
    UPSTREAM_FAILED = "upstream_failed"
    SKIPPED = "skipped"
    REMOVED = "removed"

    @classmethod
    def is_terminal_state(cls, state: str) -> bool:
        """Check if the DAG state is terminal (completed or failed)"""
        return state in {cls.SUCCESS.value, cls.FAILED.value, cls.UPSTREAM_FAILED.value, cls.SKIPPED.value, cls.REMOVED.value}
    
    @classmethod
    def is_failure_state(cls, state: str) -> bool:
        """Check if the DAG state indicates failure"""
        return state in {cls.FAILED.value, cls.UPSTREAM_FAILED.value}

class ReviewInput(BaseModel):
    """Review input model with validation"""
    text: str = Field(..., min_length=1, max_length=5000)
    rating: int = Field(..., ge=1, le=5)
    verified_purchase: bool

    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or just whitespace')
        return v

class S3Client:
    """S3/MinIO client wrapper"""
    def __init__(self, settings: Settings):
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.MINIO_ENDPOINT,
            aws_access_key_id=settings.MINIO_ACCESS_KEY,
            aws_secret_access_key=settings.MINIO_SECRET_KEY,
        )
        self.bucket = settings.MINIO_BUCKET

    def get_prediction_result(self, key: str) -> Dict[str, Any]:
        """Retrieve prediction result from S3/MinIO"""
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.error(f"Error retrieving prediction result: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve prediction result"
            )

class AirflowClient:
    """Airflow API client"""
    def __init__(self, settings: Settings):
        self.base_url = settings.AIRFLOW_API_URL
        self.auth = HTTPBasicAuth(settings.AIRFLOW_USERNAME, settings.AIRFLOW_PASSWORD)
        self.headers = {"Content-Type": "application/json"}

    def trigger_dag(self, dag_id: str, review_data: Dict[str, Any], task_id: str) -> str:
        """Trigger an Airflow DAG run"""
        url = f"{self.base_url}/dags/{dag_id}/dagRuns"
        payload = {
            "conf": {
                "review_data": json.dumps(review_data),
                "task_id": task_id
            }
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                auth=self.auth
            )
            response.raise_for_status()
            return response.json()["dag_run_id"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to trigger DAG: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to trigger Airflow DAG: {str(e)}"
            )

    def get_dag_status(self, dag_id: str, dag_run_id: str) -> str:
        """Get DAG run status"""
        url = f"{self.base_url}/dags/{dag_id}/dagRuns/{dag_run_id}"
        try:
            response = requests.get(
                url,
                headers=self.headers,
                auth=self.auth
            )
            response.raise_for_status()
            return response.json()["state"]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get DAG status: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch DAG status: {str(e)}"
            )

    def check_health(self) -> bool:
        """Check Airflow API health"""
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.headers,
                auth=self.auth
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

class ReviewAnalysisAPI:
    """Main API application class"""
    def __init__(self):
        self.app = FastAPI(
            title="Review Analysis API",
            description="API for analyzing product reviews using Airflow pipelines",
            version="1.0.0"
        )
        self.settings = get_settings()
        self.s3_client = S3Client(self.settings)
        self.airflow_client = AirflowClient(self.settings)
        self.setup_routes()

    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            if self.airflow_client.check_health():
                return {"status": "healthy", "message": "Airflow API is reachable"}
            raise HTTPException(
                status_code=503,
                detail="Airflow API is not reachable"
            )

        @self.app.post("/predict")
        async def predict(
            review: ReviewInput,
            background_tasks: BackgroundTasks
        ) -> Dict[str, Any]:
            """Prediction endpoint"""
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Trigger DAG
            dag_run_id = self.airflow_client.trigger_dag(
                "review_processing_pipeline",
                review.dict(),
                task_id
            )

            # Wait for completion
            start_time = time.time()
            while time.time() - start_time < self.settings.DAG_TIMEOUT:
                status = self.airflow_client.get_dag_status(
                    "review_processing_pipeline",
                    dag_run_id
                )
                
                if DagStatus.is_terminal_state(status):
                    if DagStatus.is_failure_state(status):
                        raise HTTPException(
                            status_code=500,
                            detail=f"DAG run failed with status: {status}"
                        )
                    elif status == DagStatus.SUCCESS.value:
                        prediction = self.s3_client.get_prediction_result(
                            self.settings.RESULT_KEY
                        )
                        return {
                            "task_id": task_id,
                            "prediction": prediction,
                            "status": "success"
                        }
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"DAG run ended with unexpected status: {status}"
                        )
                
                logger.info(f"DAG status: {status}, waiting...")
                time.sleep(5)

            raise HTTPException(
                status_code=500,
                detail="DAG run timed out"
            )

# Initialize API
app = ReviewAnalysisAPI().app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)