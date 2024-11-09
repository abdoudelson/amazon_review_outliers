from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any
import requests
import json
import os
import time
from datetime import datetime
import logging
import base64
from datetime import timezone
from requests.auth import HTTPBasicAuth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Review Analysis API")

# Airflow API configuration
AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://airflow-webserver:8080/api/v1")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")


class ReviewInput(BaseModel):
    text: str
    rating: int
    verified_purchase: bool


def get_airflow_headers():
    """Generate headers for Airflow API requests with basic auth"""
    logger.info(f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}")
    logger.info("Creds")
    auth = base64.b64encode(f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}".encode()).decode(
        "utf-8"
    )
    return {"Content-Type": "application/json", "Authorization": f"Basic {auth}"}


def trigger_airflow_dag(review_data: Dict[str, Any], task_id: str):
    """Trigger Airflow DAG and return the run_id for monitoring"""
    dag_id = "review_processing_pipeline"
    endpoint = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
    logical_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    payload = {
        "conf": {"review_data": review_data, "task_id": task_id},
        "logical_date": logical_date,
    }

    response = requests.post(
        endpoint,
        json=payload,
        headers={"Content-Type": "application/json"},
        auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
    )
    if response.status_code == 200:
        return response.json()["dag_run_id"]
    else:
        logger.error(
            f"Failed to trigger DAG: Status {response.status_code}, Response {response.text}"
        )
        raise HTTPException(status_code=500, detail="Failed to trigger Airflow DAG.")


def wait_for_dag_completion(dag_run_id: str, dag_id: str, timeout: int = 300):
    """Poll Airflow for DAG completion status"""
    start_time = time.time()
    endpoint = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns/{dag_run_id}"

    while time.time() - start_time < timeout:
        response = requests.get(
            endpoint,
            headers={"Content-Type": "application/json"},
            auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
        )
        if response.status_code == 200:
            status = response.json()["state"]
            if status == "success":
                return response.json()["conf"]["prediction"]
            elif status == "failed":
                logger.error("DAG run failed.")
                raise HTTPException(status_code=500, detail="DAG run failed.")
        else:
            logger.error(
                f"Failed to fetch DAG status: Status {response.status_code}, Response {response.text}"
            )
            raise HTTPException(status_code=500, detail="Failed to fetch DAG status.")

        time.sleep(5)  # Wait 5 seconds before checking again

    raise HTTPException(status_code=500, detail="DAG run timed out.")


@app.get("/health")
async def health_check():
    """Endpoint to check Airflow API connectivity"""
    endpoint = f"{AIRFLOW_API_URL}/health"
    response = requests.get(
        endpoint,
        headers={"Content-Type": "application/json"},
        auth=HTTPBasicAuth(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
    )
    if response.status_code == 200:
        return {"status": "Airflow API is reachable"}
    else:
        logger.error(
            f"Airflow health check failed: Status {response.status_code}, Response {response.text}"
        )
        raise HTTPException(status_code=500, detail="Airflow API is not reachable")


@app.post("/predict")
async def predict(review: ReviewInput, background_tasks: BackgroundTasks):
    """Endpoint to trigger Airflow DAG for prediction and wait for response"""
    task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    review_data = review.dict()
    logger.info(f"Review recieved {review}")
    # Trigger Airflow DAG
    try:
        dag_run_id = trigger_airflow_dag(review_data, task_id)
    except HTTPException as e:
        logger.error(f"Error triggering Airflow DAG: {e.detail}")
        raise e

    # Wait for DAG completion and retrieve prediction
    try:
        prediction = wait_for_dag_completion(dag_run_id, "review_processing_pipeline")
    except HTTPException as e:
        logger.error(f"Error during DAG completion: {e.detail}")
        raise e

    return {"task_id": task_id, "prediction": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
