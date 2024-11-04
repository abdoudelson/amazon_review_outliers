import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_model(model_name):
    # Load test data
    X_test = pd.read_csv("data/test_data.csv")

    # Load model from MLflow
    model = mlflow.sklearn.load_model(f"models:/{model_name}_model/latest")

    # Predict outliers
    predictions = model.predict(X_test)
    # Assume ground truth if available for evaluation
    y_true = [1] * len(predictions)  # Example: All assumed to be normal

    # Evaluate using accuracy as a simple metric
    accuracy = accuracy_score(y_true, predictions)
    print(f"{model_name} accuracy on test set: {accuracy}")

    # Log metrics to MLflow
    with mlflow.start_run(run_name=f"{model_name}_evaluation"):
        mlflow.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    evaluate_model("IsolationForest")
    evaluate_model("LocalOutlierFactor")
